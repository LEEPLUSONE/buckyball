#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LENET_MAGIC 0x4c4e5438u
#define LENET_CHANNEL_VERSION 3u
#define LAYER_COUNT 5
#define TILE 16
#define MAX_CHANNELS 256
#define MAX_N_PADDED 128
#define MAX_BANK_ROWS 256
#define MAX_CONV_ROWS 576

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t input_count;
  uint32_t layer_count;
  uint32_t sample_count;
} PayloadHeader;

typedef struct {
  uint32_t k;
  uint32_t n;
  uint32_t n_padded;
  uint32_t weight_count;
  uint32_t input_scale_count;
  uint32_t weight_scale_count;
  uint32_t output_scale_count;
  uint32_t alignment_count;
  uint32_t bias_count;
} LayerHeader;

typedef struct {
  uint32_t label;
  uint32_t fp32_class;
  uint32_t quant_class;
  int8_t expected_q[TILE];
} SampleHeader;

typedef struct {
  LayerHeader header;
  int8_t *weight;
  float *input_multiplier;
  float *weight_multiplier;
  float *output_multiplier;
  float *input_step;
  float *output_step;
  float *alignment;
  float *requant;
  int32_t *bias;
} Layer;

_Static_assert(sizeof(PayloadHeader) == 20, "payload header layout changed");
_Static_assert(sizeof(LayerHeader) == 36, "layer header layout changed");
_Static_assert(sizeof(SampleHeader) == 28, "sample header layout changed");

static unsigned long fp2int_issues;
static unsigned long int2fp_issues;
static unsigned long matrix_issues;
static unsigned long mmio_table_loads;
static int verify_numerics = 1;

static float scale_staging[MAX_CHANNELS] __attribute__((aligned(64)));
static float fp_staging[MAX_BANK_ROWS * TILE] __attribute__((aligned(64)));
static float fp_output_staging[MAX_BANK_ROWS * TILE]
    __attribute__((aligned(64)));
static int8_t i8_staging[MAX_BANK_ROWS * TILE] __attribute__((aligned(64)));
static int8_t i8_output_staging[MAX_BANK_ROWS * TILE]
    __attribute__((aligned(64)));
static int32_t i32_staging[MAX_BANK_ROWS * TILE] __attribute__((aligned(64)));
static int32_t i32_output_staging[MAX_BANK_ROWS * TILE]
    __attribute__((aligned(64)));
static int32_t linear_products[TILE * MAX_N_PADDED]
    __attribute__((aligned(64)));
static uint8_t file_staging[4096] __attribute__((aligned(4096)));

static void die(const char *message) {
  fprintf(stderr, "[LeNet/Pebble/channel] %s\n", message);
  exit(1);
}

static void *checked_malloc(size_t bytes) {
  void *result = malloc(bytes);
  if (result == NULL)
    die("out of memory");
  return result;
}

static void *checked_calloc(size_t count, size_t size) {
  void *result = calloc(count, size);
  if (result == NULL)
    die("out of memory");
  return result;
}

static void checked_read(void *destination, size_t size, size_t count,
                         FILE *stream, const char *what) {
  if (size != 0 && count > SIZE_MAX / size)
    die("payload read size overflow");
  size_t bytes = size * count;
  uint8_t *output = (uint8_t *)destination;
  size_t offset = 0;
  // BEMU's fast guest-address path cannot DMA a host read directly across
  // every heap-page boundary.  Read through one page-aligned static page and
  // let ordinary guest stores copy into the final object.
  while (offset < bytes) {
    size_t chunk = bytes - offset;
    if (chunk > sizeof(file_staging))
      chunk = sizeof(file_staging);
    size_t actual = fread(file_staging, 1, chunk, stream);
    if (actual != chunk) {
      fprintf(stderr,
              "[LeNet/Pebble/channel] failed to read %s: got=%lu "
              "expected=%lu offset=%ld\n",
              what, (unsigned long)actual, (unsigned long)chunk, ftell(stream));
      exit(1);
    }
    memcpy(output + offset, file_staging, chunk);
    offset += chunk;
  }
}

static uint32_t fp32_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } converted = {.value = value};
  return converted.bits;
}

static int32_t rne_i32(float value) {
  int32_t result;
  __asm__ volatile("fcvt.w.s %0, %1, rne" : "=r"(result) : "f"(value));
  return result;
}

static int8_t scalar_quantize(float value, float multiplier) {
  float scaled = value * multiplier;
  if (scaled >= 127.0f)
    return 127;
  if (scaled <= -128.0f)
    return -128;
  return (int8_t)rne_i32(scaled);
}

static int32_t clamp_i32(int64_t value) {
  if (value > INT32_MAX)
    return INT32_MAX;
  if (value < INT32_MIN)
    return INT32_MIN;
  return (int32_t)value;
}

static void load_scale_table(uint32_t owner_bank, const float *scales,
                             size_t count) {
  if (count == 0 || count > MAX_CHANNELS)
    die("MMIO scale table must contain 1..256 channels");
  size_t padded = (count + TILE - 1) / TILE * TILE;
  for (size_t index = 0; index < padded; ++index)
    scale_staging[index] = index < count ? scales[index] : 1.0f;
  bb_mvin_mmio((uintptr_t)scale_staging, 0, padded / 4, 16);
  bb_fence();
  // In Pebble BEMU one bound MMIO region row exposes 1 KiB, exactly 256 FP32
  // entries.  table_offset is relative to this source-bank-owned region.
  bb_mmio_set(owner_bank, 0, 1);
  bb_fence();
  ++mmio_table_loads;
}

static void pebble_fp2int_channel(const float *input, int8_t *output,
                                  int positions, int channels, int stride,
                                  const float *multipliers) {
  if (positions <= 0 || channels <= 0 || channels > MAX_CHANNELS ||
      stride < channels)
    die("invalid FP2INT channel layout");

  bb_mem_alloc(0, 1, 4);
  bb_mem_alloc(1, 1, 1);
  load_scale_table(0, multipliers, channels);
  for (int channel0 = 0; channel0 < channels; channel0 += TILE) {
    int valid_channels =
        channels - channel0 < TILE ? channels - channel0 : TILE;
    for (int position0 = 0; position0 < positions; position0 += MAX_BANK_ROWS) {
      int rows = positions - position0;
      if (rows > MAX_BANK_ROWS)
        rows = MAX_BANK_ROWS;
      memset(fp_staging, 0, (size_t)rows * TILE * sizeof(float));
      for (int row = 0; row < rows; ++row)
        for (int lane = 0; lane < valid_channels; ++lane)
          fp_staging[row * TILE + lane] =
              input[(position0 + row) * stride + channel0 + lane];
      bb_mvin((uintptr_t)fp_staging, 0, rows, 1);
      bb_fp2int_ex(0, 1, rows, 0, BB_SCALE_PER_CHANNEL, channel0 * 4);
      ++fp2int_issues;
      bb_mvout((uintptr_t)i8_output_staging, 1, rows, 1);
      bb_fence();
      for (int row = 0; row < rows; ++row) {
        for (int lane = 0; lane < valid_channels; ++lane) {
          int channel = channel0 + lane;
          int8_t actual = i8_output_staging[row * TILE + lane];
          output[(position0 + row) * stride + channel] = actual;
          if (verify_numerics) {
            int8_t expected =
                scalar_quantize(input[(position0 + row) * stride + channel],
                                multipliers[channel]);
            if (actual != expected)
              die("per-channel FP2INT lane/offset mismatch");
          }
        }
      }
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
}

static void pebble_int2fp_channel(const int8_t *input, float *output,
                                  int positions, int channels, int stride,
                                  const float *steps) {
  if (positions <= 0 || channels <= 0 || channels > MAX_CHANNELS ||
      stride < channels)
    die("invalid INT2FP channel layout");

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  load_scale_table(0, steps, channels);
  for (int channel0 = 0; channel0 < channels; channel0 += TILE) {
    int valid_channels =
        channels - channel0 < TILE ? channels - channel0 : TILE;
    for (int position0 = 0; position0 < positions; position0 += MAX_BANK_ROWS) {
      int rows = positions - position0;
      if (rows > MAX_BANK_ROWS)
        rows = MAX_BANK_ROWS;
      memset(i8_staging, 0, (size_t)rows * TILE);
      for (int row = 0; row < rows; ++row)
        for (int lane = 0; lane < valid_channels; ++lane)
          i8_staging[row * TILE + lane] =
              input[(position0 + row) * stride + channel0 + lane];
      bb_mvin((uintptr_t)i8_staging, 0, rows, 1);
      bb_int2fp_scale_ex(0, 1, rows, 0, BB_SCALE_PER_CHANNEL, channel0 * 4);
      ++int2fp_issues;
      // MVOUT uses accumulator-depth encoding for a four-group bank once the
      // logical row count exceeds the 16-row matrix tile.
      int mvout_depth = rows > TILE ? rows * 4 : rows;
      bb_mvout((uintptr_t)fp_output_staging, 1, mvout_depth, 1);
      bb_fence();
      for (int row = 0; row < rows; ++row)
        for (int lane = 0; lane < valid_channels; ++lane) {
          int channel = channel0 + lane;
          float actual = fp_output_staging[row * TILE + lane];
          output[(position0 + row) * stride + channel] = actual;
          if (verify_numerics) {
            float expected =
                (float)input[(position0 + row) * stride + channel] *
                steps[channel];
            if (fp32_bits(actual) != fp32_bits(expected)) {
              printf("[LeNet/Pebble/channel] int2fp position=%d channel=%d "
                     "input=%d step=%f step_bits=0x%08x got=%f "
                     "got_bits=0x%08x expected=%f expected_bits=0x%08x\n",
                     position0 + row, channel,
                     input[(position0 + row) * stride + channel],
                     steps[channel], fp32_bits(steps[channel]), actual,
                     fp32_bits(actual), expected, fp32_bits(expected));
              die("per-channel INT2FP lane/offset mismatch");
            }
          }
        }
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
}

static int8_t *rescale_channel(const int8_t *input, int positions, int channels,
                               const float *previous_steps,
                               const float *next_multipliers) {
  size_t count = (size_t)positions * channels;
  float *real = (float *)checked_malloc(count * sizeof(float));
  int8_t *output = (int8_t *)checked_malloc(count);
  pebble_int2fp_channel(input, real, positions, channels, channels,
                        previous_steps);
  pebble_fp2int_channel(real, output, positions, channels, channels,
                        next_multipliers);
  free(real);
  return output;
}

static int32_t *pebble_matmul_slice(const int8_t *a, const int8_t *b, int m,
                                    int n, int k_total, int b_stride,
                                    int k_begin, int k_count) {
  int32_t *output = (int32_t *)checked_calloc((size_t)m * n, sizeof(int32_t));
  int8_t a_tile[TILE * TILE] __attribute__((aligned(64)));
  int8_t b_tile[TILE * TILE] __attribute__((aligned(64)));
  int32_t c_tile[TILE * TILE] __attribute__((aligned(64)));

  if (k_begin < 0 || k_count <= 0 || k_begin + k_count > k_total)
    die("invalid MATRIX reduction slice");
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  for (int m0 = 0; m0 < m; m0 += TILE) {
    int mt = m - m0 < TILE ? m - m0 : TILE;
    for (int n0 = 0; n0 < n; n0 += TILE) {
      int nt = n - n0 < TILE ? n - n0 : TILE;
      for (int local_k0 = 0; local_k0 < k_count; local_k0 += TILE) {
        int kt = k_count - local_k0 < TILE ? k_count - local_k0 : TILE;
        int k0 = k_begin + local_k0;
        memset(a_tile, 0, sizeof(a_tile));
        memset(b_tile, 0, sizeof(b_tile));
        memset(c_tile, 0, sizeof(c_tile));
        for (int row = 0; row < mt; ++row)
          for (int kk = 0; kk < kt; ++kk)
            a_tile[row * TILE + kk] = a[(m0 + row) * k_total + k0 + kk];
        for (int kk = 0; kk < kt; ++kk)
          for (int lane = 0; lane < nt; ++lane)
            b_tile[kk * TILE + lane] = b[(k0 + kk) * b_stride + n0 + lane];
        bb_mvin((uintptr_t)a_tile, 0, mt, 1);
        bb_mvin((uintptr_t)b_tile, 1, kt, 1);
        bb_matrix_mnk(0, 1, 2, mt, nt, kt);
        ++matrix_issues;
        bb_mvout((uintptr_t)c_tile, 2, mt, 1);
        bb_fence();
        for (int row = 0; row < mt; ++row)
          for (int lane = 0; lane < nt; ++lane)
            output[(m0 + row) * n + n0 + lane] += c_tile[row * TILE + lane];
      }
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return output;
}

static void align_conv_partial(const int32_t *partial, int64_t *accumulator,
                               int m, int n, const float *ratios) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  load_scale_table(0, ratios, n);
  for (int n0 = 0; n0 < n; n0 += TILE) {
    int nt = n - n0 < TILE ? n - n0 : TILE;
    for (int row0 = 0; row0 < m; row0 += MAX_BANK_ROWS) {
      int rows = m - row0;
      if (rows > MAX_BANK_ROWS)
        rows = MAX_BANK_ROWS;
      memset(i32_staging, 0, (size_t)rows * TILE * sizeof(int32_t));
      for (int row = 0; row < rows; ++row)
        for (int lane = 0; lane < nt; ++lane)
          i32_staging[row * TILE + lane] =
              partial[(row0 + row) * n + n0 + lane];
      // The legacy INT32/FP32 col=1 conversion layout stores one 16-lane
      // logical row as four 128-bit SRAM/DMA rows.
      bb_mvin((uintptr_t)i32_staging, 0, rows * 4, 1);
      bb_int2fp_scale_ex(0, 1, rows, 0, BB_SCALE_PER_CHANNEL, n0 * 4);
      ++int2fp_issues;
      bb_fp2int_ex(1, 2, rows, fp32_bits(1.0f), BB_SCALE_PER_TENSOR, 0);
      ++fp2int_issues;
      bb_mvout((uintptr_t)i32_output_staging, 2, rows * 4, 1);
      bb_fence();
      for (int row = 0; row < rows; ++row)
        for (int lane = 0; lane < nt; ++lane) {
          int channel = n0 + lane;
          int32_t aligned = i32_output_staging[row * TILE + lane];
          accumulator[(row0 + row) * n + channel] += aligned;
          if (verify_numerics) {
            int32_t expected = rne_i32(
                (float)partial[(row0 + row) * n + channel] * ratios[channel]);
            if (aligned != expected) {
              printf("[LeNet/Pebble/channel] align row=%d channel=%d "
                     "partial=%d ratio=%f bits=0x%08x got=%d expected=%d\n",
                     row0 + row, channel, partial[(row0 + row) * n + channel],
                     ratios[channel], fp32_bits(ratios[channel]), aligned,
                     expected);
              die("per-channel accumulator alignment mismatch");
            }
          }
        }
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
}

static int64_t *pebble_linear_channel(const int8_t *input, const Layer *layer) {
  int k = (int)layer->header.k;
  int n = (int)layer->header.n;
  int n_padded = (int)layer->header.n_padded;
  int64_t *accumulator = (int64_t *)checked_calloc((size_t)n, sizeof(int64_t));
  int8_t a_tile[TILE * TILE] __attribute__((aligned(64)));
  int8_t b_tile[TILE * TILE] __attribute__((aligned(64)));
  int32_t c_tile[TILE * TILE] __attribute__((aligned(64)));

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mem_alloc(3, 1, 1);
  bb_mem_alloc(4, 1, 1);
  bb_mem_alloc(5, 1, 1);
  for (int k0 = 0; k0 < k; k0 += TILE) {
    int kt = k - k0 < TILE ? k - k0 : TILE;
    memset(linear_products, 0, sizeof(linear_products));
    for (int n0 = 0; n0 < n; n0 += TILE) {
      int nt = n - n0 < TILE ? n - n0 : TILE;
      memset(a_tile, 0, sizeof(a_tile));
      memset(b_tile, 0, sizeof(b_tile));
      memset(c_tile, 0, sizeof(c_tile));
      for (int row = 0; row < kt; ++row) {
        a_tile[row * TILE + row] = input[k0 + row];
        for (int lane = 0; lane < nt; ++lane)
          b_tile[row * TILE + lane] =
              layer->weight[(k0 + row) * n_padded + n0 + lane];
      }
      bb_mvin((uintptr_t)a_tile, 0, kt, 1);
      bb_mvin((uintptr_t)b_tile, 1, kt, 1);
      bb_matrix_mnk(0, 1, 2, kt, nt, kt);
      ++matrix_issues;
      bb_mvout((uintptr_t)c_tile, 2, kt, 1);
      bb_fence();
      for (int row = 0; row < kt; ++row)
        for (int lane = 0; lane < nt; ++lane)
          linear_products[row * MAX_N_PADDED + n0 + lane] =
              c_tile[row * TILE + lane];
    }

    for (int row = 0; row < kt; ++row) {
      const float *ratios = layer->alignment + (size_t)(k0 + row) * n;
      load_scale_table(3, ratios, n);
      for (int n0 = 0; n0 < n; n0 += TILE) {
        int nt = n - n0 < TILE ? n - n0 : TILE;
        memset(i32_staging, 0, TILE * sizeof(int32_t));
        for (int lane = 0; lane < nt; ++lane)
          i32_staging[lane] = linear_products[row * MAX_N_PADDED + n0 + lane];
        bb_mvin((uintptr_t)i32_staging, 3, 4, 1);
        bb_int2fp_scale_ex(3, 4, 1, 0, BB_SCALE_PER_CHANNEL, n0 * 4);
        ++int2fp_issues;
        bb_fp2int_ex(4, 5, 1, fp32_bits(1.0f), BB_SCALE_PER_TENSOR, 0);
        ++fp2int_issues;
        bb_mvout((uintptr_t)i32_output_staging, 5, 4, 1);
        bb_fence();
        for (int lane = 0; lane < nt; ++lane) {
          int channel = n0 + lane;
          int32_t aligned = i32_output_staging[lane];
          accumulator[channel] += aligned;
          if (verify_numerics) {
            int32_t expected =
                rne_i32((float)linear_products[row * MAX_N_PADDED + channel] *
                        ratios[channel]);
            if (aligned != expected)
              die("linear per-channel alignment mismatch");
          }
        }
      }
    }
  }
  for (int channel = 0; channel < n; ++channel)
    accumulator[channel] += layer->bias[channel];
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  bb_mem_release(4);
  bb_mem_release(5);
  return accumulator;
}

static int8_t *pebble_requantize_channel(const int64_t *accumulator, int m,
                                         const Layer *layer) {
  int n = (int)layer->header.n;
  int8_t *output = (int8_t *)checked_malloc((size_t)m * n);
  bb_mem_alloc(0, 1, 4);
  bb_mem_alloc(1, 1, 1);
  load_scale_table(0, layer->requant, n);
  for (int n0 = 0; n0 < n; n0 += TILE) {
    int nt = n - n0 < TILE ? n - n0 : TILE;
    for (int row0 = 0; row0 < m; row0 += MAX_BANK_ROWS) {
      int rows = m - row0;
      if (rows > MAX_BANK_ROWS)
        rows = MAX_BANK_ROWS;
      memset(i32_staging, 0, (size_t)rows * TILE * sizeof(int32_t));
      for (int row = 0; row < rows; ++row)
        for (int lane = 0; lane < nt; ++lane) {
          int channel = n0 + lane;
          i32_staging[row * TILE + lane] =
              clamp_i32(accumulator[(row0 + row) * n + channel] +
                        (m == 1 ? 0 : layer->bias[channel]));
        }
      bb_mvin((uintptr_t)i32_staging, 0, rows, 1);
      bb_int_convert_ex(0, 1, rows, BB_INT_OUTPUT_INT8, 0, BB_SCALE_PER_CHANNEL,
                        n0 * 4);
      ++int2fp_issues;
      bb_mvout((uintptr_t)i8_output_staging, 1, rows, 1);
      bb_fence();
      for (int row = 0; row < rows; ++row)
        for (int lane = 0; lane < nt; ++lane) {
          int channel = n0 + lane;
          int8_t actual = i8_output_staging[row * TILE + lane];
          output[(row0 + row) * n + channel] = actual;
          if (verify_numerics) {
            int8_t expected = scalar_quantize(
                (float)i32_staging[row * TILE + lane], layer->requant[channel]);
            if (actual != expected)
              die("requant table offset/lane mismatch");
          }
        }
    }
  }
  bb_mem_release(0);
  bb_mem_release(1);
  return output;
}

static int8_t *im2col(const int8_t *input, int height, int width, int channels,
                      int kernel) {
  int output_h = height - kernel + 1;
  int output_w = width - kernel + 1;
  int k = channels * kernel * kernel;
  int8_t *result = (int8_t *)checked_malloc((size_t)output_h * output_w * k);
  int row = 0;
  for (int oh = 0; oh < output_h; ++oh)
    for (int ow = 0; ow < output_w; ++ow) {
      int column = 0;
      for (int channel = 0; channel < channels; ++channel)
        for (int kh = 0; kh < kernel; ++kh)
          for (int kw = 0; kw < kernel; ++kw)
            result[row * k + column++] =
                input[((oh + kh) * width + ow + kw) * channels + channel];
      ++row;
    }
  return result;
}

static int8_t *run_conv(const int8_t *input, int height, int width,
                        int channels, const Layer *layer) {
  int output_h = height - 5 + 1;
  int output_w = width - 5 + 1;
  int m = output_h * output_w;
  int kernel_area = (int)layer->header.k / channels;
  int64_t *accumulator =
      (int64_t *)checked_calloc((size_t)m * layer->header.n, sizeof(int64_t));
  int8_t *columns = im2col(input, height, width, channels, 5);
  for (int channel = 0; channel < channels; ++channel) {
    int32_t *partial = pebble_matmul_slice(
        columns, layer->weight, m, (int)layer->header.n, (int)layer->header.k,
        (int)layer->header.n_padded, channel * kernel_area, kernel_area);
    align_conv_partial(partial, accumulator, m, (int)layer->header.n,
                       layer->alignment + (size_t)channel * layer->header.n);
    free(partial);
  }
  int8_t *output = pebble_requantize_channel(accumulator, m, layer);
  free(columns);
  free(accumulator);
  return output;
}

static int8_t *run_linear(const int8_t *input, const Layer *layer) {
  int64_t *accumulator = pebble_linear_channel(input, layer);
  int8_t *output = pebble_requantize_channel(accumulator, 1, layer);
  free(accumulator);
  return output;
}

static void relu_int8(int8_t *values, size_t count) {
  for (size_t index = 0; index < count; ++index)
    if (values[index] < 0)
      values[index] = 0;
}

static int8_t *maxpool2(const int8_t *input, int height, int width,
                        int channels) {
  int output_h = height / 2;
  int output_w = width / 2;
  int8_t *output =
      (int8_t *)checked_malloc((size_t)output_h * output_w * channels);
  for (int oh = 0; oh < output_h; ++oh)
    for (int ow = 0; ow < output_w; ++ow)
      for (int channel = 0; channel < channels; ++channel) {
        int8_t maximum = -128;
        for (int kh = 0; kh < 2; ++kh)
          for (int kw = 0; kw < 2; ++kw) {
            int8_t value =
                input[((oh * 2 + kh) * width + ow * 2 + kw) * channels +
                      channel];
            if (value > maximum)
              maximum = value;
          }
        output[(oh * output_w + ow) * channels + channel] = maximum;
      }
  return output;
}

static float *flatten_nchw_float(const float *nhwc, int height, int width,
                                 int channels) {
  float *output = (float *)checked_malloc((size_t)height * width * channels *
                                          sizeof(float));
  int index = 0;
  for (int channel = 0; channel < channels; ++channel)
    for (int h = 0; h < height; ++h)
      for (int w = 0; w < width; ++w)
        output[index++] = nhwc[(h * width + w) * channels + channel];
  return output;
}

static FILE *load_payload(const char *path, PayloadHeader *payload,
                          Layer layers[LAYER_COUNT]) {
  FILE *stream = fopen(path, "rb");
  if (stream == NULL) {
    fprintf(stderr, "[LeNet/Pebble/channel] cannot open payload: %s\n", path);
    exit(1);
  }
  checked_read(payload, sizeof(*payload), 1, stream, "payload header");
  if (payload->magic != LENET_MAGIC ||
      payload->version != LENET_CHANNEL_VERSION ||
      payload->input_count != 28 * 28 || payload->layer_count != LAYER_COUNT ||
      payload->sample_count == 0)
    die("invalid per-channel evaluation payload header");

  for (int index = 0; index < LAYER_COUNT; ++index) {
    Layer *layer = &layers[index];
    checked_read(&layer->header, sizeof(layer->header), 1, stream,
                 "layer header");
    if (layer->header.weight_count !=
            layer->header.k * layer->header.n_padded ||
        layer->header.weight_scale_count != layer->header.n ||
        layer->header.output_scale_count != layer->header.n ||
        layer->header.alignment_count !=
            layer->header.input_scale_count * layer->header.n ||
        layer->header.bias_count != layer->header.n ||
        layer->header.input_scale_count > MAX_CHANNELS ||
        layer->header.n > MAX_CHANNELS ||
        layer->header.n_padded > MAX_N_PADDED ||
        layer->header.n_padded % TILE != 0)
      die("invalid per-channel layer dimensions");
    printf("[LeNet/Pebble/channel] layer%d k=%u n=%u input_scales=%u "
           "alignment=%u\n",
           index, layer->header.k, layer->header.n,
           layer->header.input_scale_count, layer->header.alignment_count);

    float *weight = (float *)checked_malloc((size_t)layer->header.weight_count *
                                            sizeof(float));
    layer->weight =
        (int8_t *)checked_calloc(layer->header.weight_count, sizeof(int8_t));
    layer->input_multiplier = (float *)checked_malloc(
        (size_t)layer->header.input_scale_count * sizeof(float));
    layer->weight_multiplier = (float *)checked_malloc(
        (size_t)layer->header.weight_scale_count * sizeof(float));
    layer->output_multiplier = (float *)checked_malloc(
        (size_t)layer->header.output_scale_count * sizeof(float));
    layer->input_step = (float *)checked_malloc(
        (size_t)layer->header.input_scale_count * sizeof(float));
    layer->output_step = (float *)checked_malloc(
        (size_t)layer->header.output_scale_count * sizeof(float));
    layer->alignment = (float *)checked_malloc(
        (size_t)layer->header.alignment_count * sizeof(float));
    layer->requant = (float *)checked_malloc(
        (size_t)layer->header.output_scale_count * sizeof(float));
    layer->bias = (int32_t *)checked_malloc((size_t)layer->header.bias_count *
                                            sizeof(int32_t));

    checked_read(weight, sizeof(float), layer->header.weight_count, stream,
                 "layer weight");
    checked_read(layer->input_multiplier, sizeof(float),
                 layer->header.input_scale_count, stream, "input scales");
    checked_read(layer->weight_multiplier, sizeof(float),
                 layer->header.weight_scale_count, stream, "weight scales");
    checked_read(layer->output_multiplier, sizeof(float),
                 layer->header.output_scale_count, stream, "output scales");
    checked_read(layer->input_step, sizeof(float),
                 layer->header.input_scale_count, stream, "input steps");
    checked_read(layer->output_step, sizeof(float),
                 layer->header.output_scale_count, stream, "output steps");
    checked_read(layer->alignment, sizeof(float), layer->header.alignment_count,
                 stream, "alignment scales");
    checked_read(layer->requant, sizeof(float),
                 layer->header.output_scale_count, stream, "requant scales");
    checked_read(layer->bias, sizeof(int32_t), layer->header.bias_count, stream,
                 "quantized bias");
    pebble_fp2int_channel(weight, layer->weight, (int)layer->header.k,
                          (int)layer->header.n, (int)layer->header.n_padded,
                          layer->weight_multiplier);
    free(weight);
  }
  return stream;
}

static int8_t *run_inference(const float *input, Layer layers[LAYER_COUNT]) {
  int8_t *qinput = (int8_t *)checked_malloc(28 * 28);
  pebble_fp2int_channel(input, qinput, 28 * 28, 1, 1,
                        layers[0].input_multiplier);

  int8_t *conv1 = run_conv(qinput, 28, 28, 1, &layers[0]);
  free(qinput);
  relu_int8(conv1, 24 * 24 * 6);
  int8_t *pool1 = maxpool2(conv1, 24, 24, 6);
  free(conv1);
  int8_t *conv2_input = rescale_channel(
      pool1, 12 * 12, 6, layers[0].output_step, layers[1].input_multiplier);
  free(pool1);

  int8_t *conv2 = run_conv(conv2_input, 12, 12, 6, &layers[1]);
  free(conv2_input);
  relu_int8(conv2, 8 * 8 * 16);
  int8_t *pool2 = maxpool2(conv2, 8, 8, 16);
  free(conv2);
  float *fc1_nhwc = (float *)checked_malloc(4 * 4 * 16 * sizeof(float));
  pebble_int2fp_channel(pool2, fc1_nhwc, 4 * 4, 16, 16, layers[1].output_step);
  free(pool2);
  float *fc1_real = flatten_nchw_float(fc1_nhwc, 4, 4, 16);
  free(fc1_nhwc);
  int8_t *fc1_input = (int8_t *)checked_malloc(256);
  pebble_fp2int_channel(fc1_real, fc1_input, 1, 256, 256,
                        layers[2].input_multiplier);
  free(fc1_real);

  int8_t *fc1 = run_linear(fc1_input, &layers[2]);
  free(fc1_input);
  relu_int8(fc1, 120);
  int8_t *fc2_input = rescale_channel(fc1, 1, 120, layers[2].output_step,
                                      layers[3].input_multiplier);
  free(fc1);
  int8_t *fc2 = run_linear(fc2_input, &layers[3]);
  free(fc2_input);
  relu_int8(fc2, 84);
  int8_t *fc3_input = rescale_channel(fc2, 1, 84, layers[3].output_step,
                                      layers[4].input_multiplier);
  free(fc2);
  int8_t *fc3 = run_linear(fc3_input, &layers[4]);
  free(fc3_input);
  return fc3;
}

static void free_layers(Layer layers[LAYER_COUNT]) {
  for (int index = 0; index < LAYER_COUNT; ++index) {
    free(layers[index].weight);
    free(layers[index].input_multiplier);
    free(layers[index].weight_multiplier);
    free(layers[index].output_multiplier);
    free(layers[index].input_step);
    free(layers[index].output_step);
    free(layers[index].alignment);
    free(layers[index].requant);
    free(layers[index].bias);
  }
}

int main(void) {
  const char *payload_path = "lenet_int8_channel_eval_payload.bin";
  PayloadHeader payload;
  Layer layers[LAYER_COUNT];
  memset(layers, 0, sizeof(layers));
  printf("[LeNet/Pebble/channel] loading %s\n", payload_path);
  FILE *stream = load_payload(payload_path, &payload, layers);
  float *input =
      (float *)checked_malloc((size_t)payload.input_count * sizeof(float));
  float logits[TILE] __attribute__((aligned(64)));
  unsigned long fp32_correct = 0;
  unsigned long int8_correct = 0;
  unsigned long prediction_agreement = 0;
  unsigned long python_quant_agreement = 0;

  for (uint32_t sample = 0; sample < payload.sample_count; ++sample) {
    SampleHeader expected;
    checked_read(&expected, sizeof(expected), 1, stream, "sample header");
    checked_read(input, sizeof(float), payload.input_count, stream,
                 "sample input");
    int8_t *q = run_inference(input, layers);
    memset(logits, 0, sizeof(logits));
    pebble_int2fp_channel(q, logits, 1, 10, 10, layers[4].output_step);
    int classification = 0;
    for (int channel = 1; channel < 10; ++channel)
      if (logits[channel] > logits[classification])
        classification = channel;

    if ((uint32_t)classification == expected.label)
      ++int8_correct;
    if (expected.fp32_class == expected.label)
      ++fp32_correct;
    if ((uint32_t)classification == expected.fp32_class)
      ++prediction_agreement;
    if ((uint32_t)classification == expected.quant_class &&
        memcmp(q, expected.expected_q, 10) == 0)
      ++python_quant_agreement;
    else {
      printf("[LeNet/Pebble/channel] Python mismatch sample=%u class=%d "
             "expected=%u\n",
             sample, classification, expected.quant_class);
      for (int channel = 0; channel < 10; ++channel)
        printf("  q[%d]=%d expected=%d logit=%f\n", channel, q[channel],
               expected.expected_q[channel], logits[channel]);
      free(q);
      die("Pebble BEMU differs from Python per-channel reference");
    }
    free(q);
    verify_numerics = 0;
    if ((sample + 1) % 100 == 0 || sample + 1 == payload.sample_count)
      printf("[LeNet/Pebble/channel] progress=%u/%u int8_correct=%lu\n",
             sample + 1, payload.sample_count, int8_correct);
  }
  if (fgetc(stream) != EOF)
    die("payload has unexpected trailing data");
  fclose(stream);

  double samples = (double)payload.sample_count;
  printf("[LeNet/Pebble/channel] samples=%u\n", payload.sample_count);
  printf("[LeNet/Pebble/channel] FP32 Top1=%lu/%u %.4f%%\n", fp32_correct,
         payload.sample_count, 100.0 * fp32_correct / samples);
  printf("[LeNet/Pebble/channel] Pebble INT8 Top1=%lu/%u %.4f%%\n",
         int8_correct, payload.sample_count, 100.0 * int8_correct / samples);
  printf("[LeNet/Pebble/channel] Top1 loss=%.4f percentage-points\n",
         100.0 * ((double)fp32_correct - (double)int8_correct) / samples);
  printf("[LeNet/Pebble/channel] FP32 prediction agreement=%lu/%u %.4f%%\n",
         prediction_agreement, payload.sample_count,
         100.0 * prediction_agreement / samples);
  printf("[LeNet/Pebble/channel] Python INT8 exact agreement=%lu/%u %.4f%%\n",
         python_quant_agreement, payload.sample_count,
         100.0 * python_quant_agreement / samples);
  printf("[LeNet/Pebble/channel] issues: FP2INT=%lu INT2FP/requant=%lu "
         "MATRIX=%lu MMIO-table-load=%lu\n",
         fp2int_issues, int2fp_issues, matrix_issues, mmio_table_loads);

  free(input);
  free_layers(layers);
  return 0;
}
