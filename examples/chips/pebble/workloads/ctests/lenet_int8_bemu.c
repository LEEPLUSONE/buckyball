#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LENET_MAGIC 0x4c4e5438u
#define LENET_VERSION 1u
#define LAYER_COUNT 5
#define TILE 16
#define MAX_BANK_ROWS 256
#define MAX_REQUANT_ROWS 576

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t input_count;
  uint32_t layer_count;
  uint32_t expected_class;
  int8_t expected_q[TILE];
  float expected_logits[TILE];
} PayloadHeader;

typedef struct {
  uint32_t k;
  uint32_t n;
  uint32_t n_padded;
  uint32_t weight_count;
  uint32_t input_multiplier_bits;
  uint32_t weight_multiplier_bits;
  uint32_t output_multiplier_bits;
  uint32_t requant_multiplier_bits;
  uint32_t output_step_bits;
  uint32_t bias_count;
} LayerHeader;

_Static_assert(sizeof(PayloadHeader) == 100, "payload header layout changed");
_Static_assert(sizeof(LayerHeader) == 40, "layer header layout changed");

typedef struct {
  LayerHeader header;
  int8_t *weight;
  int32_t *bias;
} Layer;

static unsigned long fp2int_issues;
static unsigned long int2fp_issues;
static unsigned long matrix_issues;
static int verify_numerics = 1;
static int32_t requant_packed[MAX_REQUANT_ROWS * TILE]
    __attribute__((aligned(64)));
static int8_t requant_output[MAX_REQUANT_ROWS * TILE]
    __attribute__((aligned(64)));

static void print_stats(const char *name, const int8_t *values, size_t count);
static void print_stats_i32(const char *name, const int32_t *values,
                            size_t count);

static void die(const char *message) {
  fprintf(stderr, "[LeNet/Pebble] %s\n", message);
  exit(1);
}

static void *checked_malloc(size_t bytes) {
  void *result = malloc(bytes);
  if (result == NULL)
    die("out of memory");
  return result;
}

static void checked_read(void *destination, size_t size, size_t count,
                         FILE *stream, const char *what) {
  if (fread(destination, size, count, stream) != count) {
    fprintf(stderr, "[LeNet/Pebble] failed to read %s\n", what);
    exit(1);
  }
}

static void pebble_fp2int(const float *input, int8_t *output, size_t count,
                          uint32_t multiplier_bits) {
  if (count == 0 || count % TILE != 0)
    die("FP2INT element count must be a non-zero multiple of 16");

  bb_mem_alloc(0, 1, 4);
  bb_mem_alloc(1, 1, 1);
  size_t offset = 0;
  while (offset < count) {
    size_t rows = (count - offset) / TILE;
    if (rows > MAX_BANK_ROWS)
      rows = MAX_BANK_ROWS;
    bb_mvin((uintptr_t)(input + offset), 0, rows, 1);
    bb_fp2int_ex(0, 1, rows, multiplier_bits, BB_SCALE_PER_TENSOR, 0);
    ++fp2int_issues;
    bb_mvout((uintptr_t)(output + offset), 1, rows, 1);
    bb_fence();
    offset += rows * TILE;
  }
  bb_mem_release(0);
  bb_mem_release(1);
}

static void pebble_rescale_int8(const int8_t *input, int8_t *output,
                                size_t count, uint32_t dequant_bits,
                                uint32_t quant_bits) {
  if (count == 0 || count % TILE != 0)
    die("rescale element count must be a non-zero multiple of 16");

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mem_alloc(2, 1, 1);
  size_t offset = 0;
  while (offset < count) {
    size_t rows = (count - offset) / TILE;
    if (rows > MAX_BANK_ROWS)
      rows = MAX_BANK_ROWS;
    bb_mvin((uintptr_t)(input + offset), 0, rows, 1);
    bb_int2fp_scale_ex(0, 1, rows, dequant_bits, BB_SCALE_PER_TENSOR, 0);
    ++int2fp_issues;
    bb_fp2int_ex(1, 2, rows, quant_bits, BB_SCALE_PER_TENSOR, 0);
    ++fp2int_issues;
    bb_mvout((uintptr_t)(output + offset), 2, rows, 1);
    bb_fence();
    offset += rows * TILE;
  }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
}

static void pebble_int8_to_fp32(const int8_t *input, float *output,
                                size_t count, uint32_t multiplier_bits) {
  if (count == 0 || count % TILE != 0)
    die("INT2FP element count must be a non-zero multiple of 16");
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 4);
  bb_mvin((uintptr_t)input, 0, count / TILE, 1);
  bb_int2fp_scale_ex(0, 1, count / TILE, multiplier_bits, BB_SCALE_PER_TENSOR,
                     0);
  ++int2fp_issues;
  bb_mvout((uintptr_t)output, 1, count / TILE, 1);
  bb_fence();
  bb_mem_release(0);
  bb_mem_release(1);
}

static int8_t scalar_requantize_rne(int32_t value, uint32_t multiplier_bits) {
  union {
    uint32_t bits;
    float value;
  } multiplier = {.bits = multiplier_bits};
  float scaled = (float)value * multiplier.value;
  if (scaled >= 127.0f)
    return 127;
  if (scaled <= -128.0f)
    return -128;
  int32_t rounded;
  __asm__ volatile("fcvt.w.s %0, %1, rne" : "=r"(rounded) : "f"(scaled));
  return (int8_t)rounded;
}

static int32_t *pebble_matmul(const int8_t *a, const int8_t *b, int m, int n,
                              int k, int b_stride) {
  int32_t *output = (int32_t *)calloc((size_t)m * n, sizeof(int32_t));
  if (output == NULL)
    die("out of memory for matmul output");

  int8_t a_tile[TILE * TILE] __attribute__((aligned(64)));
  int8_t b_tile[TILE * TILE] __attribute__((aligned(64)));
  int32_t c_tile[TILE * TILE] __attribute__((aligned(64)));

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);

  for (int m0 = 0; m0 < m; m0 += TILE) {
    int mt = m - m0 < TILE ? m - m0 : TILE;
    for (int n0 = 0; n0 < n; n0 += TILE) {
      int nt = n - n0 < TILE ? n - n0 : TILE;
      for (int k0 = 0; k0 < k; k0 += TILE) {
        int kt = k - k0 < TILE ? k - k0 : TILE;
        memset(a_tile, 0, sizeof(a_tile));
        memset(b_tile, 0, sizeof(b_tile));
        memset(c_tile, 0, sizeof(c_tile));
        for (int i = 0; i < mt; ++i)
          for (int kk = 0; kk < kt; ++kk)
            a_tile[i * TILE + kk] = a[(m0 + i) * k + k0 + kk];
        for (int kk = 0; kk < kt; ++kk)
          for (int j = 0; j < nt; ++j)
            b_tile[kk * TILE + j] = b[(k0 + kk) * b_stride + n0 + j];

        bb_mvin((uintptr_t)a_tile, 0, mt, 1);
        bb_mvin((uintptr_t)b_tile, 1, kt, 1);
        bb_matrix_mnk(0, 1, 2, mt, nt, kt);
        ++matrix_issues;
        bb_mvout((uintptr_t)c_tile, 2, mt, 1);
        bb_fence();
        for (int i = 0; i < mt; ++i)
          for (int j = 0; j < nt; ++j)
            output[(m0 + i) * n + n0 + j] += c_tile[i * TILE + j];
      }
    }
  }

  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);

  if (verify_numerics) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        int32_t expected = 0;
        for (int kk = 0; kk < k; ++kk)
          expected += (int32_t)a[i * k + kk] * (int32_t)b[kk * b_stride + j];
        if (output[i * n + j] != expected) {
          printf("[LeNet/Pebble] MATRIX mismatch m=%d n=%d k=%d at (%d,%d): "
                 "got=%d expected=%d\n",
                 m, n, k, i, j, output[i * n + j], expected);
          die("MATRIX tile accumulation check failed");
        }
      }
    }
  }
  return output;
}

static int8_t *pebble_requantize(const int32_t *accumulator,
                                 const int32_t *bias, int m, int n,
                                 uint32_t multiplier_bits) {
  int8_t *output = (int8_t *)checked_malloc((size_t)m * n);
  if (m > MAX_REQUANT_ROWS)
    die("requant row count exceeds static staging buffer");
  int32_t *packed = requant_packed;
  int8_t *packed_output = requant_output;

  bb_mem_alloc(0, 1, 4);
  bb_mem_alloc(1, 1, 1);
  for (int n0 = 0; n0 < n; n0 += TILE) {
    int nt = n - n0 < TILE ? n - n0 : TILE;
    memset(packed, 0, (size_t)m * TILE * sizeof(int32_t));
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < nt; ++j)
        packed[i * TILE + j] = accumulator[i * n + n0 + j] + bias[n0 + j];

    int row0 = 0;
    while (row0 < m) {
      int rows = m - row0;
      if (rows > MAX_BANK_ROWS)
        rows = MAX_BANK_ROWS;
      bb_mvin((uintptr_t)(packed + row0 * TILE), 0, rows, 1);
      bb_int_convert_ex(0, 1, rows, BB_INT_OUTPUT_INT8, multiplier_bits,
                        BB_SCALE_PER_TENSOR, 0);
      ++int2fp_issues;
      bb_mvout((uintptr_t)(packed_output + row0 * TILE), 1, rows, 1);
      bb_fence();
      row0 += rows;
    }
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < nt; ++j) {
        output[i * n + n0 + j] = packed_output[i * TILE + j];
        if (verify_numerics) {
          int8_t expected =
              scalar_requantize_rne(packed[i * TILE + j], multiplier_bits);
          if (output[i * n + n0 + j] != expected) {
            printf("[LeNet/Pebble] requant mismatch row=%d channel=%d acc=%d "
                   "got=%d expected=%d multiplier=0x%08x\n",
                   i, n0 + j, packed[i * TILE + j], output[i * n + n0 + j],
                   expected, multiplier_bits);
            die("INT32-to-INT8 requant check failed");
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
  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
      int column = 0;
      for (int channel = 0; channel < channels; ++channel)
        for (int kh = 0; kh < kernel; ++kh)
          for (int kw = 0; kw < kernel; ++kw)
            result[row * k + column++] =
                input[((oh + kh) * width + ow + kw) * channels + channel];
      ++row;
    }
  }
  return result;
}

static int8_t *run_conv(const int8_t *input, int height, int width,
                        int channels, const Layer *layer) {
  static int conv_index;
  int output_h = height - 5 + 1;
  int output_w = width - 5 + 1;
  int m = output_h * output_w;
  int8_t *columns = im2col(input, height, width, channels, 5);
  if (conv_index == 0)
    print_stats("conv1_im2col_q", columns, (size_t)m * layer->header.k);
  int32_t *accumulator =
      pebble_matmul(columns, layer->weight, m, layer->header.n, layer->header.k,
                    layer->header.n_padded);
  if (conv_index == 0) {
    int32_t *with_bias = (int32_t *)checked_malloc((size_t)m * layer->header.n *
                                                   sizeof(int32_t));
    for (int i = 0; i < m; ++i)
      for (uint32_t j = 0; j < layer->header.n; ++j)
        with_bias[i * layer->header.n + j] =
            accumulator[i * layer->header.n + j] + layer->bias[j];
    print_stats_i32("conv1_acc", with_bias, (size_t)m * layer->header.n);
    free(with_bias);
  }
  int8_t *output =
      pebble_requantize(accumulator, layer->bias, m, layer->header.n,
                        layer->header.requant_multiplier_bits);
  free(columns);
  free(accumulator);
  ++conv_index;
  return output;
}

static int8_t *run_linear(const int8_t *input, const Layer *layer) {
  int32_t *accumulator = pebble_matmul(input, layer->weight, 1, layer->header.n,
                                       layer->header.k, layer->header.n_padded);
  int8_t *output =
      pebble_requantize(accumulator, layer->bias, 1, layer->header.n,
                        layer->header.requant_multiplier_bits);
  free(accumulator);
  return output;
}

static void relu_int8(int8_t *values, size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (values[i] < 0)
      values[i] = 0;
}

static void print_stats(const char *name, const int8_t *values, size_t count) {
  long sum = 0;
  int minimum = 127;
  int maximum = -128;
  for (size_t i = 0; i < count; ++i) {
    int value = values[i];
    sum += value;
    if (value < minimum)
      minimum = value;
    if (value > maximum)
      maximum = value;
  }
  printf("[LeNet/Pebble] %-20s count=%lu sum=%ld min=%d max=%d first=", name,
         (unsigned long)count, sum, minimum, maximum);
  size_t first_count = count < 8 ? count : 8;
  for (size_t i = 0; i < first_count; ++i)
    printf("%s%d", i ? "," : "", values[i]);
  printf("\n");
}

static void print_stats_i32(const char *name, const int32_t *values,
                            size_t count) {
  long long sum = 0;
  int32_t minimum = INT32_MAX;
  int32_t maximum = INT32_MIN;
  for (size_t i = 0; i < count; ++i) {
    sum += values[i];
    if (values[i] < minimum)
      minimum = values[i];
    if (values[i] > maximum)
      maximum = values[i];
  }
  printf("[LeNet/Pebble] %-20s count=%lu sum=%lld min=%d max=%d first=", name,
         (unsigned long)count, sum, minimum, maximum);
  size_t first_count = count < 8 ? count : 8;
  for (size_t i = 0; i < first_count; ++i)
    printf("%s%d", i ? "," : "", values[i]);
  printf("\n");
}

static int8_t *maxpool2(const int8_t *input, int height, int width,
                        int channels) {
  int output_h = height / 2;
  int output_w = width / 2;
  int8_t *output =
      (int8_t *)checked_malloc((size_t)output_h * output_w * channels);
  for (int oh = 0; oh < output_h; ++oh) {
    for (int ow = 0; ow < output_w; ++ow) {
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
    }
  }
  return output;
}

static int8_t *rescale_with_padding(const int8_t *input, size_t count,
                                    uint32_t dequant_bits,
                                    uint32_t quant_bits) {
  size_t padded_count = (count + TILE - 1) / TILE * TILE;
  int8_t *padded_input = (int8_t *)calloc(padded_count, 1);
  int8_t *padded_output = (int8_t *)checked_malloc(padded_count);
  int8_t *result = (int8_t *)checked_malloc(count);
  if (padded_input == NULL)
    die("out of memory for rescale input");
  memcpy(padded_input, input, count);
  pebble_rescale_int8(padded_input, padded_output, padded_count, dequant_bits,
                      quant_bits);
  memcpy(result, padded_output, count);
  free(padded_input);
  free(padded_output);
  return result;
}

static int8_t *flatten_nchw(const int8_t *nhwc, int height, int width,
                            int channels) {
  int8_t *output = (int8_t *)checked_malloc((size_t)height * width * channels);
  int index = 0;
  for (int channel = 0; channel < channels; ++channel)
    for (int h = 0; h < height; ++h)
      for (int w = 0; w < width; ++w)
        output[index++] = nhwc[(h * width + w) * channels + channel];
  return output;
}

static void load_payload(const char *path, PayloadHeader *payload,
                         float **input, Layer layers[LAYER_COUNT]) {
  static const char *const layer_names[LAYER_COUNT] = {
      "conv1_weight_q", "conv2_weight_q", "fc1_weight_q", "fc2_weight_q",
      "fc3_weight_q"};
  FILE *stream = fopen(path, "rb");
  if (stream == NULL) {
    fprintf(stderr, "[LeNet/Pebble] cannot open payload: %s\n", path);
    exit(1);
  }
  checked_read(payload, sizeof(*payload), 1, stream, "payload header");
  if (payload->magic != LENET_MAGIC || payload->version != LENET_VERSION ||
      payload->input_count != 28 * 28 || payload->layer_count != LAYER_COUNT)
    die("invalid payload header");

  *input =
      (float *)checked_malloc((size_t)payload->input_count * sizeof(float));
  checked_read(*input, sizeof(float), payload->input_count, stream, "input");

  for (int index = 0; index < LAYER_COUNT; ++index) {
    Layer *layer = &layers[index];
    checked_read(&layer->header, sizeof(layer->header), 1, stream,
                 "layer header");
    if (layer->header.weight_count !=
            layer->header.k * layer->header.n_padded ||
        layer->header.bias_count != layer->header.n ||
        layer->header.n_padded % TILE != 0)
      die("invalid layer dimensions in payload");

    float *weight = (float *)checked_malloc((size_t)layer->header.weight_count *
                                            sizeof(float));
    layer->weight =
        (int8_t *)checked_malloc((size_t)layer->header.weight_count);
    layer->bias = (int32_t *)checked_malloc((size_t)layer->header.bias_count *
                                            sizeof(int32_t));
    checked_read(weight, sizeof(float), layer->header.weight_count, stream,
                 "layer weight");
    checked_read(layer->bias, sizeof(int32_t), layer->header.bias_count, stream,
                 "layer bias");
    pebble_fp2int(weight, layer->weight, layer->header.weight_count,
                  layer->header.weight_multiplier_bits);
    print_stats(layer_names[index], layer->weight, layer->header.weight_count);
    long bias_sum = 0;
    for (uint32_t i = 0; i < layer->header.bias_count; ++i)
      bias_sum += layer->bias[i];
    printf("[LeNet/Pebble] layer%d bias_sum=%ld requant_bits=0x%08x\n", index,
           bias_sum, layer->header.requant_multiplier_bits);
    free(weight);
  }
  if (fgetc(stream) != EOF)
    die("payload has unexpected trailing data");
  fclose(stream);
}

int main(int argc, char **argv) {
  const char *payload_path = argc > 1 ? argv[1] : "lenet_int8_payload.bin";
  PayloadHeader payload;
  Layer layers[LAYER_COUNT];
  float *input_fp32 = NULL;
  memset(layers, 0, sizeof(layers));

  printf("[LeNet/Pebble] loading %s\n", payload_path);
  load_payload(payload_path, &payload, &input_fp32, layers);

  int8_t *qinput = (int8_t *)checked_malloc(payload.input_count);
  pebble_fp2int(input_fp32, qinput, payload.input_count,
                layers[0].header.input_multiplier_bits);
  print_stats("input_q", qinput, payload.input_count);
  free(input_fp32);

  int8_t *conv1 = run_conv(qinput, 28, 28, 1, &layers[0]);
  print_stats("conv1_q", conv1, 24 * 24 * 6);
  free(qinput);
  relu_int8(conv1, 24 * 24 * 6);
  int8_t *pool1 = maxpool2(conv1, 24, 24, 6);
  print_stats("pool1_q", pool1, 12 * 12 * 6);
  free(conv1);

  int8_t *conv2_input = rescale_with_padding(
      pool1, 12 * 12 * 6, layers[0].header.output_step_bits,
      layers[1].header.input_multiplier_bits);
  print_stats("conv2_input_q", conv2_input, 12 * 12 * 6);
  free(pool1);
  int8_t *conv2 = run_conv(conv2_input, 12, 12, 6, &layers[1]);
  print_stats("conv2_q", conv2, 8 * 8 * 16);
  free(conv2_input);
  relu_int8(conv2, 8 * 8 * 16);
  int8_t *pool2 = maxpool2(conv2, 8, 8, 16);
  print_stats("pool2_q", pool2, 4 * 4 * 16);
  free(conv2);

  int8_t *fc1_nhwc =
      rescale_with_padding(pool2, 4 * 4 * 16, layers[1].header.output_step_bits,
                           layers[2].header.input_multiplier_bits);
  print_stats("fc1_input_nhwc_q", fc1_nhwc, 4 * 4 * 16);
  free(pool2);
  int8_t *fc1_input = flatten_nchw(fc1_nhwc, 4, 4, 16);
  print_stats("fc1_input_q", fc1_input, 256);
  free(fc1_nhwc);
  int8_t *fc1 = run_linear(fc1_input, &layers[2]);
  print_stats("fc1_q", fc1, 120);
  free(fc1_input);
  relu_int8(fc1, 120);

  int8_t *fc2_input =
      rescale_with_padding(fc1, 120, layers[2].header.output_step_bits,
                           layers[3].header.input_multiplier_bits);
  print_stats("fc2_input_q", fc2_input, 120);
  free(fc1);
  int8_t *fc2 = run_linear(fc2_input, &layers[3]);
  print_stats("fc2_q", fc2, 84);
  free(fc2_input);
  relu_int8(fc2, 84);

  int8_t *fc3_input =
      rescale_with_padding(fc2, 84, layers[3].header.output_step_bits,
                           layers[4].header.input_multiplier_bits);
  print_stats("fc3_input_q", fc3_input, 84);
  free(fc2);
  int8_t *fc3 = run_linear(fc3_input, &layers[4]);
  print_stats("fc3_q", fc3, 10);
  free(fc3_input);

  int8_t q_padded[TILE] __attribute__((aligned(64))) = {0};
  float logits[TILE] __attribute__((aligned(64))) = {0};
  memcpy(q_padded, fc3, 10);
  pebble_int8_to_fp32(q_padded, logits, TILE,
                      layers[4].header.output_step_bits);

  int passed = memcmp(q_padded, payload.expected_q, 10) == 0;
  if (!passed) {
    for (int i = 0; i < 10; ++i)
      if (q_padded[i] != payload.expected_q[i])
        printf("[LeNet/Pebble] logit %d q=%d expected=%d\n", i, q_padded[i],
               payload.expected_q[i]);
  }
  if (memcmp(logits, payload.expected_logits, 10 * sizeof(float)) != 0) {
    printf("[LeNet/Pebble] dequantized logits differ from Python golden\n");
    passed = 0;
  }

  int classification = 0;
  for (int i = 1; i < 10; ++i)
    if (q_padded[i] > q_padded[classification])
      classification = i;
  if ((uint32_t)classification != payload.expected_class)
    passed = 0;

  printf("[LeNet/Pebble] quantized logits:");
  for (int i = 0; i < 10; ++i)
    printf(" %d", q_padded[i]);
  printf("\n[LeNet/Pebble] classification=%d expected=%u\n", classification,
         payload.expected_class);
  printf("[LeNet/Pebble] issues: FP2INT=%lu INT2FP/requant=%lu MATRIX=%lu\n",
         fp2int_issues, int2fp_issues, matrix_issues);
  printf("[LeNet/Pebble] full INT8 LeNet %s\n", passed ? "PASSED" : "FAILED");

  free(fc3);
  for (int index = 0; index < LAYER_COUNT; ++index) {
    free(layers[index].weight);
    free(layers[index].bias);
  }
  return passed ? 0 : 1;
}
