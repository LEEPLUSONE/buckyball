#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdint.h>
#include <stdio.h>

#define LANES 16

static uint32_t fp_input[LANES] __attribute__((aligned(64))) = {
    0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
    0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
    0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
};

static int32_t int_input[LANES] __attribute__((aligned(64))) = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
};

static float channel_scales[LANES] __attribute__((aligned(64))) = {
    0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f,
    4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f,
};

static int8_t q_instruction[LANES] __attribute__((aligned(64)));
static int8_t q_tensor[LANES] __attribute__((aligned(64)));
static int8_t q_channel[LANES] __attribute__((aligned(64)));
static float fp_instruction[LANES] __attribute__((aligned(64)));
static float fp_tensor[LANES] __attribute__((aligned(64)));
static float fp_channel[LANES] __attribute__((aligned(64)));
static int8_t requant_channel[LANES] __attribute__((aligned(64)));

static int check_i8(const char *name, const int8_t *actual,
                    const int8_t *expected) {
  int passed = 1;
  for (int lane = 0; lane < LANES; ++lane) {
    if (actual[lane] != expected[lane]) {
      printf("%s lane %d: got %d expected %d\n", name, lane, actual[lane],
             expected[lane]);
      passed = 0;
    }
  }
  return passed;
}

static int check_fp32(const char *name, const float *actual,
                      const float *expected) {
  int passed = 1;
  for (int lane = 0; lane < LANES; ++lane) {
    if (actual[lane] != expected[lane]) {
      printf("%s lane %d: got %f expected %f\n", name, lane, actual[lane],
             expected[lane]);
      passed = 0;
    }
  }
  return passed;
}

static void load_scale_table(uint32_t owner_bank) {
  bb_mvin_mmio((uintptr_t)channel_scales, 0, 4, 16);
  // One MMIO region row is 1 KiB; this test consumes its first 64 bytes.
  bb_mmio_set(owner_bank, 0, 1);
}

static int test_fp2int(void) {
  const uint32_t src = 0;
  const uint32_t dst_instruction = 1;
  const uint32_t dst_tensor = 2;
  const uint32_t dst_channel = 3;
  const int8_t expected_instruction[LANES] = {
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  };
  const int8_t expected_tensor[LANES] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  const int8_t expected_channel[LANES] = {
      0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8,
  };

  bb_mem_alloc(src, 1, 4);
  bb_mem_alloc(dst_instruction, 1, 1);
  bb_mem_alloc(dst_tensor, 1, 1);
  bb_mem_alloc(dst_channel, 1, 1);
  bb_mvin((uintptr_t)fp_input, src, 1, 1);
  load_scale_table(src);

  bb_fp2int_ex(src, dst_instruction, 1, 0x40000000, BB_SCALE_PER_INSTRUCTION,
               0);
  bb_fp2int_ex(src, dst_tensor, 1, 0x3f000000, BB_SCALE_PER_TENSOR, 0);
  bb_fp2int_ex(src, dst_channel, 1, 0, BB_SCALE_PER_CHANNEL, 0);
  bb_mvout((uintptr_t)q_instruction, dst_instruction, 1, 1);
  bb_mvout((uintptr_t)q_tensor, dst_tensor, 1, 1);
  bb_mvout((uintptr_t)q_channel, dst_channel, 1, 1);
  bb_fence();

  int passed =
      check_i8("fp2int/per-instruction", q_instruction, expected_instruction);
  passed &= check_i8("fp2int/per-tensor", q_tensor, expected_tensor);
  passed &= check_i8("fp2int/per-channel", q_channel, expected_channel);

  bb_mem_release(src);
  bb_mem_release(dst_instruction);
  bb_mem_release(dst_tensor);
  bb_mem_release(dst_channel);
  return passed;
}

static int test_int2fp(void) {
  const uint32_t src = 0;
  const uint32_t dst_instruction = 1;
  const uint32_t dst_tensor = 2;
  const uint32_t dst_channel = 3;
  const uint32_t dst_requant = 4;
  float expected_instruction[LANES];
  float expected_tensor[LANES];
  float expected_channel[LANES];
  int8_t expected_requant[LANES];

  for (int lane = 0; lane < LANES; ++lane) {
    expected_instruction[lane] = 2.0f;
    expected_tensor[lane] = 8.0f;
    expected_channel[lane] = 4.0f * channel_scales[lane];
    expected_requant[lane] = (int8_t)expected_channel[lane];
  }

  bb_mem_alloc(src, 1, 4);
  bb_mem_alloc(dst_instruction, 1, 4);
  bb_mem_alloc(dst_tensor, 1, 4);
  bb_mem_alloc(dst_channel, 1, 4);
  bb_mem_alloc(dst_requant, 1, 1);
  bb_mvin((uintptr_t)int_input, src, 1, 1);
  load_scale_table(src);

  bb_int2fp_scale_ex(src, dst_instruction, 1, 0x3f000000,
                     BB_SCALE_PER_INSTRUCTION, 0);
  bb_int2fp_scale_ex(src, dst_tensor, 1, 0x40000000, BB_SCALE_PER_TENSOR, 0);
  bb_int2fp_scale_ex(src, dst_channel, 1, 0, BB_SCALE_PER_CHANNEL, 0);
  bb_int_convert_ex(src, dst_requant, 1, BB_INT_OUTPUT_INT8, 0,
                    BB_SCALE_PER_CHANNEL, 0);
  bb_mvout((uintptr_t)fp_instruction, dst_instruction, 1, 1);
  bb_mvout((uintptr_t)fp_tensor, dst_tensor, 1, 1);
  bb_mvout((uintptr_t)fp_channel, dst_channel, 1, 1);
  bb_mvout((uintptr_t)requant_channel, dst_requant, 1, 1);
  bb_fence();

  int passed = check_fp32("int2fp/per-instruction", fp_instruction,
                          expected_instruction);
  passed &= check_fp32("int2fp/per-tensor", fp_tensor, expected_tensor);
  passed &= check_fp32("int2fp/per-channel", fp_channel, expected_channel);
  passed &= check_i8("requant/per-channel", requant_channel, expected_requant);
  return passed;
}

int main(void) {
  int passed = test_fp2int();
  passed &= test_int2fp();
  printf("Quant scale granularity test %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
