#ifndef _BB_INT2FP_H_
#define _BB_INT2FP_H_

#include "isa.h"

#define BB_INT2FP_FUNC7 52

#define BB_INT_OUTPUT_FP32 0
#define BB_INT_OUTPUT_INT8 1
#define BB_INT_OUTPUT_MODE(mode) FIELD((uint64_t)(mode), 32, 33)

// Explicit conversion and scale policy.  In per-channel mode, scale_fp32 is
// ignored and table_offset selects a 16-entry FP32 table in BANK0's MMIO
// region.  BB_INT_OUTPUT_INT8 performs INT32 -> INT8 requantization.
#define bb_int_convert_ex(bank_id, wr_bank_id, iter, output_mode, scale_fp32,  \
                          granularity, table_offset)                           \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),              \
      (FIELD((uint64_t)(scale_fp32), 0, 31) |                                  \
       BB_INT_OUTPUT_MODE(output_mode) | BB_SCALE_GRANULARITY(granularity) |   \
       BB_SCALE_TABLE_OFFSET(table_offset)),                                   \
      BB_INT2FP_FUNC7)

#define bb_int2fp_scale_ex(bank_id, wr_bank_id, iter, scale_fp32, granularity, \
                           table_offset)                                       \
  bb_int_convert_ex(bank_id, wr_bank_id, iter, BB_INT_OUTPUT_FP32, scale_fp32, \
                    granularity, table_offset)

// bb_int2fp(bank_id, wr_bank_id, iter, scale_fp32)
// scale_fp32 is a 32-bit FP32 value passed as uint32_t bit pattern
// Encoding: rs1 = banks | iter
//           rs2 = FIELD(scale_fp32, 0, 31)
#define bb_int2fp(bank_id, wr_bank_id, iter, scale_fp32)                       \
  bb_int2fp_scale_ex(bank_id, wr_bank_id, iter, scale_fp32,                    \
                     BB_SCALE_PER_INSTRUCTION, 0)

#endif // _BB_INT2FP_H_
