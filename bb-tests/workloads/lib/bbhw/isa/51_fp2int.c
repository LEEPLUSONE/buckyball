#ifndef _BB_FP2INT_H_
#define _BB_FP2INT_H_

#include "isa.h"

#define BB_FP2INT_FUNC7 51

// Extended encoding:
//   rs2[31:0]  = scalar FP32 multiplier (ignored for per-channel)
//   rs2[35:34] = BB_SCALE_PER_*
//   rs2[63:36] = per-channel MMIO scale-table byte offset
#define bb_fp2int_ex(bank_id, wr_bank_id, iter, scale_fp32, granularity,       \
                     table_offset)                                             \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_BANK2(wr_bank_id) | BB_ITER(iter)),              \
      (FIELD((uint64_t)(scale_fp32), 0, 31) |                                  \
       BB_SCALE_GRANULARITY(granularity) |                                     \
       BB_SCALE_TABLE_OFFSET(table_offset)),                                   \
      BB_FP2INT_FUNC7)

// Backward-compatible scalar form.  One invocation carries one multiplier;
// the compiler decides whether it is reused per tensor or changed per command.
#define bb_fp2int(bank_id, wr_bank_id, iter, scale_fp32)                       \
  bb_fp2int_ex(bank_id, wr_bank_id, iter, scale_fp32,                          \
               BB_SCALE_PER_INSTRUCTION, 0)

#endif // _BB_FP2INT_H_
