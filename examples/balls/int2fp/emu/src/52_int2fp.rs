//===- 52_int2fp.rs - INT conversion instruction --------------------------===//

use super::super::bank::{BANK_NUM, BANK_SIZE};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};
use super::{quant_model, quant_scale};

const OUTPUT_MODE_SHIFT: u32 = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputMode {
    Fp32 = 0,
    Int8 = 1,
}

impl OutputMode {
    fn decode(xs2: u64) -> Self {
        match (xs2 >> OUTPUT_MODE_SHIFT) & 0x3 {
            0 => Self::Fp32,
            1 => Self::Int8,
            mode => panic!("int2fp: reserved output mode {mode}"),
        }
    }
}

pub struct Int2Fp;

impl Instruction for Int2Fp {
    const FUNCT: u32 = 52;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let src = rs1_b0(xs1);
        let dst = rs1_b2(xs1);
        let depth = rs1_iter(xs1) as usize;

        if src >= BANK_NUM as u64 || dst >= BANK_NUM as u64 {
            panic!("int2fp: invalid bank_id");
        }
        if depth == 0 {
            panic!("int2fp: iter must be > 0");
        }

        let sc = ctx.cfgs[src as usize];
        let dc = ctx.cfgs[dst as usize];
        if !sc.allocated || !dc.allocated {
            panic!("int2fp: bank not allocated");
        }

        let scale_owner_bank = src as usize;
        match (OutputMode::decode(xs2), sc.cols, dc.cols) {
            // Legacy/debug scalar-bank layout: sixteen INT32/FP32 elements are
            // stored contiguously for each logical row.
            (OutputMode::Fp32, 1, 1) => {
                let ps = pbank(ctx.bank_map, src);
                let pd = pbank(ctx.bank_map, dst);
                for row in 0..depth {
                    let base = row * 64;
                    if base + 64 > BANK_SIZE {
                        panic!("int2fp: out of range");
                    }
                    for lane in 0..16 {
                        let off = base + lane * 4;
                        let value =
                            i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                        let scale_bits = quant_scale::multiplier_bits(
                            xs2,
                            lane,
                            scale_owner_bank,
                            ctx.mmio_banks,
                            ctx.mmio_region_table,
                        );
                        let output = quant_model::dequantize_int32_to_fp32_bits(value, scale_bits);
                        ctx.banks[pd][off..off + 4].copy_from_slice(&output.to_le_bytes());
                    }
                }
            }
            // Legacy INT8 -> FP32 expansion.  Destination groups preserve the
            // logical lane order: element 4*g+l is group g, lane l.
            (OutputMode::Fp32, 1, 4) => {
                let ps = pbank(ctx.bank_map, src);
                for row in 0..depth {
                    let src_base = row * 16;
                    if src_base + 16 > BANK_SIZE {
                        panic!("int2fp: out of range");
                    }
                    for group in 0..4u64 {
                        let pd = pbank_group(ctx.bank_map, dst, group);
                        let dst_base = row * 16;
                        if dst_base + 16 > BANK_SIZE {
                            panic!("int2fp: out of range");
                        }
                        for lane in 0..4usize {
                            let logical_lane = group as usize * 4 + lane;
                            let value = ctx.banks[ps][src_base + logical_lane] as i8 as i32;
                            let scale_bits = quant_scale::multiplier_bits(
                                xs2,
                                logical_lane,
                                scale_owner_bank,
                                ctx.mmio_banks,
                                ctx.mmio_region_table,
                            );
                            let output =
                                quant_model::dequantize_int32_to_fp32_bits(value, scale_bits);
                            let off = dst_base + lane * 4;
                            ctx.banks[pd][off..off + 4].copy_from_slice(&output.to_le_bytes());
                        }
                    }
                }
            }
            // Main dequant path: four INT32 groups become four FP32 groups.
            (OutputMode::Fp32, 4, 4) => {
                for group in 0..4u64 {
                    let ps = pbank_group(ctx.bank_map, src, group);
                    let pd = pbank_group(ctx.bank_map, dst, group);
                    for row in 0..depth {
                        let base = row * 16;
                        if base + 16 > BANK_SIZE {
                            panic!("int2fp: out of range");
                        }
                        for lane in 0..4usize {
                            let logical_lane = group as usize * 4 + lane;
                            let off = base + lane * 4;
                            let value =
                                i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                            let scale_bits = quant_scale::multiplier_bits(
                                xs2,
                                logical_lane,
                                scale_owner_bank,
                                ctx.mmio_banks,
                                ctx.mmio_region_table,
                            );
                            let output =
                                quant_model::dequantize_int32_to_fp32_bits(value, scale_bits);
                            ctx.banks[pd][off..off + 4].copy_from_slice(&output.to_le_bytes());
                        }
                    }
                }
            }
            // Full-INT8 boundary: four INT32 groups are requantized and packed
            // into one 128-bit INT8 row.
            (OutputMode::Int8, 4, 1) => {
                let pd = pbank(ctx.bank_map, dst);
                for row in 0..depth {
                    let base = row * 16;
                    if base + 16 > BANK_SIZE {
                        panic!("int2fp: out of range");
                    }
                    for group in 0..4u64 {
                        let ps = pbank_group(ctx.bank_map, src, group);
                        for lane in 0..4usize {
                            let logical_lane = group as usize * 4 + lane;
                            let off = base + lane * 4;
                            let value =
                                i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                            let scale_bits = quant_scale::multiplier_bits(
                                xs2,
                                logical_lane,
                                scale_owner_bank,
                                ctx.mmio_banks,
                                ctx.mmio_region_table,
                            );
                            ctx.banks[pd][base + logical_lane] =
                                quant_model::requantize_int32_to_int8(value, scale_bits) as u8;
                        }
                    }
                }
            }
            (mode, src_cols, dst_cols) => {
                panic!(
                    "int2fp: unsupported output mode {mode:?} for layout src_cols={src_cols} dst_cols={dst_cols}"
                );
            }
        }
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        rs1_iter(xs1).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_mode_decode_is_explicit() {
        assert_eq!(OutputMode::decode(0), OutputMode::Fp32);
        assert_eq!(
            OutputMode::decode(1u64 << OUTPUT_MODE_SHIFT),
            OutputMode::Int8
        );
    }

    #[test]
    #[should_panic(expected = "reserved output mode")]
    fn reserved_output_mode_is_rejected() {
        let _ = OutputMode::decode(2u64 << OUTPUT_MODE_SHIFT);
    }
}
