//===- 52_int2fp.rs - INT2FP instruction (INT32 to FP32) -------------------===//

use super::super::bank::{bank_num, bank_size};
use super::decode::{pbank, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

mod model;

pub struct Int2Fp;

impl Instruction for Int2Fp {
    const FUNCT: u32 = 52;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let src = rs1_b0(xs1);
        let dst = rs1_b2(xs1);
        let depth = rs1_iter(xs1) as usize;

        if src >= bank_num() as u64 || dst >= bank_num() as u64 {
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

        let scale_bits = (xs2 & 0xffff_ffff) as u32;

        if (sc.cols, dc.cols) != (1, 1) {
            panic!(
                "int2fp: INT32-to-FP32 requires src_cols=1 and dst_cols=1, got src_cols={} dst_cols={}",
                sc.cols, dc.cols
            );
        }

        let ps = pbank(ctx.bank_map, src);
        let pd = pbank(ctx.bank_map, dst);
        for i in 0..depth {
            let base = i * 16;
            if base + 16 > bank_size() {
                panic!("int2fp: out of range");
            }
            for lane in 0..4 {
                let off = base + lane * 4;
                let v = i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                let o = model::int2fp_fp32_bits(v, scale_bits);
                ctx.banks[pd][off..off + 4].copy_from_slice(&o.to_le_bytes());
            }
        }
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        rs1_iter(xs1).max(1)
    }
}
