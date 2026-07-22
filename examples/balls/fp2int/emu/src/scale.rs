//! Scale-granularity decoding shared by the quantization instructions.
//!
//! `per-instruction` and `per-tensor` use the scalar FP32 multiplier in
//! `rs2[31:0]`.  The distinction is kept in the encoding so traces and future
//! compiler lowering can preserve the requested policy.  `per-channel` reads
//! sixteen little-endian FP32 multipliers from the MMIO region bound to the
//! source bank; each logical-row lane uses one multiplier.

use super::instruction::MmioRegion;

const MMIO_BANK_BYTES: usize = 1024;
const MMIO_BANK_COUNT: usize = 16;
const MMIO_BYTES: usize = MMIO_BANK_BYTES * MMIO_BANK_COUNT;

pub const SCALE_GRANULARITY_SHIFT: u32 = 34;
pub const SCALE_TABLE_OFFSET_SHIFT: u32 = 36;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleGranularity {
    PerInstruction = 0,
    PerTensor = 1,
    PerChannel = 2,
}

impl ScaleGranularity {
    pub fn decode(xs2: u64) -> Self {
        match (xs2 >> SCALE_GRANULARITY_SHIFT) & 0x3 {
            0 => Self::PerInstruction,
            1 => Self::PerTensor,
            2 => Self::PerChannel,
            mode => panic!("quant: reserved scale granularity {mode}"),
        }
    }
}

fn mmio_u32(
    mmio_banks: &[[u8; MMIO_BANK_BYTES]; MMIO_BANK_COUNT],
    mmio_region_table: &[MmioRegion; 32],
    meta_bank: usize,
    rel_addr: usize,
) -> u32 {
    let region = mmio_region_table
        .get(meta_bank)
        .unwrap_or_else(|| panic!("quant: invalid scale owner bank {meta_bank}"));
    if !region.valid {
        panic!("quant: no MMIO scale table bound to bank {meta_bank}");
    }

    let region_bytes = region.size_rows as usize * MMIO_BANK_BYTES;
    let rel_end = rel_addr
        .checked_add(4)
        .unwrap_or_else(|| panic!("quant: scale-table address overflow"));
    if rel_end > region_bytes {
        panic!(
      "quant: scale-table range [{rel_addr}, {rel_end}) exceeds bound region size {region_bytes}"
    );
    }

    let abs_addr = region.mmio_addr as usize + rel_addr;
    let abs_end = abs_addr
        .checked_add(4)
        .unwrap_or_else(|| panic!("quant: scale-table address overflow"));
    if abs_end > MMIO_BYTES {
        panic!("quant: scale-table absolute range [{abs_addr}, {abs_end}) is out of MMIO");
    }

    let mut bytes = [0u8; 4];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let address = abs_addr + index;
        *byte = mmio_banks[address / MMIO_BANK_BYTES][address % MMIO_BANK_BYTES];
    }
    u32::from_le_bytes(bytes)
}

pub fn multiplier_bits(
    xs2: u64,
    lane: usize,
    scale_owner_bank: usize,
    mmio_banks: &[[u8; MMIO_BANK_BYTES]; MMIO_BANK_COUNT],
    mmio_region_table: &[MmioRegion; 32],
) -> u32 {
    match ScaleGranularity::decode(xs2) {
        ScaleGranularity::PerInstruction | ScaleGranularity::PerTensor => xs2 as u32,
        ScaleGranularity::PerChannel => {
            if lane >= 16 {
                panic!("quant: per-channel lane {lane} is outside a 16-element logical row");
            }
            let table_offset = (xs2 >> SCALE_TABLE_OFFSET_SHIFT) as usize;
            if !table_offset.is_multiple_of(4) {
                panic!("quant: scale-table offset {table_offset} is not FP32-aligned");
            }
            mmio_u32(
                mmio_banks,
                mmio_region_table,
                scale_owner_bank,
                table_offset + lane * 4,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(granularity: ScaleGranularity, scalar: u32, offset: u32) -> u64 {
        u64::from(scalar)
            | ((granularity as u64) << SCALE_GRANULARITY_SHIFT)
            | (u64::from(offset) << SCALE_TABLE_OFFSET_SHIFT)
    }

    #[test]
    fn scalar_modes_keep_the_immediate_multiplier() {
        let banks = [[0u8; MMIO_BANK_BYTES]; MMIO_BANK_COUNT];
        let regions = [MmioRegion::default(); 32];
        let scale = 0x3fc0_0000;
        for mode in [
            ScaleGranularity::PerInstruction,
            ScaleGranularity::PerTensor,
        ] {
            assert_eq!(
                multiplier_bits(encode(mode, scale, 0), 9, 0, &banks, &regions),
                scale
            );
        }
    }

    #[test]
    fn per_channel_uses_lane_and_table_offset() {
        let mut banks = [[0u8; MMIO_BANK_BYTES]; MMIO_BANK_COUNT];
        let mut regions = [MmioRegion::default(); 32];
        regions[3] = MmioRegion {
            valid: true,
            mmio_addr: 32,
            size_rows: 1,
        };
        let table_offset = 64u32;
        for lane in 0..16usize {
            let bits = (1.0f32 + lane as f32 / 4.0).to_bits().to_le_bytes();
            let address = 32 + table_offset as usize + lane * 4;
            banks[address / MMIO_BANK_BYTES]
                [address % MMIO_BANK_BYTES..address % MMIO_BANK_BYTES + 4]
                .copy_from_slice(&bits);
        }

        let xs2 = encode(ScaleGranularity::PerChannel, 0, table_offset);
        assert_eq!(
            multiplier_bits(xs2, 0, 3, &banks, &regions),
            1.0f32.to_bits()
        );
        assert_eq!(
            multiplier_bits(xs2, 7, 3, &banks, &regions),
            2.75f32.to_bits()
        );
        assert_eq!(
            multiplier_bits(xs2, 15, 3, &banks, &regions),
            4.75f32.to_bits()
        );
    }

    #[test]
    #[should_panic(expected = "reserved scale granularity")]
    fn reserved_mode_is_rejected() {
        let _ = ScaleGranularity::decode(3u64 << SCALE_GRANULARITY_SHIFT);
    }
}
