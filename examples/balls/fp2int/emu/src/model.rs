//! Numeric reference shared by FP2INT and INT2FP in BEMU.
//!
//! Rust `f32` operations use binary32 round-to-nearest, ties-to-even.  The
//! explicit `round_ties_even` below defines the final integer conversion.

pub const INT8_MIN: i32 = -128;
pub const INT8_MAX: i32 = 127;

fn multiplier(scale_bits: u32) -> f32 {
    let value = f32::from_bits(scale_bits);
    if !value.is_finite() || !value.is_normal() || !value.is_sign_positive() {
        panic!("quant: multiplier must be a positive normal finite FP32 value");
    }
    value
}

fn normalized_input(fp_bits: u32) -> f32 {
    let value = f32::from_bits(fp_bits);
    if value.is_subnormal() {
        value.signum() * 0.0
    } else {
        value
    }
}

fn saturate_i8(value: f32) -> i8 {
    if value.is_nan() {
        return 0;
    }
    if value >= INT8_MAX as f32 {
        return INT8_MAX as i8;
    }
    if value <= INT8_MIN as f32 {
        return INT8_MIN as i8;
    }
    value as i8
}

pub fn fp2int_i32_bits(fp_bits: u32, scale_bits: u32) -> i32 {
    let scale = multiplier(scale_bits);
    let input = normalized_input(fp_bits);
    if input.is_nan() {
        return 0;
    }
    if input == f32::INFINITY {
        return i32::MAX;
    }
    if input == f32::NEG_INFINITY {
        return i32::MIN;
    }
    (input * scale).round_ties_even() as i32
}

pub fn fp2int_i8_bits(fp_bits: u32, scale_bits: u32) -> i8 {
    let scale = multiplier(scale_bits);
    let input = normalized_input(fp_bits);
    if input == f32::INFINITY {
        return INT8_MAX as i8;
    }
    if input == f32::NEG_INFINITY {
        return INT8_MIN as i8;
    }
    saturate_i8((input * scale).round_ties_even())
}

pub fn dequantize_int32_to_fp32_bits(value: i32, scale_bits: u32) -> u32 {
    ((value as f32) * multiplier(scale_bits)).to_bits()
}

pub fn requantize_int32_to_int8(value: i32, scale_bits: u32) -> i8 {
    saturate_i8(((value as f32) * multiplier(scale_bits)).round_ties_even())
}

#[allow(dead_code)]
pub fn fp2int_i32_word(input: [u32; 4], scale_bits: u32) -> [i32; 4] {
    input.map(|value| fp2int_i32_bits(value, scale_bits))
}

#[allow(dead_code)]
pub fn fp2int_i8_group(input: [u32; 4], scale_bits: u32) -> [i8; 4] {
    input.map(|value| fp2int_i8_bits(value, scale_bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp2int_uses_rne() {
        let one = 1.0f32.to_bits();
        let inputs = [0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5];
        let expected = [0i8, 2, 2, 0, -2, -2];
        for (input, expected) in inputs.iter().copied().zip(expected) {
            assert_eq!(fp2int_i8_bits(input.to_bits(), one), expected);
        }
    }

    #[test]
    fn fp2int_special_values_and_saturation() {
        let one = 1.0f32.to_bits();
        assert_eq!(fp2int_i8_bits(f32::NAN.to_bits(), one), 0);
        assert_eq!(fp2int_i8_bits(f32::INFINITY.to_bits(), one), 127);
        assert_eq!(fp2int_i8_bits(f32::NEG_INFINITY.to_bits(), one), -128);
        assert_eq!(fp2int_i8_bits(127.5f32.to_bits(), one), 127);
        assert_eq!(fp2int_i8_bits((-128.5f32).to_bits(), one), -128);
    }

    #[test]
    fn int32_conversion_uses_binary32_rne() {
        let one = 1.0f32.to_bits();
        assert_eq!(
            dequantize_int32_to_fp32_bits((1 << 24) + 1, one),
            ((1 << 24) as f32).to_bits()
        );
        assert_eq!(requantize_int32_to_int8(3, 0.5f32.to_bits()), 2);
        assert_eq!(requantize_int32_to_int8(-3, 0.5f32.to_bits()), -2);
    }

    #[test]
    #[should_panic(expected = "positive normal finite")]
    fn illegal_multiplier_is_rejected() {
        let _ = fp2int_i8_bits(1.0f32.to_bits(), 0);
    }
}
