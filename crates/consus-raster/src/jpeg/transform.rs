//! JPEG sample reconstruction derived from RITK's first-party codec.

use std::f64::consts::{PI, SQRT_2};
use std::sync::LazyLock;

pub(super) const BLOCK_SIDE: usize = 8;
pub(super) const BLOCK_CELLS: usize = 64;

/// Zig-zag position to natural row-major coefficient index (T.81 Annex A).
pub(super) const ZIGZAG: [usize; BLOCK_CELLS] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

static COSINE: LazyLock<[[f64; BLOCK_SIDE]; BLOCK_SIDE]> = LazyLock::new(|| {
    let mut basis = [[0.0; BLOCK_SIDE]; BLOCK_SIDE];
    for (frequency, row) in basis.iter_mut().enumerate() {
        let scale = if frequency == 0 { 1.0 / SQRT_2 } else { 1.0 };
        for (position, value) in row.iter_mut().enumerate() {
            let angle_numerator = u32::try_from((2 * position + 1) * frequency)
                .expect("invariant: 8-point basis numerator fits u32");
            let angle_denominator = u32::try_from(2 * BLOCK_SIDE)
                .expect("invariant: 8-point basis denominator fits u32");
            *value = scale * (f64::from(angle_numerator) * PI / f64::from(angle_denominator)).cos();
        }
    }
    basis
});

/// Reconstruct one dequantized 8×8 block according to T.81 §A.3.3.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded IDCT samples are clamped to the exact u8 domain before conversion"
)]
pub(super) fn reconstruct(coefficients: &[i32], quantization: &[u16; 64]) -> [u8; 64] {
    let basis = &*COSINE;
    let mut dequantized = [0.0_f64; BLOCK_CELLS];
    for (index, value) in dequantized.iter_mut().enumerate() {
        *value = f64::from(coefficients[index]) * f64::from(quantization[index]);
    }
    let mut rows = [0.0_f64; BLOCK_CELLS];
    for y in 0..BLOCK_SIDE {
        for x in 0..BLOCK_SIDE {
            rows[y * BLOCK_SIDE + x] = (0..BLOCK_SIDE)
                .map(|u| basis[u][x] * dequantized[y * BLOCK_SIDE + u])
                .sum::<f64>()
                * 0.5;
        }
    }
    let mut output = [0_u8; BLOCK_CELLS];
    for y in 0..BLOCK_SIDE {
        for x in 0..BLOCK_SIDE {
            let sample = (0..BLOCK_SIDE)
                .map(|v| basis[v][y] * rows[v * BLOCK_SIDE + x])
                .sum::<f64>()
                .mul_add(0.5, 128.0)
                .round()
                .clamp(0.0, 255.0);
            output[y * BLOCK_SIDE + x] = sample as u8;
        }
    }
    output
}

/// Convert JFIF BT.601 YCbCr to RGB using RITK's signed 16.8 coefficients.
pub(super) fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = i32::from(y);
    let cb = i32::from(cb) - 128;
    let cr = i32::from(cr) - 128;
    let red = y + ((359 * cr + 128) >> 8);
    let green = y - ((88 * cb + 183 * cr + 128) >> 8);
    let blue = y + ((454 * cb + 128) >> 8);
    [
        u8::try_from(red.clamp(0, 255)).expect("invariant: clamped red channel fits u8"),
        u8::try_from(green.clamp(0, 255)).expect("invariant: clamped green channel fits u8"),
        u8::try_from(blue.clamp(0, 255)).expect("invariant: clamped blue channel fits u8"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_block_reconstructs_constant_samples() {
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 80;
        let output = reconstruct(&coefficients, &[1; 64]);
        assert_eq!(output, [138; 64]);
    }

    #[test]
    fn neutral_chroma_preserves_luminance() {
        assert_eq!(ycbcr_to_rgb(0, 128, 128), [0, 0, 0]);
        assert_eq!(ycbcr_to_rgb(128, 128, 128), [128, 128, 128]);
        assert_eq!(ycbcr_to_rgb(255, 128, 128), [255, 255, 255]);
    }

    #[test]
    fn first_horizontal_ac_basis_matches_analytical_samples() {
        let mut coefficients = [0_i32; 64];
        coefficients[1] = 1;
        let mut quantization = [1_u16; 64];
        quantization[1] = 8;
        let output = reconstruct(&coefficients, &quantization);
        let expected_row = [129, 129, 129, 128, 128, 127, 127, 127];
        for row in output.chunks_exact(8) {
            assert_eq!(row, expected_row);
        }
    }
}
