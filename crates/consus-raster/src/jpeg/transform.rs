//! JPEG dequantization, level shifting and color reconstruction.

use apollo_dctdst_core::{DctIiiPlan, Normalization};
use std::sync::LazyLock;

pub(super) const BLOCK_SIDE: usize = 8;
pub(super) const BLOCK_CELLS: usize = 64;
const _: () = assert!(BLOCK_CELLS == BLOCK_SIDE * BLOCK_SIDE);

/// Zig-zag position to natural row-major coefficient index (T.81 Annex A).
pub(super) const ZIGZAG: [usize; BLOCK_CELLS] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

static INVERSE: LazyLock<DctIiiPlan<f64, BLOCK_SIDE>> = LazyLock::new(|| {
    DctIiiPlan::new(Normalization::Orthonormal).expect("invariant: JPEG block side is nonzero")
});

/// Reconstruct one dequantized 8×8 block according to T.81 §A.3.3.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded IDCT samples are clamped to the validated JPEG precision before conversion"
)]
pub(super) fn reconstruct(
    coefficients: &[i32; BLOCK_CELLS],
    quantization: &[u16; BLOCK_CELLS],
    precision: u8,
) -> [u16; BLOCK_CELLS] {
    debug_assert!(matches!(precision, 8 | 12));
    let midpoint = 1_u16 << (precision - 1);
    let maximum = midpoint * 2 - 1;
    let inverse = &*INVERSE;
    let mut dequantized = [0.0_f64; BLOCK_CELLS];
    for ((value, coefficient), quantizer) in
        dequantized.iter_mut().zip(coefficients).zip(quantization)
    {
        *value = f64::from(*coefficient) * f64::from(*quantizer);
    }
    let mut rows = [[0.0; BLOCK_SIDE]; BLOCK_SIDE];
    for (input, output) in dequantized
        .as_chunks::<BLOCK_SIDE>()
        .0
        .iter()
        .zip(&mut rows)
    {
        inverse.transform(input, output);
    }
    let mut output = [0_u16; BLOCK_CELLS];
    for column in 0..BLOCK_SIDE {
        let coefficients = rows.map(|row| row[column]);
        let mut samples = [0.0; BLOCK_SIDE];
        inverse.transform(&coefficients, &mut samples);
        for (row, sample) in output
            .as_chunks_mut::<BLOCK_SIDE>()
            .0
            .iter_mut()
            .zip(samples)
        {
            row[column] = (sample + f64::from(midpoint))
                .round()
                .clamp(0.0, f64::from(maximum)) as u16;
        }
    }
    output
}

/// Convert JFIF BT.601 YCbCr to RGB using RITK's signed 16.8 coefficients.
pub(super) fn ycbcr_to_rgb(y: u16, cb: u16, cr: u16, precision: u8) -> [u16; 3] {
    debug_assert!(matches!(precision, 8 | 12));
    let midpoint = 1_i32 << (precision - 1);
    let maximum = midpoint * 2 - 1;
    let y = i32::from(y);
    let cb = i32::from(cb) - midpoint;
    let cr = i32::from(cr) - midpoint;
    let red = y + ((359 * cr + 128) >> 8);
    let green = y - ((88 * cb + 183 * cr + 128) >> 8);
    let blue = y + ((454 * cb + 128) >> 8);
    [
        u16::try_from(red.clamp(0, maximum)).expect("invariant: clamped red channel fits u16"),
        u16::try_from(green.clamp(0, maximum)).expect("invariant: clamped green channel fits u16"),
        u16::try_from(blue.clamp(0, maximum)).expect("invariant: clamped blue channel fits u16"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_block_reconstructs_constant_samples() {
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 80;
        let output = reconstruct(&coefficients, &[1; 64], 8);
        assert_eq!(output, [138; 64]);
    }

    #[test]
    fn neutral_chroma_preserves_luminance() {
        assert_eq!(ycbcr_to_rgb(0, 128, 128, 8), [0, 0, 0]);
        assert_eq!(ycbcr_to_rgb(128, 128, 128, 8), [128, 128, 128]);
        assert_eq!(ycbcr_to_rgb(255, 128, 128, 8), [255, 255, 255]);
    }

    #[test]
    fn first_horizontal_ac_basis_matches_analytical_samples() {
        let mut coefficients = [0_i32; 64];
        coefficients[1] = 1;
        let mut quantization = [1_u16; 64];
        quantization[1] = 8;
        let output = reconstruct(&coefficients, &quantization, 8);
        let expected_row = [129, 129, 129, 128, 128, 127, 127, 127];
        for row in output.chunks_exact(8) {
            assert_eq!(row, expected_row);
        }
    }
}
