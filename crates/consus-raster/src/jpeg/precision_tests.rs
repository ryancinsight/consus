use super::{decode, working_storage_bound};
use crate::{DecodeErrorKind, DecodeLimits, PixelFormat};

const BLOCK_SIDE: usize = 8;

mod metadata;

fn push_segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    bytes.extend_from_slice(&[0xff, marker]);
    let length = u16::try_from(payload.len() + 2).expect("invariant: test segment length fits u16");
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn push_wide_quantization(bytes: &mut Vec<u8>) {
    let mut payload = Vec::with_capacity(129);
    payload.push(0x10);
    for _ in 0..64 {
        payload.extend_from_slice(&1_u16.to_be_bytes());
    }
    push_segment(bytes, 0xdb, &payload);
}

fn push_frame(bytes: &mut Vec<u8>, marker: u8, component_ids: &[u8]) {
    let mut payload = vec![12, 0, 8, 0, 8];
    payload
        .push(u8::try_from(component_ids.len()).expect("invariant: test component count fits u8"));
    for &id in component_ids {
        payload.extend_from_slice(&[id, 0x11, 0]);
    }
    push_segment(bytes, marker, &payload);
}

fn push_huffman_tables(bytes: &mut Vec<u8>) {
    let mut payload = Vec::new();
    payload.push(0x00);
    payload.extend_from_slice(&[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&[0, 14, 15]);
    payload.push(0x10);
    payload.extend_from_slice(&[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&[0, 0x0e]);
    push_segment(bytes, 0xc4, &payload);
}

fn push_scan_header(bytes: &mut Vec<u8>, component_ids: &[u8], start: u8, end: u8) {
    let mut payload = Vec::new();
    payload
        .push(u8::try_from(component_ids.len()).expect("invariant: test component count fits u8"));
    for &id in component_ids {
        payload.extend_from_slice(&[id, 0]);
    }
    payload.extend_from_slice(&[start, end, 0]);
    push_segment(bytes, 0xda, &payload);
}

struct EntropyBits {
    bits: Vec<bool>,
}

impl EntropyBits {
    fn new() -> Self {
        Self { bits: Vec::new() }
    }

    fn push(&mut self, value: u16, width: u8) {
        for shift in (0..width).rev() {
            self.bits.push(value & (1 << shift) != 0);
        }
    }

    fn finish(mut self, bytes: &mut Vec<u8>) {
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(true);
        }
        for bits in self.bits.chunks_exact(8) {
            let byte = bits
                .iter()
                .fold(0_u8, |byte, bit| (byte << 1) | u8::from(*bit));
            bytes.push(byte);
            if byte == 0xff {
                bytes.push(0);
            }
        }
    }
}

fn gray_ac_fixture(progressive: bool) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8];
    push_wide_quantization(&mut bytes);
    push_frame(&mut bytes, if progressive { 0xc2 } else { 0xc1 }, &[1]);
    push_huffman_tables(&mut bytes);
    if progressive {
        push_scan_header(&mut bytes, &[1], 0, 0);
        let mut dc = EntropyBits::new();
        dc.push(0, 2);
        dc.finish(&mut bytes);
        push_scan_header(&mut bytes, &[1], 1, 63);
        let mut ac = EntropyBits::new();
        ac.push(1, 2);
        ac.push(8192, 14);
        ac.push(0, 2);
        ac.finish(&mut bytes);
    } else {
        push_scan_header(&mut bytes, &[1], 0, 63);
        let mut entropy = EntropyBits::new();
        entropy.push(0, 2);
        entropy.push(1, 2);
        entropy.push(8192, 14);
        entropy.push(0, 2);
        entropy.finish(&mut bytes);
    }
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn direct_rgb_fixture() -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8];
    push_wide_quantization(&mut bytes);
    push_frame(&mut bytes, 0xc1, b"RGB");
    push_huffman_tables(&mut bytes);
    push_scan_header(&mut bytes, b"RGB", 0, 63);
    let mut entropy = EntropyBits::new();
    entropy.push(2, 2);
    entropy.push(0x3fff, 15);
    entropy.push(0, 2);
    entropy.push(0, 2);
    entropy.push(0, 2);
    entropy.push(1, 2);
    entropy.push(0x3ff8, 14);
    entropy.push(0, 2);
    entropy.finish(&mut bytes);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn lossless_midpoint_fixture(precision: u8, point_transform: u8) -> Vec<u8> {
    let mut bytes = vec![
        0xff, 0xd8, 0xff, 0xc3, 0, 11, precision, 0, 1, 0, 1, 1, 1, 0x11, 0,
    ];
    let mut table = vec![0];
    table.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    table.push(0);
    push_segment(&mut bytes, 0xc4, &table);
    push_segment(&mut bytes, 0xda, &[1, 1, 0, 1, 0, point_transform]);
    bytes.push(0x7f);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn limits(bytes: &[u8]) -> DecodeLimits {
    DecodeLimits {
        max_encoded_bytes: bytes.len(),
        max_dimension: 8,
        max_pixels: 64,
        max_working_bytes: working_storage_bound(8, 8)
            .expect("invariant: test dimensions have a finite storage bound"),
    }
}

fn wide_samples(bytes: &[u8]) -> Vec<u16> {
    assert!(bytes.len().is_multiple_of(2));
    bytes
        .chunks_exact(2)
        .map(|sample| u16::from_ne_bytes([sample[0], sample[1]]))
        .collect()
}

fn fixture_limits(bytes: &[u8], width: u32, height: u32) -> DecodeLimits {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|count| usize::try_from(count).ok())
        .expect("invariant: fixture pixel count fits usize");
    DecodeLimits {
        max_encoded_bytes: bytes.len(),
        max_dimension: width.max(height),
        max_pixels: pixels,
        max_working_bytes: working_storage_bound(width, height)
            .expect("invariant: fixture dimensions have a finite storage bound"),
    }
}

const DCT_PRECISION_FIXTURES: [(&str, &[u8]); 8] = [
    (
        "precision-reference-sequential-arithmetic",
        include_bytes!("../../tests/fixtures/jpeg/precision-reference-sequential-arithmetic.jpg"),
    ),
    (
        "precision-reference-sequential-huffman",
        include_bytes!("../../tests/fixtures/jpeg/precision-reference-sequential-huffman.jpg"),
    ),
    (
        "precision-reference-progressive-arithmetic",
        include_bytes!("../../tests/fixtures/jpeg/precision-reference-progressive-arithmetic.jpg"),
    ),
    (
        "precision-reference-progressive-huffman",
        include_bytes!("../../tests/fixtures/jpeg/precision-reference-progressive-huffman.jpg"),
    ),
    (
        "precision-turbo-sequential-arithmetic",
        include_bytes!("../../tests/fixtures/jpeg/precision-turbo-sequential-arithmetic.jpg"),
    ),
    (
        "precision-turbo-sequential-huffman",
        include_bytes!("../../tests/fixtures/jpeg/precision-turbo-sequential-huffman.jpg"),
    ),
    (
        "precision-turbo-progressive-arithmetic",
        include_bytes!("../../tests/fixtures/jpeg/precision-turbo-progressive-arithmetic.jpg"),
    ),
    (
        "precision-turbo-progressive-huffman",
        include_bytes!("../../tests/fixtures/jpeg/precision-turbo-progressive-huffman.jpg"),
    ),
];

const LOSSLESS_PRECISION_FIXTURES: [(u8, u8, &str, &[u8]); 14] = [
    (
        1,
        0,
        "precision-lossless-predictor-1-point-0-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-1-point-0-huffman.jpg"
        ),
    ),
    (
        1,
        3,
        "precision-lossless-predictor-1-point-3-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-1-point-3-huffman.jpg"
        ),
    ),
    (
        2,
        0,
        "precision-lossless-predictor-2-point-0-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-2-point-0-huffman.jpg"
        ),
    ),
    (
        2,
        3,
        "precision-lossless-predictor-2-point-3-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-2-point-3-huffman.jpg"
        ),
    ),
    (
        3,
        0,
        "precision-lossless-predictor-3-point-0-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-3-point-0-huffman.jpg"
        ),
    ),
    (
        3,
        3,
        "precision-lossless-predictor-3-point-3-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-3-point-3-huffman.jpg"
        ),
    ),
    (
        4,
        0,
        "precision-lossless-predictor-4-point-0-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-4-point-0-huffman.jpg"
        ),
    ),
    (
        4,
        3,
        "precision-lossless-predictor-4-point-3-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-4-point-3-huffman.jpg"
        ),
    ),
    (
        5,
        0,
        "precision-lossless-predictor-5-point-0-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-5-point-0-huffman.jpg"
        ),
    ),
    (
        5,
        3,
        "precision-lossless-predictor-5-point-3-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-5-point-3-huffman.jpg"
        ),
    ),
    (
        6,
        0,
        "precision-lossless-predictor-6-point-0-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-6-point-0-huffman.jpg"
        ),
    ),
    (
        6,
        3,
        "precision-lossless-predictor-6-point-3-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-6-point-3-huffman.jpg"
        ),
    ),
    (
        7,
        0,
        "precision-lossless-predictor-7-point-0-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-7-point-0-huffman.jpg"
        ),
    ),
    (
        7,
        3,
        "precision-lossless-predictor-7-point-3-huffman",
        include_bytes!(
            "../../tests/fixtures/jpeg/precision-lossless-predictor-7-point-3-huffman.jpg"
        ),
    ),
];

const LOSSLESS_SAMPLES: [u16; 20] = [
    0, 4095, 2048, 1, 4094, 4095, 0, 2049, 4094, 1, 2048, 1, 0, 4095, 2049, 4094, 2048, 4095, 0, 1,
];

#[test]
fn sequential_and_progressive_twelve_bit_ac_reconstruct_the_reference_grid() {
    // Independent libjpeg-turbo 3.1.2 float-IDCT output for a centered block
    // whose first horizontal AC coefficient is 8192 and whose quantizers are 1.
    let expected_row = [3468, 3252, 2853, 2331, 1765, 1243, 844, 628];
    let expected: Vec<u16> = expected_row.into_iter().cycle().take(64).collect();
    for progressive in [false, true] {
        let encoded = gray_ac_fixture(progressive);
        let decoded = decode(&encoded, limits(&encoded)).expect("valid 12-bit DCT fixture");
        assert_eq!(decoded.format(), PixelFormat::GrayWide);
        assert_eq!(decoded.sample_precision(), 12);
        assert_eq!((decoded.width(), decoded.height()), (8, 8));
        assert_eq!(wide_samples(decoded.pixels()), expected);
    }
}

#[test]
fn twelve_bit_direct_rgb_preserves_full_range_dc_samples() {
    let encoded = direct_rgb_fixture();
    let decoded = decode(&encoded, limits(&encoded)).expect("valid direct RGB fixture");
    assert_eq!(decoded.format(), PixelFormat::RgbWide);
    assert_eq!(decoded.sample_precision(), 12);
    let expected: Vec<u16> = [0, 2048, 4095]
        .into_iter()
        .cycle()
        .take(BLOCK_SIDE * BLOCK_SIDE * 3)
        .collect();
    assert_eq!(wide_samples(decoded.pixels()), expected);
}

#[test]
fn independent_twelve_bit_dct_encoders_reconstruct_exact_tiles() {
    let row: Vec<u16> = [0_u16, 2048, 4095]
        .into_iter()
        .flat_map(|sample| [sample; BLOCK_SIDE])
        .collect();
    let expected = row.repeat(BLOCK_SIDE);

    for (name, bytes) in DCT_PRECISION_FIXTURES {
        let decoded = decode(bytes, fixture_limits(bytes, 24, 8))
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(decoded.format(), PixelFormat::GrayWide, "{name}");
        assert_eq!(decoded.sample_precision(), 12, "{name}");
        assert_eq!((decoded.width(), decoded.height()), (24, 8), "{}", name);
        assert_eq!(wide_samples(decoded.pixels()), expected, "{name}");
    }
}

#[test]
fn all_lossless_predictors_apply_point_transform_exactly() {
    for (predictor, point_transform, name, bytes) in LOSSLESS_PRECISION_FIXTURES {
        let decoded = decode(bytes, fixture_limits(bytes, 5, 4))
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let expected: Vec<u16> = LOSSLESS_SAMPLES
            .into_iter()
            .map(|sample| (sample >> point_transform) << point_transform)
            .collect();
        assert_eq!(decoded.format(), PixelFormat::GrayWide, "{name}");
        assert_eq!(decoded.sample_precision(), 12, "{name}");
        assert_eq!((decoded.width(), decoded.height()), (5, 4), "{}", name);
        assert_eq!(
            wide_samples(decoded.pixels()),
            expected,
            "predictor {predictor}, point transform {point_transform}"
        );
    }
}

#[test]
fn lossless_low_precisions_and_point_transform_preserve_meaningful_bits() {
    for (precision, point_transform, expected) in [(2, 0, 2), (2, 1, 2), (7, 0, 64), (7, 6, 64)] {
        let encoded = lossless_midpoint_fixture(precision, point_transform);
        let decoded =
            decode(&encoded, limits(&encoded)).expect("valid low-precision lossless fixture");
        assert_eq!(decoded.format(), PixelFormat::Gray);
        assert_eq!(decoded.sample_precision(), precision);
        assert_eq!(decoded.pixels(), &[expected]);
    }
}

#[test]
fn frame_and_quantization_precision_rules_are_enforced() {
    let encoded = gray_ac_fixture(false);
    let sof = encoded
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xc1])
        .expect("invariant: fixture has SOF1");

    let mut baseline_twelve = encoded.clone();
    baseline_twelve[sof + 1] = 0xc0;
    assert_eq!(
        decode(&baseline_twelve, limits(&baseline_twelve))
            .expect_err("SOF0 cannot carry 12-bit samples")
            .kind(),
        DecodeErrorKind::Unsupported
    );

    let mut eleven_bit = encoded.clone();
    eleven_bit[sof + 4] = 11;
    assert_eq!(
        decode(&eleven_bit, limits(&eleven_bit))
            .expect_err("SOF1 supports only 8- or 12-bit samples")
            .kind(),
        DecodeErrorKind::Unsupported
    );

    let dqt = encoded
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xdb])
        .expect("invariant: fixture has DQT");
    let mut invalid_quantizer_precision = encoded.clone();
    invalid_quantizer_precision[dqt + 4] = 0x20;
    assert_eq!(
        decode(
            &invalid_quantizer_precision,
            limits(&invalid_quantizer_precision)
        )
        .expect_err("DQT supports only 8- or 16-bit values")
        .kind(),
        DecodeErrorKind::Malformed
    );

    let mut wide_table_for_eight_bit = encoded;
    wide_table_for_eight_bit[sof + 4] = 8;
    assert_eq!(
        decode(&wide_table_for_eight_bit, limits(&wide_table_for_eight_bit))
            .expect_err("8-bit DCT cannot use a 16-bit quantization table")
            .kind(),
        DecodeErrorKind::Malformed
    );
}
