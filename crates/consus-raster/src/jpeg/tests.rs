use super::*;
use crate::{DecodeErrorKind, PixelFormat};
use jpeg_encoder::{ColorType, Encoder};
use proptest::prelude::*;
use proptest::test_runner::{RngSeed, TestCaseResult};

fn limits() -> DecodeLimits {
    DecodeLimits {
        max_encoded_bytes: 1 << 20,
        max_dimension: 4096,
        max_pixels: 1 << 20,
        max_working_bytes: 16 << 20,
    }
}

fn from_hex(encoded: &str) -> Vec<u8> {
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = str::from_utf8(pair).expect("fixture hex is ASCII");
            u8::from_str_radix(text, 16).expect("fixture hex byte")
        })
        .collect()
}

#[test]
fn gray_encoder_validates_arguments_and_round_trips_extremes() {
    for sample in [0, 255] {
        let encoded = encode_gray(&[sample; 64], 8, 8, 100).expect("valid grayscale image");
        let decoded = decode(&encoded, limits()).expect("encoded image decodes");
        assert_eq!((decoded.width(), decoded.height()), (8, 8));
        assert_eq!(decoded.format(), PixelFormat::Gray);
        assert_eq!(decoded.pixels(), &[sample; 64]);
    }
    for result in [
        encode_gray(&[], 0, 1, 90),
        encode_gray(&[], 1, 0, 90),
        encode_gray(&[0], 1, 1, 0),
        encode_gray(&[0], 1, 1, 101),
        encode_gray(&[], 1, 1, 90),
    ] {
        assert_eq!(
            result.expect_err("invalid encoder arguments").kind(),
            DecodeErrorKind::Malformed
        );
    }
}

#[test]
fn every_prefix_and_reterminated_cut_is_rejected() {
    let encoded = encode_gray(&[73; 64], 8, 8, 90).expect("fixture");
    for end in 0..encoded.len() {
        assert!(
            decode(&encoded[..end], limits()).is_err(),
            "accepted prefix {end}"
        );
    }
    for end in 2..encoded.len() - 2 {
        let mut cut = encoded[..end].to_vec();
        cut.extend_from_slice(&[0xff, 0xd9]);
        assert!(
            decode(&cut, limits()).is_err(),
            "accepted reterminated cut {end}"
        );
    }
}

#[test]
fn every_zero_limit_rejects_before_decode() {
    let encoded = encode_gray(&[0; 64], 8, 8, 90).expect("fixture");
    for rejected in [
        DecodeLimits {
            max_encoded_bytes: 0,
            ..limits()
        },
        DecodeLimits {
            max_dimension: 0,
            ..limits()
        },
        DecodeLimits {
            max_pixels: 0,
            ..limits()
        },
        DecodeLimits {
            max_working_bytes: 0,
            ..limits()
        },
    ] {
        assert_eq!(
            decode(&encoded, rejected)
                .expect_err("zero policy bound")
                .kind(),
            DecodeErrorKind::TooLarge
        );
    }
}

#[test]
fn working_storage_bound_covers_decoded_storage() {
    assert_eq!(working_storage_bound(8, 8), Ok(7488));
    assert_eq!(
        working_storage_bound(0, 8).expect_err("zero width").kind(),
        DecodeErrorKind::Malformed
    );
}

fn lossless_fixture(precision: u8, category: u8, magnitude: u16) -> Vec<u8> {
    let mut entropy = Vec::new();
    if category == 16 {
        entropy.push(0x7f);
    } else {
        let bit_count = usize::from(category) + 1;
        let value = u32::from(magnitude) << (32 - bit_count);
        let entropy_bytes = bit_count.div_ceil(8);
        for index in 0..entropy_bytes {
            entropy.push(
                u8::try_from((value >> (24 - index * 8)) & 0xff)
                    .expect("invariant: shifted entropy byte fits u8"),
            );
        }
        if let Some(last) = entropy.last_mut() {
            let unused = entropy_bytes * 8 - bit_count;
            *last |= u8::try_from((1_u16 << unused) - 1)
                .expect("invariant: at most seven pad bits fit u8");
        }
    }
    lossless_stream(
        LosslessParameters {
            precision,
            width: 1,
            height: 1,
            predictor: 1,
            point_transform: 0,
        },
        &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[category],
        &entropy,
    )
}

#[derive(Clone, Copy)]
struct LosslessParameters {
    precision: u8,
    width: u16,
    height: u16,
    predictor: u8,
    point_transform: u8,
}

fn lossless_stream(
    parameters: LosslessParameters,
    counts: &[u8; 16],
    symbols: &[u8],
    entropy: &[u8],
) -> Vec<u8> {
    let LosslessParameters {
        precision,
        width,
        height,
        predictor,
        point_transform,
    } = parameters;
    let mut bytes = vec![0xff, 0xd8, 0xff, 0xc3, 0x00, 0x0b, precision];
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[0x01, 0x01, 0x11, 0x00, 0xff, 0xc4]);
    let table_length = u16::try_from(2 + 1 + counts.len() + symbols.len()).expect("small table");
    bytes.extend_from_slice(&table_length.to_be_bytes());
    bytes.push(0);
    bytes.extend_from_slice(counts);
    bytes.extend_from_slice(symbols);
    bytes.extend_from_slice(&[
        0xff,
        0xda,
        0x00,
        0x08,
        0x01,
        0x01,
        0x00,
        predictor,
        0x00,
        point_transform,
    ]);
    for byte in entropy {
        bytes.push(*byte);
        if *byte == 0xff {
            bytes.push(0);
        }
    }
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

#[test]
fn lossless_grayscale_preserves_eight_and_sixteen_bit_samples() {
    let eight = decode(&lossless_fixture(8, 0, 0), limits()).expect("8-bit lossless");
    assert_eq!(eight.format(), PixelFormat::Gray);
    assert_eq!(eight.pixels(), &[128]);

    // Initial predictor is 32768. Category 15 magnitude 0x5234 extends to
    // +21044, which is not the desired 0x1234. Encode -28108 as magnitude
    // 4659 (0x1233): 32768 - 28108 = 4660 = 0x1234.
    let wide = decode(&lossless_fixture(16, 15, 0x1233), limits()).expect("16-bit lossless");
    assert_eq!(wide.format(), PixelFormat::GrayWide);
    assert_eq!(wide.pixels(), &0x1234_u16.to_ne_bytes());

    let zero = decode(&lossless_fixture(16, 16, 0), limits()).expect("category-16 zero");
    assert_eq!(zero.pixels(), &0_u16.to_ne_bytes());

    let maximum = decode(&lossless_fixture(16, 15, 0x7fff), limits()).expect("maximum sample");
    assert_eq!(maximum.pixels(), &u16::MAX.to_ne_bytes());
}

#[test]
fn lossless_prediction_wraps_modulo_sixteen_bits() {
    let bytes = lossless_stream(
        LosslessParameters {
            precision: 16,
            width: 2,
            height: 1,
            predictor: 1,
            point_transform: 0,
        },
        &[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[15, 1],
        &[0x7f, 0xff, 0xbf],
    );
    let decoded = decode(&bytes, limits()).expect("65535 + 1 wraps to zero");
    let expected: Vec<u8> = [u16::MAX, 0]
        .into_iter()
        .flat_map(u16::to_ne_bytes)
        .collect();
    assert_eq!(decoded.pixels(), expected);
}

#[test]
fn independent_sequential_and_refined_progressive_fixtures_match() {
    let sequential = from_hex(super::fixture_data::SEQUENTIAL);
    let progressive = from_hex(super::fixture_data::PROGRESSIVE);
    let sequential = decode(&sequential, limits()).expect("independent sequential fixture");
    let progressive = decode(&progressive, limits()).expect("independent progressive fixture");
    assert_eq!((sequential.width(), sequential.height()), (17, 25));
    assert_eq!(progressive, sequential);
}

#[test]
fn dct_ac_fixture_matches_analytical_pixel_oracle() {
    let mut encoded = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0x00, 1, 8];
    encoded.extend_from_slice(&[1; 62]);
    encoded.extend_from_slice(&[
        0xff, 0xc0, 0x00, 0x0b, 8, 0, 8, 0, 8, 1, 1, 0x11, 0, 0xff, 0xc4, 0x00, 0x14, 0, 1,
    ]);
    encoded.extend_from_slice(&[0; 15]);
    encoded.extend_from_slice(&[0, 0xff, 0xc4, 0x00, 0x15, 0x10, 1, 1]);
    encoded.extend_from_slice(&[0; 14]);
    encoded.extend_from_slice(&[
        1, 0, 0xff, 0xda, 0x00, 0x08, 1, 1, 0, 0, 63, 0, 0x37, 0xff, 0xd9,
    ]);
    let decoded = decode(&encoded, limits()).expect("single AC coefficient fixture");
    let expected_row = [129, 129, 129, 128, 128, 127, 127, 127];
    for row in decoded.pixels().chunks_exact(8) {
        assert_eq!(row, expected_row);
    }
}

#[test]
fn extended_sequential_reuses_the_sequential_coefficient_path() {
    let mut encoded = Vec::new();
    Encoder::new(&mut encoded, 100)
        .encode(&[90; 64], 8, 8, ColorType::Luma)
        .expect("sequential fixture");
    let frame = encoded
        .windows(2)
        .position(|window| window == [0xff, 0xc0])
        .expect("encoder emits SOF0");
    let expected = decode(&encoded, limits()).expect("SOF0");
    encoded[frame + 1] = 0xc1;
    assert_eq!(decode(&encoded, limits()).expect("SOF1"), expected);
}

#[test]
fn cmyk_decodes_to_rgb_pixels() {
    // The encoder maps cyan to YCbCrK (76,85,255,255). The 16.8 inverse
    // reconstructs (254,0,0), so Adobe transform 2 yields (1,255,255).
    // This is exact integer color conversion, not a lossy tolerance.
    for (input, cmyk, ycck) in [
        ([255, 0, 0, 0], [0, 255, 255], [1, 255, 255]),
        ([0, 0, 0, 0], [255, 255, 255], [255, 255, 255]),
        ([0, 0, 0, 255], [0, 0, 0], [0, 0, 0]),
    ] {
        for (format, expected) in [(ColorType::Cmyk, cmyk), (ColorType::CmykAsYcck, ycck)] {
            let mut encoded = Vec::new();
            Encoder::new(&mut encoded, 100)
                .encode(&input.repeat(64), 8, 8, format)
                .expect("CMYK fixture");
            let decoded = decode(&encoded, limits()).expect("CMYK decode");
            assert_eq!(decoded.format(), PixelFormat::Rgb);
            assert_eq!(decoded.pixels(), expected.repeat(64));
        }
    }
}

#[test]
fn lossless_restart_intervals_reset_prediction_and_validate_sequence() {
    let mut base = lossless_stream(
        LosslessParameters {
            precision: 8,
            width: 2,
            height: 4,
            predictor: 2,
            point_transform: 0,
        },
        &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[1],
        &[0x55, 0x55],
    );
    let entropy_start = base.len() - 4;
    base.insert(entropy_start + 1, 0xff);
    base.insert(entropy_start + 2, 0xd0);
    let sos = base
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("SOS");
    let mut encoded = base;
    encoded.splice(sos..sos, [0xff, 0xdd, 0x00, 0x04, 0x00, 0x04]);
    let decoded = decode(&encoded, limits()).expect("restart interval");
    assert_eq!(decoded.pixels(), &[129, 130, 130, 131, 129, 130, 130, 131]);

    let restart = encoded
        .windows(2)
        .position(|window| window == [0xff, 0xd0])
        .expect("RST0");
    encoded[restart + 1] = 0xd1;
    assert_eq!(
        decode(&encoded, limits())
            .expect_err("wrong restart sequence")
            .kind(),
        DecodeErrorKind::Malformed
    );
}

#[test]
fn lossless_point_transform_accepts_full_precision_range() {
    let encoded = lossless_stream(
        LosslessParameters {
            precision: 16,
            width: 1,
            height: 1,
            predictor: 1,
            point_transform: 15,
        },
        &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[0],
        &[0x7f],
    );
    let decoded = decode(&encoded, limits()).expect("point transform 15");
    assert_eq!(decoded.pixels(), &32_768_u16.to_ne_bytes());
}

#[test]
fn coarse_progressive_dc_stream_is_a_valid_image() {
    let mut encoded = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0x00];
    encoded.extend_from_slice(&[1; 64]);
    encoded.extend_from_slice(&[
        0xff, 0xc2, 0x00, 0x0b, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4,
        0x00, 0x14, 0x00, 0x01,
    ]);
    encoded.extend_from_slice(&[0; 15]);
    encoded.extend_from_slice(&[
        0x00, 0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x7f, 0xff, 0xd9,
    ]);
    let decoded = decode(&encoded, limits()).expect("coarse DC image");
    assert_eq!(decoded.pixels(), &[128]);
}

#[test]
fn resource_limits_hold_at_boundaries() {
    let encoded = encode_gray(&[0; 64], 8, 8, 100).expect("fixture");
    let exact = DecodeLimits {
        max_encoded_bytes: encoded.len(),
        max_dimension: 8,
        max_pixels: 64,
        max_working_bytes: 4096,
    };
    assert_eq!(
        decode(&encoded, exact).expect("exact limits").pixels(),
        &[0; 64]
    );
    for rejected in [
        DecodeLimits {
            max_encoded_bytes: encoded.len() - 1,
            ..exact
        },
        DecodeLimits {
            max_dimension: 7,
            ..exact
        },
        DecodeLimits {
            max_pixels: 63,
            ..exact
        },
        DecodeLimits {
            max_working_bytes: 383,
            ..exact
        },
    ] {
        assert_eq!(
            decode(&encoded, rejected)
                .expect_err("limit is one below need")
                .kind(),
            DecodeErrorKind::TooLarge
        );
    }
}

fn assert_decoded_contract(image: &DecodedImage, policy: DecodeLimits) -> TestCaseResult {
    let pixels = usize::try_from(image.width())
        .expect("invariant: u32 width fits usize on supported targets")
        .checked_mul(
            usize::try_from(image.height())
                .expect("invariant: u32 height fits usize on supported targets"),
        )
        .expect("invariant: decoder validated the bounded pixel count");
    prop_assert!(image.width() <= policy.max_dimension);
    prop_assert!(image.height() <= policy.max_dimension);
    prop_assert!(pixels <= policy.max_pixels);
    let bytes_per_pixel = match image.format() {
        PixelFormat::Gray => 1,
        PixelFormat::GrayWide => 2,
        PixelFormat::Rgb => 3,
    };
    prop_assert_eq!(
        image.pixels().len(),
        pixels
            .checked_mul(bytes_per_pixel)
            .expect("invariant: accepted output length fits usize")
    );
    prop_assert!(image.pixels().len() <= policy.max_working_bytes);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_iters: 1024,
        rng_seed: RngSeed::Fixed(0x434f_4e53_5553),
        ..ProptestConfig::default()
    })]

    #[test]
    fn arbitrary_bytes_never_escape_decode_contract(
        bytes in prop::collection::vec(any::<u8>(), 0..2048)
    ) {
        let policy = DecodeLimits {
            max_encoded_bytes: 2048,
            max_dimension: 64,
            max_pixels: 4096,
            max_working_bytes: 1 << 20,
        };
        if let Ok(image) = decode(&bytes, policy) {
            assert_decoded_contract(&image, policy)?;
        }
    }

    #[test]
    fn fixture_mutations_never_escape_decode_contract(
        index in any::<usize>(),
        value in any::<u8>()
    ) {
        let mut bytes = from_hex(super::fixture_data::SEQUENTIAL);
        let position = index % bytes.len();
        bytes[position] = value;
        let policy = limits();
        if let Ok(image) = decode(&bytes, policy) {
            assert_decoded_contract(&image, policy)?;
        }
    }
}
