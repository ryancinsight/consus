use crate::DecodeLimits;
use crate::jpeg::decode;

use super::{
    enforce_dct_magnitude, enforce_signed_category, signed_dct_magnitude, signed_magnitude,
};

struct Fixture {
    name: &'static str,
    arithmetic: &'static [u8],
    huffman: &'static [u8],
}

macro_rules! fixture {
    ($name:literal) => {
        Fixture {
            name: $name,
            arithmetic: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/jpeg/",
                $name,
                "-arithmetic.jpg"
            )),
            huffman: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/jpeg/",
                $name,
                "-huffman.jpg"
            )),
        }
    };
}

const FIXTURES: [Fixture; 12] = [
    fixture!("dc-sequential"),
    fixture!("dc-progressive"),
    fixture!("dc-restart"),
    fixture!("dc-progressive-restart"),
    fixture!("ac-sequential"),
    fixture!("ac-progressive"),
    fixture!("ac-restart"),
    fixture!("ac-progressive-restart"),
    fixture!("sparse-sequential"),
    fixture!("sparse-progressive"),
    fixture!("sparse-restart"),
    fixture!("sparse-progressive-restart"),
];

fn limits() -> DecodeLimits {
    DecodeLimits {
        max_encoded_bytes: 1 << 20,
        max_dimension: 4096,
        max_pixels: 1 << 20,
        max_working_bytes: 16 << 20,
    }
}

#[test]
fn arithmetic_processes_match_independent_huffman_coefficient_peers() {
    for fixture in FIXTURES {
        let arithmetic = decode(fixture.arithmetic, limits())
            .unwrap_or_else(|error| panic!("{} arithmetic: {error:?}", fixture.name));
        let huffman = decode(fixture.huffman, limits())
            .unwrap_or_else(|error| panic!("{} Huffman: {error:?}", fixture.name));
        assert_eq!(arithmetic, huffman, "{} pixels", fixture.name);
    }
}

#[test]
fn color_arithmetic_processes_match_the_canonical_huffman_image() {
    let huffman = decode(
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/jpeg/color-transcode-sequential-huffman.jpg"
        )),
        limits(),
    )
    .expect("valid canonical color Huffman fixture");
    let variants: [(&str, &[u8]); 4] = [
        (
            "sequential",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/jpeg/color-transcode-sequential-arithmetic.jpg"
            )),
        ),
        (
            "sequential restart",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/jpeg/color-transcode-sequential-restart-arithmetic.jpg"
            )),
        ),
        (
            "progressive",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/jpeg/color-transcode-progressive-arithmetic.jpg"
            )),
        ),
        (
            "progressive restart",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/jpeg/color-transcode-progressive-restart-arithmetic.jpg"
            )),
        ),
    ];

    assert_eq!((huffman.width(), huffman.height()), (48, 32));
    assert_eq!(huffman.format(), crate::PixelFormat::Rgb);
    assert_eq!(huffman.sample_precision(), 8);
    assert_eq!(huffman.compression(), crate::Compression::Lossy);
    let first = *huffman
        .pixels()
        .first()
        .expect("invariant: a 48-by-32 RGB image contains samples");
    assert!(huffman.pixels().iter().any(|&sample| sample != first));

    for (name, bytes) in variants {
        let arithmetic =
            decode(bytes, limits()).unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(
            (arithmetic.width(), arithmetic.height()),
            (48, 32),
            "{name}"
        );
        assert_eq!(arithmetic.format(), crate::PixelFormat::Rgb, "{name}");
        assert_eq!(arithmetic.sample_precision(), 8, "{name}");
        assert_eq!(
            arithmetic.compression(),
            crate::Compression::Lossy,
            "{name}"
        );
        assert_eq!(arithmetic, huffman, "{name} full image");
    }
}

#[test]
fn manufactured_dc_and_ac_cases_match_analytical_pixels() {
    let dc_row: Vec<u8> = [0_u8, 128, 255]
        .into_iter()
        .flat_map(|sample| [sample; 8])
        .collect();
    let dc_pixels: Vec<u8> = dc_row.repeat(8);
    let ac_row = [129, 129, 129, 128, 128, 127, 127, 127];
    let ac_pixels = ac_row.repeat(8);

    for fixture in FIXTURES {
        let decoded = decode(fixture.arithmetic, limits())
            .unwrap_or_else(|error| panic!("{} arithmetic: {error:?}", fixture.name));
        assert_eq!(decoded.compression(), crate::Compression::Lossy);
        if fixture.name.starts_with("dc-") {
            assert_eq!(decoded.pixels(), dc_pixels, "{} DC pixels", fixture.name);
        } else if fixture.name.starts_with("ac-") {
            assert_eq!(decoded.pixels(), ac_pixels, "{} AC pixels", fixture.name);
        }
    }
}

#[test]
fn arithmetic_prefixes_require_physical_scan_termination() {
    for fixture in FIXTURES {
        for end in 0..fixture.arithmetic.len() {
            let error = decode(&fixture.arithmetic[..end], limits()).unwrap_err();
            assert!(matches!(
                error.kind(),
                crate::DecodeErrorKind::Malformed | crate::DecodeErrorKind::Unsupported
            ));
        }
    }
}

#[test]
fn arithmetic_fill_before_stuffing_zero_is_rejected() {
    let fixture = fixture!("dc-sequential");
    let mut bytes = fixture.arithmetic.to_vec();
    let stuffed = bytes
        .windows(2)
        .position(|window| window == [0xff, 0x00])
        .expect("invariant: the fixture contains stuffed entropy");
    bytes.insert(stuffed, 0xff);

    let error = decode(&bytes, limits()).expect_err("FF FF 00 is not entropy stuffing");
    assert_eq!(error.kind(), crate::DecodeErrorKind::Malformed);
}

proptest::proptest! {
    #[test]
    fn mutated_arithmetic_streams_preserve_bounded_output(
        case in 0_usize..12,
        offset in proptest::prelude::any::<usize>(),
        replacement in proptest::prelude::any::<u8>(),
    ) {
        let fixture = &FIXTURES[case];
        let mut bytes = fixture.arithmetic.to_vec();
        let index = offset % bytes.len();
        bytes[index] = replacement;
        // Mutations outside the format domain may be rejected. Every accepted
        // mutation must still satisfy the bounded raster representation.
        if let Ok(image) = decode(&bytes, limits()) {
            let channels = match image.format() {
                crate::PixelFormat::Gray => 1,
                crate::PixelFormat::GrayWide => 2,
                crate::PixelFormat::Rgb => 3,
                crate::PixelFormat::RgbWide => 6,
            };
            let pixels = u64::from(image.width()) * u64::from(image.height());
            proptest::prop_assert!(pixels <= u64::try_from(limits().max_pixels).unwrap());
            proptest::prop_assert_eq!(u64::try_from(image.pixels().len()).unwrap(), pixels * channels);
            proptest::prop_assert!((2..=16).contains(&image.sample_precision()));
        }
    }
}

#[test]
fn dct_categories_exclude_the_negative_power_of_two_endpoint() {
    assert_eq!(enforce_dct_magnitude(-2047, 11), Ok(()));
    assert_eq!(enforce_dct_magnitude(2047, 11), Ok(()));
    assert!(enforce_dct_magnitude(-2048, 11).is_err());
    assert!(enforce_dct_magnitude(2048, 11).is_err());

    assert_eq!(signed_dct_magnitude(16_383, true, 14), Ok(-16_383));
    assert_eq!(signed_dct_magnitude(16_383, false, 14), Ok(16_383));
    assert!(signed_dct_magnitude(16_384, true, 14).is_err());
    assert!(signed_dct_magnitude(16_384, false, 14).is_err());
}

#[test]
fn predictor_and_lossless_ranges_retain_the_negative_endpoint() {
    assert_eq!(enforce_signed_category(-2048, 11), Ok(()));
    assert_eq!(enforce_signed_category(2047, 11), Ok(()));
    assert!(enforce_signed_category(-2049, 11).is_err());
    assert!(enforce_signed_category(2048, 11).is_err());

    assert_eq!(signed_magnitude(32_768, true, 15), Ok(-32_768));
    assert!(signed_magnitude(32_768, false, 15).is_err());
}
