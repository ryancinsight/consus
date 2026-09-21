use crate::{DecodeLimits, PixelFormat};

#[test]
fn lossless_scan_rejects_ac_selectors_for_both_entropy_methods() {
    for source in [
        include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-0-arithmetic.jpg")
            .as_slice(),
        include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-0-huffman.jpg")
            .as_slice(),
    ] {
        let scan = source
            .windows(2)
            .position(|marker| marker == [0xff, 0xda])
            .unwrap();
        for selector in 1..=3 {
            let mut encoded = source.to_vec();
            encoded[scan + 6] = selector;
            let error = crate::jpeg::decode(
                &encoded,
                DecodeLimits {
                    max_encoded_bytes: 4096,
                    max_dimension: 5,
                    max_pixels: 20,
                    max_working_bytes: crate::jpeg::working_storage_bound(5, 4).unwrap(),
                },
            )
            .unwrap_err();
            assert_eq!(error.kind(), crate::DecodeErrorKind::Malformed);
        }
    }
}

#[test]
fn independent_lossless_streams_preserve_samples_and_restarts() {
    let cases: &[(u8, &[u8])] = &[
        (
            2,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-2-restart-0-arithmetic.jpg")
                as &[u8],
        ),
        (
            2,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-2-restart-0-huffman.jpg")
                as &[u8],
        ),
        (
            2,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-2-restart-5-arithmetic.jpg")
                as &[u8],
        ),
        (
            2,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-2-restart-5-huffman.jpg")
                as &[u8],
        ),
        (
            2,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-2-restart-10-arithmetic.jpg")
                as &[u8],
        ),
        (
            2,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-2-restart-10-huffman.jpg")
                as &[u8],
        ),
        (
            8,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-0-arithmetic.jpg")
                as &[u8],
        ),
        (
            8,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-0-huffman.jpg")
                as &[u8],
        ),
        (
            8,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-5-arithmetic.jpg")
                as &[u8],
        ),
        (
            8,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-5-huffman.jpg")
                as &[u8],
        ),
        (
            8,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-10-arithmetic.jpg")
                as &[u8],
        ),
        (
            8,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-8-restart-10-huffman.jpg")
                as &[u8],
        ),
        (
            12,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-12-restart-0-arithmetic.jpg")
                as &[u8],
        ),
        (
            12,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-12-restart-0-huffman.jpg")
                as &[u8],
        ),
        (
            12,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-12-restart-5-arithmetic.jpg")
                as &[u8],
        ),
        (
            12,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-12-restart-5-huffman.jpg")
                as &[u8],
        ),
        (
            12,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-12-restart-10-arithmetic.jpg")
                as &[u8],
        ),
        (
            12,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-12-restart-10-huffman.jpg")
                as &[u8],
        ),
        (
            16,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-16-restart-0-arithmetic.jpg")
                as &[u8],
        ),
        (
            16,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-16-restart-0-huffman.jpg")
                as &[u8],
        ),
        (
            16,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-16-restart-5-arithmetic.jpg")
                as &[u8],
        ),
        (
            16,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-16-restart-5-huffman.jpg")
                as &[u8],
        ),
        (
            16,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-16-restart-10-arithmetic.jpg")
                as &[u8],
        ),
        (
            16,
            include_bytes!("../../../../tests/fixtures/jpeg/lossless-16-restart-10-huffman.jpg")
                as &[u8],
        ),
    ];
    for &(precision, encoded) in cases {
        let expected = expected_samples(precision);
        let image = crate::jpeg::decode(
            encoded,
            DecodeLimits {
                max_encoded_bytes: 4096,
                max_dimension: 5,
                max_pixels: 20,
                max_working_bytes: crate::jpeg::working_storage_bound(5, 4).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(
            (image.width(), image.height(), image.sample_precision()),
            (5, 4, precision)
        );
        assert_eq!(image.compression(), crate::Compression::Lossless);
        let samples: Vec<u16> = if precision > 8 {
            assert_eq!(image.format(), PixelFormat::GrayWide);
            image
                .pixels()
                .chunks_exact(2)
                .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
                .collect()
        } else {
            assert_eq!(image.format(), PixelFormat::Gray);
            image.pixels().iter().copied().map(u16::from).collect()
        };
        assert_eq!(
            samples, expected,
            "precision {precision}, encoded {encoded:?}"
        );
    }
}

fn expected_samples(precision: u8) -> &'static [u16] {
    match precision {
        2 => &[0, 3, 2, 1, 2, 3, 0, 3, 2, 1, 2, 1, 0, 3, 3, 2, 2, 3, 0, 1],
        8 => &[
            0, 255, 128, 1, 254, 255, 0, 129, 254, 1, 128, 1, 0, 255, 129, 254, 128, 255, 0, 1,
        ],
        12 => &[
            0, 4095, 2048, 1, 4094, 4095, 0, 2049, 4094, 1, 2048, 1, 0, 4095, 2049, 4094, 2048,
            4095, 0, 1,
        ],
        16 => &[
            0, 65535, 32768, 1, 65534, 65535, 0, 32769, 65534, 1, 32768, 1, 0, 65535, 32769, 65534,
            32768, 65535, 0, 1,
        ],
        _ => panic!("fixture precision must be declared"),
    }
}
