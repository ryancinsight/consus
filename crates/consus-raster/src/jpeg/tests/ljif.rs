use super::*;

const LJIF: &[u8] = b"\xff\xe0\x00\x0fLJIF\0\x01\0\0\0\0\0\0\0";

fn insert_app0(encoded: &[u8], marker: &[u8]) -> Vec<u8> {
    [&encoded[..2], marker, &encoded[2..]].concat()
}

fn traced_case<Input, Output>(
    arrange: impl FnOnce() -> Input,
    act: impl FnOnce(Input) -> Output,
    assert: impl FnOnce(Output),
) {
    let subscriber = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let _test = tracing::info_span!("jpeg_ljif").entered();
        let input = tracing::info_span!("arrange").in_scope(arrange);
        let output = tracing::info_span!("act").in_scope(|| act(input));
        tracing::info_span!("assert").in_scope(|| assert(output));
    });
}

#[test]
fn grayscale_lossless_ljif_preserves_integer_samples() {
    for (category, magnitude, expected) in [(15, 0x1233, 0x1234_u16), (15, 0x7fff, u16::MAX)] {
        traced_case(
            || insert_app0(&lossless_fixture(16, category, magnitude), LJIF),
            |encoded| decode(&encoded, limits()),
            |result| {
                let decoded = result.expect("16-bit LJIF");
                // The initial predictor is 32768. Category 15 differences
                // -28108 and +32767 reconstruct 4660 and 65535 respectively.
                assert_eq!((decoded.width(), decoded.height()), (1, 1));
                assert_eq!(decoded.format(), PixelFormat::GrayWide);
                assert_eq!(decoded.pixels(), &expected.to_ne_bytes());
            },
        );
    }
}

#[test]
fn ljif_rejects_bad_length_mode_and_duplicate_header() {
    let mut unknown_mode = LJIF.to_vec();
    *unknown_mode.last_mut().expect("LJIF mode") = 2;
    let short_body = b"\xff\xe0\x00\x0eLJIF\0\x01\0\0\0\0\0\0";
    let duplicate = [LJIF, LJIF].concat();

    for (marker, expected) in [
        (unknown_mode.as_slice(), DecodeErrorKind::Unsupported),
        (short_body.as_slice(), DecodeErrorKind::Malformed),
        (duplicate.as_slice(), DecodeErrorKind::Malformed),
    ] {
        traced_case(
            || insert_app0(&lossless_fixture(16, 15, 0x1233), marker),
            |encoded| decode(&encoded, limits()),
            |result| {
                assert_eq!(
                    result.expect_err("unsupported or malformed LJIF").kind(),
                    expected
                )
            },
        );
    }
}

#[test]
fn ljif_rejects_a_valid_dct_frame() {
    traced_case(
        || {
            let encoded = from_hex(super::super::fixture_data::SEQUENTIAL);
            let [0xff, 0xd8, 0xff, 0xe0, length_hi, length_lo, ..] = encoded.as_slice() else {
                panic!("sequential fixture starts with JFIF APP0");
            };
            let app0_end = 4 + usize::from(u16::from_be_bytes([*length_hi, *length_lo]));
            let without_jfif = [&encoded[..2], &encoded[app0_end..]].concat();
            (insert_app0(&without_jfif, LJIF), without_jfif)
        },
        |(with_ljif, without_jfif)| {
            (
                decode(&with_ljif, limits()),
                decode(&without_jfif, limits()),
            )
        },
        |(with_ljif, without_jfif)| {
            let decoded = without_jfif.expect("valid sequential DCT");
            assert_eq!((decoded.width(), decoded.height()), (17, 25));
            assert_eq!(
                with_ljif.expect_err("LJIF requires SOF3").kind(),
                DecodeErrorKind::Unsupported
            );
        },
    );
}
