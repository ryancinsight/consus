use super::fixture_limits;
use crate::DecodeErrorKind;
use crate::jpeg::decode;

#[test]
fn empty_exif_ifd_and_jpeg_xt_metadata_are_rejected() {
    let bytes =
        include_bytes!("../../../tests/fixtures/jpeg/precision-reference-empty-ifd-arithmetic.jpg");
    let error = decode(bytes, fixture_limits(bytes, 24, 8))
        .expect_err("TIFF 6.0 requires at least one entry in every IFD");
    assert_eq!(error.kind(), DecodeErrorKind::Malformed);

    let [0xff, 0xd8, 0xff, 0xe1, length_high, length_low, tail @ ..] = bytes.as_slice() else {
        panic!("invariant: negative fixture begins with SOI and APP1")
    };
    let payload_length = usize::from(u16::from_be_bytes([*length_high, *length_low]))
        .checked_sub(2)
        .expect("invariant: APP1 length includes its two-byte length field");
    let after_app1 = tail
        .get(payload_length..)
        .expect("invariant: APP1 payload is contained in the fixture");
    let mut without_app1 = Vec::with_capacity(2 + after_app1.len());
    without_app1.extend_from_slice(&[0xff, 0xd8]);
    without_app1.extend_from_slice(after_app1);
    let error = decode(&without_app1, fixture_limits(&without_app1, 24, 8))
        .expect_err("JPEG XT APP11 metadata is outside the admitted subset");
    assert_eq!(error.kind(), DecodeErrorKind::Unsupported);
}

#[test]
fn empty_arithmetic_conditioning_segment_is_malformed() {
    let source = include_bytes!(
        "../../../tests/fixtures/jpeg/precision-reference-progressive-arithmetic.jpg"
    );
    let mut bytes = source.to_vec();
    let scan = bytes
        .windows(2)
        .position(|window| window == [0xff, 0xda])
        .expect("invariant: progressive fixture contains a scan header");
    assert_eq!(
        bytes.splice(scan..scan, [0xff, 0xcc, 0x00, 0x02]).count(),
        0
    );

    let error = decode(&bytes, fixture_limits(&bytes, 24, 8))
        .expect_err("T.81 DAC segments contain one or more conditioning tables");
    assert_eq!(error.kind(), DecodeErrorKind::Malformed);
}
