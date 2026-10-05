//! Lossless scan-header selector coverage.
//!
//! `Ss` is the scan header's predictor selector. T.81 defines 1 through 7;
//! DICOM additionally defines `Ss = 0` for `JpegLosslessNonHierarchical`,
//! meaning no selector and a plain `Rb` prediction.

use super::{decode, limits, lossless_scan_fixture};

#[test]
fn dicom_non_hierarchical_ss_zero_decodes() {
    // DICOM's `JpegLosslessNonHierarchical` writes `Ss = 0`. This decoder used
    // to reject that outright, which made every such stream undecodable here --
    // a DICOM-completeness gap, not malformed-input rejection.
    let bytes = lossless_scan_fixture(8, 0, 0);
    let decoded = decode(&bytes, limits(&bytes)).expect("Ss = 0 must decode");
    assert_eq!(decoded.width(), 1);
    assert_eq!(decoded.height(), 1);
}

#[test]
fn ss_zero_and_ss_two_reconstruct_identically() {
    // `Ss = 0` carries no selector and DICOM defines it as a plain `Rb`
    // prediction, which is exactly what selector 2 asks for. Same entropy data,
    // so the two must agree.
    let zero = lossless_scan_fixture(8, 0, 0);
    let two = lossless_scan_fixture(8, 0, 2);
    assert_eq!(
        decode(&zero, limits(&zero))
            .expect("Ss = 0 decodes")
            .pixels(),
        decode(&two, limits(&two)).expect("Ss = 2 decodes").pixels(),
        "Ss = 0 and Ss = 2 must reconstruct the same samples"
    );
}

#[test]
fn ss_above_seven_is_still_rejected() {
    // Widening `Ss` must not widen it past the standard: 8 and above remain
    // malformed, so a truncated or corrupt scan header is still caught.
    let bytes = lossless_scan_fixture(8, 0, 8);
    assert!(
        decode(&bytes, limits(&bytes)).is_err(),
        "Ss = 8 exceeds every defined selector and must stay rejected"
    );
}
