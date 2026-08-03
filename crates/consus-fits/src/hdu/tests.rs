use alloc::vec::Vec;

use super::*;

use crate::datastructure::{
    FitsBlockAlignment, FitsDataSpan, FitsHeaderBlock, FitsHeaderCardCount,
};
use crate::file::parse_extension_header_bytes;
use crate::header::parse_header_bytes;
use crate::types::HduType;

fn card(text: &str) -> [u8; 80] {
    assert!(text.len() <= 80);
    let mut raw = [b' '; 80];
    raw[..text.len()].copy_from_slice(text.as_bytes());
    raw
}

fn header_bytes(cards: &[&str]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for text in cards {
        bytes.extend_from_slice(&card(text));
    }
    let padded_len = FitsBlockAlignment::padded_len(bytes.len());
    bytes.resize(padded_len, b' ');
    bytes
}

fn primary_hdu() -> FitsHdu {
    let bytes = header_bytes(&[
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    1",
        "NAXIS1  =                    4",
        "END",
    ]);
    let header = parse_header_bytes(&bytes).unwrap();
    FitsHdu::from_header(
        FitsHduIndex::new(0),
        header,
        FitsHeaderBlock::new(FitsHeaderCardCount::new(5)),
        FitsDataSpan::new(2880, 4).unwrap(),
    )
    .unwrap()
}

fn image_extension_hdu() -> FitsHdu {
    let bytes = header_bytes(&[
        "XTENSION= 'IMAGE   '",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        "NAXIS1  =                    2",
        "NAXIS2  =                    3",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "END",
    ]);
    let header = parse_extension_header_bytes(&bytes).unwrap();
    FitsHdu::from_header(
        FitsHduIndex::new(1),
        header,
        FitsHeaderBlock::new(FitsHeaderCardCount::new(7)),
        FitsDataSpan::new(5760, 12).unwrap(),
    )
    .unwrap()
}

fn ascii_table_hdu() -> FitsHdu {
    let bytes = header_bytes(&[
        "XTENSION= 'TABLE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                    8",
        "NAXIS2  =                    2",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TFORM1  = 'A8      '",
        "END",
    ]);
    let header = parse_extension_header_bytes(&bytes).unwrap();
    FitsHdu::from_header(
        FitsHduIndex::new(1),
        header,
        FitsHeaderBlock::new(FitsHeaderCardCount::new(9)),
        FitsDataSpan::new(5760, 16).unwrap(),
    )
    .unwrap()
}

fn binary_table_hdu() -> FitsHdu {
    let bytes = header_bytes(&[
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                    4",
        "NAXIS2  =                    3",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TFORM1  = '1J      '",
        "END",
    ]);
    let header = parse_extension_header_bytes(&bytes).unwrap();
    FitsHdu::from_header(
        FitsHduIndex::new(1),
        header,
        FitsHeaderBlock::new(FitsHeaderCardCount::new(9)),
        FitsDataSpan::new(5760, 12).unwrap(),
    )
    .unwrap()
}

#[test]
fn hdu_kind_derives_from_position_and_xtension() {
    assert_eq!(
        FitsHduKind::from_position_and_xtension(FitsHduIndex::new(0), None).unwrap(),
        FitsHduKind::Primary
    );
    assert_eq!(
        FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("IMAGE")).unwrap(),
        FitsHduKind::ImageExtension
    );
    assert_eq!(
        FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("TABLE")).unwrap(),
        FitsHduKind::AsciiTableExtension
    );
    assert_eq!(
        FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("BINTABLE")).unwrap(),
        FitsHduKind::BinaryTableExtension
    );
}

#[test]
fn hdu_kind_rejects_invalid_primary_and_extension_forms() {
    assert!(FitsHduKind::from_position_and_xtension(FitsHduIndex::new(0), Some("IMAGE")).is_err());
    assert!(FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), None).is_err());
    assert!(
        FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("A3DTABLE")).is_err()
    );
}

#[test]
fn parses_primary_hdu_descriptor() {
    let hdu = primary_hdu();
    assert!(hdu.is_primary());
    assert!(hdu.is_image());
    assert_eq!(hdu.index().get(), 0);
    assert_eq!(hdu.hdu_type(), HduType::Primary);
    assert_eq!(hdu.image().unwrap().axis_lengths(), &[4]);
    assert_eq!(hdu.data_span().logical_len(), 4);
}

#[test]
fn parses_image_extension_descriptor() {
    let hdu = image_extension_hdu();
    assert!(hdu.is_extension());
    assert!(hdu.is_image());
    assert_eq!(hdu.kind(), FitsHduKind::ImageExtension);
    assert_eq!(hdu.hdu_type(), HduType::Image);
    assert_eq!(hdu.image().unwrap().axis_lengths(), &[2, 3]);
}

#[test]
fn parses_ascii_table_extension_descriptor() {
    let hdu = ascii_table_hdu();
    assert!(hdu.is_ascii_table());
    assert_eq!(hdu.kind(), FitsHduKind::AsciiTableExtension);
    assert_eq!(hdu.hdu_type(), HduType::Table);
    assert_eq!(hdu.ascii_table().unwrap().rows(), 2);
    assert_eq!(hdu.ascii_table().unwrap().row_len(), 8);
}

#[test]
fn parses_binary_table_extension_descriptor() {
    let hdu = binary_table_hdu();
    assert!(hdu.is_binary_table());
    assert_eq!(hdu.kind(), FitsHduKind::BinaryTableExtension);
    assert_eq!(hdu.hdu_type(), HduType::BinTable);
    assert_eq!(hdu.binary_table().unwrap().rows(), 3);
    assert_eq!(hdu.binary_table().unwrap().row_len(), 4);
}

#[test]
fn sequence_requires_primary_first_and_contiguous_indices() {
    let primary = primary_hdu();
    let image = image_extension_hdu();

    let sequence = FitsHduSequence::new(vec![primary.clone(), image.clone()]).unwrap();
    assert_eq!(sequence.len(), 2);
    assert!(sequence.primary().unwrap().is_primary());
    assert_eq!(
        sequence.get(FitsHduIndex::new(1)).unwrap().kind(),
        image.kind()
    );

    let invalid_first = FitsHduSequence::new(vec![image.clone()]);
    assert!(invalid_first.is_err());

    let invalid_gap = FitsHduSequence::new(vec![
        primary,
        FitsHdu::new(
            FitsHduIndex::new(2),
            image.kind(),
            image.header().clone(),
            image.header_block(),
            image.data_span(),
            image.payload().clone(),
        ),
    ]);
    assert!(invalid_gap.is_err());
}

#[test]
fn sequence_push_preserves_invariants() {
    let mut sequence = FitsHduSequence::empty();
    sequence.push(primary_hdu()).unwrap();
    sequence.push(image_extension_hdu()).unwrap();

    assert_eq!(sequence.len(), 2);
    assert!(sequence.primary().unwrap().is_primary());

    let invalid_primary_again = sequence.push(primary_hdu());
    assert!(invalid_primary_again.is_err());
}
