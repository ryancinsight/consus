use super::*;
use consus_core::{Error, Selection};
use consus_io::MemCursor;

use crate::datastructure::{FitsBlockAlignment, FitsDataSpan};
use crate::header::parse_header_bytes;
use crate::types::Bitpix;

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

#[test]
fn parses_standard_image_descriptor() {
    let bytes = header_bytes(&[
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        "NAXIS1  =                    3",
        "NAXIS2  =                    2",
        "END",
    ]);

    let header = parse_header_bytes(&bytes).unwrap();
    let descriptor = FitsImageDescriptor::from_header(&header).unwrap();

    assert_eq!(descriptor.bitpix(), Bitpix::I16);
    assert_eq!(descriptor.axis_lengths(), &[3, 2]);
    assert_eq!(descriptor.rank(), 2);
    assert_eq!(descriptor.num_image_elements(), 6);
    assert_eq!(descriptor.logical_data_len().unwrap(), 12);
    assert_eq!(descriptor.scaling(), FitsImageScaling::identity());
    assert!(!descriptor.is_random_groups());
}

#[test]
fn parses_scaling_keywords() {
    let bytes = header_bytes(&[
        "SIMPLE  =                    T",
        "BITPIX  =                  -32",
        "NAXIS   =                    1",
        "NAXIS1  =                    4",
        "BSCALE  =                  2.5",
        "BZERO   =                 -1.0",
        "BLANK   =                 -999",
        "END",
    ]);

    let header = parse_header_bytes(&bytes).unwrap();
    let descriptor = FitsImageDescriptor::from_header(&header).unwrap();

    assert_eq!(descriptor.bitpix(), Bitpix::F32);
    assert_eq!(descriptor.logical_data_len().unwrap(), 16);
    assert_eq!(
        descriptor.scaling(),
        FitsImageScaling {
            bscale: 2.5,
            bzero: -1.0,
            blank: Some(-999),
        }
    );
}

#[test]
fn parses_random_groups_descriptor() {
    let bytes = header_bytes(&[
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    3",
        "NAXIS1  =                    0",
        "NAXIS2  =                    5",
        "NAXIS3  =                    7",
        "GROUPS  =                    T",
        "PCOUNT  =                    2",
        "GCOUNT  =                    3",
        "END",
    ]);

    let header = parse_header_bytes(&bytes).unwrap();
    let descriptor = FitsImageDescriptor::from_header(&header).unwrap();

    assert!(descriptor.is_random_groups());
    assert_eq!(
        descriptor.random_groups(),
        Some(FitsRandomGroups {
            parameter_count: 2,
            group_count: 3,
        })
    );
    assert_eq!(descriptor.logical_data_len().unwrap(), (2 + 35) * 3 * 2);
}

#[test]
fn rejects_random_groups_without_zero_naxis1() {
    let bytes = header_bytes(&[
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                    4",
        "NAXIS2  =                    5",
        "GROUPS  =                    T",
        "PCOUNT  =                    1",
        "GCOUNT  =                    2",
        "END",
    ]);

    let header = parse_header_bytes(&bytes).unwrap();
    let error = FitsImageDescriptor::from_header(&header).unwrap_err();
    assert!(matches!(error, Error::InvalidFormat { .. }));
}

#[test]
fn reads_full_raw_image_payload() {
    let descriptor =
        FitsImageDescriptor::new(Bitpix::U8, vec![4], FitsImageScaling::identity(), None);
    let span = FitsDataSpan::new(0, 4).unwrap();
    let image = FitsImageData::new(descriptor, span);

    let reader = MemCursor::from_bytes(vec![10, 20, 30, 40]);
    let mut buf = [0u8; 4];
    let read = image.read_all(&reader, &mut buf).unwrap();

    assert_eq!(read, 4);
    assert_eq!(buf, [10, 20, 30, 40]);
}

#[test]
fn read_selection_supports_all_and_none_only() {
    let descriptor =
        FitsImageDescriptor::new(Bitpix::U8, vec![2, 2], FitsImageScaling::identity(), None);
    let span = FitsDataSpan::new(0, 4).unwrap();
    let image = FitsImageData::new(descriptor, span);

    let reader = MemCursor::from_bytes(vec![1, 2, 3, 4]);
    let mut buf = [0u8; 4];

    let read = image
        .read_selection(&reader, &Selection::All, &mut buf)
        .unwrap();
    assert_eq!(read, 4);
    assert_eq!(buf, [1, 2, 3, 4]);

    let read_none = image
        .read_selection(&reader, &Selection::None, &mut buf)
        .unwrap();
    assert_eq!(read_none, 0);
}
