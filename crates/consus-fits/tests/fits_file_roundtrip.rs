#![cfg(feature = "alloc")]

use consus_core::{FileRead, FileWrite, Hyperslab, HyperslabDim, Selection, Shape};
use consus_fits::FitsFile;
use consus_io::MemCursor;

const FITS_CARD_LEN: usize = 80;
const FITS_BLOCK_LEN: usize = 2880;

fn card(text: &str) -> [u8; FITS_CARD_LEN] {
    assert!(text.len() <= FITS_CARD_LEN);
    let mut raw = [b' '; FITS_CARD_LEN];
    raw[..text.len()].copy_from_slice(text.as_bytes());
    raw
}

fn append_header(bytes: &mut Vec<u8>, cards: &[&str]) {
    let start = bytes.len();
    for text in cards {
        bytes.extend_from_slice(&card(text));
    }

    let header_len = bytes.len() - start;
    let padded_len = padded_len(header_len);
    bytes.resize(start + padded_len, b' ');
}

fn append_data(bytes: &mut Vec<u8>, data: &[u8]) {
    bytes.extend_from_slice(data);
    let padded = padded_len(data.len());
    bytes.resize(bytes.len() + (padded - data.len()), 0);
}

fn padded_len(len: usize) -> usize {
    let remainder = len % FITS_BLOCK_LEN;
    if remainder == 0 {
        len
    } else {
        len + (FITS_BLOCK_LEN - remainder)
    }
}

fn primary_image_and_bintable_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();

    append_header(
        &mut bytes,
        &[
            "SIMPLE  =                    T / standard FITS primary HDU",
            "BITPIX  =                    8 / unsigned byte image",
            "NAXIS   =                    2 / rank",
            "NAXIS1  =                    4 / axis 1 length",
            "NAXIS2  =                    2 / axis 2 length",
            "OBJECT  = 'ROUNDTRIP'           / integration test payload",
            "END",
        ],
    );
    append_data(&mut bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);

    append_header(
        &mut bytes,
        &[
            "XTENSION= 'BINTABLE'           / binary table extension",
            "BITPIX  =                    8 / table bytes",
            "NAXIS   =                    2 / table rank",
            "NAXIS1  =                    4 / row length",
            "NAXIS2  =                    3 / row count",
            "PCOUNT  =                    0 / no heap",
            "GCOUNT  =                    1 / one table",
            "TFIELDS =                    2 / two columns",
            "TTYPE1  = 'X       '           / first column",
            "TFORM1  = '1I      '           / 16-bit integer",
            "TTYPE2  = 'Y       '           / second column",
            "TFORM2  = '1I      '           / 16-bit integer",
            "END",
        ],
    );
    append_data(
        &mut bytes,
        &[10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],
    );

    bytes
}

fn primary_random_groups_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();

    append_header(
        &mut bytes,
        &[
            "SIMPLE  =                    T / standard FITS primary HDU",
            "BITPIX  =                   16 / signed 16-bit values",
            "NAXIS   =                    3 / random groups rank",
            "NAXIS1  =                    0 / required for random groups",
            "NAXIS2  =                    2 / image axis 1",
            "NAXIS3  =                    3 / image axis 2",
            "GROUPS  =                    T / legacy random groups",
            "PCOUNT  =                    1 / one parameter per group",
            "GCOUNT  =                    2 / two groups",
            "END",
        ],
    );

    // (PCOUNT + NAXIS2*NAXIS3) * GCOUNT * element_size
    // = (1 + 2*3) * 2 * 2 = 28 bytes
    append_data(
        &mut bytes,
        &[
            0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, //
            0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14,
        ],
    );

    bytes
}

#[test]
fn round_trips_primary_image_and_binary_table_reference_style_hdus() {
    let cursor = MemCursor::from_bytes(primary_image_and_bintable_bytes());
    let mut file = FitsFile::open_mut(cursor).expect("FITS file must scan");

    assert_eq!(file.format(), "fits");
    assert_eq!(file.hdu_count(), 2);
    assert_eq!(file.num_children_at("/").unwrap(), 2);
    assert_eq!(
        file.dataset_shape("/PRIMARY").unwrap(),
        Shape::fixed(&[4, 2])
    );
    assert_eq!(file.dataset_shape("/HDU/1").unwrap(), Shape::fixed(&[3]));

    let mut primary = [0u8; 8];
    let read = file
        .read_dataset_raw("/PRIMARY", &Selection::All, &mut primary)
        .expect("primary image read must succeed");
    assert_eq!(read, 8);
    assert_eq!(primary, [1, 2, 3, 4, 5, 6, 7, 8]);

    let mut table = [0u8; 12];
    let read = file
        .read_dataset_raw("/HDU/1", &Selection::All, &mut table)
        .expect("binary table read must succeed");
    assert_eq!(read, 12);
    assert_eq!(table, [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61]);

    file.write_dataset_raw("/PRIMARY", &Selection::All, &[8, 7, 6, 5, 4, 3, 2, 1])
        .expect("primary image overwrite must succeed");

    let row_selection = Selection::Hyperslab(Hyperslab::new(&[HyperslabDim {
        start: 1,
        stride: 1,
        count: 2,
        block: 1,
    }]));
    file.write_dataset_raw("/HDU/1", &row_selection, &[90, 91, 92, 93, 94, 95, 96, 97])
        .expect("binary table row overwrite must succeed");

    let mut primary_after = [0u8; 8];
    let read = file
        .read_dataset_raw("/PRIMARY", &Selection::All, &mut primary_after)
        .expect("primary image reread must succeed");
    assert_eq!(read, 8);
    assert_eq!(primary_after, [8, 7, 6, 5, 4, 3, 2, 1]);

    let mut table_after = [0u8; 12];
    let read = file
        .read_dataset_raw("/HDU/1", &Selection::All, &mut table_after)
        .expect("binary table reread must succeed");
    assert_eq!(read, 12);
    assert_eq!(
        table_after,
        [10, 11, 20, 21, 90, 91, 92, 93, 94, 95, 96, 97]
    );

    file.flush().expect("flush must succeed");
}

#[test]
fn scans_and_reads_legacy_random_groups_primary_hdu() {
    let cursor = MemCursor::from_bytes(primary_random_groups_bytes());
    let file = FitsFile::open(cursor).expect("random-groups FITS file must scan");

    assert_eq!(file.hdu_count(), 1);
    assert!(file.exists("/PRIMARY").unwrap());

    let mut payload = [0u8; 28];
    let read = file
        .read_dataset_raw("/PRIMARY", &Selection::All, &mut payload)
        .expect("random-groups payload read must succeed");

    assert_eq!(read, 28);
    assert_eq!(
        payload,
        [
            0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, //
            0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14
        ]
    );
}

#[test]
fn preserves_fits_block_padding_after_roundtrip_writes() {
    let original = primary_image_and_bintable_bytes();
    let original_len = original.len();

    let cursor = MemCursor::from_bytes(original);
    let mut file = FitsFile::open_mut(cursor).expect("FITS file must scan");

    file.write_dataset_raw(
        "/PRIMARY",
        &Selection::All,
        &[42, 43, 44, 45, 46, 47, 48, 49],
    )
    .expect("primary overwrite must succeed");
    file.write_dataset_raw(
        "/HDU/1",
        &Selection::All,
        &[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    )
    .expect("table overwrite must succeed");
    file.flush().expect("flush must succeed");

    let reopened = FitsFile::open(file.io().clone()).expect("rewrapped FITS file must rescan");
    assert_eq!(reopened.hdu_count(), 2);

    let mut primary = [0u8; 8];
    reopened
        .read_dataset_raw("/PRIMARY", &Selection::All, &mut primary)
        .expect("primary reread must succeed");
    assert_eq!(primary, [42, 43, 44, 45, 46, 47, 48, 49]);

    let mut table = [0u8; 12];
    reopened
        .read_dataset_raw("/HDU/1", &Selection::All, &mut table)
        .expect("table reread must succeed");
    assert_eq!(table, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]);

    assert_eq!(reopened.io().as_bytes().len(), original_len);
    assert_eq!(reopened.io().as_bytes().len() % FITS_BLOCK_LEN, 0);
}
