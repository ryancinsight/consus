//! Value-semantic tests for the `reader` module.
//!
//! All byte arrays are analytically derived from Thrift compact binary and
//! Parquet wire specifications. Derivations are recorded as inline comments.

use super::page::merge_column_values;
use super::{ColumnPageDecoder, ParquetReader, max_levels_for_field};
use crate::encoding::column::ColumnValues;
use crate::encoding::compression::CompressionCodec;
use crate::schema::field::{FieldDescriptor, FieldId};
use crate::schema::physical::ParquetPhysicalType;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Minimal single-column INT32 Parquet file (98 bytes).
///
/// Layout: PAR1(4) | page_header(17) | values(12) | footer(57) | len(4) | PAR1(4)
/// data_page_offset=4, total_compressed=29, footer_offset=33, footer_len=57.
fn make_synthetic_parquet_file() -> alloc::vec::Vec<u8> {
    alloc::vec![
        // Leading PAR1 magic
        b'P', b'A', b'R', b'1',
        // DataPage v1 header (17 bytes):
        //   type_=DATA_PAGE(0): 0x15 0x00
        //   uncompressed=12 zigzag(24)=0x18: 0x15 0x18
        //   compressed=12: 0x15 0x18
        //   field 5 DataPageHeader STRUCT delta=2: (2<<4)|0x0C=0x2C
        //   num_values=3 zigzag(6)=0x06: 0x15 0x06
        //   encoding=PLAIN=0: 0x15 0x00  def_enc=0: 0x15 0x00  rep_enc=0: 0x15 0x00
        //   DPH stop: 0x00  PH stop: 0x00
        0x15, 0x00, 0x15, 0x18, 0x15, 0x18, 0x2C, 0x15, 0x06, 0x15, 0x00, 0x15, 0x00, 0x15, 0x00,
        0x00, 0x00, // INT32 values [10, 20, 30] little-endian (12 bytes)
        0x0A, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00,
        // FileMetadata footer (57 bytes)
        0x15, 0x04, // version=2: field 1 I32 zigzag(4)=2
        0x19, 0x2C, // schema list: field 2 LIST (2<<4)|STRUCT=0x2C → 2 structs
        // Root SchemaElement: field 4 Binary "schema"(6) field 5 I32 num_children=1 stop
        0x48, 0x06, 0x73, 0x63, 0x68, 0x65, 0x6D, 0x61, 0x15, 0x02, 0x00,
        // "x": field 1 I32 type_=INT32=1 zigzag(2)  field 3 I32 rep=0 delta=2  field 4 "x"  stop
        0x15, 0x02, 0x25, 0x00, 0x18, 0x01, 0x78, 0x00, 0x16,
        0x06, // num_rows=3: field 3 I64 zigzag(6)=3
        0x19, 0x1C, // row_groups list: 1 struct
        0x19, 0x1C, // columns list: 1 struct
        0x26, 0x08, // file_offset=4: field 2 I64 delta=2 zigzag(8)=4
        0x1C, // meta_data STRUCT: field 3 delta=1
        0x15, 0x02, // CM.type_=INT32=1
        0x19, 0x15, 0x00, // CM.encodings=[PLAIN=0]
        0x19, 0x18, 0x01, 0x78, // CM.path=["x"]
        0x15, 0x00, // CM.codec=0 UNCOMPRESSED
        0x16, 0x06, // CM.num_values=3 zigzag(6)=3
        0x16, 0x3A, // CM.total_uncompressed=29 zigzag(58)=0x3A
        0x16, 0x3A, // CM.total_compressed=29
        0x26, 0x08, // CM.data_page_offset=4
        0x00, 0x00, // CM stop, CC stop
        0x16, 0x3A, // RG.total_byte_size=29
        0x16, 0x06, // RG.num_rows=3
        0x00, 0x00, // RG stop, FM stop
        0x3B, 0x00, 0x00, 0x00, // footer_len=59=0x3B LE u32
        b'P', b'A', b'R', b'1', // trailing PAR1
    ]
}

/// DataPage v1 header (17 bytes): type_=0 uncomp=12 comp=12 num_values=3 PLAIN.
const PLAIN_HDR_12_3: &[u8] = &[
    0x15, 0x00, 0x15, 0x18, 0x15, 0x18, 0x2C, 0x15, 0x06, 0x15, 0x00, 0x15, 0x00, 0x15, 0x00, 0x00,
    0x00,
];

/// INT32 PLAIN values [10, 20, 30] little-endian (12 bytes).
const INT32_10_20_30: &[u8] = &[
    0x0A, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00,
];

// ── max_levels_for_field ─────────────────────────────────────────────────────

#[test]
fn max_levels_required_is_zero_zero() {
    let f = FieldDescriptor::required(FieldId::new(1), "x", ParquetPhysicalType::Int32);
    assert_eq!(max_levels_for_field(&f), (0, 0));
}

#[test]
fn max_levels_optional_is_zero_one() {
    let f = FieldDescriptor::optional(FieldId::new(1), "x", ParquetPhysicalType::Int32, None);
    assert_eq!(max_levels_for_field(&f), (0, 1));
}

#[test]
fn max_levels_repeated_is_one_one() {
    let f = FieldDescriptor::repeated(FieldId::new(1), "x", ParquetPhysicalType::Int32);
    assert_eq!(max_levels_for_field(&f), (1, 1));
}

// ── merge_column_values ──────────────────────────────────────────────────────

#[test]
fn merge_empty_returns_invalid_format() {
    let err = merge_column_values(alloc::vec![]).unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn merge_single_part_returns_same() {
    let v = ColumnValues::Int32(alloc::vec![7, 8]);
    let merged = merge_column_values(alloc::vec![v.clone()]).unwrap();
    assert_eq!(merged, v);
}

#[test]
fn merge_two_i32_parts_concatenates() {
    let a = ColumnValues::Int32(alloc::vec![1, 2]);
    let b = ColumnValues::Int32(alloc::vec![3, 4]);
    let merged = merge_column_values(alloc::vec![a, b]).unwrap();
    assert!(matches!(&merged, ColumnValues::Int32(x) if *x == alloc::vec![1, 2, 3, 4]));
}

#[test]
fn merge_three_byte_array_parts_concatenates() {
    let mk = |b: u8| ColumnValues::ByteArray(alloc::vec![alloc::vec![b]]);
    let merged = merge_column_values(alloc::vec![mk(1), mk(2), mk(3)]).unwrap();
    assert_eq!(merged.len(), 3);
    assert!(matches!(&merged, ColumnValues::ByteArray(x)
        if *x == alloc::vec![alloc::vec![1u8], alloc::vec![2u8], alloc::vec![3u8]]));
}

#[test]
fn merge_type_mismatch_returns_datatype_mismatch() {
    let a = ColumnValues::Int32(alloc::vec![1]);
    let b = ColumnValues::Int64(alloc::vec![2]);
    let err = merge_column_values(alloc::vec![a, b]).unwrap_err();
    assert!(matches!(err, consus_core::Error::DatatypeMismatch { .. }));
}

// ── ColumnPageDecoder ────────────────────────────────────────────────────────

#[test]
fn decoder_single_plain_i32_page() {
    let mut chunk = alloc::vec![];
    chunk.extend_from_slice(PLAIN_HDR_12_3);
    chunk.extend_from_slice(INT32_10_20_30);
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let v = dec.decode_pages_from_chunk_bytes(&chunk).unwrap();
    assert_eq!(v.len(), 3);
    assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![10, 20, 30]));
}

#[test]
fn decoder_two_data_pages_concatenated() {
    // Page 2: INT32 [40, 50, 60] (12 bytes).
    let v2: &[u8] = &[
        0x28, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00, 0x3C, 0x00, 0x00, 0x00,
    ];
    let mut chunk = alloc::vec![];
    chunk.extend_from_slice(PLAIN_HDR_12_3);
    chunk.extend_from_slice(INT32_10_20_30);
    chunk.extend_from_slice(PLAIN_HDR_12_3);
    chunk.extend_from_slice(v2);
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let v = dec.decode_pages_from_chunk_bytes(&chunk).unwrap();
    assert_eq!(v.len(), 6);
    assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![10, 20, 30, 40, 50, 60]));
}

#[test]
fn decoder_dictionary_page_then_rle_dict_data_page() {
    // Dict page header (13 bytes):
    //   type_=DICTIONARY_PAGE(2) zigzag(4)=0x04: 0x15 0x04
    //   uncomp=12 comp=12
    //   field 7 DPH STRUCT delta=4: (4<<4)|0x0C=0x4C
    //   num_values=3 zigzag(6)=0x06  encoding=0 PLAIN  DPH stop  PH stop
    //
    // Data page header (17 bytes):
    //   type_=DATA_PAGE(0)  uncomp=4 comp=4
    //   DPH: num_values=4 zigzag(8)=0x08  encoding=RLE_DICT(8) zigzag(16)=0x10
    //   def_enc=0  rep_enc=0  DPH stop  PH stop
    //
    // RLE_DICT bytes: bit_width=2 | 1 bit-packed group | indices[1,2,0,1]→0x49
    //   → dict[1,2,0,1] = [20,30,10,20]
    let chunk: &[u8] = &[
        // Dict page header
        0x15, 0x04, 0x15, 0x18, 0x15, 0x18, 0x4C, 0x15, 0x06, 0x15, 0x00, 0x00, 0x00,
        // Dict values INT32 [10,20,30]
        0x0A, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00,
        // Data page header
        0x15, 0x00, 0x15, 0x08, 0x15, 0x08, 0x2C, 0x15, 0x08, 0x15, 0x10, 0x15, 0x00, 0x15, 0x00,
        0x00, 0x00, // RLE_DICT data
        0x02, 0x01, 0x49, 0x00,
    ];
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let v = dec.decode_pages_from_chunk_bytes(chunk).unwrap();
    assert_eq!(v.len(), 4);
    assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![20, 30, 10, 20]));
}

#[test]
fn decoder_data_page_v2_is_compressed_false() {
    // DataPage V2 header (22 bytes):
    //   type_=DATA_PAGE_V2(3) zigzag(6)=0x06: 0x15 0x06
    //   uncomp=12 comp=12
    //   field 8 V2H STRUCT delta=5 from 3: (5<<4)|0x0C=0x5C
    //   num_values=3  num_nulls=0  num_rows=3  encoding=PLAIN=0
    //   def_byte_len=0  rep_byte_len=0
    //   is_compressed=false: delta=1 BOOL_FALSE nibble=0x02 → (1<<4)|0x02=0x12
    //   V2H stop  PH stop
    let chunk: &[u8] = &[
        0x15, 0x06, 0x15, 0x18, 0x15, 0x18, 0x5C, 0x15, 0x06, // num_values=3
        0x15, 0x00, // num_nulls=0
        0x15, 0x06, // num_rows=3
        0x15, 0x00, // encoding=PLAIN
        0x15, 0x00, // def_byte_len=0
        0x15, 0x00, // rep_byte_len=0
        0x12, // is_compressed=false
        0x00, 0x00, // V2H stop, PH stop
        // INT32 [10, 20, 30]
        0x0A, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00,
    ];
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let v = dec.decode_pages_from_chunk_bytes(chunk).unwrap();
    assert_eq!(v.len(), 3);
    assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![10, 20, 30]));
}

#[test]
fn decoder_empty_chunk_returns_invalid_format() {
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let err = dec.decode_pages_from_chunk_bytes(&[]).unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn decoder_truncated_payload_returns_buffer_too_small() {
    // Header says compressed=12 but only 4 bytes follow.
    let mut chunk = alloc::vec![];
    chunk.extend_from_slice(PLAIN_HDR_12_3);
    chunk.extend_from_slice(&INT32_10_20_30[..4]);
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let err = dec.decode_pages_from_chunk_bytes(&chunk).unwrap_err();
    assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
}

// ── ParquetReader ────────────────────────────────────────────────────────────

#[test]
fn reader_new_decodes_metadata_and_dataset() {
    let file = make_synthetic_parquet_file();
    let r = ParquetReader::new(&file).unwrap();
    assert_eq!(r.metadata().version, 2);
    assert_eq!(r.metadata().num_rows, 3);
    assert_eq!(r.metadata().row_groups.len(), 1);
    assert_eq!(r.dataset().total_rows(), 3);
    assert_eq!(r.dataset().column_count(), 1);
}

#[test]
fn reader_read_column_chunk_returns_correct_i32_values() {
    let file = make_synthetic_parquet_file();
    let r = ParquetReader::new(&file).unwrap();
    let v = r.read_column_chunk(0, 0).unwrap();
    assert_eq!(v.len(), 3);
    assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![10, 20, 30]));
}

#[test]
fn reader_out_of_bounds_row_group_returns_invalid_format() {
    let file = make_synthetic_parquet_file();
    let r = ParquetReader::new(&file).unwrap();
    let err = r.read_column_chunk(1, 0).unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn reader_out_of_bounds_column_returns_invalid_format() {
    let file = make_synthetic_parquet_file();
    let r = ParquetReader::new(&file).unwrap();
    let err = r.read_column_chunk(0, 1).unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn reader_invalid_footer_magic_returns_error() {
    let err = ParquetReader::new(b"this is not a parquet file!!").unwrap_err();
    assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
}

#[test]
fn reader_too_short_returns_buffer_too_small() {
    let err = ParquetReader::new(b"PAR1").unwrap_err();
    assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
}
