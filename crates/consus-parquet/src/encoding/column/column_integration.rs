//! Integration tests for [`decode_compressed_column_values`].

use super::*;
use crate::encoding::compression::CompressionCodec;

#[test]
fn decode_compressed_uncompressed_i32_matches_decode_column_values() {
    let b = [
        0x01u8, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,
    ];
    let v = decode_compressed_column_values(
        &b,
        3,
        0,
        ParquetPhysicalType::Int32,
        None,
        CompressionCodec::Uncompressed,
        b.len(),
    )
    .unwrap();
    assert_eq!(v.len(), 3);
    assert!(matches!(v, ColumnValues::Int32(ref x) if *x == alloc::vec![1, -1, i32::MAX]));
}

#[test]
fn decode_compressed_brotli_returns_unsupported() {
    let err = decode_compressed_column_values(
        &[0x01u8, 0x02, 0x03],
        1,
        0,
        ParquetPhysicalType::Int32,
        None,
        CompressionCodec::Brotli,
        4,
    )
    .unwrap_err();
    assert!(matches!(err, consus_core::Error::UnsupportedFeature { .. }));
}

#[cfg(feature = "gzip")]
#[test]
fn decode_compressed_gzip_i32_round_trip() {
    use flate2::read::DeflateEncoder;
    use std::io::Read;
    let original: Vec<u8> = alloc::vec![
        0x0Au8, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00
    ];
    let mut encoder = DeflateEncoder::new(&original[..], flate2::Compression::new(6));
    let mut compressed = Vec::new();
    encoder
        .read_to_end(&mut compressed)
        .expect("deflate encode must succeed");
    let v = decode_compressed_column_values(
        &compressed,
        3,
        0,
        ParquetPhysicalType::Int32,
        None,
        CompressionCodec::Gzip,
        original.len(),
    )
    .unwrap();
    assert_eq!(v.len(), 3);
    assert!(matches!(v, ColumnValues::Int32(ref x) if *x == alloc::vec![10, 20, 30]));
}

#[cfg(feature = "gzip")]
#[test]
fn decode_compressed_malformed_gzip_returns_compression_error() {
    let garbage: Vec<u8> = alloc::vec![0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x01, 0x02, 0x03];
    let err = decode_compressed_column_values(
        &garbage,
        3,
        0,
        ParquetPhysicalType::Int32,
        None,
        CompressionCodec::Gzip,
        12,
    )
    .unwrap_err();
    assert!(matches!(err, consus_core::Error::CompressionError { .. }));
}

#[cfg(feature = "zstd")]
#[test]
fn decode_compressed_zstd_i32_round_trip() {
    let original: Vec<u8> = alloc::vec![
        0x0Au8, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00
    ];
    let compressed = zstd::bulk::compress(&original, 3).expect("zstd compress must succeed");
    let v = decode_compressed_column_values(
        &compressed,
        3,
        0,
        ParquetPhysicalType::Int32,
        None,
        CompressionCodec::Zstd,
        original.len(),
    )
    .unwrap();
    assert_eq!(v.len(), 3);
    assert!(matches!(v, ColumnValues::Int32(ref x) if *x == alloc::vec![10, 20, 30]));
}

#[cfg(feature = "snappy")]
#[test]
fn decode_compressed_snappy_rle_dict_round_trip() {
    let dict = ColumnValues::Int32(alloc::vec![10, 20, 30]);
    let original: Vec<u8> = alloc::vec![0x02u8, 0x01, 0x49, 0x00];
    let mut encoder = snap::raw::Encoder::new();
    let compressed = encoder
        .compress_vec(&original)
        .expect("snappy compress must succeed");
    let v = decode_compressed_column_values(
        &compressed,
        4,
        8,
        ParquetPhysicalType::Int32,
        Some(&dict),
        CompressionCodec::Snappy,
        original.len(),
    )
    .unwrap();
    assert_eq!(v.len(), 4);
    assert!(matches!(v, ColumnValues::Int32(ref x) if *x == alloc::vec![20, 30, 10, 20]));
}

#[cfg(feature = "lz4")]
#[test]
fn decode_compressed_lz4_i32_round_trip() {
    let original: Vec<u8> = alloc::vec![
        0x0Au8, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00
    ];
    let compressed = lz4_flex::compress_prepend_size(&original);
    let v = decode_compressed_column_values(
        &compressed,
        3,
        0,
        ParquetPhysicalType::Int32,
        None,
        CompressionCodec::Lz4,
        original.len(),
    )
    .unwrap();
    assert_eq!(v.len(), 3);
    assert!(matches!(v, ColumnValues::Int32(ref x) if *x == alloc::vec![10, 20, 30]));
}
