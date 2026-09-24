//! Single-chunk read/write round-trip and bounds tests.

use super::super::*;
use crate::chunk::error::ChunkError;
use crate::metadata::{ArrayMetadata, ChunkKeyEncoding, Codec, FillValue, ZarrVersion};
use crate::store::InMemoryStore;

#[test]
fn read_write_chunk() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![100, 100],
        chunks: vec![10, 10],
        dtype: "<f8".to_string(),
        fill_value: FillValue::Float("NaN".to_string()),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let chunk_data: Vec<u8> = (0..100)
        .flat_map(|i| {
            let val = i as f64;
            val.to_le_bytes()
        })
        .collect();

    // Write chunk
    write_chunk(&mut store, "test_array", &[0u64, 0u64], &meta, &chunk_data).unwrap();

    // Read chunk
    let read_data = read_chunk(&store, "test_array", &[0u64, 0u64], &meta).unwrap();
    assert_eq!(read_data, chunk_data);
}

#[test]
fn read_write_chunk_with_compression() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![100, 100],
        chunks: vec![10, 10],
        dtype: "<f8".to_string(),
        fill_value: FillValue::Float("NaN".to_string()),
        order: 'C',
        codecs: vec![Codec {
            name: "gzip".to_string(),
            configuration: vec![("level".to_string(), "1".to_string())],
        }],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let chunk_data: Vec<u8> = (0..100)
        .flat_map(|i| {
            let val = i as f64;
            val.to_le_bytes()
        })
        .collect();

    // Write chunk with compression
    write_chunk(&mut store, "test_array", &[0u64, 0u64], &meta, &chunk_data).unwrap();

    // Read chunk and decompress
    let read_data = read_chunk(&store, "test_array", &[0u64, 0u64], &meta).unwrap();
    assert_eq!(read_data, chunk_data);
}

#[test]
fn read_chunk_out_of_bounds() {
    let store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![10, 10],
        chunks: vec![5, 5],
        dtype: "<f8".to_string(),
        fill_value: FillValue::Float("NaN".to_string()),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let result = read_chunk(&store, "test_array", &[2u64, 0u64], &meta);
    assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

    let result = read_chunk(&store, "test_array", &[0u64, 2u64], &meta);
    assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

    let result = read_chunk(&store, "test_array", &[2u64, 2u64], &meta);
    assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));
}

#[test]
fn write_chunk_out_of_bounds() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![10, 10],
        chunks: vec![5, 5],
        dtype: "<f8".to_string(),
        fill_value: FillValue::Float("NaN".to_string()),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let chunk_data: Vec<u8> = (0..25)
        .flat_map(|i| {
            let val = i as f64;
            val.to_le_bytes()
        })
        .collect();

    let result = write_chunk(&mut store, "test_array", &[2u64, 0u64], &meta, &chunk_data);
    assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

    let result = write_chunk(&mut store, "test_array", &[0u64, 2u64], &meta, &chunk_data);
    assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

    let result = write_chunk(&mut store, "test_array", &[2u64, 2u64], &meta, &chunk_data);
    assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));
}

#[test]
fn uninitialized_chunk_returns_empty() {
    let store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![10, 10],
        chunks: vec![5, 5],
        dtype: "<f8".to_string(),
        fill_value: FillValue::Float("NaN".to_string()),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    // Chunk [0, 0] was never written, should return Uninitialized
    let result = read_chunk(&store, "test_array", &[0u64, 0u64], &meta);
    assert!(matches!(result, Err(ChunkError::Uninitialized)));
}
