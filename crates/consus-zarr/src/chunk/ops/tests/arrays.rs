//! Array-level selection read/write tests.

use super::super::*;
use crate::chunk::error::ChunkError;
use crate::chunk::selection::{Selection, SelectionStep};
use crate::metadata::{ArrayMetadata, ChunkKeyEncoding, FillValue, ZarrVersion};
use crate::store::InMemoryStore;

#[test]
fn read_array_partial_selection_contiguous_across_chunks() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![4, 4],
        chunks: vec![2, 2],
        dtype: "<i4".to_string(),
        fill_value: FillValue::Int(0),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let data: Vec<u8> = (0..16i32).flat_map(|value| value.to_le_bytes()).collect();
    write_array(&mut store, "test_array", &meta, &data).unwrap();

    let selection = Selection::from_steps(vec![
        SelectionStep {
            start: 1,
            count: 2,
            stride: 1,
        },
        SelectionStep {
            start: 1,
            count: 3,
            stride: 1,
        },
    ]);

    let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
    let values: Vec<i32> = read_data
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(values, vec![5, 6, 7, 9, 10, 11]);
}

#[test]
fn read_array_partial_selection_strided_across_chunks() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![4, 4],
        chunks: vec![2, 2],
        dtype: "<i4".to_string(),
        fill_value: FillValue::Int(0),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let data: Vec<u8> = (0..16i32).flat_map(|value| value.to_le_bytes()).collect();
    write_array(&mut store, "test_array", &meta, &data).unwrap();

    let selection = Selection::from_steps(vec![
        SelectionStep {
            start: 0,
            count: 2,
            stride: 2,
        },
        SelectionStep {
            start: 1,
            count: 2,
            stride: 2,
        },
    ]);

    let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
    let values: Vec<i32> = read_data
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(values, vec![1, 3, 9, 11]);
}

#[test]
fn read_array_partial_selection_uninitialized_chunk_uses_fill_value() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![4, 4],
        chunks: vec![2, 2],
        dtype: "<i4".to_string(),
        fill_value: FillValue::Int(-1),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let initialized_chunk: Vec<u8> = [0i32, 1, 4, 5]
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    write_chunk(
        &mut store,
        "test_array",
        &[0u64, 0u64],
        &meta,
        &initialized_chunk,
    )
    .unwrap();

    let selection = Selection::from_steps(vec![
        SelectionStep {
            start: 0,
            count: 2,
            stride: 1,
        },
        SelectionStep {
            start: 0,
            count: 4,
            stride: 1,
        },
    ]);

    let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
    let values: Vec<i32> = read_data
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(values, vec![0, 1, -1, -1, 4, 5, -1, -1]);
}

#[test]
fn write_array_selection_contiguous_across_chunks() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![4, 4],
        chunks: vec![2, 2],
        dtype: "<i4".to_string(),
        fill_value: FillValue::Int(0),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let initial: Vec<u8> = (0..16i32).flat_map(|value| value.to_le_bytes()).collect();
    write_array(&mut store, "test_array", &meta, &initial).unwrap();

    let selection = Selection::from_steps(vec![
        SelectionStep {
            start: 1,
            count: 2,
            stride: 1,
        },
        SelectionStep {
            start: 1,
            count: 2,
            stride: 1,
        },
    ]);
    let patch: Vec<u8> = [100i32, 101, 102, 103]
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    write_array_selection(&mut store, "test_array", &selection, &meta, &patch).unwrap();

    let read_back = read_array(&store, "test_array", &Selection::full(2), &meta).unwrap();
    let values: Vec<i32> = read_back
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(
        values,
        vec![0, 1, 2, 3, 4, 100, 101, 7, 8, 102, 103, 11, 12, 13, 14, 15]
    );
}

#[test]
fn write_array_selection_strided_across_chunks() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![4, 4],
        chunks: vec![2, 2],
        dtype: "<i4".to_string(),
        fill_value: FillValue::Int(0),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let initial: Vec<u8> = (0..16i32).flat_map(|value| value.to_le_bytes()).collect();
    write_array(&mut store, "test_array", &meta, &initial).unwrap();

    let selection = Selection::from_steps(vec![
        SelectionStep {
            start: 0,
            count: 2,
            stride: 2,
        },
        SelectionStep {
            start: 1,
            count: 2,
            stride: 2,
        },
    ]);
    let patch: Vec<u8> = [200i32, 201, 202, 203]
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    write_array_selection(&mut store, "test_array", &selection, &meta, &patch).unwrap();

    let read_back = read_array(&store, "test_array", &Selection::full(2), &meta).unwrap();
    let values: Vec<i32> = read_back
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(
        values,
        vec![0, 200, 2, 201, 4, 5, 6, 7, 8, 202, 10, 203, 12, 13, 14, 15]
    );
}

#[test]
fn write_array_selection_uninitialized_chunks_materialize_fill_value() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![4, 4],
        chunks: vec![2, 2],
        dtype: "<i4".to_string(),
        fill_value: FillValue::Int(-1),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let selection = Selection::from_steps(vec![
        SelectionStep {
            start: 1,
            count: 2,
            stride: 1,
        },
        SelectionStep {
            start: 1,
            count: 2,
            stride: 1,
        },
    ]);
    let patch: Vec<u8> = [50i32, 51, 52, 53]
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    write_array_selection(&mut store, "test_array", &selection, &meta, &patch).unwrap();

    let read_back = read_array(&store, "test_array", &Selection::full(2), &meta).unwrap();
    let values: Vec<i32> = read_back
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(
        values,
        vec![
            -1, -1, -1, -1, -1, 50, 51, -1, -1, 52, 53, -1, -1, -1, -1, -1
        ]
    );
}

#[test]
fn write_array_selection_rejects_invalid_input_length() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![4, 4],
        chunks: vec![2, 2],
        dtype: "<i4".to_string(),
        fill_value: FillValue::Int(0),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let selection = Selection::from_steps(vec![
        SelectionStep {
            start: 1,
            count: 2,
            stride: 1,
        },
        SelectionStep {
            start: 1,
            count: 2,
            stride: 1,
        },
    ]);
    let invalid_patch: Vec<u8> = [1i32, 2, 3]
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    let result = write_array_selection(&mut store, "test_array", &selection, &meta, &invalid_patch);
    assert!(matches!(result, Err(ChunkError::UnexpectedLength)));
}

#[test]
fn write_array_and_read_back() {
    let mut store = InMemoryStore::new();
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![20, 20],
        chunks: vec![10, 10],
        dtype: "<f4".to_string(),
        fill_value: FillValue::Float("0.0".to_string()),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };

    let data: Vec<u8> = (0..400)
        .flat_map(|i| {
            let val = i as f32;
            val.to_le_bytes()
        })
        .collect();

    write_array(&mut store, "test_array", &meta, &data).unwrap();

    let selection = Selection::full(2);
    let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
    assert_eq!(read_data, data);
}
