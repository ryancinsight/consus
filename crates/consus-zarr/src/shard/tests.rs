use super::*;
use crate::metadata::{ArrayMetadata, ChunkKeyEncoding, Codec, FillValue, ZarrVersion};

fn make_meta(shape: Vec<usize>, chunks: Vec<usize>) -> ArrayMetadata {
    ArrayMetadata {
        version: ZarrVersion::V3,
        shape,
        chunks,
        dtype: alloc::string::String::from("float64"),
        fill_value: FillValue::Float(alloc::string::String::from("0.0")),
        order: 'C',
        codecs: alloc::vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    }
}

#[test]
fn sharding_config_total_inner_chunks() {
    let cfg = ShardingConfig {
        inner_chunk_shape: vec![2, 2],
        inner_codecs: vec![],
        index_codecs: vec![],
    };
    assert_eq!(cfg.total_inner_chunks(&[4, 4]), 4);
    assert_eq!(cfg.total_inner_chunks(&[4, 6]), 6);
}

#[test]
fn sharding_config_index_size_bytes() {
    let cfg = ShardingConfig {
        inner_chunk_shape: vec![2, 2],
        inner_codecs: vec![],
        index_codecs: vec![],
    };
    assert_eq!(cfg.index_size_bytes(&[4, 4]), 64);
}

#[test]
fn sharding_config_inner_chunks_per_dim_partial() {
    let cfg = ShardingConfig {
        inner_chunk_shape: vec![2, 3],
        inner_codecs: vec![],
        index_codecs: vec![],
    };
    assert_eq!(cfg.inner_chunks_per_dim(&[5, 7]), vec![3, 3]);
}

#[test]
fn inner_linear_index_correctness() {
    assert_eq!(inner_linear_index(&[1, 2], &[2, 3]), 5);
    assert_eq!(inner_linear_index(&[0, 0], &[2, 3]), 0);
    assert_eq!(inner_linear_index(&[1, 0], &[2, 3]), 3);
}

#[test]
fn write_shard_and_read_inner_chunk() {
    let mut inner_chunks = alloc::collections::BTreeMap::new();
    inner_chunks.insert(0usize, vec![42u8, 0, 0, 0, 43, 0, 0, 0]);
    inner_chunks.insert(2usize, vec![99u8, 0, 0, 0, 100, 0, 0, 0]);
    let total = 4usize;
    let shard = write_shard(&inner_chunks, total);
    assert_eq!(shard.len(), 16 + 64);
    let c0 = read_inner_chunk_from_shard(&shard, 0, total, &[], 1).unwrap();
    assert_eq!(c0, vec![42u8, 0, 0, 0, 43, 0, 0, 0]);
    let c1 = read_inner_chunk_from_shard(&shard, 1, total, &[], 1).unwrap();
    assert!(c1.is_empty(), "uninitialized chunk must return empty vec");
    let c2 = read_inner_chunk_from_shard(&shard, 2, total, &[], 1).unwrap();
    assert_eq!(c2, vec![99u8, 0, 0, 0, 100, 0, 0, 0]);
    let c3 = read_inner_chunk_from_shard(&shard, 3, total, &[], 1).unwrap();
    assert!(c3.is_empty(), "uninitialized chunk must return empty vec");
}

#[test]
fn write_shard_all_uninitialized() {
    let inner_chunks = alloc::collections::BTreeMap::new();
    let total = 4usize;
    let shard = write_shard(&inner_chunks, total);
    assert_eq!(shard.len(), 64);
    for i in 0..total {
        let c = read_inner_chunk_from_shard(&shard, i, total, &[], 1).unwrap();
        assert!(c.is_empty());
    }
}

#[test]
fn extract_sharding_config_basic() {
    let codec = Codec {
        name: alloc::string::String::from("sharding_indexed"),
        configuration: vec![
            (
                alloc::string::String::from("chunk_shape"),
                alloc::string::String::from("[2,2]"),
            ),
            (
                alloc::string::String::from("codecs"),
                alloc::string::String::from(
                    r#"[{"name":"bytes","configuration":{"endian":"little"}}]"#,
                ),
            ),
            (
                alloc::string::String::from("index_codecs"),
                alloc::string::String::from("[]"),
            ),
        ],
    };
    let cfg = extract_sharding_config(&[codec]);
    assert!(cfg.is_some());
    let cfg = cfg.unwrap();
    assert_eq!(cfg.inner_chunk_shape, vec![2, 2]);
    assert_eq!(cfg.inner_codecs.len(), 1);
    assert_eq!(cfg.inner_codecs[0].name, "bytes");
    assert_eq!(cfg.index_codecs.len(), 0);
}

#[test]
fn extract_sharding_config_returns_none_for_non_sharding() {
    let codec = Codec {
        name: alloc::string::String::from("bytes"),
        configuration: vec![],
    };
    assert!(extract_sharding_config(&[codec]).is_none());
    assert!(extract_sharding_config(&[]).is_none());
}

#[test]
fn make_meta_helper_compiles() {
    let meta = make_meta(vec![8, 8], vec![4, 4]);
    assert_eq!(meta.shape, vec![8, 8]);
    assert_eq!(meta.chunks, vec![4, 4]);
}
