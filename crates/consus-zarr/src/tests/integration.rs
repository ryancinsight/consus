//! Integration tests for consus-zarr.
//!
//! Tests round-trip encoding/decoding, store backends, metadata
//! parsing, and Python zarr compatibility.

#![cfg(feature = "alloc")]

use consus_zarr::metadata::{
    ArrayMetadata, ArrayMetadataV2, ArrayMetadataV3, AttributeValue, ChunkKeyEncoding, Codec,
    ConsolidatedMetadata, ConsolidatedMetadataV3, FillValue, GroupMetadataV2, ZarrVersion,
};
use consus_zarr::store::{InMemoryStore, Store};

// ---------------------------------------------------------------------------
// Helper: build a minimal ArrayMetadata for testing.
// ---------------------------------------------------------------------------

fn make_meta_v2() -> ArrayMetadata {
    ArrayMetadata {
        version: ZarrVersion::V2,
        shape: vec![100, 100],
        chunks: vec![10, 10],
        dtype: "<f8".to_string(),
        fill_value: FillValue::Float(0.0.to_string()),
        order: 'C',
        codecs: vec![Codec {
            name: "gzip".to_string(),
            configuration: vec![("level".to_string(), "1".to_string())],
        }],
        chunk_key_encoding: ChunkKeyEncoding {
            name: "default".to_string(),
            separator: '/',
        },
    }
}

fn make_meta_v3() -> ArrayMetadata {
    ArrayMetadata {
        version: ZarrVersion::V3,
        shape: vec![100, 100],
        chunks: vec![10, 10],
        dtype: "float64".to_string(),
        fill_value: FillValue::Float(0.0.to_string()),
        order: 'C',
        codecs: vec![
            Codec {
                name: "bytes".to_string(),
                configuration: vec![("endian".to_string(), "little".to_string())],
            },
            Codec {
                name: "gzip".to_string(),
                configuration: vec![("level".to_string(), "1".to_string())],
            },
        ],
        chunk_key_encoding: ChunkKeyEncoding {
            name: "default".to_string(),
            separator: '/',
        },
    }
}

// ---------------------------------------------------------------------------
// Zarr v2 metadata round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_v2_zarray_parse_and_serialize() {
    let json = r#"{
  "zarr_format": 2,
  "shape": [10, 20],
  "chunks": [5, 10],
  "dtype": "<f4",
  "fill_value": 0.0,
  "order": "C",
  "compressor": {"id": "zlib", "configuration": {"level": 6}},
  "filters": null
}"#;

    let parsed = ArrayMetadataV2::parse(json).expect("must parse .zarray");
    assert_eq!(parsed.zarr_format, 2);
    assert_eq!(parsed.shape, &[10, 20]);
    assert_eq!(parsed.chunks, &[5, 10]);
    assert_eq!(parsed.dtype, "<f4");

    let serialized = parsed.to_json().expect("must serialize .zarray");
    let reparsed = ArrayMetadataV2::parse(&serialized).expect("re-parse after serialize");
    assert_eq!(reparsed.shape, parsed.shape);
    assert_eq!(reparsed.chunks, parsed.chunks);
}

#[test]
fn test_v2_zarray_null_compressor() {
    let json = r#"{
  "zarr_format": 2,
  "shape": [5],
  "chunks": [5],
  "dtype": "<i2",
  "fill_value": -1,
  "order": "C",
  "compressor": null,
  "filters": null
}"#;

    let parsed = ArrayMetadataV2::parse(json).expect("must parse");
    assert!(parsed.compressor.is_none());
    let canon = parsed.to_canonical();
    assert!(canon.codecs.is_empty()); // no compression
}

#[test]
fn test_v2_zgroup_parse_and_canonical() {
    let json = r#"{"zarr_format": 2}"#;
    let parsed = GroupMetadataV2::parse(json).expect("must parse .zgroup");
    assert_eq!(parsed.zarr_format, 2);
    let canon = parsed.to_canonical();
    assert_eq!(canon.version, ZarrVersion::V2);
    assert!(canon.attributes.is_empty());
}

#[test]
fn test_v2_fill_value_special() {
    // NaN fill value
    let json = r#"{
  "zarr_format": 2,
  "shape": [1],
  "chunks": [1],
  "dtype": "<f8",
  "fill_value": "NaN",
  "order": "C",
  "compressor": null,
  "filters": null
}"#;
    let parsed = ArrayMetadataV2::parse(json).expect("must parse");
    let canon = parsed.to_canonical();
    assert!(matches!(canon.fill_value, FillValue::Float(ref s) if s == "NaN"));

    // Infinity fill value
    let json_inf = r#"{
  "zarr_format": 2,
  "shape": [1],
  "chunks": [1],
  "dtype": "<f8",
  "fill_value": "Infinity",
  "order": "C",
  "compressor": null,
  "filters": null
}"#;
    let parsed_inf = ArrayMetadataV2::parse(json_inf).expect("must parse");
    let canon_inf = parsed_inf.to_canonical();
    assert!(matches!(canon_inf.fill_value, FillValue::Float(ref s) if s == "Infinity"));
}

// ---------------------------------------------------------------------------
// Zarr v3 metadata round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_v3_zarr_json_array_parse_and_serialize() {
    let json = r#"{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [50, 50],
  "data_type": "int32",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [10, 10]}
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {"separator": "/"}
  },
  "codecs": [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "gzip", "configuration": {"level": 3}}
  ],
  "fill_value": -999,
  "order": "C"
}"#;

    let parsed = consus_zarr::metadata::v3::ZarrJson::parse(json).expect("must parse zarr.json");
    let serialized = parsed.to_json().expect("must serialize zarr.json");
    let reparsed =
        consus_zarr::metadata::v3::ZarrJson::parse(&serialized).expect("re-parse after serialize");
    let canon = reparsed.to_array_canonical().expect("must be array");
    assert_eq!(canon.shape, &[50, 50]);
    assert_eq!(canon.dtype, "int32");
}

#[test]
fn test_v3_zarr_json_group_parse() {
    let json = r#"{
  "zarr_format": 3,
  "node_type": "group",
  "codecs": []
}"#;

    let parsed =
        consus_zarr::metadata::v3::ZarrJson::parse(json).expect("must parse group zarr.json");
    let canon = parsed.to_group_canonical().expect("must be group");
    assert_eq!(canon.version, ZarrVersion::V3);
    assert!(canon.attributes.is_empty());
}

#[test]
fn test_v3_to_canonical_preserves_dtype() {
    let json = r#"{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [8],
  "data_type": "float32",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [8]}
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {"separator": "/"}
  },
  "codecs": [],
  "fill_value": 0.0,
  "order": "C"
}"#;

    let v3 = consus_zarr::metadata::v3::ZarrJson::parse(json).unwrap();
    let canon = v3.to_array_canonical().unwrap();
    assert_eq!(canon.dtype, "float32");
    assert_eq!(canon.shape, &[8]);
}

// ---------------------------------------------------------------------------
// Consolidated metadata
// ---------------------------------------------------------------------------

#[test]
fn test_consolidated_v3_roundtrip() {
    let json = r#"{
  "zarr_format": 3,
  "metadata": [
    {
      "path": ".",
      "data": {
        "zarr_format": 3,
        "node_type": "group",
        "codecs": []
      }
    },
    {
      "path": "data/array",
      "data": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100],
        "data_type": "float32",
        "chunk_grid": {
          "name": "regular",
          "configuration": {"chunk_shape": [10]}
        },
        "chunk_key_encoding": {
          "name": "default",
          "configuration": {"separator": "/"}
        },
        "codecs": [],
        "fill_value": 0.0,
        "order": "C"
      }
    }
  ]
}"#;

    let parsed = ConsolidatedMetadataV3::parse(json).expect("must parse");
    assert_eq!(parsed.zarr_format, 3);
    assert_eq!(parsed.metadata.len(), 2);

    let canon = parsed.to_canonical();
    assert_eq!(canon.len(), 2);
    assert_eq!(canon.group_paths(), &["."]);
    assert_eq!(canon.array_paths(), &["data/array"]);

    let root = canon.get(".").expect("root must exist");
    assert_eq!(root.path, ".");
    assert!(matches!(
        root.metadata,
        consus_zarr::metadata::consolidated::NodeMetadata::Group(_)
    ));

    let arr = canon.get("data/array").expect("array must exist");
    assert!(matches!(
        arr.metadata,
        consus_zarr::metadata::consolidated::NodeMetadata::Array(_)
    ));
}

// ---------------------------------------------------------------------------
// In-memory store
// ---------------------------------------------------------------------------

#[test]
fn test_in_memory_store_set_get() {
    let mut store = InMemoryStore::new();
    store.set("arr/.zarray", b"{}").unwrap();
    let data = store.get("arr/.zarray").unwrap();
    assert_eq!(&data, b"{}");
}

#[test]
fn test_in_memory_store_delete() {
    let mut store = InMemoryStore::new();
    store.set("key", b"value").unwrap();
    store.delete("key").unwrap();
    let err = store.get("key").unwrap_err();
    assert!(matches!(err, consus_core::Error::NotFound { .. }));
}

#[test]
fn test_in_memory_store_list() {
    let mut store = InMemoryStore::new();
    store.set("arr/.zarray", b"{}").unwrap();
    store.set("arr/c/0.0", b"chunk").unwrap();
    store.set("arr/c/1.0", b"chunk").unwrap();
    store.set("other/.zgroup", b"{}").unwrap();

    let mut keys = store.list("arr/").unwrap();
    keys.sort();
    assert_eq!(keys, &["arr/.zarray", "arr/c/0.0", "arr/c/1.0"]);
}

#[test]
fn test_in_memory_store_contains() {
    let mut store = InMemoryStore::new();
    store.set("key", b"val").unwrap();
    assert!(store.contains("key").unwrap());
    assert!(!store.contains("missing").unwrap());
}

#[test]
fn test_in_memory_store_missing_is_not_found() {
    let store = InMemoryStore::new();
    let err = store.get("nonexistent").unwrap_err();
    assert!(matches!(err, consus_core::Error::NotFound { .. }));
}

// ---------------------------------------------------------------------------
// ArrayMetadata utilities
// ---------------------------------------------------------------------------

#[test]
fn test_array_metadata_chunk_grid() {
    let meta = make_meta_v2();
    assert_eq!(meta.chunk_grid(), vec![10, 10]);
    assert_eq!(meta.total_chunks(), 100);
    assert_eq!(meta.num_elements(), 10_000);
}

#[test]
fn test_array_metadata_chunk_grid_v3() {
    let meta = make_meta_v3();
    assert_eq!(meta.chunk_grid(), vec![10, 10]);
    assert_eq!(meta.total_chunks(), 100);
}

#[test]
fn test_dtype_to_element_size() {
    use consus_zarr::metadata::dtype_to_element_size;

    // v3 named types
    assert_eq!(dtype_to_element_size("float64"), Some(8));
    assert_eq!(dtype_to_element_size("int32"), Some(4));
    assert_eq!(dtype_to_element_size("bool"), Some(1));

    // v2 numpy dtype strings
    assert_eq!(dtype_to_element_size("<f8"), Some(8));
    assert_eq!(dtype_to_element_size("<i4"), Some(4));
    assert_eq!(dtype_to_element_size("|S10"), Some(10));

    // Variable-length types
    assert_eq!(dtype_to_element_size("vlen<unicode>"), None);
    assert_eq!(dtype_to_element_size("utf8"), None);
}

#[test]
fn test_codec_compression_level_helpers() {
    let gzip = Codec {
        name: "gzip".to_string(),
        configuration: vec![("level".to_string(), "9".to_string())],
    };
    assert_eq!(gzip.gzip_level(), Some(9));

    let zstd = Codec {
        name: "zstd".to_string(),
        configuration: vec![("level".to_string(), "-3".to_string())],
    };
    assert_eq!(zstd.zstd_level(), Some(-3));

    let bytes = Codec {
        name: "bytes".to_string(),
        configuration: vec![("endian".to_string(), "native".to_string())],
    };
    assert!(bytes.is_identity());
    assert_eq!(bytes.bytes_endian(), Some("native"));
}

#[test]
fn test_fill_value_to_canonical() {
    use consus_zarr::metadata::FillValue;
    use consus_zarr::metadata::v2::{FillValueJson, SpecialFillValue};

    // Null -> Default
    assert_eq!(FillValueJson::Null.to_fill_value(), FillValue::Default);

    // Int
    assert_eq!(FillValueJson::Int(42).to_fill_value(), FillValue::Int(42));

    // Bool
    assert_eq!(
        FillValueJson::Bool(true).to_fill_value(),
        FillValue::Bool(true)
    );

    // Special NaN
    assert_eq!(
        FillValueJson::Special(SpecialFillValue::NaN).to_fill_value(),
        FillValue::Float("NaN".to_string())
    );

    // Special Infinity
    assert_eq!(
        FillValueJson::Special(SpecialFillValue::Infinity).to_fill_value(),
        FillValue::Float("Infinity".to_string())
    );
}

// ---------------------------------------------------------------------------
// Attribute value
// ---------------------------------------------------------------------------

#[test]
fn test_attribute_value_num_elements() {
    assert_eq!(AttributeValue::Int(42).num_elements(), 1);
    assert_eq!(AttributeValue::IntArray(vec![1, 2, 3]).num_elements(), 3);
    assert_eq!(AttributeValue::FloatArray(vec![1.0, 2.0]).num_elements(), 2);
}

// ---------------------------------------------------------------------------
// Zarr v2 .zattrs round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_zattrs_parse_and_serialize() {
    use consus_zarr::metadata::v2::{parse_zattrs, serialize_zattrs};

    let attrs = vec![
        ("temperature".to_string(), AttributeValue::Float(298.15)),
        (
            "dimensions".to_string(),
            AttributeValue::IntArray(vec![3, 512, 512]),
        ),
        (
            "name".to_string(),
            AttributeValue::String("run_001".to_string()),
        ),
    ];

    let json = serialize_zattrs(&attrs).expect("serialize must succeed");
    let parsed = parse_zattrs(&json).expect("parse must succeed");
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].0, "temperature");
    assert_eq!(parsed[2].0, "name");
}

// ---------------------------------------------------------------------------
// Sharding: linear index computation
// ---------------------------------------------------------------------------

#[test]
fn test_shard_linear_index_1d() {
    use consus_zarr::shard::compute_linear_index;
    let grid = vec![10];
    assert_eq!(compute_linear_index(&[0], &grid), 0);
    assert_eq!(compute_linear_index(&[5], &grid), 5);
    assert_eq!(compute_linear_index(&[9], &grid), 9);
}

#[test]
fn test_shard_linear_index_2d() {
    use consus_zarr::shard::compute_linear_index;
    let grid = vec![3, 4]; // 12 total chunks, row-major
    assert_eq!(compute_linear_index(&[0, 0], &grid), 0);
    assert_eq!(compute_linear_index(&[0, 1], &grid), 1);
    assert_eq!(compute_linear_index(&[0, 3], &grid), 3);
    assert_eq!(compute_linear_index(&[1, 0], &grid), 4);
    assert_eq!(compute_linear_index(&[2, 3], &grid), 11);
}

#[test]
fn test_shard_linear_index_3d() {
    use consus_zarr::shard::compute_linear_index;
    let grid = vec![2, 3, 4]; // 24 total chunks
    assert_eq!(compute_linear_index(&[0, 0, 0], &grid), 0);
    assert_eq!(compute_linear_index(&[0, 0, 1], &grid), 1);
    assert_eq!(compute_linear_index(&[0, 1, 0], &grid), 4);
    assert_eq!(compute_linear_index(&[1, 0, 0], &grid), 12);
}

// ---------------------------------------------------------------------------
// ShardConfig
// ---------------------------------------------------------------------------

#[test]
fn test_shard_config_from_json_config() {
    use consus_zarr::shard::ShardConfig;

    let json = serde_json::json!({
        "name": "sharding",
        "configuration": {
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "gzip", "configuration": {"level": 1}}
            ],
            "index_codec": {"name": "crc32"},
            "chunk_boundary": 64
        }
    });

    let cfg = ShardConfig::from_json_config(
        json.get("name").and_then(|v| v.as_str()).unwrap_or(""),
        json.get("configuration").and_then(|v| v.as_object()),
    );

    assert!(cfg.is_some());
    let cfg = cfg.unwrap();
    assert_eq!(cfg.codecs.len(), 2);
    assert!(cfg.index_codec.is_some());
    assert_eq!(cfg.chunk_boundary, 64);
}

#[test]
fn test_shard_config_index_chunk_size() {
    use consus_zarr::shard::ShardConfig;

    let cfg = ShardConfig {
        codecs: vec![],
        index_codec: None,
        chunk_boundary: 64,
        raw_configuration: None,
    };

    let meta = make_meta_v3();
    // 10x10 = 100 chunks, 16 bytes each = 1600
    assert_eq!(cfg.index_chunk_size(&meta), Some(1600));
}

// ---------------------------------------------------------------------------
// ShardIndexReader
// ---------------------------------------------------------------------------

#[test]
fn test_shard_index_reader_basic() {
    use consus_zarr::shard::ShardIndexReader;

    let mut index = vec![0u8; 16 * 4];
    // Chunk (0,0): offset 100, length 50
    index[0..8].copy_from_slice(&100u64.to_le_bytes());
    index[8..16].copy_from_slice(&50u64.to_le_bytes());
    // Chunk (0,1): offset 200, length 75
    index[16..24].copy_from_slice(&200u64.to_le_bytes());
    index[24..32].copy_from_slice(&75u64.to_le_bytes());
    // Chunk (1,0): uninitialized
    // Chunk (1,1): offset 500, length 100
    index[48..56].copy_from_slice(&500u64.to_le_bytes());
    index[56..64].copy_from_slice(&100u64.to_le_bytes());

    let reader = ShardIndexReader::new(&index, vec![2, 2]);
    assert_eq!(reader.get_chunk_entry(&[0, 0]), Some((100, 50)));
    assert_eq!(reader.get_chunk_entry(&[0, 1]), Some((200, 75)));
    assert_eq!(reader.get_chunk_entry(&[1, 0]), Some((0, 0)));
    assert_eq!(reader.get_chunk_entry(&[1, 1]), Some((500, 100)));
    assert_eq!(reader.num_chunks(), 4);
}

#[test]
fn test_shard_index_reader_out_of_bounds() {
    use consus_zarr::shard::ShardIndexReader;

    let index = vec![0u8; 16 * 4];
    let reader = ShardIndexReader::new(&index, vec![2, 2]);
    assert!(reader.get_chunk_entry(&[2, 0]).is_none());
    assert!(reader.get_chunk_entry(&[0]).is_none()); // wrong rank
}

// ---------------------------------------------------------------------------
// Codec pipeline round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_codec_pipeline_gzip_roundtrip() {
    use consus_zarr::codec::{CodecPipeline, default_registry};
    use consus_zarr::metadata::Codec;

    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: "gzip".to_string(),
        configuration: vec![("level".to_string(), "1".to_string())],
    });

    let input = b"The quick brown fox jumps over the lazy dog";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");
    assert_eq!(&decompressed, input);
    // Compressed size should be smaller for this input
    assert!(compressed.len() < input.len());
}

#[test]
fn test_codec_pipeline_bytes_is_identity() {
    use consus_zarr::codec::{CodecPipeline, default_registry};
    use consus_zarr::metadata::Codec;

    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: "bytes".to_string(),
        configuration: vec![("endian".to_string(), "native".to_string())],
    });

    let input = b"raw chunk bytes";
    let result = pipeline.compress(input, registry).unwrap();
    assert_eq!(&result, input);
    let decompressed = pipeline.decompress(&result, registry).unwrap();
    assert_eq!(&decompressed, input);
}

#[test]
fn test_codec_pipeline_zstd_roundtrip() {
    use consus_zarr::codec::{CodecPipeline, default_registry};
    use consus_zarr::metadata::Codec;

    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: "zstd".to_string(),
        configuration: vec![("level".to_string(), "1".to_string())],
    });

    let input = b"Zstandard is a real-time compression algorithm";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");
    assert_eq!(&decompressed, input);
}

// ---------------------------------------------------------------------------
// Version detection
// ---------------------------------------------------------------------------

#[test]
fn test_zarr_version_detection() {
    let v2_json = r#"{
  "zarr_format": 2,
  "shape": [10],
  "chunks": [10],
  "dtype": "<f8",
  "fill_value": 0.0,
  "order": "C",
  "compressor": null,
  "filters": null
}"#;

    let v3_json = r#"{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [10],
  "data_type": "float64",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [10]}
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {"separator": "/"}
  },
  "codecs": [],
  "fill_value": 0.0,
  "order": "C"
}"#;

    let v2_parsed = ArrayMetadataV2::parse(v2_json).unwrap();
    assert_eq!(v2_parsed.zarr_format, 2);

    let v3_parsed = consus_zarr::metadata::v3::ZarrJson::parse(v3_json).unwrap();
    let v3_canonical = v3_parsed.to_array_canonical().unwrap();
    assert_eq!(v3_canonical.version, ZarrVersion::V3);
}

// ---------------------------------------------------------------------------
// HDF5 filter ID to codec name mapping
// ---------------------------------------------------------------------------

#[test]
fn test_hdf5_filter_id_mapping() {
    use consus_zarr::metadata::v2::hdf5_filter_id_to_name;

    assert_eq!(hdf5_filter_id_to_name(1), "deflate");
    assert_eq!(hdf5_filter_id_to_name(32004), "lz4");
    assert_eq!(hdf5_filter_id_to_name(32015), "zstd");
    assert_eq!(hdf5_filter_id_to_name(32001), "blosc");
    assert_eq!(hdf5_filter_id_to_name(999), "filter999");
}

// ---------------------------------------------------------------------------
// Chunk key encoding
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_key_encoding_v2_default() {
    use consus_zarr::chunk::ChunkKeySeparator;
    use consus_zarr::chunk::chunk_key;

    // v3 default: slash-separated with c/ prefix
    let key = chunk_key(&[3, 1, 4], ChunkKeySeparator::Slash);
    assert_eq!(key, "c/3/1/4");

    // v2 compat: dot-separated
    let key_dot = chunk_key(&[3, 1, 4], ChunkKeySeparator::Dot);
    assert_eq!(key_dot, "3.1.4");
}

#[test]
fn test_chunk_key_encoding_scalar() {
    use consus_zarr::chunk::ChunkKeySeparator;
    use consus_zarr::chunk::chunk_key;

    // Scalar chunk (0D) — empty string
    let key = chunk_key(&[], ChunkKeySeparator::Dot);
    assert_eq!(key, "");
}

// ---------------------------------------------------------------------------
// Consolidated entry lookup
// ---------------------------------------------------------------------------

#[test]
fn test_consolidated_entry_lookup() {
    let json = r#"{
  "zarr_format": 2,
  "metadata": [
    {"path": "a", "type": "group", "data": {"zarr_format": 2}},
    {"path": "b", "type": "group", "data": {"zarr_format": 2}}
  ]
}"#;

    let v2 = consus_zarr::metadata::consolidated::ConsolidatedMetadataV2::parse(json)
        .expect("must parse");
    let canon = v2.to_canonical();
    assert!(canon.get("a").is_some());
    assert!(canon.get("b").is_some());
    assert!(canon.get("c").is_none());
}

// ---------------------------------------------------------------------------
// Array metadata with attributes from consolidated v2
// ---------------------------------------------------------------------------

#[test]
fn test_consolidated_v2_with_attributes() {
    let json = r#"{
  "zarr_format": 2,
  "metadata": [
    {
      "path": "arr",
      "type": "array",
      "data": {
        "zarr_format": 2,
        "shape": [5],
        "chunks": [5],
        "dtype": "<f4",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
      },
      "attributes": "{\"units\": \"K\", \"scale\": 1.0}"
    }
  ]
}"#;

    let v2 = consus_zarr::metadata::consolidated::ConsolidatedMetadataV2::parse(json)
        .expect("must parse");
    let canon = v2.to_canonical();
    let entry = canon.get("arr").expect("entry must exist");
    let attrs = entry.attributes();
    assert_eq!(attrs.len(), 2);
    assert!(attrs.iter().any(|(k, _)| k == "units"));
    assert!(attrs.iter().any(|(k, _)| k == "scale"));
}

// ---------------------------------------------------------------------------
// Zarr v2 compressor blosc parse
// ---------------------------------------------------------------------------

#[test]
fn test_v2_blosc_compressor() {
    let json = r#"{
  "zarr_format": 2,
  "shape": [100],
  "chunks": [10],
  "dtype": "<f4",
  "fill_value": 0.0,
  "order": "C",
  "compressor": {"id": "blosc", "configuration": {"level": 1, "cname": "lz4"}},
  "filters": null
}"#;

    let parsed = ArrayMetadataV2::parse(json).expect("must parse");
    let canon = parsed.to_canonical();
    assert_eq!(canon.codecs.len(), 1);
    assert_eq!(canon.codecs[0].name, "blosc");
}

// ---------------------------------------------------------------------------
// V3 dimension names
// ---------------------------------------------------------------------------

#[test]
fn test_v3_dimension_names() {
    let json = r#"{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [3, 512, 512],
  "data_type": "float32",
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [1, 128, 128]}
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {"separator": "/"}
  },
  "codecs": [],
  "fill_value": 0.0,
  "order": "C",
  "dimension_names": ["time", "y", "x"]
}"#;

    let v3 = consus_zarr::metadata::v3::ZarrJson::parse(json).expect("must parse");
    match &v3 {
        consus_zarr::metadata::v3::ZarrJson::Array {
            dimension_names, ..
        } => {
            assert_eq!(dimension_names.as_ref().unwrap(), &["time", "y", "x"]);
        }
        _ => panic!("expected Array node"),
    }
}

// ---------------------------------------------------------------------------
// Empty consolidated metadata
// ---------------------------------------------------------------------------

#[test]
fn test_empty_consolidated_metadata() {
    let json = r#"{
  "zarr_format": 3,
  "metadata": []
}"#;

    let v3 = ConsolidatedMetadataV3::parse(json).expect("must parse");
    let canon = v3.to_canonical();
    assert!(canon.is_empty());
    assert!(canon.array_paths().is_empty());
    assert!(canon.group_paths().is_empty());
}
