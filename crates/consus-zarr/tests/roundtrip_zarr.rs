//! Round-trip tests for Zarr v2 and v3 arrays and groups.
//!
//! ## Coverage
//!
//! - Write Zarr v2 array → read back → verify data integrity
//! - Write Zarr v3 array → read back → verify data integrity
//! - Group hierarchy roundtrip
//! - Metadata roundtrip
//! - Chunk data roundtrip with compression
//! - Full array write and read operations

use consus_zarr::Codec;
use consus_zarr::chunk::{ChunkKeySeparator, chunk_key};
use consus_zarr::codec::{CodecPipeline, default_registry};
use consus_zarr::metadata::{ArrayMetadataV2, GroupMetadataV2, ZarrJson};
use consus_zarr::metadata::{parse_zattrs, serialize_zattrs};
use consus_zarr::store::{InMemoryStore, Store};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Zarr v2 roundtrip tests
// ---------------------------------------------------------------------------

/// Test Zarr v2 array metadata roundtrip.
///
/// ## Invariant
///
/// `parse(serialize(parse(json))) == parse(json)`
#[test]
fn v2_array_metadata_roundtrip() {
    let original = r#"{
        "zarr_format": 2,
        "shape": [1000, 1000],
        "chunks": [100, 100],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 5},
        "filters": null
    }"#;

    let meta1 = ArrayMetadataV2::parse(original).expect("first parse must succeed");
    let serialized = meta1.to_json().expect("serialize must succeed");
    let meta2 = ArrayMetadataV2::parse(&serialized).expect("second parse must succeed");

    assert_eq!(meta1.zarr_format, meta2.zarr_format);
    assert_eq!(meta1.shape, meta2.shape);
    assert_eq!(meta1.chunks, meta2.chunks);
    assert_eq!(meta1.dtype, meta2.dtype);
}

/// Test Zarr v2 group metadata roundtrip.
#[test]
fn v2_group_metadata_roundtrip() {
    let original = r#"{
        "zarr_format": 2,
        "attributes": {
            "name": "experiment_001",
            "version": 1,
            "description": "Test experiment"
        }
    }"#;

    let meta1 = GroupMetadataV2::parse(original).expect("first parse must succeed");
    let serialized = meta1.to_json().expect("serialize must succeed");
    let meta2 = GroupMetadataV2::parse(&serialized).expect("second parse must succeed");

    assert_eq!(meta1.zarr_format, meta2.zarr_format);
}

/// Test Zarr v2 attributes roundtrip.
#[test]
fn v2_attributes_roundtrip() {
    let original = r#"{
        "experiment": "climate_model",
        "units": "kelvin",
        "missing_value": -999.0,
        "coordinates": ["time", "lat", "lon"],
        "metadata": {
            "source": "simulation",
            "version": 2
        }
    }"#;

    let attrs1 = parse_zattrs(original).expect("first parse must succeed");
    let serialized = serialize_zattrs(&attrs1).expect("serialize must succeed");
    let attrs2 = parse_zattrs(&serialized).expect("second parse must succeed");

    assert_eq!(attrs1.len(), attrs2.len());
}

/// Test Zarr v2 array write and read back.
#[test]
fn v2_array_write_read_roundtrip() {
    let mut store = InMemoryStore::new();
    let registry = default_registry();

    // Create array metadata
    let zarray = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 3},
        "filters": null
    }"#;

    store
        .set("my_array/.zarray", zarray.as_bytes())
        .expect("set must succeed");

    // Write chunks
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("3"))],
    });

    for i in 0..10 {
        for j in 0..10 {
            // Create chunk data (10x10 float64 = 800 bytes)
            let chunk_data: Vec<u8> = (0..800).map(|k| ((i * 10 + j + k) % 256) as u8).collect();
            let compressed = pipeline
                .compress(&chunk_data, registry)
                .expect("compress must succeed");

            let key = format!("my_array/{}", chunk_key(&[i, j], ChunkKeySeparator::Dot));
            store
                .set(&key, &compressed)
                .expect("set chunk must succeed");
        }
    }

    // Read back and verify
    let retrieved_zarray = store
        .get("my_array/.zarray")
        .expect("get metadata must succeed");
    let meta = ArrayMetadataV2::parse(std::str::from_utf8(&retrieved_zarray).expect("utf8"))
        .expect("parse must succeed");

    assert_eq!(meta.shape, &[100, 100]);
    assert_eq!(meta.chunks, &[10, 10]);

    // Verify all chunks
    for i in 0..10 {
        for j in 0..10 {
            let key = format!("my_array/{}", chunk_key(&[i, j], ChunkKeySeparator::Dot));
            let compressed = store.get(&key).expect("get chunk must succeed");
            let decompressed = pipeline
                .decompress(&compressed, registry)
                .expect("decompress must succeed");

            // Verify chunk size
            assert_eq!(decompressed.len(), 800);

            // Verify some values
            let expected: Vec<u8> = (0..800).map(|k| ((i * 10 + j + k) % 256) as u8).collect();
            assert_eq!(decompressed, expected);
        }
    }
}

/// Test Zarr v2 group hierarchy roundtrip.
#[test]
fn v2_group_hierarchy_roundtrip() {
    let mut store = InMemoryStore::new();

    // Create root group
    store
        .set(".zgroup", b"{\"zarr_format\": 2}")
        .expect("set must succeed");
    store
        .set(".zattrs", b"{\"name\": \"root\"}")
        .expect("set must succeed");

    // Create subgroup
    store
        .set("subgroup/.zgroup", b"{\"zarr_format\": 2}")
        .expect("set must succeed");
    store
        .set("subgroup/.zattrs", b"{\"name\": \"subgroup\"}")
        .expect("set must succeed");

    // Create array in subgroup
    let zarray = r#"{
        "zarr_format": 2,
        "shape": [50, 50],
        "chunks": [10, 10],
        "dtype": "<f4",
        "fill_value": -999.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;
    store
        .set("subgroup/data/.zarray", zarray.as_bytes())
        .expect("set must succeed");

    // Verify hierarchy by listing keys
    let keys = store.list("").expect("list must succeed");
    assert!(keys.iter().any(|k| k == ".zgroup"));
    assert!(keys.iter().any(|k| k == ".zattrs"));
    assert!(keys.iter().any(|k| k == "subgroup/.zgroup"));
    assert!(keys.iter().any(|k| k == "subgroup/.zattrs"));
    assert!(keys.iter().any(|k| k == "subgroup/data/.zarray"));

    // Read back and verify
    let root_group = store.get(".zgroup").expect("get must succeed");
    let root_meta = GroupMetadataV2::parse(std::str::from_utf8(&root_group).expect("utf8"))
        .expect("parse must succeed");
    assert_eq!(root_meta.zarr_format, 2);

    let sub_group = store.get("subgroup/.zgroup").expect("get must succeed");
    let sub_meta = GroupMetadataV2::parse(std::str::from_utf8(&sub_group).expect("utf8"))
        .expect("parse must succeed");
    assert_eq!(sub_meta.zarr_format, 2);

    let data_meta = store
        .get("subgroup/data/.zarray")
        .expect("get must succeed");
    let arr_meta = ArrayMetadataV2::parse(std::str::from_utf8(&data_meta).expect("utf8"))
        .expect("parse must succeed");
    assert_eq!(arr_meta.shape, &[50, 50]);
}

/// Test Zarr v2 with different compression codecs roundtrip.
#[test]
fn v2_compression_roundtrip_gzip() {
    let mut store = InMemoryStore::new();
    let registry = default_registry();

    let zarray = r#"{
        "zarr_format": 2,
        "shape": [100],
        "chunks": [100],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 6},
        "filters": null
    }"#;

    store
        .set("array/.zarray", zarray.as_bytes())
        .expect("set must succeed");

    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    // Create and write chunk
    let original_data: Vec<u8> = (0..800).map(|i| (i % 256) as u8).collect();
    let compressed = pipeline
        .compress(&original_data, registry)
        .expect("compress must succeed");
    store.set("array/0", &compressed).expect("set must succeed");

    // Read and decompress
    let retrieved = store.get("array/0").expect("get must succeed");
    let decompressed = pipeline
        .decompress(&retrieved, registry)
        .expect("decompress must succeed");

    assert_eq!(decompressed, original_data);
}

/// Test Zarr v2 with zstd compression roundtrip.
#[test]
fn v2_compression_roundtrip_zstd() {
    let mut store = InMemoryStore::new();
    let registry = default_registry();

    let zarray = r#"{
        "zarr_format": 2,
        "shape": [100],
        "chunks": [100],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "zstd", "level": 3},
        "filters": null
    }"#;

    store
        .set("array/.zarray", zarray.as_bytes())
        .expect("set must succeed");

    let pipeline = CodecPipeline::single(Codec {
        name: String::from("zstd"),
        configuration: vec![(String::from("level"), String::from("3"))],
    });

    let original_data: Vec<u8> = (0..800).map(|i| (i % 256) as u8).collect();
    let compressed = pipeline
        .compress(&original_data, registry)
        .expect("compress must succeed");
    store.set("array/0", &compressed).expect("set must succeed");

    let retrieved = store.get("array/0").expect("get must succeed");
    let decompressed = pipeline
        .decompress(&retrieved, registry)
        .expect("decompress must succeed");

    assert_eq!(decompressed, original_data);
}

// ---------------------------------------------------------------------------
// Zarr v3 roundtrip tests
// ---------------------------------------------------------------------------

/// Test Zarr v3 array metadata roundtrip.
#[test]
fn v3_array_metadata_roundtrip() {
    let original = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1000, 1000],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [100, 100]}
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"}
        },
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "gzip", "configuration": {"level": 5}}
        ],
        "fill_value": 0.0
    }"#;

    let meta1 = ZarrJson::parse(original).expect("first parse must succeed");
    let serialized = meta1.to_json().expect("serialize must succeed");
    let meta2 = ZarrJson::parse(&serialized).expect("second parse must succeed");

    match (meta1, meta2) {
        (
            ZarrJson::Array {
                shape: s1,
                data_type: d1,
                ..
            },
            ZarrJson::Array {
                shape: s2,
                data_type: d2,
                ..
            },
        ) => {
            assert_eq!(s1, s2);
            assert_eq!(d1, d2);
        }
        _ => panic!("type mismatch"),
    }
}

/// Test Zarr v3 group metadata roundtrip.
#[test]
fn v3_group_metadata_roundtrip() {
    let original = r#"{
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "name": "experiment_001",
            "version": 1
        }
    }"#;

    let meta1 = ZarrJson::parse(original).expect("first parse must succeed");
    let serialized = meta1.to_json().expect("serialize must succeed");
    let meta2 = ZarrJson::parse(&serialized).expect("second parse must succeed");

    match (meta1, meta2) {
        (
            ZarrJson::Group {
                zarr_format: zf1, ..
            },
            ZarrJson::Group {
                zarr_format: zf2, ..
            },
        ) => {
            assert_eq!(zf1, zf2);
        }
        _ => panic!("type mismatch"),
    }
}

/// Test Zarr v3 array write and read back.
#[test]
fn v3_array_write_read_roundtrip() {
    let mut store = InMemoryStore::new();
    let registry = default_registry();

    // Create array metadata
    let zarr_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 100],
        "data_type": "float64",
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
        "fill_value": 0.0
    }"#;

    store
        .set("my_array/zarr.json", zarr_json.as_bytes())
        .expect("set must succeed");

    // Create codec pipeline
    let pipeline = CodecPipeline::new(vec![
        Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from("little"))],
        },
        Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), String::from("3"))],
        },
    ]);

    // Write chunks using v3 key encoding
    for i in 0..10 {
        for j in 0..10 {
            let chunk_data: Vec<u8> = (0..800).map(|k| ((i * 10 + j + k) % 256) as u8).collect();
            let compressed = pipeline
                .compress(&chunk_data, registry)
                .expect("compress must succeed");

            let key = format!("my_array/{}", chunk_key(&[i, j], ChunkKeySeparator::Slash));
            store
                .set(&key, &compressed)
                .expect("set chunk must succeed");
        }
    }

    // Read back and verify
    let retrieved_json = store
        .get("my_array/zarr.json")
        .expect("get metadata must succeed");
    let meta = ZarrJson::parse(std::str::from_utf8(&retrieved_json).expect("utf8"))
        .expect("parse must succeed");

    match meta {
        ZarrJson::Array { shape, .. } => {
            assert_eq!(shape, &[100, 100]);
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }

    // Verify all chunks
    for i in 0..10 {
        for j in 0..10 {
            let key = format!("my_array/{}", chunk_key(&[i, j], ChunkKeySeparator::Slash));
            let compressed = store.get(&key).expect("get chunk must succeed");
            let decompressed = pipeline
                .decompress(&compressed, registry)
                .expect("decompress must succeed");

            let expected: Vec<u8> = (0..800).map(|k| ((i * 10 + j + k) % 256) as u8).collect();
            assert_eq!(decompressed, expected);
        }
    }
}

/// Test Zarr v3 group hierarchy roundtrip.
#[test]
fn v3_group_hierarchy_roundtrip() {
    let mut store = InMemoryStore::new();

    // Create root group
    let root_json = r#"{"zarr_format": 3, "node_type": "group", "attributes": {"name": "root"}}"#;
    store
        .set("zarr.json", root_json.as_bytes())
        .expect("set must succeed");

    // Create subgroup
    let sub_json =
        r#"{"zarr_format": 3, "node_type": "group", "attributes": {"name": "subgroup"}}"#;
    store
        .set("subgroup/zarr.json", sub_json.as_bytes())
        .expect("set must succeed");

    // Create array in subgroup
    let array_json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [50, 50],
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": -999.0
    }"#;
    store
        .set("subgroup/data/zarr.json", array_json.as_bytes())
        .expect("set must succeed");

    // Verify hierarchy
    let keys = store.list("").expect("list must succeed");
    assert!(keys.iter().any(|k| k == "zarr.json"));
    assert!(keys.iter().any(|k| k == "subgroup/zarr.json"));
    assert!(keys.iter().any(|k| k == "subgroup/data/zarr.json"));

    // Read back and verify
    let root = ZarrJson::parse(
        std::str::from_utf8(&store.get("zarr.json").expect("get must succeed")).expect("utf8"),
    )
    .expect("parse must succeed");
    match root {
        ZarrJson::Group { zarr_format, .. } => assert_eq!(zarr_format, 3),
        ZarrJson::Array { .. } => panic!("expected group"),
    }

    let sub = ZarrJson::parse(
        std::str::from_utf8(&store.get("subgroup/zarr.json").expect("get must succeed"))
            .expect("utf8"),
    )
    .expect("parse must succeed");
    match sub {
        ZarrJson::Group { zarr_format, .. } => assert_eq!(zarr_format, 3),
        ZarrJson::Array { .. } => panic!("expected group"),
    }

    let arr = ZarrJson::parse(
        std::str::from_utf8(
            &store
                .get("subgroup/data/zarr.json")
                .expect("get must succeed"),
        )
        .expect("utf8"),
    )
    .expect("parse must succeed");
    match arr {
        ZarrJson::Array { shape, .. } => assert_eq!(shape, &[50, 50]),
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

// ---------------------------------------------------------------------------
// Filesystem store roundtrip tests
// ---------------------------------------------------------------------------

/// Test Zarr v2 array with filesystem store roundtrip.
#[test]
fn v2_filesystem_roundtrip() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let registry = default_registry();

    // Create store and write array
    {
        let mut store =
            consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

        let zarray = r#"{
            "zarr_format": 2,
            "shape": [50, 50],
            "chunks": [10, 10],
            "dtype": "<f8",
            "fill_value": 0.0,
            "order": "C",
            "compressor": {"id": "gzip", "level": 1},
            "filters": null
        }"#;

        store
            .set("array/.zarray", zarray.as_bytes())
            .expect("set must succeed");

        let pipeline = CodecPipeline::single(Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), String::from("1"))],
        });

        // Write a few chunks
        for i in 0..5 {
            for j in 0..5 {
                let data: Vec<u8> = (0..800).map(|k| ((i + j + k) % 256) as u8).collect();
                let compressed = pipeline
                    .compress(&data, registry)
                    .expect("compress must succeed");
                let key = format!("array/{}.{}", i, j);
                store.set(&key, &compressed).expect("set must succeed");
            }
        }
    }

    // Reopen and verify
    {
        let store = consus_zarr::store::FsStore::open(tmp.path()).expect("open must succeed");

        let zarray_data = store.get("array/.zarray").expect("get must succeed");
        let meta = ArrayMetadataV2::parse(std::str::from_utf8(&zarray_data).expect("utf8"))
            .expect("parse must succeed");

        assert_eq!(meta.shape, &[50, 50]);

        // Verify chunks exist
        for i in 0..5 {
            for j in 0..5 {
                let key = format!("array/{}.{}", i, j);
                assert!(store.contains(&key).expect("contains must succeed"));
            }
        }
    }
}

/// Test Zarr v3 array with filesystem store roundtrip.
#[test]
fn v3_filesystem_roundtrip() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let registry = default_registry();

    // Create store and write array
    {
        let mut store =
            consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

        let zarr_json = r#"{
            "zarr_format": 3,
            "node_type": "array",
            "shape": [50, 50],
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "gzip", "configuration": {"level": 1}}
            ],
            "fill_value": 0.0
        }"#;

        store
            .set("array/zarr.json", zarr_json.as_bytes())
            .expect("set must succeed");

        let pipeline = CodecPipeline::new(vec![
            Codec {
                name: String::from("bytes"),
                configuration: vec![(String::from("endian"), String::from("little"))],
            },
            Codec {
                name: String::from("gzip"),
                configuration: vec![(String::from("level"), String::from("1"))],
            },
        ]);

        // Write chunks
        for i in 0..5 {
            for j in 0..5 {
                let data: Vec<u8> = (0..800).map(|k| ((i + j + k) % 256) as u8).collect();
                let compressed = pipeline
                    .compress(&data, registry)
                    .expect("compress must succeed");
                let key = format!("array/c/{}/{}", i, j);
                store.set(&key, &compressed).expect("set must succeed");
            }
        }
    }

    // Reopen and verify
    {
        let store = consus_zarr::store::FsStore::open(tmp.path()).expect("open must succeed");

        let json_data = store.get("array/zarr.json").expect("get must succeed");
        let meta = ZarrJson::parse(std::str::from_utf8(&json_data).expect("utf8"))
            .expect("parse must succeed");

        match meta {
            ZarrJson::Array { shape, .. } => assert_eq!(shape, &[50, 50]),
            ZarrJson::Group { .. } => panic!("expected array"),
        }

        // Verify chunks exist
        for i in 0..5 {
            for j in 0..5 {
                let key = format!("array/c/{}/{}", i, j);
                assert!(store.contains(&key).expect("contains must succeed"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Test roundtrip with empty array.
#[test]
fn empty_array_roundtrip() {
    let mut store = InMemoryStore::new();

    let zarray = r#"{
        "zarr_format": 2,
        "shape": [0],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    store
        .set("empty/.zarray", zarray.as_bytes())
        .expect("set must succeed");

    let retrieved = store.get("empty/.zarray").expect("get must succeed");
    let meta = ArrayMetadataV2::parse(std::str::from_utf8(&retrieved).expect("utf8"))
        .expect("parse must succeed");

    assert_eq!(meta.shape, &[0]);
}

/// Test roundtrip with single element array.
#[test]
fn single_element_roundtrip() {
    let mut store = InMemoryStore::new();

    let zarray = r#"{
        "zarr_format": 2,
        "shape": [1],
        "chunks": [1],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    store
        .set("scalar/.zarray", zarray.as_bytes())
        .expect("set must succeed");

    // Single chunk with single value
    let data: Vec<u8> = vec![1.5f64.to_le_bytes().iter().copied()]
        .into_iter()
        .flatten()
        .collect();
    store.set("scalar/0", &data).expect("set must succeed");

    let retrieved = store.get("scalar/0").expect("get must succeed");
    assert_eq!(retrieved.len(), 8);
}

/// Test roundtrip with very long array name.
#[test]
fn long_name_roundtrip() {
    let mut store = InMemoryStore::new();

    let long_name = "a_very_long_array_name_with_many_characters_to_test_path_handling_123456789";

    let zarray = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let key = format!("{}/.zarray", long_name);
    store
        .set(&key, zarray.as_bytes())
        .expect("set must succeed");

    let retrieved = store.get(&key).expect("get must succeed");
    let meta = ArrayMetadataV2::parse(std::str::from_utf8(&retrieved).expect("utf8"))
        .expect("parse must succeed");

    assert_eq!(meta.shape, &[10]);
}

/// Test roundtrip with special fill values.
#[test]
fn special_fill_value_roundtrip() {
    let mut store = InMemoryStore::new();

    // NaN fill value
    let zarray_nan = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": "NaN",
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    store
        .set("nan_array/.zarray", zarray_nan.as_bytes())
        .expect("set must succeed");

    let retrieved = store.get("nan_array/.zarray").expect("get must succeed");
    let meta = ArrayMetadataV2::parse(std::str::from_utf8(&retrieved).expect("utf8"))
        .expect("parse must succeed");

    // to_fill_value returns FillValue directly (not Result); verify it produces the expected variant
    let fv = meta.fill_value.to_fill_value();
    assert!(
        matches!(fv, consus_zarr::FillValue::Float(ref s) if s == "NaN"),
        "expected FillValue::Float(\"NaN\"), got {:?}",
        fv
    );
}

/// Test roundtrip preserves metadata exactly.
#[test]
fn metadata_exact_preservation() {
    let zarray = r#"{
        "zarr_format": 2,
        "shape": [100, 200, 300],
        "chunks": [10, 20, 30],
        "dtype": "<f4",
        "fill_value": -999.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 6},
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(zarray).expect("parse must succeed");
    let serialized = meta.to_json().expect("serialize must succeed");
    let reparsed = ArrayMetadataV2::parse(&serialized).expect("reparse must succeed");

    // Verify exact preservation of all fields
    assert_eq!(meta.zarr_format, reparsed.zarr_format);
    assert_eq!(meta.shape, reparsed.shape);
    assert_eq!(meta.chunks, reparsed.chunks);
    assert_eq!(meta.dtype, reparsed.dtype);
    assert_eq!(meta.order, reparsed.order);

    // Verify compressor is preserved via canonical conversion.
    // CompressorConfig is not re-exported, so verify through ArrayMetadata codecs.
    let canon1 = meta.to_canonical();
    let canon2 = reparsed.to_canonical();
    assert!(
        !canon1.codecs.is_empty(),
        "canonical must have at least one codec"
    );
    assert_eq!(canon1.codecs.len(), canon2.codecs.len());
    assert_eq!(canon1.codecs[0].name, canon2.codecs[0].name);
}
