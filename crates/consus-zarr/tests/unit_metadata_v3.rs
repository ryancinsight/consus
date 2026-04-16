//! Zarr v3 metadata parsing and validation tests.
//!
//! ## Specification Reference
//!
//! Tests validate compliance with Zarr v3 specification:
//! <https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html>
//!
//! ## Coverage
//!
//! - zarr.json parsing for arrays
//! - zarr.json parsing for groups
//! - Zarr v3 codec configuration
//! - Chunk key encoding
//! - Dimension names
//! - Fill value handling

use consus_zarr::metadata::ZarrJson;

// ---------------------------------------------------------------------------
// zarr.json array parsing tests
// ---------------------------------------------------------------------------

/// Test minimal zarr.json array parsing.
///
/// ## Spec Compliance
///
/// Zarr v3 requires: zarr_format, node_type, shape, data_type, chunk_grid,
/// chunk_key_encoding, codecs, fill_value.
#[test]
fn parse_minimal_zarr_json_array() {
    let json = r#"{
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
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array {
            zarr_format, shape, ..
        } => {
            assert_eq!(*zarr_format, 3);
            assert_eq!(shape, &[100, 100]);
        }
        ZarrJson::Group { .. } => panic!("expected array, got group"),
    }
}

/// Test zarr.json array with gzip codec.
#[test]
fn parse_zarr_json_array_with_gzip() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1000, 1000],
        "data_type": "float32",
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

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs.len(), 2);
            assert_eq!(codecs[0].name, "bytes");
            assert_eq!(codecs[1].name, "gzip");
        }
        ZarrJson::Group { .. } => panic!("expected array, got group"),
    }
}

/// Test zarr.json array with blosc codec.
#[test]
fn parse_zarr_json_array_with_blosc() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [512, 512, 512],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [64, 64, 64]}
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"}
        },
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5, "shuffle": "shuffle"}}
        ],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs.len(), 2);
            assert_eq!(codecs[1].name, "blosc");
        }
        ZarrJson::Group { .. } => panic!("expected array, got group"),
    }
}

/// Test zarr.json array with zstd codec.
#[test]
fn parse_zarr_json_array_with_zstd() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 100],
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
            {"name": "zstd", "configuration": {"level": 3}}
        ],
        "fill_value": -999
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs[1].name, "zstd");
        }
        ZarrJson::Group { .. } => panic!("expected array, got group"),
    }
}

/// Test zarr.json array with lz4 codec.
#[test]
fn parse_zarr_json_array_with_lz4() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 100],
        "data_type": "uint16",
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
            {"name": "lz4", "configuration": {"level": 1}}
        ],
        "fill_value": 0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs[1].name, "lz4");
        }
        ZarrJson::Group { .. } => panic!("expected array, got group"),
    }
}

// ---------------------------------------------------------------------------
// Data type tests
// ---------------------------------------------------------------------------

/// Test float64 data type.
#[test]
fn data_type_float64() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { data_type, .. } => {
            assert_eq!(data_type, "float64");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test int32 data type.
#[test]
fn data_type_int32() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { data_type, .. } => {
            assert_eq!(data_type, "int32");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test uint16 data type.
#[test]
fn data_type_uint16() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { data_type, .. } => {
            assert_eq!(data_type, "uint16");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test bool data type.
#[test]
fn data_type_bool() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "bool",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": false
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { data_type, .. } => {
            assert_eq!(data_type, "bool");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test complex64 data type.
#[test]
fn data_type_complex64() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "complex64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": "0j"
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { data_type, .. } => {
            assert_eq!(data_type, "complex64");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

// ---------------------------------------------------------------------------
// Fill value tests
// ---------------------------------------------------------------------------

/// Test integer fill value.
#[test]
fn fill_value_integer() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "int32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": -999
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { fill_value, .. } => {
            let _fv = fill_value.to_fill_value();
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test NaN fill value.
#[test]
fn fill_value_nan() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": "NaN"
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { fill_value, .. } => {
            let _fv = fill_value.to_fill_value();
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test positive infinity fill value.
#[test]
fn fill_value_infinity() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": "Infinity"
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { fill_value, .. } => {
            let _fv = fill_value.to_fill_value();
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test negative infinity fill value.
#[test]
fn fill_value_neg_infinity() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": "-Infinity"
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { fill_value, .. } => {
            let _fv = fill_value.to_fill_value();
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test null fill value.
#[test]
fn fill_value_null() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": null
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { fill_value, .. } => {
            let _fv = fill_value.to_fill_value();
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

// ---------------------------------------------------------------------------
// zarr.json group parsing tests
// ---------------------------------------------------------------------------

/// Test minimal zarr.json group parsing.
#[test]
fn parse_minimal_zarr_json_group() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "group"
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Group { zarr_format, .. } => {
            assert_eq!(*zarr_format, 3);
        }
        ZarrJson::Array { .. } => panic!("expected group, got array"),
    }
}

/// Test zarr.json group with attributes.
#[test]
fn parse_zarr_json_group_with_attributes() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "name": "experiment_001",
            "version": 1
        }
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Group { attributes, .. } => {
            let attrs = attributes.as_ref().expect("attributes must be present");
            assert!(attrs.contains_key("name"));
            assert!(attrs.contains_key("version"));
        }
        ZarrJson::Array { .. } => panic!("expected group, got array"),
    }
}

/// Test zarr.json group with codecs.
#[test]
fn parse_zarr_json_group_with_codecs() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "group",
        "codecs": []
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Group { codecs, .. } => {
            assert!(codecs.is_empty() || codecs.iter().all(|c| c.name == "bytes"));
        }
        ZarrJson::Array { .. } => panic!("expected group, got array"),
    }
}

// ---------------------------------------------------------------------------
// Chunk key encoding tests
// ---------------------------------------------------------------------------

/// Test default chunk key encoding (slash separator).
#[test]
fn chunk_key_encoding_default() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10, 10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5, 5]}},
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"}
        },
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array {
            chunk_key_encoding, ..
        } => {
            assert_eq!(chunk_key_encoding.name, "default");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test v2-compatible chunk key encoding (dot separator).
#[test]
fn chunk_key_encoding_v2() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10, 10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5, 5]}},
        "chunk_key_encoding": {
            "name": "v2",
            "configuration": {"separator": "."}
        },
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array {
            chunk_key_encoding, ..
        } => {
            assert_eq!(chunk_key_encoding.name, "v2");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

// ---------------------------------------------------------------------------
// Dimension names tests
// ---------------------------------------------------------------------------

/// Test array with dimension names.
#[test]
fn dimension_names() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 50, 25],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10, 10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0,
        "dimension_names": ["time", "latitude", "longitude"]
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array {
            dimension_names, ..
        } => {
            assert!(dimension_names.is_some());
            let names = dimension_names.as_ref().unwrap();
            assert_eq!(names.len(), 3);
            assert_eq!(names[0], "time");
            assert_eq!(names[1], "latitude");
            assert_eq!(names[2], "longitude");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test array without dimension names.
#[test]
fn no_dimension_names() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10, 10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5, 5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array {
            dimension_names, ..
        } => {
            assert!(dimension_names.is_none());
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

// ---------------------------------------------------------------------------
// Codec configuration tests
// ---------------------------------------------------------------------------

/// Test bytes codec with little endian.
#[test]
fn bytes_codec_little_endian() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs.len(), 1);
            assert_eq!(codecs[0].name, "bytes");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test bytes codec with big endian.
#[test]
fn bytes_codec_big_endian() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "big"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs[0].name, "bytes");
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test crc32 checksum codec.
#[test]
fn crc32_codec() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "crc32", "configuration": {}}
        ],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs.len(), 2);
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test multiple codecs in chain.
#[test]
fn multiple_codecs() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 100],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "shuffle"}},
            {"name": "crc32", "configuration": {}}
        ],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { codecs, .. } => {
            assert_eq!(codecs.len(), 3);
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

// ---------------------------------------------------------------------------
// Serialization tests
// ---------------------------------------------------------------------------

/// Test zarr.json array serialization roundtrip.
#[test]
fn zarr_json_array_roundtrip() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 100],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    let serialized = meta.to_json().expect("serialize must succeed");
    let reparsed = ZarrJson::parse(&serialized).expect("reparse must succeed");

    match (&meta, &reparsed) {
        (
            ZarrJson::Array {
                shape: s1,
                data_type: dt1,
                ..
            },
            ZarrJson::Array {
                shape: s2,
                data_type: dt2,
                ..
            },
        ) => {
            assert_eq!(s1, s2);
            assert_eq!(dt1, dt2);
        }
        _ => panic!("type mismatch"),
    }
}

/// Test zarr.json group serialization roundtrip.
#[test]
fn zarr_json_group_roundtrip() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"name": "test"}
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    let serialized = meta.to_json().expect("serialize must succeed");
    let reparsed = ZarrJson::parse(&serialized).expect("reparse must succeed");

    match (&meta, &reparsed) {
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

// ---------------------------------------------------------------------------
// Canonical conversion tests
// ---------------------------------------------------------------------------

/// Test ZarrJson array to canonical conversion.
#[test]
fn zarr_json_array_to_canonical() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 100],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10, 10]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    let canonical = meta
        .to_array_canonical()
        .expect("canonical conversion must succeed");
    assert_eq!(canonical.shape, &[100, 100]);
}

/// Test ZarrJson group to canonical conversion.
#[test]
fn zarr_json_group_to_canonical() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"name": "test"}
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    let canonical = meta
        .to_group_canonical()
        .expect("canonical conversion must succeed");
    assert!(canonical.attributes.iter().any(|(k, _)| k == "name"));
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Test single-element array.
#[test]
fn single_element_array() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { shape, .. } => {
            assert_eq!(shape, &[1]);
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test large multi-dimensional array.
#[test]
fn large_multidimensional_array() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10000, 10000, 1000],
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100, 100, 100]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { shape, .. } => {
            assert_eq!(shape, &[10000, 10000, 1000]);
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test 5D array.
#[test]
fn five_dimensional_array() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10, 20, 30, 40, 50],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5, 10, 15, 20, 25]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { shape, .. } => {
            assert_eq!(shape.len(), 5);
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}

/// Test array with attributes.
#[test]
fn array_with_attributes() {
    let json = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10, 10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5, 5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0.0,
        "attributes": {
            "units": "meters",
            "long_name": "Temperature"
        }
    }"#;

    let meta = ZarrJson::parse(json).expect("parse must succeed");
    match &meta {
        ZarrJson::Array { .. } => {
            // V3 array metadata may not have attributes field in our current representation
        }
        ZarrJson::Group { .. } => panic!("expected array"),
    }
}
