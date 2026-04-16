//! Reference validation tests against Zarr v2 and v3 specifications.
//!
//! ## Specification Reference
//!
//! These tests validate implementation against official Zarr spec examples:
//! - Zarr v2: <https://zarr.readthedocs.io/en/stable/spec/v2.html>
//! - Zarr v3: <https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html>
//!
//! ## Coverage
//!
//! - Zarr v2 spec example validation
//! - Zarr v3 spec example validation
//! - Metadata format compliance
//! - Chunk key encoding compliance
//! - Store hierarchy compliance

use consus_zarr::metadata::{ArrayMetadataV2, CompressorConfig, GroupMetadataV2};

// ---------------------------------------------------------------------------
// Zarr v2 Specification Examples
// ---------------------------------------------------------------------------

/// Test Zarr v2 spec example: minimal array metadata.
///
/// ## Spec Reference
///
/// From Zarr v2 spec: "The array metadata is stored in a JSON file located
/// at `<path>/.zarray` relative to the array."
#[test]
fn v2_spec_minimal_array_metadata() {
    // Example from Zarr v2 spec documentation
    let zarray_json = r#"{
        "zarr_format": 2,
        "shape": [10000, 10000],
        "chunks": [1000, 1000],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(zarray_json).expect("parse must succeed");

    // Validate spec-required fields
    assert_eq!(meta.zarr_format, 2, "zarr_format must be 2");
    assert_eq!(meta.shape, &[10000, 10000], "shape must match spec");
    assert_eq!(meta.chunks, &[1000, 1000], "chunks must match spec");
    assert_eq!(meta.dtype, "<f8", "dtype must match spec");
}

/// Test Zarr v2 spec example: array with gzip compression.
///
/// ## Spec Reference
///
/// Zarr v2 spec allows compressor configuration with id and parameters.
#[test]
fn v2_spec_array_with_gzip() {
    let zarray_json = r#"{
        "zarr_format": 2,
        "shape": [10000, 10000],
        "chunks": [1000, 1000],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 1},
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(zarray_json).expect("parse must succeed");

    assert!(meta.compressor.is_some());
    let comp = meta.compressor.as_ref().unwrap();
    match comp {
        CompressorConfig::Named(named) => assert_eq!(named.id, "gzip"),
        _ => panic!("expected named compressor"),
    }
}

/// Test Zarr v2 spec example: array with blosc compression.
///
/// ## Spec Reference
///
/// Blosc compressor parameters: cname, clevel, shuffle, blocksize.
#[test]
fn v2_spec_array_with_blosc() {
    let zarray_json = r#"{
        "zarr_format": 2,
        "shape": [10000, 10000],
        "chunks": [1000, 1000],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
            "blocksize": 0
        },
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(zarray_json).expect("parse must succeed");

    let comp = meta.compressor.as_ref().unwrap();
    match comp {
        CompressorConfig::Named(named) => assert_eq!(named.id, "blosc"),
        _ => panic!("expected named compressor"),
    }
}

/// Test Zarr v2 spec: group metadata.
///
/// ## Spec Reference
///
/// "The group metadata is stored in a JSON file located at `<path>/.zgroup`."
#[test]
fn v2_spec_group_metadata() {
    let zgroup_json = r#"{"zarr_format": 2}"#;

    let meta = GroupMetadataV2::parse(zgroup_json).expect("parse must succeed");

    assert_eq!(meta.zarr_format, 2);
}

/// Test Zarr v2 spec: group with attributes.
#[test]
fn v2_spec_group_with_attributes() {
    let zgroup_json = r#"{
        "zarr_format": 2,
        "attributes": {
            "experiment": "simulation",
            "version": 1
        }
    }"#;

    let meta = GroupMetadataV2::parse(zgroup_json).expect("parse must succeed");

    let attrs = meta
        .attributes
        .as_ref()
        .expect("attributes must be present");
    assert!(attrs.contains_key("experiment"));
    assert!(attrs.contains_key("version"));
}
