//! Zarr v2 metadata parsing and validation tests.
//!
//! ## Specification Reference
//!
//! Tests validate compliance with Zarr v2 specification:
//! <https://zarr.readthedocs.io/en/stable/spec/v2.html>
//!
//! ## Coverage
//!
//! - `.zarray` parsing and validation
//! - `.zgroup` parsing
//! - `.zattrs` parsing
//! - dtype string parsing (NumPy format)
//! - Compressor configuration
//! - Fill value handling (including special values)
//! - Filter configuration

use consus_zarr::metadata::{ArrayMetadataV2, CompressorConfig, FilterId, GroupMetadataV2};
use consus_zarr::metadata::{parse_zattrs, serialize_zattrs};

// ---------------------------------------------------------------------------
// .zarray parsing tests
// ---------------------------------------------------------------------------

/// Test minimal .zarray parsing with null compressor.
///
/// ## Spec Compliance
///
/// Zarr v2 requires: zarr_format, shape, chunks, dtype, fill_value, order,
/// compressor, filters. All fields must be present.
#[test]
fn parse_minimal_zarray() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.zarr_format, 2);
    assert_eq!(meta.shape, &[100, 100]);
    assert_eq!(meta.chunks, &[10, 10]);
    assert_eq!(meta.dtype, "<f8");
}

/// Test .zarray with gzip compressor configuration.
///
/// ## Spec Compliance
///
/// Compressor must have `id` field. Configuration parameters (e.g., level)
/// are codec-specific.
#[test]
fn parse_zarray_with_gzip_compressor() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [1000, 1000],
        "chunks": [100, 100],
        "dtype": "<f4",
        "fill_value": 0.0,
        "order": "F",
        "compressor": {"id": "gzip", "level": 5},
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert!(meta.compressor.is_some());
    let comp = meta.compressor.as_ref().unwrap();
    match comp {
        CompressorConfig::Named(named) => {
            assert_eq!(named.id, "gzip");
        }
        _ => panic!("expected named compressor"),
    }
}

/// Test .zarray with blosc compressor.
///
/// ## Spec Compliance
///
/// Blosc is a meta-compressor with multiple internal codecs.
/// Configuration includes: cname, clevel, shuffle, blocksize.
#[test]
fn parse_zarray_with_blosc_compressor() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [512, 512, 512],
        "chunks": [64, 64, 64],
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

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let comp = meta.compressor.as_ref().unwrap();
    match comp {
        CompressorConfig::Named(named) => {
            assert_eq!(named.id, "blosc");
        }
        _ => panic!("expected named compressor"),
    }
}

/// Test .zarray with zstd compressor.
#[test]
fn parse_zarray_with_zstd_compressor() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<i4",
        "fill_value": -999,
        "order": "C",
        "compressor": {"id": "zstd", "level": 3},
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let comp = meta.compressor.as_ref().unwrap();
    match comp {
        CompressorConfig::Named(named) => assert_eq!(named.id, "zstd"),
        _ => panic!("expected named compressor"),
    }
}

/// Test .zarray with lz4 compressor.
#[test]
fn parse_zarray_with_lz4_compressor() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<u2",
        "fill_value": 0,
        "order": "C",
        "compressor": {"id": "lz4", "level": 1},
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let comp = meta.compressor.as_ref().unwrap();
    match comp {
        CompressorConfig::Named(named) => assert_eq!(named.id, "lz4"),
        _ => panic!("expected named compressor"),
    }
}

// ---------------------------------------------------------------------------
// Dtype string parsing tests
// ---------------------------------------------------------------------------

/// Test little-endian float64 dtype parsing.
///
/// ## Spec Compliance
///
/// NumPy dtype format: `<` for little-endian, `f` for float, `8` for 8 bytes.
#[test]
fn dtype_little_endian_float64() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.dtype, "<f8");
}

/// Test big-endian int32 dtype parsing.
#[test]
fn dtype_big_endian_int32() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": ">i4",
        "fill_value": 0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.dtype, ">i4");
}

/// Test native-endian uint16 dtype parsing.
#[test]
fn dtype_native_endian_uint16() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "=u2",
        "fill_value": 0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.dtype, "=u2");
}

/// Test complex dtype (complex128 = two float64).
#[test]
fn dtype_complex128() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<c16",
        "fill_value": "0j",
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.dtype, "<c16");
}

/// Test boolean dtype.
#[test]
fn dtype_bool() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "|b1",
        "fill_value": false,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.dtype, "|b1");
}

/// Test structured dtype with fields.
///
/// Structured dtypes use a JSON array representation (`[["x", "<f4"], ...]`)
/// which is incompatible with the current `dtype: String` field. The parser
/// rejects this input.
#[test]
fn dtype_structured() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": [["x", "<f4"], ["y", "<f4"]],
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let result = ArrayMetadataV2::parse(json);
    assert!(
        result.is_err(),
        "structured dtype (JSON array) is not supported by String field"
    );
}

// ---------------------------------------------------------------------------
// Fill value tests
// ---------------------------------------------------------------------------

/// Test integer fill value.
#[test]
fn fill_value_integer() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<i4",
        "fill_value": -999,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

/// Test float fill value.
#[test]
fn fill_value_float() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": -999.5,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

/// Test NaN fill value (special IEEE 754 value).
#[test]
fn fill_value_nan() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": "NaN",
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

/// Test positive infinity fill value.
#[test]
fn fill_value_infinity() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": "Infinity",
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

/// Test negative infinity fill value.
#[test]
fn fill_value_neg_infinity() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": "-Infinity",
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

/// Test null fill value (uninitialized chunks).
#[test]
fn fill_value_null() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "<f8",
        "fill_value": null,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

/// Test boolean fill value.
#[test]
fn fill_value_bool() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "|b1",
        "fill_value": true,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

/// Test string fill value.
#[test]
fn fill_value_string() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10],
        "chunks": [10],
        "dtype": "|S10",
        "fill_value": "NODATA",
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let _fv = meta.fill_value.to_fill_value();
}

// ---------------------------------------------------------------------------
// .zgroup parsing tests
// ---------------------------------------------------------------------------

/// Test minimal .zgroup parsing.
#[test]
fn parse_zgroup_minimal() {
    let json = r#"{"zarr_format": 2}"#;

    let meta = GroupMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.zarr_format, 2);
}

/// Test .zgroup with attributes.
#[test]
fn parse_zgroup_with_attributes() {
    let json = r#"{
        "zarr_format": 2,
        "attributes": {
            "name": "experiment_001",
            "version": 1
        }
    }"#;

    let meta = GroupMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.zarr_format, 2);
    let attrs = meta
        .attributes
        .as_ref()
        .expect("attributes must be present");
    assert!(attrs.contains_key("name"));
}

// ---------------------------------------------------------------------------
// .zattrs parsing tests
// ---------------------------------------------------------------------------

/// Test .zattrs parsing with various attribute types.
#[test]
fn parse_zattrs_types() {
    let json = r#"{
        "string_attr": "value",
        "int_attr": 42,
        "float_attr": 3.14,
        "bool_attr": true,
        "array_attr": [1, 2, 3],
        "nested_attr": {
            "inner": "value"
        }
    }"#;

    let attrs = parse_zattrs(json).expect("parse must succeed");
    assert_eq!(attrs.len(), 6);
    assert!(attrs.iter().any(|(k, _)| k == "string_attr"));
    assert!(attrs.iter().any(|(k, _)| k == "int_attr"));
    assert!(attrs.iter().any(|(k, _)| k == "float_attr"));
    assert!(attrs.iter().any(|(k, _)| k == "bool_attr"));
    assert!(attrs.iter().any(|(k, _)| k == "array_attr"));
    assert!(attrs.iter().any(|(k, _)| k == "nested_attr"));
}

/// Test .zattrs roundtrip serialization.
#[test]
fn zattrs_roundtrip() {
    let original = r#"{
        "experiment": "test",
        "iteration": 5,
        "threshold": 0.75
    }"#;

    let attrs = parse_zattrs(original).expect("parse must succeed");
    let serialized = serialize_zattrs(&attrs).expect("serialize must succeed");
    let reparsed = parse_zattrs(&serialized).expect("reparse must succeed");

    assert_eq!(attrs.len(), reparsed.len());
}

/// Test empty .zattrs.
#[test]
fn parse_zattrs_empty() {
    let json = r#"{}"#;

    let attrs = parse_zattrs(json).expect("parse must succeed");
    assert!(attrs.is_empty());
}

// ---------------------------------------------------------------------------
// Filter tests
// ---------------------------------------------------------------------------

/// Test .zarray with named filter.
#[test]
fn parse_zarray_with_named_filter() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": [{"id": "shuffle", "elementsize": 8}]
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert!(meta.filters.is_some());
    let filters = meta.filters.as_ref().unwrap();
    assert_eq!(filters.len(), 1);
    match &filters[0].id {
        FilterId::Name(name) => assert_eq!(name, "shuffle"),
        _ => panic!("expected named filter"),
    }
}

/// Test .zarray with HDF5 filter ID format.
#[test]
fn parse_zarray_with_hdf5_filter_id() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": [{"id": 32008, "elementsize": 8}]
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert!(meta.filters.is_some());
}

/// Test .zarray with multiple filters.
#[test]
fn parse_zarray_with_multiple_filters() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 1},
        "filters": [
            {"id": "shuffle", "elementsize": 8},
            {"id": "delta", "dtype": "<f8"}
        ]
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let filters = meta.filters.as_ref().unwrap();
    assert_eq!(filters.len(), 2);
}

// ---------------------------------------------------------------------------
// Serialization tests
// ---------------------------------------------------------------------------

/// Test .zarray JSON serialization.
#[test]
fn zarray_serialization() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 1},
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let serialized = meta.to_json().expect("serialize must succeed");
    let reparsed = ArrayMetadataV2::parse(&serialized).expect("reparse must succeed");

    assert_eq!(meta.shape, reparsed.shape);
    assert_eq!(meta.chunks, reparsed.chunks);
    assert_eq!(meta.dtype, reparsed.dtype);
}

/// Test .zgroup JSON serialization.
#[test]
fn zgroup_serialization() {
    let json = r#"{"zarr_format": 2}"#;

    let meta = GroupMetadataV2::parse(json).expect("parse must succeed");
    let serialized = meta.to_json().expect("serialize must succeed");
    let reparsed = GroupMetadataV2::parse(&serialized).expect("reparse must succeed");

    assert_eq!(meta.zarr_format, reparsed.zarr_format);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Test single-element array shape.
#[test]
fn single_element_shape() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [1],
        "chunks": [1],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.shape, &[1]);
    assert_eq!(meta.chunks, &[1]);
}

/// Test large array shape.
#[test]
fn large_array_shape() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10000, 10000, 1000],
        "chunks": [100, 100, 100],
        "dtype": "<f4",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.shape, &[10000, 10000, 1000]);
}

/// Test 1D array.
#[test]
fn one_dimensional_array() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [1000],
        "chunks": [100],
        "dtype": "<i8",
        "fill_value": -1,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.shape.len(), 1);
}

/// Test 5D array.
#[test]
fn five_dimensional_array() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10, 20, 30, 40, 50],
        "chunks": [5, 10, 15, 20, 25],
        "dtype": "<f4",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert_eq!(meta.shape.len(), 5);
}

/// Test C order vs F order.
#[test]
fn order_c_vs_f() {
    let json_c = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let json_f = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "F",
        "compressor": null,
        "filters": null
    }"#;

    let meta_c = ArrayMetadataV2::parse(json_c).expect("parse must succeed");
    let meta_f = ArrayMetadataV2::parse(json_f).expect("parse must succeed");

    assert_eq!(meta_c.order, 'C');
    assert_eq!(meta_f.order, 'F');
}

/// Test chunk shape larger than array shape.
#[test]
fn chunks_larger_than_shape() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [10, 10],
        "chunks": [100, 100],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": null,
        "filters": null
    }"#;

    let meta = ArrayMetadataV2::parse(json).expect("parse must succeed");
    assert!(meta.chunks[0] > meta.shape[0]);
}

// ---------------------------------------------------------------------------
// Canonical conversion tests
// ---------------------------------------------------------------------------

/// Test ArrayMetadataV2 to canonical ArrayMetadata conversion.
#[test]
fn array_metadata_v2_to_canonical() {
    let json = r#"{
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "fill_value": 0.0,
        "order": "C",
        "compressor": {"id": "gzip", "level": 1},
        "filters": null
    }"#;

    let v2 = ArrayMetadataV2::parse(json).expect("parse must succeed");
    let canonical = v2.to_canonical();

    assert_eq!(canonical.shape, &[100, 100]);
    assert!(!canonical.codecs.is_empty());
}

/// Test GroupMetadataV2 to canonical GroupMetadata conversion.
#[test]
fn group_metadata_v2_to_canonical() {
    let json = r#"{"zarr_format": 2, "attributes": {"name": "test"}}"#;

    let v2 = GroupMetadataV2::parse(json).expect("parse must succeed");
    let canonical = v2.to_canonical();

    assert!(canonical.attributes.iter().any(|(k, _)| k == "name"));
}
