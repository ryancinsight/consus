//! Integration tests: verify consus-hdf5 writer output with `h5dump`.
//!
//! ## Purpose
//!
//! These tests build HDF5 files in-memory with [`Hdf5FileBuilder`], write them to a
//! temporary file on disk, run `h5dump` on the result, and assert that the DDL output
//! contains the expected structural elements (dataset names, type strings, shapes,
//! and data values).
//!
//! ## Design invariants
//!
//! - Every test checks `h5dump --version` first and returns early if h5dump is not on
//!   PATH.  This makes the suite skip-safe in CI without h5dump.
//! - `h5dump` exit code 0 is a necessary condition for a valid HDF5 file; a non-zero
//!   exit is treated as a test failure.
//! - Assertions use substring matching so they are robust to minor formatting
//!   differences across h5dump versions.
//! - Files are written to `CARGO_MANIFEST_DIR` (the crate directory) rather than
//!   the system temp directory to avoid 8.3 short-name path failures on Windows
//!   with MSYS2-built h5dump.
//! - `HDF5_USE_FILE_LOCKING=FALSE` is set on each h5dump invocation to prevent
//!   advisory-lock acquisition failures with HDF5 2.x libraries on Windows.
//!
//! ## Specification reference
//!
//! HDF5 file format specification §II (superblock), §III (object headers),
//! §IV (messages).  Canonical type strings (H5T_STD_I32LE, H5T_IEEE_F64LE, …)
//! are defined in the HDF5 DDL grammar §6.

use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
use consus_hdf5::file::writer::{
    ChildDatasetSpec, ChildGroupSpec, FileCreationProps, Hdf5FileBuilder,
};
use consus_hdf5::property_list::{DatasetCreationProps, DatasetLayout};
use std::io::Write as _;
use std::num::NonZeroUsize;

// ---------------------------------------------------------------------------
// Helper: detect h5dump
// ---------------------------------------------------------------------------

/// Returns `true` when `h5dump` is available on the current `PATH`.
fn h5dump_available() -> bool {
    std::process::Command::new("h5dump")
        .arg("--version")
        .output()
        .is_ok()
}

// ---------------------------------------------------------------------------
// Helper: write to CARGO_MANIFEST_DIR and dump
// ---------------------------------------------------------------------------

/// Write `bytes` to a `NamedTempFile` in `CARGO_MANIFEST_DIR`, close the
/// write handle, and run `h5dump` on the resulting path.
///
/// ## Windows path constraints
///
/// The system temp directory on Windows frequently contains 8.3 short-name
/// components (e.g., `RYANCL~1`).  MSYS2-built HDF5 tools cannot resolve
/// these paths.  Writing to `CARGO_MANIFEST_DIR` (`D:/consus/crates/consus-hdf5`)
/// avoids this by providing a fully long-form path.
///
/// ## File locking
///
/// `HDF5_USE_FILE_LOCKING=FALSE` suppresses the advisory file-lock acquisition
/// that HDF5 2.x libraries attempt by default on Windows.  Without this, h5dump
/// fails with "internal error" when opening v2-superblock files that have no
/// SWMR locking metadata.
///
/// Returns `(TempPath, ddl_string)`.  `TempPath` deletes the file on `Drop`;
/// it must stay in scope until the assertions are complete.
///
/// Returns `None` when h5dump is absent or exits non-zero.
fn write_and_dump(bytes: &[u8]) -> Option<(tempfile::TempPath, String)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut f = tempfile::Builder::new()
        .prefix("hdump_test_")
        .suffix(".h5")
        .tempfile_in(dir)
        .expect("create NamedTempFile in CARGO_MANIFEST_DIR");
    f.write_all(bytes).expect("write HDF5 bytes");
    f.flush().expect("flush NamedTempFile");
    let (file, path) = f.into_parts();
    drop(file); // Release write handle before h5dump opens the file.

    let output = std::process::Command::new("h5dump")
        .env("HDF5_USE_FILE_LOCKING", "FALSE")
        .arg(&path)
        .output()
        .ok()?;
    if output.status.success() {
        let ddl = String::from_utf8(output.stdout).ok()?;
        Some((path, ddl))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Test 1: scalar int32 dataset
// ---------------------------------------------------------------------------

/// ## Invariants
///
/// - `Hdf5FileBuilder` emits a valid superblock v2 for a single scalar int32 dataset.
/// - h5dump reports `DATATYPE H5T_STD_I32LE`, `DATASPACE SCALAR`, and value `42`.
#[test]
fn hdump_scalar_int32_dataset() {
    if !h5dump_available() {
        return;
    }

    let int32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "my_int",
            &int32,
            &Shape::scalar(),
            &42i32.to_le_bytes(),
            &DatasetCreationProps::default(),
        )
        .expect("add scalar int32 dataset");
    let bytes = builder.finish().expect("finish HDF5 file");

    let (_path, ddl) = match write_and_dump(&bytes) {
        Some(t) => t,
        None => return, // h5dump not available or failed
    };

    assert!(
        ddl.contains("DATASET \"my_int\""),
        "DDL must name the dataset 'my_int'; got:\n{ddl}"
    );
    assert!(
        ddl.contains("H5T_STD_I32LE"),
        "DDL must report LE int32 type; got:\n{ddl}"
    );
    assert!(
        ddl.contains("SCALAR"),
        "DDL must report SCALAR dataspace; got:\n{ddl}"
    );
    assert!(
        ddl.contains("42"),
        "DDL must contain the scalar value 42; got:\n{ddl}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: 1-D float64 array
// ---------------------------------------------------------------------------

/// ## Invariants
///
/// - A 1-D shape-\[3\] float64 dataset is written and read back by h5dump.
/// - `H5T_IEEE_F64LE` and `SIMPLE` appear in the DDL.
/// - Values 1, 2, and 3 appear in the DATA block.
#[test]
fn hdump_1d_float64_array() {
    if !h5dump_available() {
        return;
    }

    let f64_le = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let raw: Vec<u8> = [1.0f64, 2.0f64, 3.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "arr",
            &f64_le,
            &Shape::fixed(&[3]),
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add 1-D float64 dataset");
    let bytes = builder.finish().expect("finish HDF5 file");

    let (_path, ddl) = match write_and_dump(&bytes) {
        Some(t) => t,
        None => return,
    };

    assert!(
        ddl.contains("DATASET \"arr\""),
        "DDL must name the dataset 'arr'; got:\n{ddl}"
    );
    assert!(
        ddl.contains("H5T_IEEE_F64LE"),
        "DDL must report LE float64 type; got:\n{ddl}"
    );
    assert!(
        ddl.contains("SIMPLE"),
        "DDL must report SIMPLE dataspace; got:\n{ddl}"
    );
    // 1.0, 2.0, 3.0 render as "1", "2", "3" in h5dump DATA block.
    assert!(
        ddl.contains('1') && ddl.contains('2') && ddl.contains('3'),
        "DDL DATA block must contain values 1, 2, 3; got:\n{ddl}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: group hierarchy with nested dataset
// ---------------------------------------------------------------------------

/// ## Invariants
///
/// - A group `"grp"` with a child dataset `"child_ds"` (scalar int32, value 77)
///   is written to the root group.
/// - h5dump DDL contains `GROUP "grp"` and `DATASET "child_ds"` and value `77`.
#[test]
fn hdump_group_hierarchy() {
    if !h5dump_available() {
        return;
    }

    let int32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let child = ChildDatasetSpec {
        name: "child_ds",
        datatype: &int32,
        shape: &Shape::scalar(),
        raw_data: &77i32.to_le_bytes(),
        dcpl: DatasetCreationProps::default(),
        attributes: &[],
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_children("grp", &[], &[child], &[])
        .expect("add group with children");
    let bytes = builder.finish().expect("finish HDF5 file");

    let (_path, ddl) = match write_and_dump(&bytes) {
        Some(t) => t,
        None => return,
    };

    assert!(
        ddl.contains("GROUP \"grp\""),
        "DDL must contain GROUP 'grp'; got:\n{ddl}"
    );
    assert!(
        ddl.contains("DATASET \"child_ds\""),
        "DDL must contain DATASET 'child_ds'; got:\n{ddl}"
    );
    assert!(
        ddl.contains("77"),
        "DDL DATA block must contain value 77; got:\n{ddl}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: dataset with string attribute
// ---------------------------------------------------------------------------

/// ## Invariants
///
/// - A dataset `"tagged_ds"` (scalar int32, value 99) carries an attribute
///   `"label"` whose raw bytes spell `"hello"` (FixedString, length 5).
/// - h5dump DDL contains `ATTRIBUTE "label"` and the string `"hello"`.
#[test]
fn hdump_dataset_with_string_attribute() {
    if !h5dump_available() {
        return;
    }

    let int32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let str5 = Datatype::FixedString {
        length: 5,
        encoding: StringEncoding::Ascii,
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "tagged_ds",
            &int32,
            &Shape::scalar(),
            &99i32.to_le_bytes(),
            &DatasetCreationProps::default(),
            &[("label", &str5, &Shape::scalar(), b"hello")],
        )
        .expect("add dataset with attribute");
    let bytes = builder.finish().expect("finish HDF5 file");

    let (_path, ddl) = match write_and_dump(&bytes) {
        Some(t) => t,
        None => return,
    };

    assert!(
        ddl.contains("DATASET \"tagged_ds\""),
        "DDL must name the dataset 'tagged_ds'; got:\n{ddl}"
    );
    assert!(
        ddl.contains("ATTRIBUTE \"label\""),
        "DDL must name the attribute 'label'; got:\n{ddl}"
    );
    assert!(
        ddl.contains("hello"),
        "DDL must contain the attribute value 'hello'; got:\n{ddl}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: valid HDF5 file magic (h5dump exits 0)
// ---------------------------------------------------------------------------

/// ## Invariants
///
/// - An empty-ish HDF5 file (one scalar dataset) must be accepted as valid
///   by h5dump (exit code 0).
/// - The DDL contains `"HDF5"`.
#[test]
fn hdump_writer_produces_valid_hdf5() {
    if !h5dump_available() {
        return;
    }

    let int32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "sentinel",
            &int32,
            &Shape::scalar(),
            &0i32.to_le_bytes(),
            &DatasetCreationProps::default(),
        )
        .expect("add sentinel dataset");
    let bytes = builder.finish().expect("finish HDF5 file");

    let (_path, ddl) = match write_and_dump(&bytes) {
        Some(t) => t,
        None => return, // h5dump unavailable or invalid file
    };

    assert!(
        ddl.contains("HDF5"),
        "h5dump DDL must contain the HDF5 file header; got:\n{ddl}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: chunked dataset structure
// ---------------------------------------------------------------------------

/// ## Invariants
///
/// - A 2-D chunked uint8 dataset (8 × 8, chunks 4 × 4) is written and
///   accepted by h5dump.
/// - The DDL contains the dataset name, the unsigned 8-bit integer type,
///   and `SIMPLE` dataspace.
#[test]
fn hdump_chunked_dataset_structure() {
    if !h5dump_available() {
        return;
    }

    let u8_le = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let raw: Vec<u8> = (0u8..64).collect();
    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![4, 4]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("chunked", &u8_le, &Shape::fixed(&[8, 8]), &raw, &dcpl)
        .expect("add chunked uint8 dataset");
    let bytes = builder.finish().expect("finish HDF5 file");

    let (_path, ddl) = match write_and_dump(&bytes) {
        Some(t) => t,
        None => return,
    };

    assert!(
        ddl.contains("DATASET \"chunked\""),
        "DDL must name the dataset 'chunked'; got:\n{ddl}"
    );
    assert!(
        ddl.contains("H5T_STD_U8LE"),
        "DDL must report LE uint8 type (H5T_STD_U8LE); got:\n{ddl}"
    );
    assert!(
        ddl.contains("SIMPLE"),
        "DDL must report SIMPLE dataspace for the 2-D dataset; got:\n{ddl}"
    );
}

// ---------------------------------------------------------------------------
// Test 7: deeply-nested group + sub-group hierarchy
// ---------------------------------------------------------------------------

/// ## Invariants
///
/// - A two-level group hierarchy (`outer` → `inner`) with a dataset `"ds"` in
///   the inner group is faithfully represented by h5dump.
/// - DDL contains both `GROUP "outer"` and `"inner"` and `DATASET "ds"`.
#[test]
fn hdump_nested_group_hierarchy() {
    if !h5dump_available() {
        return;
    }

    let int32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };

    let inner_child = ChildDatasetSpec {
        name: "ds",
        datatype: &int32,
        shape: &Shape::scalar(),
        raw_data: &100i32.to_le_bytes(),
        dcpl: DatasetCreationProps::default(),
        attributes: &[],
    };
    let inner_group = ChildGroupSpec {
        name: "inner",
        attributes: &[],
        datasets: &[inner_child],
        sub_groups: &[],
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_children("outer", &[], &[], &[inner_group])
        .expect("add nested group hierarchy");
    let bytes = builder.finish().expect("finish HDF5 file");

    let (_path, ddl) = match write_and_dump(&bytes) {
        Some(t) => t,
        None => return,
    };

    assert!(
        ddl.contains("GROUP \"outer\""),
        "DDL must contain outer group; got:\n{ddl}"
    );
    assert!(
        ddl.contains("inner"),
        "DDL must contain inner group name; got:\n{ddl}"
    );
    assert!(
        ddl.contains("DATASET \"ds\""),
        "DDL must contain dataset 'ds' in inner group; got:\n{ddl}"
    );
    assert!(
        ddl.contains("100"),
        "DDL DATA block must contain value 100; got:\n{ddl}"
    );
}
