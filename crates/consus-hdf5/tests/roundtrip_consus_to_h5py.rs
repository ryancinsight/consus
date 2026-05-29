//! Roundtrip tests: consus-hdf5 writer → h5py reader.
//!
//! ## Purpose
//!
//! These tests verify that HDF5 files produced by consus-hdf5 can be read
//! correctly by h5py (Python HDF5 library).  Each test:
//!
//! 1. Builds an in-memory HDF5 image with `Hdf5FileBuilder`.
//! 2. Writes it to a `NamedTempFile`.
//! 3. Invokes `data/verify_consus_output.py --file <path> --case <name>`.
//! 4. Asserts exit code 0.
//!
//! All tests skip gracefully when Python or h5py is unavailable.
//!
//! ## Coverage
//!
//! | Case                    | Layout      | Dtype   | Shape     | Notes              |
//! |-------------------------|-------------|---------|-----------|---------------------|
//! | `scalar_i32`            | contiguous  | int32   | scalar    | value 42            |
//! | `contiguous_1d_f64`     | contiguous  | float64 | (4,)      | [1.5,2.5,3.5,4.5]  |
//! | `contiguous_2d_i32`     | contiguous  | int32   | (3,4)     | 0..11              |
//! | `contiguous_3d_f64`     | contiguous  | float64 | (2,3,4)   | 0.0..23.0          |
//! | `chunked_1d_i32_v1`     | chunked v1  | int32   | (12,)     | chunks (4,)        |
//! | `chunked_2d_f64_v1`     | chunked v1  | float64 | (4,6)     | chunks (2,3)       |
//! | `chunked_1d_i32_v4`     | chunked v4  | int32   | (8,)      | chunks (4,)        |
//! | `deflate_chunked_1d_i32`| chunked v1  | int32   | (8,)      | deflate level 6    |
//! | `dataset_with_attr`     | contiguous  | int32   | scalar    | attribute "answer" |
//! | `group_nested_dataset`  | contiguous  | int32   | scalar    | grp/nested_value   |

use core::num::NonZeroUsize;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use consus_core::{ByteOrder as CoreByteOrder, Compression, Datatype, Shape};
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};

// ---------------------------------------------------------------------------
// Infrastructure helpers
// ---------------------------------------------------------------------------

/// Absolute path to `data/verify_consus_output.py` in the workspace.
fn verifier_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("verify_consus_output.py")
}

/// Locate a Python interpreter that has `h5py` importable.
///
/// Returns `None` when no suitable interpreter is found, causing tests to
/// skip rather than fail.
fn find_python() -> Option<String> {
    let candidates = [
        r"D:\miniforge3\python.exe",
        "python3",
        "python",
    ];
    for candidate in &candidates {
        let available = Command::new(candidate)
            .args(["-c", "import h5py"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if available {
            return Some((*candidate).to_string());
        }
    }
    None
}

/// Write `bytes` to a `NamedTempFile` and return it (keeps the file on disk
/// until the caller drops the handle).
fn write_temp(bytes: Vec<u8>) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
    tmp.write_all(&bytes).expect("write temp file");
    tmp
}

/// Write `bytes` to a temp file, invoke the Python verifier for `case`, and
/// assert exit code 0.  Skips when Python/h5py is unavailable or the verifier
/// script does not exist.
fn verify_with_h5py(bytes: Vec<u8>, case: &str) {
    let script = verifier_script();
    if !script.exists() {
        eprintln!(
            "Skipping {case}: verifier script not found at {:?}. \
             Run `data/verify_consus_output.py` after generating it.",
            script
        );
        return;
    }
    let python = match find_python() {
        Some(p) => p,
        None => {
            eprintln!("Skipping {case}: Python interpreter with h5py not found");
            return;
        }
    };

    let tmp = write_temp(bytes);
    let output = Command::new(&python)
        .args([
            script.to_str().unwrap(),
            "--file",
            tmp.path().to_str().unwrap(),
            "--case",
            case,
        ])
        .output()
        .expect("invoke Python verifier");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "h5py verification FAILED for case {case:?}\nstdout: {stdout}\nstderr: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// Datatype constructors
// ---------------------------------------------------------------------------

fn i32_dt() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: true,
    }
}

fn f64_dt() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
    }
}

fn f32_dt() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
    }
}

fn u8_dt() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    }
}

fn i16_dt() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(16).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: true,
    }
}

// ---------------------------------------------------------------------------
// Tests: contiguous datasets
// ---------------------------------------------------------------------------

/// Write a scalar int32 dataset, value 42.  h5py must read back int32
/// scalar == 42.
///
/// ## Invariant
///
/// Scalar contiguous datasets written by consus are readable by h5py.
#[test]
fn consus_scalar_i32_readable_by_h5py() {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "scalar_i32",
            &i32_dt(),
            &Shape::scalar(),
            &42i32.to_le_bytes(),
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "scalar_i32");
}

/// Write a 1-D float64 contiguous dataset, shape (4,), values [1.5, 2.5, 3.5, 4.5].
///
/// ## Invariant
///
/// 1-D contiguous float64 datasets written by consus are readable by h5py.
#[test]
fn consus_contiguous_1d_f64_readable_by_h5py() {
    let values: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "array_1d_f64",
            &f64_dt(),
            &Shape::fixed(&[4]),
            &raw,
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "contiguous_1d_f64");
}

/// Write a 2-D int32 contiguous dataset, shape (3,4), values 0..11.
///
/// ## Invariant
///
/// 2-D contiguous int32 datasets written by consus are readable by h5py.
#[test]
fn consus_contiguous_2d_i32_readable_by_h5py() {
    let values: Vec<i32> = (0..12).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "array_2d_i32",
            &i32_dt(),
            &Shape::fixed(&[3, 4]),
            &raw,
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "contiguous_2d_i32");
}

/// Write a 3-D float64 contiguous dataset, shape (2,3,4), values 0.0..23.0.
///
/// ## Invariant
///
/// 3-D contiguous float64 datasets written by consus are readable by h5py.
#[test]
fn consus_contiguous_3d_f64_readable_by_h5py() {
    let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "array_3d_f64",
            &f64_dt(),
            &Shape::fixed(&[2, 3, 4]),
            &raw,
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "contiguous_3d_f64");
}

// ---------------------------------------------------------------------------
// Tests: chunked datasets (B-tree v1)
// ---------------------------------------------------------------------------

/// Write a chunked 1-D int32 dataset (layout v3, B-tree v1), shape (12,),
/// chunk (4,), values 0..11.
///
/// ## Invariant
///
/// B-tree v1 chunk index written by consus is readable by h5py/libhdf5.
#[test]
fn consus_chunked_1d_i32_v1_readable_by_h5py() {
    let values: Vec<i32> = (0..12).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![4]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("chunked_1d_i32", &i32_dt(), &Shape::fixed(&[12]), &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "chunked_1d_i32_v1");
}

/// Write a chunked 2-D float64 dataset (layout v3, B-tree v1), shape (4,6),
/// chunks (2,3), values 0.0..23.0.
///
/// ## Invariant
///
/// Multi-dimensional B-tree v1 chunk index written by consus is readable by
/// h5py/libhdf5.
#[test]
fn consus_chunked_2d_f64_v1_readable_by_h5py() {
    let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![2, 3]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("chunked_2d_f64", &f64_dt(), &Shape::fixed(&[4, 6]), &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "chunked_2d_f64_v1");
}

// ---------------------------------------------------------------------------
// Tests: chunked datasets (B-tree v4 / layout v4)
// ---------------------------------------------------------------------------

/// Write a chunked 1-D int32 dataset (layout v4, B-tree v2), shape (8,),
/// chunks (4,), values 0..7.
///
/// ## Invariant
///
/// B-tree v2 chunk index (layout version 4) written by consus is readable by
/// h5py/libhdf5.
#[test]
fn consus_chunked_1d_i32_v4_readable_by_h5py() {
    let values: Vec<i32> = (0..8).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![4]),
        layout_version: Some(4),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("chunked_1d_i32_v4", &i32_dt(), &Shape::fixed(&[8]), &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "chunked_1d_i32_v4");
}

// ---------------------------------------------------------------------------
// Tests: compressed chunked datasets
// ---------------------------------------------------------------------------

/// Write a deflate-compressed chunked 1-D int32 dataset (layout v3, B-tree v1),
/// shape (8,), chunks (4,), deflate level 6, values 0..7.
///
/// ## Invariant
///
/// Deflate-compressed B-tree v1 data written by consus is decompressed and
/// read correctly by h5py/libhdf5.
#[test]
fn consus_deflate_chunked_1d_i32_readable_by_h5py() {
    let values: Vec<i32> = (0..8).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![4]),
        compression: Compression::Deflate { level: 6 },
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("deflate_1d_i32", &i32_dt(), &Shape::fixed(&[8]), &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "deflate_chunked_1d_i32");
}

// ---------------------------------------------------------------------------
// Tests: dataset attributes
// ---------------------------------------------------------------------------

/// Write a scalar int32 dataset with a scalar int32 attribute "answer" = 99.
/// The dataset value is 7.
///
/// ## Invariant
///
/// Dataset attributes written by consus are readable by h5py.
#[test]
fn consus_dataset_with_scalar_attr_readable_by_h5py() {
    let dataset_raw = 7i32.to_le_bytes();
    let answer_raw = 99i32.to_le_bytes();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "dataset_with_attr",
            &i32_dt(),
            &Shape::scalar(),
            &dataset_raw,
            &DatasetCreationProps::default(),
            &[("answer", &i32_dt(), &Shape::scalar(), &answer_raw)],
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "dataset_with_attr");
}

// ---------------------------------------------------------------------------
// Tests: groups
// ---------------------------------------------------------------------------

/// Write a group "grp" containing a scalar int32 dataset "nested_value" = 77.
///
/// ## Invariant
///
/// Named groups with nested datasets written by consus are navigable and
/// readable by h5py.
#[test]
fn consus_group_nested_dataset_readable_by_h5py() {
    let nested_raw = 77i32.to_le_bytes();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let mut grp = builder.begin_group("grp");
    grp.add_dataset_with_attributes(
        "nested_value",
        &i32_dt(),
        &Shape::scalar(),
        &nested_raw,
        &DatasetCreationProps::default(),
        &[],
    )
    .unwrap();
    grp.finish_with_attributes(&[]).unwrap();

    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "group_nested_dataset");
}

// ---------------------------------------------------------------------------
// Tests: additional scalar types
// ---------------------------------------------------------------------------

/// Write a uint8 1-D contiguous dataset, shape (8,), values 0..7.
///
/// ## Invariant
///
/// uint8 contiguous datasets written by consus are readable by h5py.
#[test]
fn consus_contiguous_1d_u8_readable_by_h5py() {
    let values: Vec<u8> = (0..8).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "array_1d_u8",
            &u8_dt(),
            &Shape::fixed(&[8]),
            &values,
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "contiguous_1d_u8");
}

/// Write a float32 1-D contiguous dataset, shape (4,), values [1.0, 2.0, 3.0, 4.0].
///
/// ## Invariant
///
/// float32 contiguous datasets written by consus are readable by h5py.
#[test]
fn consus_contiguous_1d_f32_readable_by_h5py() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "array_1d_f32",
            &f32_dt(),
            &Shape::fixed(&[4]),
            &raw,
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "contiguous_1d_f32");
}

/// Write an int16 1-D contiguous dataset, shape (4,), values [-100, 0, 100, 200].
///
/// ## Invariant
///
/// int16 contiguous datasets written by consus are readable by h5py.
#[test]
fn consus_contiguous_1d_i16_readable_by_h5py() {
    let values: Vec<i16> = vec![-100, 0, 100, 200];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "array_1d_i16",
            &i16_dt(),
            &Shape::fixed(&[4]),
            &raw,
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "contiguous_1d_i16");
}

// ---------------------------------------------------------------------------
// Tests: layout v4, additional shapes
// ---------------------------------------------------------------------------

/// Write a float64 chunked 2-D dataset (layout v4), shape (4,6), chunks (2,3).
///
/// ## Invariant
///
/// v4 chunked layout for multi-dimensional datasets is readable by h5py.
#[test]
fn consus_chunked_2d_f64_v4_readable_by_h5py() {
    let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![2, 3]),
        layout_version: Some(4),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("chunked_2d_f64_v4", &f64_dt(), &Shape::fixed(&[4, 6]), &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "chunked_2d_f64_v4");
}

// ---------------------------------------------------------------------------
// Tests: multiple datasets in root group
// ---------------------------------------------------------------------------

/// Write two datasets to the root group: int32 scalar "ds_a"=1 and float64
/// scalar "ds_b"=2.5.
///
/// ## Invariant
///
/// Multiple datasets in the root group written by consus are all accessible
/// by h5py.
#[test]
fn consus_multi_dataset_root_readable_by_h5py() {
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "ds_a",
            &i32_dt(),
            &Shape::scalar(),
            &1i32.to_le_bytes(),
            &DatasetCreationProps::default(),
        )
        .unwrap();
    builder
        .add_dataset(
            "ds_b",
            &f64_dt(),
            &Shape::scalar(),
            &2.5f64.to_le_bytes(),
            &DatasetCreationProps::default(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "multi_dataset_root");
}

// ---------------------------------------------------------------------------
// Tests: deflate-compressed multi-dimensional dataset
// ---------------------------------------------------------------------------

/// Write a deflate-compressed 2-D int32 chunked dataset (layout v1),
/// shape (3,8), chunks (1,8), values 0..23.
///
/// ## Invariant
///
/// Deflate-compressed multi-dimensional chunked datasets written by consus
/// are decompressed and read correctly by h5py.
#[test]
fn consus_deflate_chunked_2d_i32_readable_by_h5py() {
    let values: Vec<i32> = (0..24).collect();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![1, 8]),
        compression: Compression::Deflate { level: 6 },
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("deflate_2d_i32", &i32_dt(), &Shape::fixed(&[3, 8]), &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();
    verify_with_h5py(bytes, "deflate_chunked_2d_i32");
}
