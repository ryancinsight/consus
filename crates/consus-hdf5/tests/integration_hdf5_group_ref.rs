//! Integration tests for `data/hdf5_group_ref_sample.h5`.
//!
//! ## Fixture manifest
//!
//! | Path | Type | Shape | Value |
//! |------|------|-------|-------|
//! | `/flat_ds`                   | Integer(i32) | scalar | `7`       |
//! | `/grp_a`                     | group        | —      | attr `tag = "grp_a"` |
//! | `/grp_a/ds_one`              | Integer(i32) | scalar | `11`      |
//! | `/grp_a/ds_two`              | Float(f64)   | (2,)   | `[1.5, 2.5]` |
//! | `/grp_b`                     | group        | —      | no attributes |
//! | `/grp_b/nested`              | group        | —      | —         |
//! | `/grp_b/nested/deep_ds`      | Integer(i32) | scalar | `42`      |
//! | `/grp_b/nested/deep_ds_float`| Float(f64)   | scalar | `3.14`    |
//!
//! ## Group navigation strategy
//!
//! Tests use `Hdf5File::open_path` for path-based navigation and
//! `Hdf5File::list_root_group` / `list_group_at` for child enumeration.
//! Attribute VLen strings are decoded via `consus_hdf5::heap::resolve_vl_references`.

use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::dataset::layout::DataLayout;
use consus_hdf5::file::{Hdf5File, reader};
use consus_hdf5::heap::resolve_vl_references;
use consus_hdf5::object_header::message_types;
use consus_io::MemCursor;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("hdf5_group_ref_sample.h5")
}

fn open_fixture() -> Option<Hdf5File<MemCursor>> {
    let path = fixture_path();
    if !path.exists() {
        eprintln!("Skipping: reference file not found at {:?}", path);
        return None;
    }
    let bytes = std::fs::read(&path).expect("read fixture file bytes");
    let cursor = MemCursor::from_bytes(bytes);
    Some(Hdf5File::open(cursor).expect("open HDF5 file"))
}

/// Read `total_bytes` of raw data from a dataset at `addr`.
///
/// Handles contiguous and compact storage layouts.
/// Contiguous: reads via `read_contiguous_dataset_bytes`.
/// Compact: extracts `compact_data` from the DATA_LAYOUT object header message.
fn read_dataset_raw_bytes(file: &Hdf5File<MemCursor>, addr: u64, total_bytes: usize) -> Vec<u8> {
    let dataset = file.dataset_at(addr).expect("read dataset metadata");
    match dataset.layout {
        StorageLayout::Contiguous => {
            let data_addr = dataset
                .data_address
                .expect("contiguous layout must have data_address");
            let mut buf = vec![0u8; total_bytes];
            file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)
                .expect("read_contiguous_dataset_bytes");
            buf
        }
        StorageLayout::Compact => {
            let header = reader::read_object_header(file.source(), addr, file.context())
                .expect("read object header for compact dataset");
            let msg = reader::find_message(&header, message_types::DATA_LAYOUT)
                .expect("DATA_LAYOUT message must exist in compact dataset object header");
            let layout = DataLayout::parse(&msg.data, file.context())
                .expect("parse DataLayout from message bytes");
            let data = layout
                .compact_data
                .expect("compact_data must be populated for compact layout");
            assert!(
                data.len() >= total_bytes,
                "compact_data length {} < expected total_bytes {}",
                data.len(),
                total_bytes
            );
            data[..total_bytes].to_vec()
        }
        other => panic!(
            "unsupported StorageLayout {:?} for read_dataset_raw_bytes at addr {:#x}",
            other, addr
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test: read `/flat_ds` as a scalar int32 dataset.
///
/// ## Invariants
///
/// - Shape is scalar (rank 0).
/// - Little-endian i32 value equals `7`.
#[test]
fn group_ref_flat_dataset() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file.open_path("/flat_ds").expect("navigate to /flat_ds");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert!(
        dataset.shape.is_scalar(),
        "/flat_ds must be a scalar dataset"
    );

    let buf = read_dataset_raw_bytes(&file, addr, 4);
    let value = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(value, 7, "/flat_ds scalar int32 value must equal 7");
}

/// Test: list root group children.
///
/// ## Invariants
///
/// - Root group contains entries named `"flat_ds"`, `"grp_a"`, and `"grp_b"`.
#[test]
fn group_ref_list_root_children() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let children = file.list_root_group().expect("list root group");
    let names: Vec<&str> = children.iter().map(|(n, _, _)| n.as_str()).collect();

    assert!(
        names.contains(&"flat_ds"),
        "root group must contain \"flat_ds\"; found: {:?}",
        names
    );
    assert!(
        names.contains(&"grp_a"),
        "root group must contain \"grp_a\"; found: {:?}",
        names
    );
    assert!(
        names.contains(&"grp_b"),
        "root group must contain \"grp_b\"; found: {:?}",
        names
    );
}

/// Test: navigate a one-level nested path and read int32 value.
///
/// ## Invariants
///
/// - `/grp_a/ds_one` is a scalar dataset.
/// - Little-endian i32 value equals `11`.
#[test]
fn group_ref_nested_path() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/grp_a/ds_one")
        .expect("navigate to /grp_a/ds_one");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert!(
        dataset.shape.is_scalar(),
        "/grp_a/ds_one must be a scalar dataset"
    );

    let buf = read_dataset_raw_bytes(&file, addr, 4);
    let value = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(value, 11, "/grp_a/ds_one int32 value must equal 11");
}

/// Test: read a 1-D float64 array dataset.
///
/// ## Invariants
///
/// - `/grp_a/ds_two` has rank 1 with 2 elements.
/// - Raw byte payload is exactly 16 bytes (2 × 8).
/// - Values interpreted as little-endian f64 are `1.5` and `2.5` (exact).
#[test]
fn group_ref_nested_float_array() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/grp_a/ds_two")
        .expect("navigate to /grp_a/ds_two");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert_eq!(
        dataset.shape.rank(),
        1,
        "/grp_a/ds_two must be 1-dimensional"
    );
    assert_eq!(
        dataset.shape.num_elements(),
        2,
        "/grp_a/ds_two must contain exactly 2 elements"
    );

    // 2 × 8 bytes = 16 bytes total for two f64 values.
    let buf = read_dataset_raw_bytes(&file, addr, 16);

    let v0 = f64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    let v1 = f64::from_le_bytes([
        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    ]);

    // 1.5 and 2.5 are exactly representable in IEEE 754 double precision.
    assert_eq!(v0, 1.5_f64, "/grp_a/ds_two element[0] must equal 1.5");
    assert_eq!(v1, 2.5_f64, "/grp_a/ds_two element[1] must equal 2.5");
}

/// Test: navigate a three-level nested path and read int32 value.
///
/// ## Invariants
///
/// - `/grp_b/nested/deep_ds` is a scalar dataset.
/// - Little-endian i32 value equals `42`.
#[test]
fn group_ref_deep_path() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/grp_b/nested/deep_ds")
        .expect("navigate to /grp_b/nested/deep_ds");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert!(
        dataset.shape.is_scalar(),
        "/grp_b/nested/deep_ds must be a scalar dataset"
    );

    let buf = read_dataset_raw_bytes(&file, addr, 4);
    let value = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(value, 42, "/grp_b/nested/deep_ds int32 value must equal 42");
}

/// Test: read a deeply nested scalar float64 dataset.
///
/// ## Invariants
///
/// - `/grp_b/nested/deep_ds_float` is a scalar dataset.
/// - Value decoded as little-endian f64 is within `1e-10` of `3.14`.
#[test]
fn group_ref_deep_float() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/grp_b/nested/deep_ds_float")
        .expect("navigate to /grp_b/nested/deep_ds_float");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert!(
        dataset.shape.is_scalar(),
        "/grp_b/nested/deep_ds_float must be a scalar dataset"
    );

    let buf = read_dataset_raw_bytes(&file, addr, 8);
    let value = f64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);

    assert!(
        (value - 3.14_f64).abs() < 1e-10,
        "/grp_b/nested/deep_ds_float must be approximately 3.14, got {:.15}",
        value
    );
}

/// Test: read variable-length string attribute from `/grp_a`.
///
/// ## Invariants
///
/// - `/grp_a` exposes attribute `"tag"`.
/// - Resolved VLen string equals `"grp_a"`.
#[test]
fn group_ref_grp_a_tag_attribute() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let grp_addr = file.open_path("/grp_a").expect("navigate to /grp_a");
    let attrs = file
        .attributes_at(grp_addr)
        .expect("read attributes on /grp_a");

    let tag = attrs
        .iter()
        .find(|a| a.name == "tag")
        .expect("attribute \"tag\" must exist on /grp_a");

    // `tag.raw_data` holds the VLen descriptor pointing to the global heap.
    let resolved = resolve_vl_references(file.source(), &tag.raw_data, file.context())
        .expect("resolve VLen reference for tag attribute");

    assert_eq!(
        resolved.len(),
        1,
        "scalar attribute must resolve to exactly 1 element"
    );
    let s = std::str::from_utf8(&resolved[0])
        .expect("tag attribute bytes must be valid UTF-8")
        .trim_end_matches('\0');
    assert_eq!(s, "grp_a", "tag attribute value must equal \"grp_a\"");
}
