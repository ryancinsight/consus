//! Integration tests for `data/hdf5_string_ref_sample.h5`.
//!
//! ## Fixture manifest
//!
//! | Path | Type | Shape | Value |
//! |------|------|-------|-------|
//! | `/fixed_str_scalar` | FixedString(8) | scalar | `b"hello\0\0\0"` |
//! | `/vlen_str_scalar`  | VariableString | scalar | `"world"` |
//! | `/fixed_str_1d`     | FixedString(4) | (3,)   | `["foo\0","bar\0","baz\0"]` |
//! | `/vlen_str_1d`      | VariableString | (3,)   | `["abc","de","fghij"]` |
//! | `/grp_with_str_attr`| group          | —      | attr `label = "test_label"` |
//! | `/grp_with_str_attr/inner_ds` | Integer(i32) | scalar | `99` |
//!
//! ## VLen decode strategy
//!
//! Variable-length strings are stored as global-heap references
//! (`4 + offset_size + 4` bytes each). Resolution uses the public
//! `consus_hdf5::heap::resolve_vl_references` function, which traverses
//! the global heap collection and returns raw string bytes per element.

use consus_core::Datatype;
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
        .join("hdf5_string_ref_sample.h5")
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
/// Handles both contiguous and compact storage layouts.
/// For contiguous: reads via `read_contiguous_dataset_bytes`.
/// For compact: parses the DATA_LAYOUT header message and returns `compact_data`.
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

/// Compute the VLen descriptor size for this file's offset width.
///
/// VLen descriptor: `uint32 sequence_length || addr heap_collection || uint32 object_index`.
fn vlen_ref_size(file: &Hdf5File<MemCursor>) -> usize {
    4 + file.context().offset_bytes() + 4
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test: read fixed-length string scalar.
///
/// ## Invariants
///
/// - Datatype is `FixedString { length: 8 }`.
/// - Shape is scalar (rank 0).
/// - Raw bytes `[0..5]` equal `b"hello"`.
/// - Total byte count is 8 (declared length).
#[test]
fn string_ref_scalar_fixed_length() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/fixed_str_scalar")
        .expect("navigate to /fixed_str_scalar");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    let str_len = match &dataset.datatype {
        Datatype::FixedString { length, .. } => {
            assert_eq!(*length, 8, "FixedString byte length must be 8");
            *length
        }
        other => panic!("expected FixedString datatype, got {:?}", other),
    };

    assert!(dataset.shape.is_scalar(), "shape must be scalar (rank 0)");

    let buf = read_dataset_raw_bytes(&file, addr, str_len);
    assert_eq!(
        &buf[0..5],
        b"hello",
        "first 5 bytes of /fixed_str_scalar must equal b\"hello\""
    );
    assert_eq!(buf.len(), 8, "total byte count must be 8");
}

/// Test: read variable-length string scalar.
///
/// ## Invariants
///
/// - Datatype is `VariableString`.
/// - Shape is scalar (rank 0).
/// - Resolved VLen string equals `"world"`.
#[test]
fn string_ref_scalar_vlen() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/vlen_str_scalar")
        .expect("navigate to /vlen_str_scalar");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert!(
        matches!(dataset.datatype, Datatype::VariableString { .. }),
        "datatype must be VariableString, got {:?}",
        dataset.datatype
    );
    assert!(dataset.shape.is_scalar(), "shape must be scalar (rank 0)");

    let ref_size = vlen_ref_size(&file);
    let raw = read_dataset_raw_bytes(&file, addr, ref_size);
    let resolved = resolve_vl_references(file.source(), &raw, file.context())
        .expect("resolve VLen reference for /vlen_str_scalar");

    assert_eq!(
        resolved.len(),
        1,
        "scalar VLen dataset must resolve to exactly 1 element"
    );
    let s = std::str::from_utf8(&resolved[0])
        .expect("VLen string bytes must be valid UTF-8")
        .trim_end_matches('\0');
    assert_eq!(s, "world", "VLen scalar string value must equal \"world\"");
}

/// Test: read fixed-length string 1-D array.
///
/// ## Invariants
///
/// - Datatype is `FixedString { length: 4 }`.
/// - Shape rank is 1 with 3 elements.
/// - Total raw byte count is 12 (3 × 4).
/// - First element raw bytes `[0..3]` equal `b"foo"`.
#[test]
fn string_ref_1d_fixed_length() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/fixed_str_1d")
        .expect("navigate to /fixed_str_1d");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    let str_len = match &dataset.datatype {
        Datatype::FixedString { length, .. } => {
            assert_eq!(*length, 4, "FixedString byte length must be 4");
            *length
        }
        other => panic!("expected FixedString datatype, got {:?}", other),
    };

    assert_eq!(dataset.shape.rank(), 1, "shape must be 1-dimensional");
    assert_eq!(
        dataset.shape.num_elements(),
        3,
        "shape must contain exactly 3 elements"
    );

    let total_bytes = dataset.shape.num_elements() * str_len;
    let buf = read_dataset_raw_bytes(&file, addr, total_bytes);

    assert_eq!(
        &buf[0..3],
        b"foo",
        "element[0] first 3 bytes must equal b\"foo\""
    );
    assert_eq!(buf.len(), 12, "total raw byte count must be 12 (3 × 4)");
}

/// Test: read variable-length string 1-D array.
///
/// ## Invariants
///
/// - Datatype is `VariableString`.
/// - Shape rank is 1 with 3 elements.
/// - Resolved element byte lengths are `[3, 2, 5]`.
/// - Resolved element values are `["abc", "de", "fghij"]`.
#[test]
fn string_ref_1d_vlen() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/vlen_str_1d")
        .expect("navigate to /vlen_str_1d");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert!(
        matches!(dataset.datatype, Datatype::VariableString { .. }),
        "datatype must be VariableString, got {:?}",
        dataset.datatype
    );
    assert_eq!(dataset.shape.rank(), 1, "shape must be 1-dimensional");
    assert_eq!(
        dataset.shape.num_elements(),
        3,
        "shape must contain exactly 3 elements"
    );

    let ref_size = vlen_ref_size(&file);
    let total_bytes = dataset.shape.num_elements() * ref_size;
    let raw = read_dataset_raw_bytes(&file, addr, total_bytes);

    let resolved = resolve_vl_references(file.source(), &raw, file.context())
        .expect("resolve VLen references for /vlen_str_1d");

    assert_eq!(resolved.len(), 3, "must resolve exactly 3 VLen elements");

    // Byte-length assertions for ["abc", "de", "fghij"].
    assert_eq!(
        resolved[0].len(),
        3,
        "element[0] length must be 3 (\"abc\")"
    );
    assert_eq!(resolved[1].len(), 2, "element[1] length must be 2 (\"de\")");
    assert_eq!(
        resolved[2].len(),
        5,
        "element[2] length must be 5 (\"fghij\")"
    );

    // Value assertions.
    assert_eq!(&resolved[0], b"abc", "element[0] must equal b\"abc\"");
    assert_eq!(&resolved[1], b"de", "element[1] must equal b\"de\"");
    assert_eq!(&resolved[2], b"fghij", "element[2] must equal b\"fghij\"");
}

/// Test: read variable-length string attribute from a group.
///
/// ## Invariants
///
/// - `/grp_with_str_attr` exposes attribute `"label"`.
/// - Resolved VLen string equals `"test_label"`.
#[test]
fn string_ref_group_str_attribute() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let grp_addr = file
        .open_path("/grp_with_str_attr")
        .expect("navigate to /grp_with_str_attr");
    let attrs = file
        .attributes_at(grp_addr)
        .expect("read attributes on /grp_with_str_attr");

    let label = attrs
        .iter()
        .find(|a| a.name == "label")
        .expect("attribute \"label\" must exist on /grp_with_str_attr");

    // `label.raw_data` holds the VLen descriptor pointing to the global heap.
    let resolved = resolve_vl_references(file.source(), &label.raw_data, file.context())
        .expect("resolve VLen reference for label attribute");

    assert_eq!(
        resolved.len(),
        1,
        "scalar attribute must resolve to exactly 1 element"
    );
    let s = std::str::from_utf8(&resolved[0])
        .expect("label attribute bytes must be valid UTF-8")
        .trim_end_matches('\0');
    assert_eq!(
        s, "test_label",
        "label attribute value must equal \"test_label\""
    );
}

/// Test: read int32 scalar dataset nested inside a group.
///
/// ## Invariants
///
/// - `/grp_with_str_attr/inner_ds` is a scalar dataset.
/// - Raw 4 bytes decoded as little-endian i32 equal `99`.
#[test]
fn string_ref_group_inner_dataset() {
    let file = match open_fixture() {
        Some(f) => f,
        None => return,
    };

    let addr = file
        .open_path("/grp_with_str_attr/inner_ds")
        .expect("navigate to /grp_with_str_attr/inner_ds");
    let dataset = file.dataset_at(addr).expect("read dataset metadata");

    assert!(
        dataset.shape.is_scalar(),
        "/grp_with_str_attr/inner_ds must be a scalar dataset"
    );

    let buf = read_dataset_raw_bytes(&file, addr, 4);
    let value = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(
        value, 99,
        "/grp_with_str_attr/inner_ds int32 value must equal 99"
    );
}
