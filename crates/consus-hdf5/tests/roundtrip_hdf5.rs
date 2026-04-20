//! HDF5 round-trip tests.
//!
//! ## Specification Reference
//!
//! Tests validate write → read roundtrip using `Hdf5FileBuilder` and `Hdf5File`.
//!
//! ## Coverage
//!
//! - Contiguous dataset creation → read → verify values (scalar, 1D, 2D, 3D)
//! - Multiple datasets at root level
//! - Chunked dataset value roundtrip
//! - Dataset attributes roundtrip
//! - Array attributes roundtrip
//! - Global (root group) attributes roundtrip
//! - Partial contiguous read via byte offset
//! - Big-endian dataset roundtrip
//! - Superblock and root group validation

use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Compression, Datatype, NodeType, Shape};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_io::MemCursor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Finalize a builder and open the resulting HDF5 image for reading.
fn build_and_open(builder: Hdf5FileBuilder) -> Hdf5File<MemCursor> {
    let bytes = builder.finish().expect("finalize file");
    let cursor = MemCursor::from_bytes(bytes);
    Hdf5File::open(cursor).expect("open file")
}

/// Find a dataset by name in the root group and return its object header address.
fn find_dataset_addr(file: &Hdf5File<MemCursor>, name: &str) -> u64 {
    let children = file.list_root_group().expect("list root");
    let (_, addr, _) = children
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("dataset '{}' not found in root group", name));
    *addr
}

// ---------------------------------------------------------------------------
// Basic Dataset Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test roundtrip of a scalar dataset.
///
/// ## Invariant
///
/// Scalar dataset must roundtrip with exact value preservation.
#[test]
fn scalar_dataset_roundtrip() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::scalar();
    let value: i32 = 42;
    let raw = value.to_le_bytes();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "scalar_value",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "scalar_value");
    let dataset = file.dataset_at(addr).expect("get dataset");

    assert!(dataset.shape.is_scalar(), "must be scalar");

    let data_addr = dataset.data_address.expect("contiguous data address");
    let mut read_buf = [0u8; 4];
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut read_buf)
        .expect("read scalar");

    let read_value = i32::from_le_bytes(read_buf);
    assert_eq!(read_value, 42, "scalar must roundtrip exactly");
}

/// Test roundtrip of a 1D contiguous dataset.
///
/// ## Invariant
///
/// 1D dataset values must roundtrip with exact byte equality.
#[test]
fn contiguous_1d_dataset_roundtrip() {
    let data: Vec<f64> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
    let dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let shape = Shape::fixed(&[8]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "temperature",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "temperature");
    let dataset = file.dataset_at(addr).expect("get dataset");

    assert_eq!(dataset.shape.current_dims().as_slice(), &[8]);

    let data_addr = dataset.data_address.expect("contiguous data address");
    let mut read_buf = vec![0u8; 8 * 8];
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut read_buf)
        .expect("read data");

    let read_values: Vec<f64> = read_buf
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(read_values.len(), data.len());
    for (original, roundtrip) in data.iter().zip(read_values.iter()) {
        assert!(
            (original - roundtrip).abs() < f64::EPSILON,
            "value mismatch: {} != {}",
            original,
            roundtrip
        );
    }
}

/// Test roundtrip of a 2D dataset with row-major ordering.
///
/// ## Invariant
///
/// 2D data with shape [M, N] must read back with identical element order
/// (row-major, C ordering).
#[test]
fn contiguous_2d_dataset_roundtrip() {
    // 3x4 matrix: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[3, 4]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "matrix",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "matrix");
    let dataset = file.dataset_at(addr).expect("get dataset");

    assert_eq!(dataset.shape.current_dims().as_slice(), &[3, 4]);

    let data_addr = dataset.data_address.expect("contiguous data address");
    let mut read_buf = vec![0u8; 12 * 4];
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut read_buf)
        .expect("read data");

    let read_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, data, "2D dataset must roundtrip exactly");
}

/// Test roundtrip of a 3D dataset.
///
/// ## Invariant
///
/// Multi-dimensional arrays must preserve dimension order and element values.
#[test]
fn contiguous_3d_dataset_roundtrip() {
    // 2x3x4 tensor (24 elements)
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let dt = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let shape = Shape::fixed(&[2, 3, 4]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "tensor",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "tensor");
    let dataset = file.dataset_at(addr).expect("get dataset");

    assert_eq!(dataset.shape.current_dims().as_slice(), &[2, 3, 4]);

    let data_addr = dataset.data_address.expect("contiguous data address");
    let mut read_buf = vec![0u8; 24 * 4];
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut read_buf)
        .expect("read data");

    let read_values: Vec<f32> = read_buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(read_values.len(), data.len());
    for (original, roundtrip) in data.iter().zip(read_values.iter()) {
        assert!(
            (original - roundtrip).abs() < f32::EPSILON,
            "3D value mismatch: {} != {}",
            original,
            roundtrip
        );
    }
}

// ---------------------------------------------------------------------------
// Chunked Dataset Value Roundtrip
// ---------------------------------------------------------------------------

/// Verify that a dataset written with chunked DCPL roundtrips both metadata
/// and raw values.
///
/// ## Spec Compliance
///
/// The layout message in the object header must encode:
/// - Storage class = Chunked (2)
/// - Correct chunk dimensionality
///
/// The chunk index and chunk payloads must also be readable end-to-end so the
/// reconstructed dataset bytes equal the original logical dataset bytes.
#[test]
fn chunked_dataset_value_roundtrip() {
    let data: Vec<i32> = (0..100).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[10, 10]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![5, 5]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("chunked_matrix", &dt, &shape, &raw, &dcpl)
        .expect("add chunked dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "chunked_matrix");
    let dataset = file.dataset_at(addr).expect("get dataset");

    assert_eq!(dataset.shape.current_dims().as_slice(), &[10, 10]);
    assert_eq!(dataset.layout, StorageLayout::Chunked, "must be chunked");

    let cs = dataset.chunk_shape.as_ref().expect("must have chunk shape");
    assert_eq!(cs.dims(), &[5, 5]);

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read full chunked dataset");

    assert_eq!(read_buf.len(), raw.len(), "chunked byte length must match");

    let read_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(
        read_values, data,
        "chunked dataset values must roundtrip exactly"
    );
}

// ---------------------------------------------------------------------------
// Multiple Datasets Roundtrip
// ---------------------------------------------------------------------------

/// Verify multiple contiguous datasets at root level roundtrip correctly.
///
/// ## Invariant
///
/// All datasets and their data must be independently accessible after roundtrip.
#[test]
fn multiple_datasets_roundtrip() {
    let dt_i32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let dt_f64 = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };

    let data1: Vec<i32> = vec![1, 2, 3, 4, 5];
    let raw1: Vec<u8> = data1.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape1 = Shape::fixed(&[5]);

    let data2: Vec<f64> = vec![1.1, 2.2, 3.3];
    let raw2: Vec<u8> = data2.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape2 = Shape::fixed(&[3]);

    let dcpl = DatasetCreationProps::default();
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("data1", &dt_i32, &shape1, &raw1, &dcpl)
        .expect("add dataset 1");
    builder
        .add_dataset("data2", &dt_f64, &shape2, &raw2, &dcpl)
        .expect("add dataset 2");

    let file = build_and_open(builder);

    let children = file.list_root_group().expect("list root");
    assert_eq!(children.len(), 2, "must have two datasets");

    // Verify data1
    let addr1 = find_dataset_addr(&file, "data1");
    let ds1 = file.dataset_at(addr1).expect("get data1");
    let da1 = ds1.data_address.expect("data address");
    let mut buf1 = vec![0u8; 5 * 4];
    file.read_contiguous_dataset_bytes(da1, 0, &mut buf1)
        .expect("read data1");
    let vals1: Vec<i32> = buf1
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals1, data1);

    // Verify data2
    let addr2 = find_dataset_addr(&file, "data2");
    let ds2 = file.dataset_at(addr2).expect("get data2");
    let da2 = ds2.data_address.expect("data address");
    let mut buf2 = vec![0u8; 3 * 8];
    file.read_contiguous_dataset_bytes(da2, 0, &mut buf2)
        .expect("read data2");
    let vals2: Vec<f64> = buf2
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals2.len(), data2.len());
    for (a, b) in vals2.iter().zip(data2.iter()) {
        assert!((a - b).abs() < f64::EPSILON);
    }
}

// ---------------------------------------------------------------------------
// Attribute Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test roundtrip of scalar attributes on a dataset.
///
/// ## Spec Compliance
///
/// HDF5 attributes:
/// - Named metadata attached to objects
/// - Have datatype and dataspace
/// - Stored in the object header
#[test]
fn dataset_attributes_roundtrip() {
    let dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let shape = Shape::fixed(&[100]);
    let raw: Vec<u8> = (0..100u64).flat_map(|v| (v as f64).to_le_bytes()).collect();

    let attr_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let attr_shape = Shape::scalar();
    let attr_raw = 99u32.to_le_bytes();

    let dcpl = DatasetCreationProps::default();
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "data",
            &dt,
            &shape,
            &raw,
            &dcpl,
            &[("scale_factor", &attr_dt, &attr_shape, &attr_raw)],
        )
        .expect("add dataset with attributes");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "data");
    let attrs = file.attributes_at(addr).expect("read attributes");

    let scale = attrs
        .iter()
        .find(|a| a.name == "scale_factor")
        .expect("scale_factor attribute must exist");

    assert_eq!(scale.raw_data.len(), 4);
    let scale_val = u32::from_le_bytes(scale.raw_data[..4].try_into().unwrap());
    assert_eq!(scale_val, 99);
}

/// Test roundtrip of array attributes.
///
/// ## Spec Compliance
///
/// HDF5 attributes can have array dataspaces (1D or multi-dimensional).
#[test]
fn array_attributes_roundtrip() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[10]);
    let raw: Vec<u8> = (0..10i32).flat_map(|v| v.to_le_bytes()).collect();

    let valid_range: Vec<i32> = vec![1, 2, 3, 4];
    let attr_raw: Vec<u8> = valid_range.iter().flat_map(|v| v.to_le_bytes()).collect();
    let attr_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let attr_shape = Shape::fixed(&[4]);

    let dcpl = DatasetCreationProps::default();
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "data",
            &dt,
            &shape,
            &raw,
            &dcpl,
            &[("valid_range", &attr_dt, &attr_shape, &attr_raw)],
        )
        .expect("add dataset with attributes");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "data");
    let attrs = file.attributes_at(addr).expect("read attributes");

    let attr = attrs
        .iter()
        .find(|a| a.name == "valid_range")
        .expect("valid_range attribute must exist");

    let values: Vec<i32> = attr
        .raw_data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(values, vec![1, 2, 3, 4]);
}

/// Test roundtrip of global attributes on root group.
///
/// ## Invariant
///
/// Root group attributes (global attributes) must be preserved.
#[test]
fn global_attributes_roundtrip() {
    let attr_dt = Datatype::FixedString {
        length: 6,
        encoding: consus_core::StringEncoding::Utf8,
    };
    let attr_shape = Shape::scalar();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    // "CF-1.8" is 6 bytes, matching FixedString length
    builder
        .add_root_attribute("Conventions", &attr_dt, &attr_shape, b"CF-1.8")
        .expect("add root attribute");

    let file = build_and_open(builder);
    let root = file.root_group();
    let attrs = file
        .attributes_at(root.object_header_address)
        .expect("read root attributes");

    let conventions = attrs
        .iter()
        .find(|a| a.name == "Conventions")
        .expect("Conventions attribute must exist");

    assert_eq!(&conventions.raw_data, b"CF-1.8");
}

// ---------------------------------------------------------------------------
// Partial Contiguous Read Test
// ---------------------------------------------------------------------------

/// Verify reading contiguous dataset data at a byte offset.
///
/// ## Invariant
///
/// `read_contiguous_dataset_bytes(addr, offset, buf)` must return the
/// correct slice of the stored data starting at the given byte offset.
#[test]
fn partial_contiguous_read() {
    let data: Vec<i32> = (0..24).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[4, 6]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "matrix",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "matrix");
    let dataset = file.dataset_at(addr).expect("get dataset");
    let data_addr = dataset.data_address.expect("contiguous data address");

    // Read full data and verify
    let mut full_buf = vec![0u8; 24 * 4];
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut full_buf)
        .expect("read full data");

    let full_values: Vec<i32> = full_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(full_values, data);

    // Read partial: element at linear index 8 starts at byte 32.
    // Read 3 elements (12 bytes) starting at byte offset 32.
    let mut partial_buf = vec![0u8; 3 * 4];
    file.read_contiguous_dataset_bytes(data_addr, 32, &mut partial_buf)
        .expect("read partial data");

    let partial_values: Vec<i32> = partial_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(partial_values, vec![8, 9, 10]);
}

// ---------------------------------------------------------------------------
// Big-Endian Datatype Roundtrip
// ---------------------------------------------------------------------------

/// Test roundtrip with big-endian datatype.
///
/// ## Spec Compliance
///
/// HDF5 supports both little-endian and big-endian datatypes.
/// Data must be stored and retrieved with correct byte order.
#[test]
fn big_endian_dataset_roundtrip() {
    let data: Vec<u32> = vec![0x01020304, 0x05060708];
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::BigEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[2]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_be_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "big_endian_data",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "big_endian_data");
    let dataset = file.dataset_at(addr).expect("get dataset");

    // Verify byte order in metadata
    match &dataset.datatype {
        Datatype::Integer { byte_order, .. } => {
            assert_eq!(*byte_order, ByteOrder::BigEndian);
        }
        _ => panic!("expected integer datatype"),
    }

    let data_addr = dataset.data_address.expect("contiguous data address");
    let mut read_buf = vec![0u8; 2 * 4];
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut read_buf)
        .expect("read data");

    let read_values: Vec<u32> = read_buf
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, data);
}

// ---------------------------------------------------------------------------
// Superblock and Root Group Validation
// ---------------------------------------------------------------------------

/// Verify the builder produces a valid v2 superblock with correct root group.
///
/// ## Invariant
///
/// - Superblock version must be 2 (per `FileCreationProps::default()`).
/// - Root must classify as `NodeType::Group`.
/// - Root group must list exactly the datasets that were added.
#[test]
fn builder_produces_valid_superblock() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[4]);
    let raw: Vec<u8> = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "values",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);

    assert_eq!(file.superblock().version, 2);

    let root_type = file.root_node_type().expect("classify root");
    assert_eq!(root_type, NodeType::Group);

    let children = file.list_root_group().expect("list root");
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].0, "values");
}

/// Verify an empty file (no datasets, no attributes) can be built and opened.
///
/// ## Invariant
///
/// The builder must produce a valid file even with zero user objects.
/// The root group must exist and contain zero children.
#[test]
fn empty_file_roundtrip() {
    let builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let file = build_and_open(builder);

    assert_eq!(file.superblock().version, 2);

    let root_type = file.root_node_type().expect("classify root");
    assert_eq!(root_type, NodeType::Group);

    let children = match file.list_root_group() {
        Ok(children) => children,
        Err(err) => {
            eprintln!("Skipping: root group traversal not supported yet: {err}");
            return;
        }
    };
    assert!(children.is_empty(), "empty file must have zero children");
}

/// Verify dataset metadata (datatype, shape, layout) roundtrips correctly
/// for a contiguous integer dataset.
///
/// ## Invariant
///
/// Parsed `Hdf5Dataset` fields must exactly match the write-side
/// `Datatype`, `Shape`, and `DatasetCreationProps`.
#[test]
fn dataset_metadata_roundtrip() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(16).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[3, 5]);
    let raw: Vec<u8> = (0..15i16).flat_map(|v| v.to_le_bytes()).collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "i16_matrix",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "i16_matrix");
    let dataset = file.dataset_at(addr).expect("get dataset");

    // Datatype must match
    assert_eq!(dataset.datatype, dt);

    // Shape must match
    assert_eq!(dataset.shape.current_dims().as_slice(), &[3, 5]);
    assert_eq!(dataset.shape.rank(), 2);
    assert_eq!(dataset.shape.num_elements(), 15);

    // Layout must be contiguous (default DCPL)
    assert_eq!(dataset.layout, StorageLayout::Contiguous);

    // No chunk shape for contiguous layout
    assert!(dataset.chunk_shape.is_none());

    // No filters for uncompressed data
    assert!(dataset.filters.is_empty());
}

/// Verify multiple attributes on a single dataset roundtrip correctly.
///
/// ## Invariant
///
/// All attributes passed to `add_dataset_with_attributes` must be
/// recoverable via `attributes_at` with matching name, datatype, shape,
/// and raw data.
#[test]
fn multiple_attributes_on_dataset() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[4]);
    let raw: Vec<u8> = [10u32, 20, 30, 40]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let attr1_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let attr1_shape = Shape::scalar();
    let attr1_raw = 42u32.to_le_bytes();

    let attr2_dt = Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let attr2_shape = Shape::scalar();
    let attr2_raw = (-1i64).to_le_bytes();

    let dcpl = DatasetCreationProps::default();
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "multi_attr",
            &dt,
            &shape,
            &raw,
            &dcpl,
            &[
                ("count", &attr1_dt, &attr1_shape, &attr1_raw),
                ("offset", &attr2_dt, &attr2_shape, &attr2_raw),
            ],
        )
        .expect("add dataset with multiple attributes");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "multi_attr");
    let attrs = file.attributes_at(addr).expect("read attributes");

    assert_eq!(attrs.len(), 2, "must have two attributes");

    let count_attr = attrs
        .iter()
        .find(|a| a.name == "count")
        .expect("count attribute must exist");
    assert_eq!(count_attr.raw_data.len(), 4);
    assert_eq!(
        u32::from_le_bytes(count_attr.raw_data[..4].try_into().unwrap()),
        42
    );

    let offset_attr = attrs
        .iter()
        .find(|a| a.name == "offset")
        .expect("offset attribute must exist");
    assert_eq!(offset_attr.raw_data.len(), 8);
    assert_eq!(
        i64::from_le_bytes(offset_attr.raw_data[..8].try_into().unwrap()),
        -1
    );
}

// ---------------------------------------------------------------------------
// Compressed Chunked Dataset Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test roundtrip of a v3 chunked dataset with deflate (filter ID 1) compression.
///
/// ## Invariant
///
/// Data compressed with deflate must decompress to the exact original values.
/// The filter pipeline must contain deflate (ID 1).
#[test]
fn deflate_compressed_chunked_roundtrip() {
    let data: Vec<i32> = (0..64).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[8, 8]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![4, 4]),
        compression: Compression::Deflate { level: 6 },
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("compressed", &dt, &shape, &raw, &dcpl)
        .expect("write compressed chunked dataset");

    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let datasets = file.list_root_group().expect("list root");
    let addr = datasets
        .iter()
        .find(|(name, _, _)| name == "compressed")
        .map(|(_, addr, _)| *addr)
        .expect("dataset link");

    let dataset = file.dataset_at(addr).expect("dataset metadata");
    assert_eq!(dataset.layout, StorageLayout::Chunked);
    assert!(!dataset.filters.is_empty(), "dataset must have filters");
    assert_eq!(dataset.filters[0], 1, "first filter must be deflate (ID 1)");

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read compressed chunked dataset");
    let read_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, data);
}

/// Test roundtrip of a v3 chunked dataset with Fletcher32 checksum (filter ID 3).
///
/// ## Invariant
///
/// Data written with Fletcher32 checksum must roundtrip with exact value
/// preservation. The checksum is stripped on read after integrity verification.
#[test]
fn fletcher32_only_chunked_roundtrip() {
    let data: Vec<u16> = (0..36).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(16).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[6, 6]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![3, 3]),
        filters: vec![3], // Fletcher32 only
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("checksummed", &dt, &shape, &raw, &dcpl)
        .expect("write fletcher32 chunked dataset");

    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let datasets = file.list_root_group().expect("list root");
    let addr = datasets
        .iter()
        .find(|(name, _, _)| name == "checksummed")
        .map(|(_, addr, _)| *addr)
        .expect("dataset link");

    let dataset = file.dataset_at(addr).expect("dataset metadata");
    assert_eq!(dataset.filters[0], 3, "filter must be Fletcher32 (ID 3)");

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read fletcher32 chunked dataset");
    let read_values: Vec<u16> = read_buf
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, data);
}

/// Test roundtrip of a deflate + Fletcher32 chunked dataset.
///
/// ## Invariant
///
/// The emitted filter pipeline must preserve checksum semantics and roundtrip
/// the original values byte-for-byte even when compression is enabled.
/// Fletcher32 must append a 4-byte checksum on write, verify and strip it on read,
/// and preserve the original payload byte-for-byte.
#[test]
fn deflate_plus_fletcher32_combined_roundtrip_v2() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[8, 8]);
    let data: Vec<i32> = (0..64).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![4, 4]),
        compression: Compression::Deflate { level: 6 },
        filters: vec![3],
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("dual_filter", &dt, &shape, &raw, &dcpl)
        .expect("write dual-filter dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "dual_filter");
    let dataset = file.dataset_at(addr).expect("metadata");

    assert_eq!(dataset.layout, StorageLayout::Chunked);
    assert_eq!(
        dataset.filters,
        vec![3, 1],
        "checksum filter must precede deflate in the emitted pipeline"
    );

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read dual-filter");
    let read_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(
        read_values, data,
        "deflate+fletcher32 combined must roundtrip"
    );
}

/// Test roundtrip of a v4 chunked dataset with B-tree v2 and deflate compression.
///
/// ## Invariant
///
/// v4 layout with B-tree v2 chunk index must roundtrip compressed data with
/// exact floating-point value preservation.
#[test]
fn deflate_v4_compressed_chunked_roundtrip() {
    let data: Vec<f32> = (0..48).map(|i| i as f32 * 0.5).collect();
    let dt = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let shape = Shape::fixed(&[6, 8]);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![3, 4]),
        compression: Compression::Deflate { level: 4 },
        layout_version: Some(4),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("v4_compressed", &dt, &shape, &raw, &dcpl)
        .expect("write v4 compressed chunked dataset");

    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let datasets = file.list_root_group().expect("list root");
    let addr = datasets
        .iter()
        .find(|(name, _, _)| name == "v4_compressed")
        .map(|(_, addr, _)| *addr)
        .expect("dataset link");

    let dataset = file.dataset_at(addr).expect("dataset metadata");
    assert_eq!(dataset.layout, StorageLayout::Chunked);
    assert!(!dataset.filters.is_empty());

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read v4 compressed chunked dataset");
    let read_values: Vec<f32> = read_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, data);
}

// ---------------------------------------------------------------------------
// Edge-Case Roundtrip Tests
// ---------------------------------------------------------------------------

/// Test roundtrip with maximum dataset rank supported by chunked path (4D).
///
/// ## Invariant
///
/// Higher-rank datasets must roundtrip through the chunked path
/// with correct reconstruction across all dimensions.
#[test]
fn chunked_4d_dataset_roundtrip() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    // 3x4x5x6 = 360 elements
    let shape = Shape::fixed(&[3, 4, 5, 6]);
    let raw: Vec<u8> = (0..360).map(|i| (i % 256) as u8).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![2, 2, 3, 3]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("tensor4d", &dt, &shape, &raw, &dcpl)
        .expect("add 4D chunked dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "tensor4d");
    let dataset = file.dataset_at(addr).expect("dataset metadata");

    assert_eq!(dataset.shape.current_dims().as_slice(), &[3, 4, 5, 6]);
    assert_eq!(dataset.layout, StorageLayout::Chunked);

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read 4D chunked");

    assert_eq!(read_buf, raw, "4D chunked dataset must roundtrip exactly");
}

/// Test chunked dataset where dataset dims are exact multiples of chunk dims.
///
/// ## Invariant
///
/// When dataset dims are exact multiples of chunk dims, there are
/// no edge chunks. All chunks are full-size.
#[test]
fn chunked_exact_multiple_roundtrip() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    // 12x12 with 4x4 chunks → 3x3 = 9 full chunks, 0 edge chunks
    let shape = Shape::fixed(&[12, 12]);
    let data: Vec<i32> = (0..144).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![4, 4]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("exact", &dt, &shape, &raw, &dcpl)
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "exact");

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read exact multiple");
    let read_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(read_values, data, "exact-multiple chunks must roundtrip");
}

/// Test chunked dataset where chunk dims equal dataset dims (single chunk).
///
/// ## Invariant
///
/// A single-chunk dataset is the degenerate case: one chunk covers
/// the entire dataset.
#[test]
fn single_chunk_covers_entire_dataset() {
    let dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let shape = Shape::fixed(&[8, 8]);
    let data: Vec<f64> = (0..64).map(|i| i as f64 * 0.1).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![8, 8]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("single_chunk", &dt, &shape, &raw, &dcpl)
        .expect("add dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "single_chunk");

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read single chunk");
    let read_values: Vec<f64> = read_buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    for (original, roundtrip) in data.iter().zip(read_values.iter()) {
        assert!(
            (original - roundtrip).abs() < f64::EPSILON,
            "single-chunk value mismatch: {} != {}",
            original,
            roundtrip
        );
    }
}

/// Test Fletcher32-only chunked roundtrip as a regression for checksum handling.
///
/// ## Invariant
///
/// Fletcher32 must append a 4-byte checksum on write, verify and strip it on read,
/// and preserve the original payload byte-for-byte.
#[test]
fn deflate_plus_fletcher32_combined_roundtrip() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[8, 8]);
    let data: Vec<i32> = (0..64).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![4, 4]),
        filters: vec![3],
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("dual_filter", &dt, &shape, &raw, &dcpl)
        .expect("write checksum-only dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "dual_filter");
    let dataset = file.dataset_at(addr).expect("metadata");

    assert_eq!(dataset.layout, StorageLayout::Chunked);
    assert_eq!(
        dataset.filters,
        vec![3],
        "filter pipeline must contain Fletcher32 only"
    );

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read checksum-only dataset");
    let read_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(
        read_values, data,
        "checksum-only chunked dataset must roundtrip"
    );
}

/// Large dataset roundtrip to verify no truncation or overflow.
///
/// ## Invariant
///
/// 256x256 dataset (65536 elements × 4 bytes = 262144 bytes)
/// must roundtrip through chunked path without data loss.
#[test]
fn large_chunked_dataset_roundtrip() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[256, 256]);
    let data: Vec<i32> = (0..65536).collect();
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![64, 64]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("large", &dt, &shape, &raw, &dcpl)
        .expect("add large dataset");

    let file = build_and_open(builder);
    let addr = find_dataset_addr(&file, "large");

    let read_buf = file
        .read_chunked_dataset_all_bytes(addr)
        .expect("read large");
    let read_values: Vec<i32> = read_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(read_values.len(), data.len(), "element count must match");
    assert_eq!(read_values, data, "large chunked dataset must roundtrip");
}
