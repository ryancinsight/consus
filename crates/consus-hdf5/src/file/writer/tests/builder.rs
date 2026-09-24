//! Builder integration tests: file, group, dataset, and sub-group construction round-trips.

use super::super::*;
use crate::file::Hdf5File;
use crate::property_list::{DatasetCreationProps, FileCreationProps};
use byteorder::{ByteOrder as _, LittleEndian};
use core::num::NonZeroUsize;

/// `Hdf5FileBuilder::add_v1_group_with_children` + `finish` → `Hdf5File::open`
/// → `list_group_at` enumerates children in insertion order with correct addresses.
///
/// ## Invariants
/// - Two children "x" and "y" are returned with their written addresses.
/// - `open_path` resolves both children by name.
/// - The group address returned by the builder method is non-zero and
///   distinct from both child addresses.
#[cfg(feature = "alloc")]
#[test]
fn add_v1_group_with_children_builder_e2e() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};
    use core::num::NonZeroUsize;

    let u32_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let raw_x: Vec<u8> = 7u32.to_le_bytes().to_vec();
    let raw_y: Vec<u8> = 42u32.to_le_bytes().to_vec();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());

    // Write two scalar datasets as v2 object headers.
    let x_addr = builder
        .add_dataset(
            "__x",
            &u32_dt,
            &Shape::scalar(),
            &raw_x,
            &DatasetCreationProps::default(),
        )
        .expect("add_dataset x");
    let y_addr = builder
        .add_dataset(
            "__y",
            &u32_dt,
            &Shape::scalar(),
            &raw_y,
            &DatasetCreationProps::default(),
        )
        .expect("add_dataset y");

    // Link those addresses into a v1 group.
    let grp_addr = builder
        .add_v1_group_with_children("v1grp", &[("x", x_addr), ("y", y_addr)])
        .expect("add_v1_group_with_children");

    let bytes = builder.finish().expect("finish");
    let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");

    // list_group_at must enumerate both children in emission order.
    let children = file
        .list_group_at(grp_addr)
        .expect("list_group_at must succeed for v1 group");
    assert_eq!(children.len(), 2, "v1 group must have exactly 2 children");
    assert_eq!(children[0].0, "x", "first child name must be 'x'");
    assert_eq!(
        children[0].1, x_addr,
        "first child address must equal x_addr"
    );
    assert_eq!(children[1].0, "y", "second child name must be 'y'");
    assert_eq!(
        children[1].1, y_addr,
        "second child address must equal y_addr"
    );

    // The group address must be distinct from both child addresses.
    assert_ne!(grp_addr, x_addr, "grp_addr must not alias x_addr");
    assert_ne!(grp_addr, y_addr, "grp_addr must not alias y_addr");
}

/// `add_named_datatype` writes an object that `node_type_at` classifies as
/// `NodeType::NamedDatatype` and that `named_datatype_at` decodes back to
/// the original canonical type.
#[cfg(feature = "alloc")]
#[test]
fn add_named_datatype_creates_readable_named_type() {
    use consus_core::{ByteOrder, Datatype, NodeType};
    use core::num::NonZeroUsize;

    let expected = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    let addr = b.add_named_datatype("my_type", &expected).expect("write");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(consus_io::SliceReader::new(&bytes)).expect("open");
    assert_eq!(
        file.node_type_at(addr).expect("node_type_at"),
        NodeType::NamedDatatype,
        "object must be classified as NamedDatatype"
    );
    let got = file.named_datatype_at(addr).expect("named_datatype_at");
    assert_eq!(got, expected, "decoded datatype must match original");
}

#[cfg(feature = "alloc")]
#[test]
fn hdf5_file_builder_produces_valid_superblock() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[3]);
    let raw: Vec<u8> = [10u32, 20, 30]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("temps", &dt, &shape, &raw, &DatasetCreationProps::default())
        .unwrap();
    let bytes = builder.finish().unwrap();

    // Verify HDF5 magic bytes at offset 0.
    assert_eq!(&bytes[0..8], &crate::constants::HDF5_MAGIC);
    assert_eq!(bytes[8], 2); // superblock version 2
    // Root group address at offset 36 (12 + 3*8) must be non-zero.
    assert_ne!(LittleEndian::read_u64(&bytes[36..44]), 0);
}

#[cfg(feature = "alloc")]
#[test]
fn add_group_with_children_creates_navigable_nested_group() {
    use consus_core::{Datatype, Shape, StringEncoding};

    let species_bytes = b"Mus musculus";
    let species_dt = Datatype::FixedString {
        length: species_bytes.len(),
        encoding: StringEncoding::Ascii,
    };
    let species_shape = Shape::scalar();

    let subject_attrs: &[(&str, &Datatype, &Shape, &[u8])] =
        &[("species", &species_dt, &species_shape, species_bytes)];

    let subject_spec = ChildGroupSpec {
        name: "subject",
        attributes: subject_attrs,
        datasets: &[],
        sub_groups: &[],
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_children("general", &[], &[], &[subject_spec])
        .unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
    let addr = file
        .open_path("general/subject")
        .expect("navigate to general/subject");
    assert_ne!(addr, 0, "subject group address must be non-zero");

    let attrs = file
        .attributes_at(addr)
        .expect("read attributes at subject");
    let species_attr = attrs
        .iter()
        .find(|a| a.name == "species")
        .expect("species attribute must be present");
    assert_eq!(
        species_attr.raw_data.as_slice(),
        species_bytes,
        "species attribute raw bytes must match 'Mus musculus'"
    );
}

#[cfg(feature = "alloc")]
#[test]
fn add_group_with_children_nested_group_datasets_are_readable() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

    let f64_dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
    };
    let values_shape = Shape::fixed(&[3]);
    let raw_data: Vec<u8> = [1.0f64, 2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let raw_data_spec = ChildDatasetSpec {
        name: "raw_data",
        datatype: &f64_dt,
        shape: &values_shape,
        raw_data: &raw_data,
        dcpl: DatasetCreationProps::default(),
        attributes: &[],
    };
    let values_spec = ChildGroupSpec {
        name: "values",
        attributes: &[],
        datasets: &[raw_data_spec],
        sub_groups: &[],
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_children("data_container", &[], &[], &[values_spec])
        .unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
    let dataset_addr = file
        .open_path("data_container/values/raw_data")
        .expect("navigate to raw_data dataset");
    assert_ne!(dataset_addr, 0, "raw_data dataset address must be non-zero");

    let dataset = file
        .dataset_at(dataset_addr)
        .expect("read dataset metadata");
    let data_addr = dataset
        .data_address
        .expect("contiguous dataset must have a data_address");

    let mut buf = [0u8; 24]; // 3 × 8 bytes
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)
        .expect("read contiguous dataset bytes");

    let v0 = LittleEndian::read_f64(&buf[0..8]);
    let v1 = LittleEndian::read_f64(&buf[8..16]);
    let v2 = LittleEndian::read_f64(&buf[16..24]);
    assert_eq!(
        [v0, v1, v2],
        [1.0f64, 2.0, 3.0],
        "decoded f64 values must equal [1.0, 2.0, 3.0]"
    );
}

#[cfg(feature = "alloc")]
#[test]
fn add_group_with_attributes_still_works_after_refactor() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[2]);
    let raw: Vec<u8> = [42u32, 99].iter().flat_map(|v| v.to_le_bytes()).collect();

    let child = ChildDatasetSpec {
        name: "my_dataset",
        datatype: &dt,
        shape: &shape,
        raw_data: &raw,
        dcpl: DatasetCreationProps::default(),
        attributes: &[],
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_attributes("my_group", &[], &[child])
        .unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
    let dataset_addr = file
        .open_path("my_group/my_dataset")
        .expect("navigate to my_group/my_dataset");
    assert_ne!(dataset_addr, 0, "dataset address must be non-zero");

    let dataset = file
        .dataset_at(dataset_addr)
        .expect("read dataset metadata");
    let data_addr = dataset
        .data_address
        .expect("contiguous dataset must have a data_address");

    let mut buf = [0u8; 8]; // 2 × 4 bytes
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)
        .expect("read contiguous dataset bytes");

    assert_eq!(
        LittleEndian::read_u32(&buf[0..4]),
        42,
        "first element must be 42"
    );
    assert_eq!(
        LittleEndian::read_u32(&buf[4..8]),
        99,
        "second element must be 99"
    );
}

/// `begin_group` + `finish_with_attributes(&[])` + `finish()` produces
/// an HDF5 file where the group is navigable and classified as
/// `NodeType::Group`.
///
/// ## Invariant
///
/// An empty group written via `SubGroupBuilder` has a valid object header
/// that the reader classifies as a `NodeType::Group`.
#[cfg(feature = "alloc")]
#[test]
fn sub_group_builder_empty_finish_creates_navigable_group() {
    use consus_core::NodeType;

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let grp = builder.begin_group("empty_grp");
    grp.finish_with_attributes(&[]).unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
    let addr = file.open_path("empty_grp").expect("navigate to empty_grp");
    assert_ne!(addr, 0, "empty_grp address must be non-zero");

    let node_type = file
        .node_type_at(addr)
        .expect("node_type_at must succeed for empty_grp");
    assert_eq!(
        node_type,
        NodeType::Group,
        "empty_grp must be classified as NodeType::Group"
    );
}

/// `add_dataset_with_attributes` returns the dataset's object-header
/// address, which can be embedded verbatim in a subsequent
/// `DIMENSION_LIST` attribute byte payload and round-trips correctly.
///
/// ## Invariant
///
/// The `dim_addr` returned by the first `add_dataset_with_attributes` call
/// equals the u64 LE value read from the `DIMENSION_LIST` attribute bytes
/// on the variable dataset.
#[cfg(feature = "alloc")]
#[test]
fn sub_group_builder_dataset_address_is_reusable_in_dimlist() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, ReferenceType, Shape};

    let u32_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let f32_dt = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
    };
    let dim_raw: Vec<u8> = [0u32, 1, 2, 3]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let var_raw = vec![0u8; 4 * 4]; // 4 × f32

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let mut grp = builder.begin_group("grp");

    // Write dimension dataset; capture its object-header address.
    let dim_addr = grp
        .add_dataset_with_attributes(
            "x",
            &u32_dt,
            &Shape::fixed(&[4]),
            &dim_raw,
            &DatasetCreationProps::default(),
            &[],
        )
        .expect("write dim dataset");

    // Build DIMENSION_LIST attribute using dim_addr.
    let ref_dt = Datatype::Reference(ReferenceType::Object);
    let ref_shape = Shape::fixed(&[1]);
    let ref_data = dim_addr.to_le_bytes().to_vec();

    // Write variable dataset with DIMENSION_LIST pointing to dim_addr.
    grp.add_dataset_with_attributes(
        "var",
        &f32_dt,
        &Shape::fixed(&[4]),
        &var_raw,
        &DatasetCreationProps::default(),
        &[("DIMENSION_LIST", &ref_dt, &ref_shape, &ref_data)],
    )
    .expect("write variable dataset");

    grp.finish_with_attributes(&[]).unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
    let var_addr = file.open_path("grp/var").expect("navigate to grp/var");

    let attrs = file
        .attributes_at(var_addr)
        .expect("read attributes at grp/var");
    let dim_list_attr = attrs
        .iter()
        .find(|a| a.name == "DIMENSION_LIST")
        .expect("DIMENSION_LIST attribute must be present");

    assert!(
        dim_list_attr.raw_data.len() >= 8,
        "DIMENSION_LIST raw_data must be at least 8 bytes"
    );
    let decoded_addr = u64::from_le_bytes(dim_list_attr.raw_data[0..8].try_into().unwrap());
    assert_eq!(
        decoded_addr, dim_addr,
        "DIMENSION_LIST bytes must decode back to dim_addr={dim_addr:#x}"
    );
}

/// `begin_sub_group` creates a nested group under the parent, and datasets
/// inside the nested group are navigable after `finish`.
///
/// ## Invariant
///
/// Navigating `outer/inner/ds` reaches a readable dataset whose raw bytes
/// match the written content.
#[cfg(feature = "alloc")]
#[test]
fn sub_group_builder_nested_sub_group_roundtrip() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

    let u32_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let raw: Vec<u8> = [7u32, 42, 99]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let mut outer = builder.begin_group("outer");
    {
        let mut inner = outer.begin_sub_group("inner");
        inner
            .add_dataset_with_attributes(
                "ds",
                &u32_dt,
                &Shape::fixed(&[3]),
                &raw,
                &DatasetCreationProps::default(),
                &[],
            )
            .expect("write ds in inner");
        inner.finish_with_attributes(&[]).unwrap();
    }
    outer.finish_with_attributes(&[]).unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
    let ds_addr = file
        .open_path("outer/inner/ds")
        .expect("navigate to outer/inner/ds");
    assert_ne!(ds_addr, 0, "ds address must be non-zero");

    let dataset = file.dataset_at(ds_addr).expect("read dataset metadata");
    let data_addr = dataset
        .data_address
        .expect("contiguous dataset must have a data_address");

    let mut buf = [0u8; 12]; // 3 × u32
    file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)
        .expect("read ds bytes");
    assert_eq!(LittleEndian::read_u32(&buf[0..4]), 7, "ds[0] must be 7");
    assert_eq!(LittleEndian::read_u32(&buf[4..8]), 42, "ds[1] must be 42");
    assert_eq!(LittleEndian::read_u32(&buf[8..12]), 99, "ds[2] must be 99");
}
