//! Integration tests: netCDF-4 HDF5 read layer.
//!
//! Tests validate computed values: dimension names, sizes, variable names,
//! shapes, classification correctness for CLASS=DIMENSION_SCALE, and
//! DIMENSION_LIST-based variable-to-dimension binding.

use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
use consus_hdf5::file::Hdf5File;
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_hdf5::property_list::DatasetLayout;
use consus_io::SliceReader;
use core::num::NonZeroUsize;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn f32_dt() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

fn i32_dt() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    }
}

fn fixed_str_dt(s: &str) -> Datatype {
    Datatype::FixedString {
        length: s.len(),
        encoding: StringEncoding::Ascii,
    }
}

const CLASS_VAL: &str = "DIMENSION_SCALE";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A CLASS=DIMENSION_SCALE dataset becomes exactly one dimension.
/// Name and size are derived from the dataset name and shape.
#[test]
fn dimension_scale_detected_by_class_attr() {
    let data = vec![0u8; 5 * 4];
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset_with_attributes(
        "time",
        &f32_dt(),
        &Shape::fixed(&[5]),
        &data,
        &DatasetCreationProps::default(),
        &[(
            "CLASS",
            &fixed_str_dt(CLASS_VAL),
            &Shape::scalar(),
            CLASS_VAL.as_bytes(),
        )],
    )
    .expect("add");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.dimensions.len(), 1, "must produce exactly one dimension");
    assert_eq!(
        g.dimensions[0].name, "time",
        "dimension name must match dataset name"
    );
    assert_eq!(
        g.dimensions[0].size, 5,
        "dimension size must equal leading extent"
    );
    assert_eq!(
        g.variables.len(),
        0,
        "dim scale must not also appear as a variable"
    );
}

/// A dataset without CLASS attr becomes exactly one variable.
#[test]
fn regular_dataset_becomes_variable() {
    let data = vec![0u8; 10 * 4];
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset(
        "depth",
        &f32_dt(),
        &Shape::fixed(&[10]),
        &data,
        &DatasetCreationProps::default(),
    )
    .expect("add");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.variables.len(), 1, "must produce exactly one variable");
    assert_eq!(
        g.variables[0].name, "depth",
        "variable name must match dataset name"
    );
    assert_eq!(g.variables[0].rank(), 1, "rank must match dataset rank");
    assert_eq!(g.dimensions.len(), 0, "no dimension scales written");
}

/// One dim scale + one regular dataset are partitioned correctly.
/// Shape of the 2-D variable is verified element-by-element.
#[test]
fn dim_scale_and_variable_partition_correctly() {
    let time_data = vec![0u8; 5 * 4];
    let temp_data = vec![0u8; 5 * 3 * 4];
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset_with_attributes(
        "time",
        &f32_dt(),
        &Shape::fixed(&[5]),
        &time_data,
        &DatasetCreationProps::default(),
        &[(
            "CLASS",
            &fixed_str_dt(CLASS_VAL),
            &Shape::scalar(),
            CLASS_VAL.as_bytes(),
        )],
    )
    .expect("add time");
    b.add_dataset(
        "temperature",
        &f32_dt(),
        &Shape::fixed(&[5, 3]),
        &temp_data,
        &DatasetCreationProps::default(),
    )
    .expect("add temp");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.dimensions.len(), 1, "one dim scale");
    assert_eq!(g.dimensions[0].name, "time");
    assert_eq!(g.dimensions[0].size, 5);
    assert_eq!(g.variables.len(), 1, "one variable");
    assert_eq!(g.variables[0].name, "temperature");
    let sh = g.variables[0].shape.as_ref().expect("shape must be set");
    assert_eq!(sh.rank(), 2, "variable rank must be 2");
    let dims = sh.current_dims();
    assert_eq!(dims[0], 5, "leading dimension");
    assert_eq!(dims[1], 3, "second dimension");
}

/// NAME attribute on a dim scale overrides the dataset name.
#[test]
fn name_attr_overrides_dataset_name_for_dim() {
    let name_val = "latitude";
    let data = vec![0u8; 4 * 4];
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset_with_attributes(
        "lat",
        &f32_dt(),
        &Shape::fixed(&[4]),
        &data,
        &DatasetCreationProps::default(),
        &[
            (
                "CLASS",
                &fixed_str_dt(CLASS_VAL),
                &Shape::scalar(),
                CLASS_VAL.as_bytes(),
            ),
            (
                "NAME",
                &fixed_str_dt(name_val),
                &Shape::scalar(),
                name_val.as_bytes(),
            ),
        ],
    )
    .expect("add");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.dimensions.len(), 1);
    assert_eq!(
        g.dimensions[0].name, "latitude",
        "NAME attr value must win over dataset name lat"
    );
    assert_eq!(g.dimensions[0].size, 4);
}

/// A file with no datasets produces a fully empty model.
#[test]
fn empty_group_produces_empty_model() {
    let bytes = Hdf5FileBuilder::new(FileCreationProps::default())
        .finish()
        .expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert!(g.dimensions.is_empty(), "no dimensions in empty file");
    assert!(g.variables.is_empty(), "no variables in empty file");
    assert!(g.groups.is_empty(), "no subgroups in empty file");
}

/// Integer-typed dim scale: size equals the leading dimension extent.
#[test]
fn integer_dim_scale_size_is_correct() {
    let data: Vec<u8> = (0i32..12).flat_map(|v| v.to_le_bytes()).collect();
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset_with_attributes(
        "level",
        &i32_dt(),
        &Shape::fixed(&[12]),
        &data,
        &DatasetCreationProps::default(),
        &[(
            "CLASS",
            &fixed_str_dt(CLASS_VAL),
            &Shape::scalar(),
            CLASS_VAL.as_bytes(),
        )],
    )
    .expect("add");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.dimensions.len(), 1);
    assert_eq!(g.dimensions[0].name, "level");
    assert_eq!(g.dimensions[0].size, 12, "size must equal rank-1 extent 12");
}

/// A contiguous dataset extracted as a netCDF variable can be read back
/// through the HDF5 bridge with exact byte preservation.
#[test]
fn contiguous_variable_bytes_roundtrip() {
    let data: Vec<u8> = (0i32..6).flat_map(|v| v.to_le_bytes()).collect();
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset(
        "temperature",
        &i32_dt(),
        &Shape::fixed(&[2, 3]),
        &data,
        &DatasetCreationProps::default(),
    )
    .expect("add");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.variables.len(), 1, "must extract one variable");
    let variable = &g.variables[0];
    let read = consus_netcdf::hdf5::variable::read_variable_bytes(&file, variable)
        .expect("contiguous variable bytes");

    assert_eq!(
        read, data,
        "contiguous variable bytes must match original payload"
    );
}

/// A chunked dataset extracted as a netCDF variable can be read back
/// through the HDF5 bridge with exact logical byte assembly.
#[test]
fn chunked_variable_bytes_roundtrip() {
    let data: Vec<u8> = (0i32..16).flat_map(|v| v.to_le_bytes()).collect();
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![2, 2]),
        ..DatasetCreationProps::default()
    };
    b.add_dataset("pressure", &i32_dt(), &Shape::fixed(&[4, 4]), &data, &dcpl)
        .expect("add");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.variables.len(), 1, "must extract one variable");
    let variable = &g.variables[0];
    let read = consus_netcdf::hdf5::variable::read_variable_bytes(&file, variable)
        .expect("chunked variable bytes");

    assert_eq!(
        read, data,
        "chunked variable bytes must match original logical payload"
    );
}

/// Reading variable bytes without an HDF5 object-header address must fail.
#[test]
fn variable_bytes_require_object_header_address() {
    let data: Vec<u8> = (0i32..4).flat_map(|v| v.to_le_bytes()).collect();
    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset(
        "salinity",
        &i32_dt(),
        &Shape::fixed(&[4]),
        &data,
        &DatasetCreationProps::default(),
    )
    .expect("add");
    let bytes = b.finish().expect("finish");

    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    let mut variable = g.variables[0].clone();
    variable.object_header_address = None;

    let err = consus_netcdf::hdf5::variable::read_variable_bytes(&file, &variable)
        .expect_err("missing object header address must fail");

    let message = err.to_string();
    assert!(
        message.contains("object header"),
        "error must mention missing object header address, got: {message}"
    );
}

/// Without DIMENSION_LIST, extraction falls back to synthetic dimension names.
#[test]
fn missing_dimension_list_falls_back_to_synthetic_dimension_names() {
    let temp_data: Vec<u8> = (0i32..6).flat_map(|v| v.to_le_bytes()).collect();

    let mut b = Hdf5FileBuilder::new(FileCreationProps::default());
    b.add_dataset(
        "temperature",
        &i32_dt(),
        &Shape::fixed(&[2, 3]),
        &temp_data,
        &DatasetCreationProps::default(),
    )
    .expect("add temperature");

    let bytes = b.finish().expect("finish");
    let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
    let root = file.root_group();
    let g = consus_netcdf::extract_group(&file, root.path, root.object_header_address)
        .expect("extract_group");

    assert_eq!(g.variables.len(), 1, "must extract one variable");
    assert_eq!(
        g.variables[0].dimensions,
        vec![String::from("d0"), String::from("d1")],
        "missing DIMENSION_LIST must preserve conservative synthetic fallback"
    );
}
