//! MAT v7.3 (HDF5-backed) read tests.
//! Builds synthetic HDF5 files with MATLAB_class attributes and asserts
//! value-semantic decoding for supported MATLAB classes.
//
//! The current HDF5 test builder supports root-group attributes and
//! root-linked datasets. It does not expose nested group construction with
//! attached attributes, and it does not expose virtual-dataset authoring.
//! These tests therefore cover the v7.3 dataset-backed classes and storage
//! layouts that can be authored with the available builder surface, including
//! explicit compact-layout rejection. Virtual-layout rejection remains a
//! documented reader contract but is not directly fixture-authorable here.

use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
use consus_hdf5::file::writer::{ChildDatasetSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_hdf5::property_list::DatasetLayout;
use consus_mat::{MatArray, MatError, MatNumericClass, MatVersion, loadmat_bytes};
use core::num::NonZeroUsize;

fn fixed_ascii_attr(class: &str) -> (Datatype, Shape, Vec<u8>) {
    (
        Datatype::FixedString {
            length: class.len(),
            encoding: StringEncoding::Ascii,
        },
        Shape::scalar(),
        class.as_bytes().to_vec(),
    )
}

fn float64_datatype() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

fn uint8_datatype() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    }
}

fn uint16_datatype(byte_order: ByteOrder) -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(16).unwrap(),
        byte_order,
        signed: false,
    }
}

fn build_hdf5_double_array(values: &[f64]) -> Vec<u8> {
    let dt = float64_datatype();
    let shape = Shape::fixed(&[values.len()]);
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (attr_dt, attr_shape, attr_data) = fixed_ascii_attr("double");

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "x",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
            &[("MATLAB_class", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add double dataset failed");
    builder.finish().expect("finish failed")
}

fn build_hdf5_logical_array(values: &[u8]) -> Vec<u8> {
    let dt = uint8_datatype();
    let shape = Shape::fixed(&[values.len()]);
    let (attr_dt, attr_shape, attr_data) = fixed_ascii_attr("logical");

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "flag",
            &dt,
            &shape,
            values,
            &DatasetCreationProps::default(),
            &[("MATLAB_class", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add logical dataset failed");
    builder.finish().expect("finish failed")
}

fn build_hdf5_char_array(text: &str) -> Vec<u8> {
    build_hdf5_char_array_with_byte_order(text, ByteOrder::LittleEndian)
}

fn build_hdf5_char_array_with_byte_order(text: &str, byte_order: ByteOrder) -> Vec<u8> {
    let dt = uint16_datatype(byte_order);
    let raw: Vec<u8> = text
        .chars()
        .flat_map(|c| match byte_order {
            ByteOrder::LittleEndian => (c as u16).to_le_bytes(),
            ByteOrder::BigEndian => (c as u16).to_be_bytes(),
        })
        .collect();
    let shape = Shape::fixed(&[text.chars().count()]);
    let (attr_dt, attr_shape, attr_data) = fixed_ascii_attr("char");

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "str",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
            &[("MATLAB_class", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add char dataset failed");
    builder.finish().expect("finish failed")
}

fn build_hdf5_root_with_ignored_attribute_and_double_dataset(values: &[f64]) -> Vec<u8> {
    let dt = float64_datatype();
    let shape = Shape::fixed(&[values.len()]);
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (dataset_attr_dt, dataset_attr_shape, dataset_attr_data) = fixed_ascii_attr("double");
    let (root_attr_dt, root_attr_shape, root_attr_data) = fixed_ascii_attr("metadata");

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_root_attribute(
            "MATLAB_class",
            &root_attr_dt,
            &root_attr_shape,
            &root_attr_data,
        )
        .expect("add root attribute failed");
    builder
        .add_dataset_with_attributes(
            "x",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
            &[(
                "MATLAB_class",
                &dataset_attr_dt,
                &dataset_attr_shape,
                &dataset_attr_data,
            )],
        )
        .expect("add dataset failed");
    builder.finish().expect("finish failed")
}

fn build_hdf5_sparse_dataset() -> Vec<u8> {
    let dt = float64_datatype();
    let shape = Shape::fixed(&[2]);
    let raw: Vec<u8> = [1.0f64, 2.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (attr_dt, attr_shape, attr_data) = fixed_ascii_attr("sparse");

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "sp",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
            &[("MATLAB_class", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add sparse dataset failed");
    builder.finish().expect("finish failed")
}

fn build_hdf5_compact_double_dataset() -> Vec<u8> {
    let dt = float64_datatype();
    let shape = Shape::fixed(&[2]);
    let raw: Vec<u8> = [3.0f64, 4.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (attr_dt, attr_shape, attr_data) = fixed_ascii_attr("double");

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Compact,
        ..DatasetCreationProps::default()
    };
    builder
        .add_dataset_with_attributes(
            "compact_x",
            &dt,
            &shape,
            &raw,
            &dcpl,
            &[("MATLAB_class", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add compact dataset failed");
    builder.finish().expect("finish failed")
}

fn decode_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect()
}

#[test]
fn v73_double_array_roundtrip() {
    let hdf5_bytes = build_hdf5_double_array(&[10.0, 20.0, 30.0]);
    let mat = loadmat_bytes(&hdf5_bytes).expect("v7.3 parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "x");

    match arr {
        MatArray::Numeric(na) => {
            assert_eq!(na.class, MatNumericClass::Double);
            assert_eq!(na.shape, vec![3]);
            assert_eq!(na.numel(), 3);
            assert_eq!(decode_f64s(&na.real_data), vec![10.0, 20.0, 30.0]);
            assert!(na.imag_data.is_none());
        }
        other => panic!("expected Numeric array, got {:?}", other),
    }
}

#[test]
fn v73_logical_array_roundtrip() {
    let bytes = build_hdf5_logical_array(&[1u8, 0u8, 1u8, 1u8]);
    let mat = loadmat_bytes(&bytes).expect("v7.3 logical parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "flag");

    match arr {
        MatArray::Logical(la) => {
            assert_eq!(la.shape, vec![4]);
            assert_eq!(la.data, vec![true, false, true, true]);
        }
        other => panic!("expected Logical, got {:?}", other),
    }
}

#[test]
fn v73_char_array_roundtrip() {
    let bytes = build_hdf5_char_array("hi");
    let mat = loadmat_bytes(&bytes).expect("v7.3 char parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "str");

    match arr {
        MatArray::Char(ca) => {
            assert_eq!(ca.shape, vec![2]);
            assert_eq!(ca.data, "hi");
        }
        other => panic!("expected Char, got {:?}", other),
    }
}

#[test]
fn v73_root_attributes_are_ignored_for_variable_collection() {
    let bytes = build_hdf5_root_with_ignored_attribute_and_double_dataset(&[5.0, 6.0]);
    let mat = loadmat_bytes(&bytes).expect("v7.3 parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "x");

    match arr {
        MatArray::Numeric(na) => {
            assert_eq!(na.class, MatNumericClass::Double);
            assert_eq!(na.shape, vec![2]);
            assert_eq!(decode_f64s(&na.real_data), vec![5.0, 6.0]);
        }
        other => panic!("expected Numeric array, got {:?}", other),
    }
}

#[test]
fn v73_char_big_endian_dataset_roundtrip() {
    let bytes = build_hdf5_char_array_with_byte_order("hi", ByteOrder::BigEndian);
    let mat = loadmat_bytes(&bytes).expect("v7.3 big-endian char parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "str");

    match arr {
        MatArray::Char(ca) => {
            assert_eq!(ca.shape, vec![2]);
            assert_eq!(ca.data, "hi");
        }
        other => panic!("expected Char, got {:?}", other),
    }
}

#[test]
fn v73_sparse_dataset_returns_unsupported_feature_error() {
    let bytes = build_hdf5_sparse_dataset();
    let err = loadmat_bytes(&bytes).expect_err("v7.3 sparse dataset must return Err");

    match err {
        MatError::UnsupportedFeature(message) => {
            assert_eq!(message, "v7.3 sparse datasets are not supported");
        }
        other => panic!("expected UnsupportedFeature, got {:?}", other),
    }
}

#[test]
fn v73_compact_layout_returns_unsupported_feature_error() {
    let bytes = build_hdf5_compact_double_dataset();
    let err = loadmat_bytes(&bytes).expect_err("v7.3 compact dataset must return Err");

    match err {
        MatError::UnsupportedFeature(message) => {
            assert_eq!(message, "v7.3 compact layout");
        }
        other => panic!("expected UnsupportedFeature, got {:?}", other),
    }
}

fn build_hdf5_cell_array(values: &[f64]) -> Vec<u8> {
    let dt = float64_datatype();
    let scalar_shape = Shape::scalar();
    let (class_dt, class_shape, class_data) = fixed_ascii_attr("cell");

    let names: Vec<String> = (0..values.len()).map(|i| i.to_string()).collect();
    let raws: Vec<Vec<u8>> = values.iter().map(|v| v.to_le_bytes().to_vec()).collect();

    let specs: Vec<ChildDatasetSpec<'_>> = names
        .iter()
        .zip(raws.iter())
        .map(|(name, raw)| ChildDatasetSpec {
            name: name.as_str(),
            datatype: &dt,
            shape: &scalar_shape,
            raw_data: raw.as_slice(),
            dcpl: DatasetCreationProps::default(),
            attributes: &[],
        })
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_attributes(
            "cells",
            &[("MATLAB_class", &class_dt, &class_shape, &class_data)],
            &specs,
        )
        .expect("add cell group failed");
    builder.finish().expect("finish failed")
}

fn build_hdf5_struct_array(fields: &[(&str, f64)]) -> Vec<u8> {
    let dt = float64_datatype();
    let scalar_shape = Shape::scalar();
    let (class_dt, class_shape, class_data) = fixed_ascii_attr("struct");

    let raws: Vec<Vec<u8>> = fields.iter().map(|(_, v)| v.to_le_bytes().to_vec()).collect();

    let specs: Vec<ChildDatasetSpec<'_>> = fields
        .iter()
        .zip(raws.iter())
        .map(|(field, raw)| ChildDatasetSpec {
            name: field.0,
            datatype: &dt,
            shape: &scalar_shape,
            raw_data: raw.as_slice(),
            dcpl: DatasetCreationProps::default(),
            attributes: &[],
        })
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_attributes(
            "s",
            &[("MATLAB_class", &class_dt, &class_shape, &class_data)],
            &specs,
        )
        .expect("add struct group failed");
    builder.finish().expect("finish failed")
}

#[test]
fn v73_cell_array_roundtrip() {
    let bytes = build_hdf5_cell_array(&[10.0, 20.0]);
    let mat = loadmat_bytes(&bytes).expect("v7.3 cell parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "cells");

    match arr {
        MatArray::Cell(ca) => {
            assert_eq!(ca.shape, vec![1, 2]);
            assert_eq!(ca.cells().len(), 2);

            for (expected, cell) in [10.0f64, 20.0].iter().zip(ca.cells().iter()) {
                match cell {
                    MatArray::Numeric(na) => {
                        assert_eq!(na.class, MatNumericClass::Double);
                        let v = f64::from_le_bytes([
                            na.real_data[0], na.real_data[1], na.real_data[2], na.real_data[3],
                            na.real_data[4], na.real_data[5], na.real_data[6], na.real_data[7],
                        ]);
                        assert_eq!(v, *expected);
                    }
                    other => panic!("expected Numeric cell element, got {:?}", other),
                }
            }
        }
        other => panic!("expected Cell array, got {:?}", other),
    }
}

#[test]
fn v73_struct_array_roundtrip() {
    let bytes = build_hdf5_struct_array(&[("x", 42.0), ("y", 99.0)]);
    let mat = loadmat_bytes(&bytes).expect("v7.3 struct parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "s");

    match arr {
        MatArray::Struct(sa) => {
            assert_eq!(sa.shape, vec![1, 1]);

            let field_names: Vec<&str> = sa.field_names().collect();
            assert!(field_names.contains(&"x"), "field x must be present");
            assert!(field_names.contains(&"y"), "field y must be present");

            let read_f64 = |arr: &MatArray| -> f64 {
                if let MatArray::Numeric(na) = arr {
                    f64::from_le_bytes([
                        na.real_data[0], na.real_data[1], na.real_data[2], na.real_data[3],
                        na.real_data[4], na.real_data[5], na.real_data[6], na.real_data[7],
                    ])
                } else {
                    panic!("expected Numeric field element, got {:?}", arr)
                }
            };

            let x_vals = sa.field("x").expect("field x not found");
            assert_eq!(x_vals.len(), 1);
            assert_eq!(read_f64(&x_vals[0]), 42.0);

            let y_vals = sa.field("y").expect("field y not found");
            assert_eq!(y_vals.len(), 1);
            assert_eq!(read_f64(&y_vals[0]), 99.0);
        }
        other => panic!("expected Struct array, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Chunked dataset fixture
// ---------------------------------------------------------------------------

fn build_hdf5_chunked_double_dataset(values: &[f64], chunk_size: usize) -> Vec<u8> {
    let dt = float64_datatype();
    let shape = Shape::fixed(&[values.len()]);
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (attr_dt, attr_shape, attr_data) = fixed_ascii_attr("double");

    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![chunk_size]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "x",
            &dt,
            &shape,
            &raw,
            &dcpl,
            &[("MATLAB_class", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add chunked dataset failed");
    builder.finish().expect("finish failed")
}

#[test]
fn v73_chunked_double_array_roundtrip() {
    let values = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = build_hdf5_chunked_double_dataset(&values, 3);
    let mat = loadmat_bytes(&bytes).expect("v7.3 chunked parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "x");

    match arr {
        MatArray::Numeric(na) => {
            assert_eq!(na.class, MatNumericClass::Double);
            assert_eq!(na.shape, vec![6]);
            assert_eq!(na.numel(), 6);
            let got = decode_f64s(&na.real_data);
            assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            assert!(na.imag_data.is_none());
        }
        other => panic!("expected Numeric array, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Virtual layout rejection fixture
// ---------------------------------------------------------------------------

fn build_hdf5_virtual_dataset() -> Vec<u8> {
    let dt = float64_datatype();
    let shape = Shape::fixed(&[2]);
    let raw: Vec<u8> = [1.0f64, 2.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (attr_dt, attr_shape, attr_data) = fixed_ascii_attr("double");

    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Virtual,
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset_with_attributes(
            "virt",
            &dt,
            &shape,
            &raw,
            &dcpl,
            &[("MATLAB_class", &attr_dt, &attr_shape, &attr_data)],
        )
        .expect("add virtual dataset failed");
    builder.finish().expect("finish failed")
}

#[test]
fn v73_virtual_layout_returns_unsupported_feature_error() {
    let bytes = build_hdf5_virtual_dataset();
    let err = loadmat_bytes(&bytes).expect_err("v7.3 virtual dataset must return Err");

    match err {
        MatError::UnsupportedFeature(message) => {
            assert_eq!(message, "v7.3 virtual layout");
        }
        other => panic!("expected UnsupportedFeature, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Non-scalar struct array fixture
// ---------------------------------------------------------------------------

fn build_hdf5_struct_non_scalar(field_defs: &[(&str, Vec<f64>)]) -> Vec<u8> {
    assert!(!field_defs.is_empty(), "field_defs must not be empty");
    let n = field_defs[0].1.len();
    assert!(n > 0, "field must have at least one value");

    let dt = float64_datatype();
    let field_shape = Shape::fixed(&[n]);

    // MATLAB_dims: uint32 array [1, n] in MATLAB column-major order.
    let dims_raw: Vec<u8> = [1u32, n as u32]
        .iter()
        .flat_map(|d| d.to_le_bytes())
        .collect();
    let dims_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let dims_shape = Shape::fixed(&[2]);

    let (class_dt, class_shape, class_data) = fixed_ascii_attr("struct");

    let raws: Vec<Vec<u8>> = field_defs
        .iter()
        .map(|(_, vals)| vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();

    let specs: Vec<ChildDatasetSpec<'_>> = field_defs
        .iter()
        .zip(raws.iter())
        .map(|((name, _), raw)| ChildDatasetSpec {
            name,
            datatype: &dt,
            shape: &field_shape,
            raw_data: raw.as_slice(),
            dcpl: DatasetCreationProps::default(),
            attributes: &[],
        })
        .collect();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_attributes(
            "s",
            &[
                ("MATLAB_class", &class_dt, &class_shape, &class_data),
                ("MATLAB_dims", &dims_dt, &dims_shape, &dims_raw),
            ],
            &specs,
        )
        .expect("add non-scalar struct group failed");
    builder.finish().expect("finish failed")
}

#[test]
fn v73_struct_array_non_scalar_roundtrip() {
    let bytes = build_hdf5_struct_non_scalar(&[
        ("x", vec![10.0f64, 20.0f64]),
        ("y", vec![30.0f64, 40.0f64]),
    ]);
    let mat = loadmat_bytes(&bytes).expect("v7.3 non-scalar struct parse failed");

    assert_eq!(mat.version, MatVersion::V73);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "s");

    match arr {
        MatArray::Struct(sa) => {
            // Shape derived from MATLAB_dims = [1, 2].
            assert_eq!(sa.shape, vec![1, 2]);
            assert_eq!(sa.numel(), 2);

            let read_f64 = |a: &MatArray| -> f64 {
                if let MatArray::Numeric(na) = a {
                    f64::from_le_bytes(na.real_data[0..8].try_into().unwrap())
                } else {
                    panic!("expected Numeric field element, got {:?}", a)
                }
            };

            let x_vals = sa.field("x").expect("field x not found");
            assert_eq!(x_vals.len(), 2, "x must have 2 elements for [1,2] struct");
            assert_eq!(read_f64(&x_vals[0]), 10.0);
            assert_eq!(read_f64(&x_vals[1]), 20.0);

            let y_vals = sa.field("y").expect("field y not found");
            assert_eq!(y_vals.len(), 2, "y must have 2 elements for [1,2] struct");
            assert_eq!(read_f64(&y_vals[0]), 30.0);
            assert_eq!(read_f64(&y_vals[1]), 40.0);
        }
        other => panic!("expected Struct array, got {:?}", other),
    }
}
