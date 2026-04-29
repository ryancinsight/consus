//! Private write-path helpers for the netCDF-4 HDF5 write path.
//!
//! Defines [`DatasetTarget`], [`encode_cf_attrs`], [`write_dimension_scale`],
//! [`write_variable`], and [`write_child_group_content`].

use alloc::{collections::BTreeMap, string::String, vec, vec::Vec};
use core::num::NonZeroUsize;

use consus_core::{
    AttributeValue, ByteOrder, Datatype, ReferenceType, Result, Shape, StringEncoding,
};
use consus_hdf5::file::writer::{DatasetCreationProps, Hdf5FileBuilder, SubGroupBuilder};

use crate::conventions::{DIMENSION_SCALE_CLASS, DIMENSION_SCALE_VALUE};
use crate::dimension::NetcdfDimension;
use crate::model::NetcdfGroup;
use crate::variable::NetcdfVariable;

// ---------------------------------------------------------------------------
// DatasetTarget trait
// ---------------------------------------------------------------------------

/// Abstraction over any write target that accepts datasets with attributes.
///
/// Implemented for [`Hdf5FileBuilder`] (root-level) and [`SubGroupBuilder`]
/// (nested-group).  Allows `write_dimension_scale` and `write_variable` to
/// remain generic so one implementation serves every group depth.
///
/// Both impls delegate to the corresponding inherent method; method resolution
/// in Rust prefers inherent over trait methods, so neither impl recurses.
pub(super) trait DatasetTarget {
    fn add_dataset_with_attributes(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
        attributes: &[(&str, &Datatype, &Shape, &[u8])],
    ) -> Result<u64>;
}

impl DatasetTarget for Hdf5FileBuilder {
    fn add_dataset_with_attributes(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
        attributes: &[(&str, &Datatype, &Shape, &[u8])],
    ) -> Result<u64> {
        Hdf5FileBuilder::add_dataset_with_attributes(
            self, name, dt, shape, raw_data, dcpl, attributes,
        )
    }
}

impl DatasetTarget for SubGroupBuilder<'_> {
    fn add_dataset_with_attributes(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
        attributes: &[(&str, &Datatype, &Shape, &[u8])],
    ) -> Result<u64> {
        SubGroupBuilder::add_dataset_with_attributes(
            self, name, dt, shape, raw_data, dcpl, attributes,
        )
    }
}

// ---------------------------------------------------------------------------
// Local datatype constructors (zero-cost inline helpers)
// ---------------------------------------------------------------------------

#[inline(always)]
fn i64_le_dt() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    }
}

#[inline(always)]
fn u64_le_dt() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    }
}

#[inline(always)]
fn f64_le_dt() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

#[inline(always)]
fn fixed_str_dt(length: usize) -> Datatype {
    Datatype::FixedString {
        length,
        encoding: StringEncoding::Ascii,
    }
}

// ---------------------------------------------------------------------------
// encode_cf_attrs
// ---------------------------------------------------------------------------

/// Encode all supported [`AttributeValue`] variants to
/// `(name, Datatype, Shape, raw_bytes)` tuples.
///
/// Supported: `Int`, `Uint`, `Float`, `String`, `IntArray`, `UintArray`,
/// `FloatArray`, `StringArray` (non-empty).  `Bytes` and empty arrays are
/// silently skipped.  System keys `"DIMENSION_LIST"`, `"CLASS"`, `"NAME"`,
/// and `"_Netcdf4Dimid"` are always excluded.
pub(super) fn encode_cf_attrs(
    attrs: &[(String, AttributeValue)],
) -> Vec<(String, Datatype, Shape, Vec<u8>)> {
    const SKIP: &[&str] = &["DIMENSION_LIST", "CLASS", "NAME", "_Netcdf4Dimid"];
    let mut result: Vec<(String, Datatype, Shape, Vec<u8>)> = Vec::new();

    for (attr_name, attr_value) in attrs {
        if SKIP.contains(&attr_name.as_str()) {
            continue;
        }
        match attr_value {
            AttributeValue::Int(v) => result.push((
                attr_name.clone(),
                i64_le_dt(),
                Shape::scalar(),
                v.to_le_bytes().to_vec(),
            )),
            AttributeValue::Uint(v) => result.push((
                attr_name.clone(),
                u64_le_dt(),
                Shape::scalar(),
                v.to_le_bytes().to_vec(),
            )),
            AttributeValue::Float(v) => result.push((
                attr_name.clone(),
                f64_le_dt(),
                Shape::scalar(),
                v.to_bits().to_le_bytes().to_vec(),
            )),
            AttributeValue::String(s) => {
                let len = s.len().max(1);
                let data: Vec<u8> = if s.is_empty() {
                    vec![0u8]
                } else {
                    s.as_bytes().to_vec()
                };
                result.push((attr_name.clone(), fixed_str_dt(len), Shape::scalar(), data));
            }
            AttributeValue::IntArray(v) if !v.is_empty() => {
                let data: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                result.push((
                    attr_name.clone(),
                    i64_le_dt(),
                    Shape::fixed(&[v.len()]),
                    data,
                ));
            }
            AttributeValue::UintArray(v) if !v.is_empty() => {
                let data: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                result.push((
                    attr_name.clone(),
                    u64_le_dt(),
                    Shape::fixed(&[v.len()]),
                    data,
                ));
            }
            AttributeValue::FloatArray(v) if !v.is_empty() => {
                let data: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
                result.push((
                    attr_name.clone(),
                    f64_le_dt(),
                    Shape::fixed(&[v.len()]),
                    data,
                ));
            }
            AttributeValue::StringArray(v) if !v.is_empty() => {
                let max_len = v.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
                let mut data = vec![0u8; max_len * v.len()];
                for (i, s) in v.iter().enumerate() {
                    let bytes = s.as_bytes();
                    let copy_len = bytes.len().min(max_len);
                    data[i * max_len..i * max_len + copy_len].copy_from_slice(&bytes[..copy_len]);
                }
                result.push((
                    attr_name.clone(),
                    fixed_str_dt(max_len),
                    Shape::fixed(&[v.len()]),
                    data,
                ));
            }
            // Empty arrays and Bytes: no HDF5 encoding; skip.
            _ => {}
        }
    }
    result
}

// ---------------------------------------------------------------------------
// write_dimension_scale
// ---------------------------------------------------------------------------

/// Write one netCDF-4 dimension scale dataset into `writer`.
///
/// Emits attributes `CLASS="DIMENSION_SCALE"`, `NAME={dim.name}`, and
/// `_Netcdf4Dimid={dim_id}`.  Data payload is `[0u32, 1, …, size-1]` LE.
pub(super) fn write_dimension_scale<W: DatasetTarget>(
    writer: &mut W,
    dim: &NetcdfDimension,
    dim_id: u32,
) -> Result<u64> {
    let u32_le_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };

    let size = dim.size;
    let data_shape = if size > 0 {
        Shape::fixed(&[size])
    } else {
        Shape::scalar()
    };
    let raw_data: Vec<u8> = (0u32..size as u32).flat_map(|i| i.to_le_bytes()).collect();

    let class_dt = fixed_str_dt(DIMENSION_SCALE_VALUE.len());
    let class_shape = Shape::scalar();

    let name_len = dim.name.len().max(1);
    let name_dt = fixed_str_dt(name_len);
    let name_shape = Shape::scalar();
    let name_bytes: Vec<u8> = if dim.name.is_empty() {
        vec![0u8]
    } else {
        dim.name.as_bytes().to_vec()
    };

    let dimid_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let dimid_shape = Shape::scalar();
    let dimid_bytes: [u8; 4] = dim_id.to_le_bytes();

    writer.add_dataset_with_attributes(
        &dim.name,
        &u32_le_dt,
        &data_shape,
        &raw_data,
        &DatasetCreationProps::default(),
        &[
            (
                DIMENSION_SCALE_CLASS,
                &class_dt,
                &class_shape,
                DIMENSION_SCALE_VALUE.as_bytes(),
            ),
            ("NAME", &name_dt, &name_shape, &name_bytes),
            ("_Netcdf4Dimid", &dimid_dt, &dimid_shape, &dimid_bytes),
        ],
    )
}

// ---------------------------------------------------------------------------
// write_variable
// ---------------------------------------------------------------------------

/// Write one netCDF-4 variable dataset into `writer`.
///
/// Writes a zero-filled contiguous dataset with a `DIMENSION_LIST` attribute
/// (one 8-byte LE object reference per axis) and all CF attributes from
/// `var.attributes` via [`encode_cf_attrs`].  Scalar variables (rank 0) carry
/// no `DIMENSION_LIST`.
pub(super) fn write_variable<W: DatasetTarget>(
    writer: &mut W,
    var: &NetcdfVariable,
    dim_addrs: &BTreeMap<String, u64>,
) -> Result<u64> {
    let shape = var.shape.clone().unwrap_or_else(Shape::scalar);
    let raw_data: Vec<u8> = match var.datatype.element_size() {
        Some(elem_size) => vec![0u8; elem_size * shape.num_elements()],
        None => Vec::new(),
    };

    let rank = var.dimensions.len();
    let mut attr_list: Vec<(String, Datatype, Shape, Vec<u8>)> = Vec::new();

    if rank > 0 {
        let dim_list_data: Vec<u8> = var
            .dimensions
            .iter()
            .flat_map(|n| {
                dim_addrs
                    .get(n.as_str())
                    .copied()
                    .unwrap_or(0u64)
                    .to_le_bytes()
            })
            .collect();
        attr_list.push((
            String::from("DIMENSION_LIST"),
            Datatype::Reference(ReferenceType::Object),
            Shape::fixed(&[rank]),
            dim_list_data,
        ));
    }

    for (name, dt, s, data) in encode_cf_attrs(&var.attributes) {
        attr_list.push((name, dt, s, data));
    }

    let attr_refs: Vec<(&str, &Datatype, &Shape, &[u8])> = attr_list
        .iter()
        .map(|(n, dt, s, d)| (n.as_str(), dt, s, d.as_slice()))
        .collect();

    writer.add_dataset_with_attributes(
        &var.name,
        &var.datatype,
        &shape,
        &raw_data,
        &DatasetCreationProps::default(),
        &attr_refs,
    )
}

// ---------------------------------------------------------------------------
// write_child_group_content
// ---------------------------------------------------------------------------

/// Write the contents of `group` into the open `sub` write context.
///
/// ## Write order
///
/// 1. Dimension scales (addresses collected into `dim_addrs`).
/// 2. Variables with `DIMENSION_LIST` derived from `dim_addrs`.
/// 3. Child groups (recursive depth-first via [`SubGroupBuilder::begin_sub_group`]).
///
/// Each nested [`SubGroupBuilder`] is consumed by `finish_with_attributes`
/// before the parent builder is used again.
pub(super) fn write_child_group_content(
    sub: &mut SubGroupBuilder<'_>,
    group: &NetcdfGroup,
) -> Result<()> {
    let mut dim_addrs: BTreeMap<String, u64> = BTreeMap::new();
    for (idx, dim) in group.dimensions.iter().enumerate() {
        let addr = write_dimension_scale(sub, dim, idx as u32)?;
        dim_addrs.insert(dim.name.clone(), addr);
    }

    for var in &group.variables {
        write_variable(sub, var, &dim_addrs)?;
    }

    for child_group in &group.groups {
        let mut child_sub = sub.begin_sub_group(&child_group.name);
        write_child_group_content(&mut child_sub, child_group)?;
        let child_attr_list = encode_cf_attrs(&child_group.attributes);
        let child_attr_refs: Vec<(&str, &Datatype, &Shape, &[u8])> = child_attr_list
            .iter()
            .map(|(n, dt, s, d)| (n.as_str(), dt, s, d.as_slice()))
            .collect();
        child_sub.finish_with_attributes(&child_attr_refs)?;
    }

    Ok(())
}
