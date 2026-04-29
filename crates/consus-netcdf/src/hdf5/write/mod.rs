//! netCDF-4 HDF5 write path.
//!
//! ## Specification
//!
//! A [`NetcdfModel`] is emitted as a valid HDF5 file by:
//!
//! 1. Writing the `_nc_properties` root-group attribute marking the file as
//!    netCDF-4 (`"version=2,netcdf=4.x.x"`).
//! 2. Writing each root-group dimension as a 1-D coordinate-index dataset
//!    carrying `CLASS = "DIMENSION_SCALE"` and `NAME = {dim_name}` attributes;
//!    collecting the returned object-header address per dimension.
//! 3. Writing each root-group variable as a contiguous dataset carrying a
//!    `DIMENSION_LIST` attribute whose entries are the object-reference
//!    addresses of the corresponding dimension scale datasets in axis order.
//! 4. Writing each root-group child group recursively via
//!    [`SubGroupBuilder`], propagating dimensions, variables, and nested
//!    groups depth-first.
//!
//! ## Invariants
//!
//! - Dimension scales are written before variables so their object-header
//!   addresses are available when building `DIMENSION_LIST` payloads.
//! - `DIMENSION_LIST` entries are 8-byte little-endian object-header
//!   addresses encoded as `Datatype::Reference(ReferenceType::Object)`.
//! - Scalar variables (rank 0) do not carry a `DIMENSION_LIST` attribute.
//! - Variable data bytes are zero-filled; fill-value HDF5 message emission
//!   is deferred to a future milestone.
//! - Root-group `_nc_properties` is present in every emitted file.
//!
//! ## Supported model subset (M-043)
//!
//! - Root-level and nested-group dimensions and variables (enhanced netCDF-4
//!   model with arbitrary sub-group depth).
//! - Fixed-size element datatypes only; variable-length types emit an empty
//!   scalar dataset.
//! - All supported `AttributeValue` variants (`Int`, `Uint`, `Float`,
//!   `String`, `IntArray`, `UintArray`, `FloatArray`, `StringArray`) are
//!   propagated as HDF5 attributes on variables and groups.
//! - `Bytes`-typed attributes are skipped (no matching HDF5 datatype).

mod helpers;

use alloc::{collections::BTreeMap, string::String, vec::Vec};

use consus_core::{Datatype, Result, Shape, StringEncoding};
use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};

use crate::conventions::{NC_PROPERTIES_ATTR, NC_PROPERTIES_VALUE};
use crate::model::NetcdfModel;

use helpers::{encode_cf_attrs, write_child_group_content, write_dimension_scale, write_variable};

/// HDF5 writer for [`NetcdfModel`] instances.
///
/// Emits a valid HDF5 file encoding the netCDF-4 model (root-level and
/// nested-group dimensions + variables, arbitrary sub-group depth).
///
/// ## Example
///
/// ```rust
/// use consus_netcdf::{NetcdfModel, NetcdfWriter};
///
/// let model = NetcdfModel::default();
/// let bytes = NetcdfWriter::new().write_model(&model).unwrap();
/// assert!(!bytes.is_empty());
/// ```
pub struct NetcdfWriter {
    builder: Hdf5FileBuilder,
}

impl Default for NetcdfWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl NetcdfWriter {
    /// Create a new writer with default HDF5 file creation properties.
    #[must_use]
    pub fn new() -> Self {
        Self {
            builder: Hdf5FileBuilder::new(FileCreationProps::default()),
        }
    }

    /// Emit the complete HDF5 byte image for `model`.
    ///
    /// ## Write order
    ///
    /// 1. Root-group `_nc_properties` attribute.
    /// 2. Root dimension scale datasets (addresses collected into `dim_addrs`).
    /// 3. Root variable datasets with `DIMENSION_LIST` attributes.
    /// 4. Child groups (recursive, depth-first via [`SubGroupBuilder`]).
    ///
    /// ## Errors
    ///
    /// Returns `Error::UnsupportedFeature` if any datatype cannot be encoded
    /// by the HDF5 writer (e.g. variable-length or compound types).
    pub fn write_model(mut self, model: &NetcdfModel) -> Result<Vec<u8>> {
        // 1. Root-group _nc_properties attribute.
        let nc_props_dt = Datatype::FixedString {
            length: NC_PROPERTIES_VALUE.len(),
            encoding: StringEncoding::Ascii,
        };
        self.builder.add_root_attribute(
            NC_PROPERTIES_ATTR,
            &nc_props_dt,
            &Shape::scalar(),
            NC_PROPERTIES_VALUE.as_bytes(),
        )?;

        // 2. Root dimension scales: write each, collect dim_name → header_addr.
        let mut dim_addrs: BTreeMap<String, u64> = BTreeMap::new();
        for (idx, dim) in model.root.dimensions.iter().enumerate() {
            let addr = write_dimension_scale(&mut self.builder, dim, idx as u32)?;
            dim_addrs.insert(dim.name.clone(), addr);
        }

        // 3. Root variables: write each with DIMENSION_LIST.
        for var in &model.root.variables {
            write_variable(&mut self.builder, var, &dim_addrs)?;
        }

        // 4. Child groups (enhanced model).
        for child_group in &model.root.groups {
            let mut sub = self.builder.begin_group(&child_group.name);
            write_child_group_content(&mut sub, child_group)?;
            let child_attr_list = encode_cf_attrs(&child_group.attributes);
            let child_attr_refs: Vec<(&str, &Datatype, &Shape, &[u8])> = child_attr_list
                .iter()
                .map(|(n, dt, s, d)| (n.as_str(), dt, s, d.as_slice()))
                .collect();
            sub.finish_with_attributes(&child_attr_refs)?;
        }

        self.builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use consus_core::{ByteOrder, Datatype, Shape};
    use core::num::NonZeroUsize;

    fn f32_dt() -> Datatype {
        Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        }
    }

    fn i32_le_dt() -> Datatype {
        Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }
    }

    /// write_model on a default (empty) model succeeds and produces non-empty bytes.
    #[test]
    fn write_empty_model_produces_bytes() {
        let model = NetcdfModel::default();
        let bytes = NetcdfWriter::new()
            .write_model(&model)
            .expect("write_model must succeed for empty model");
        assert!(!bytes.is_empty(), "output must not be empty");
    }

    /// write_model with a single dimension produces valid bytes.
    #[test]
    fn write_single_dimension_produces_bytes() {
        let mut model = NetcdfModel::default();
        model
            .root
            .dimensions
            .push(crate::dimension::NetcdfDimension::new(
                String::from("time"),
                5,
            ));
        let bytes = NetcdfWriter::new()
            .write_model(&model)
            .expect("write_model must succeed with one dimension");
        assert!(!bytes.is_empty());
    }

    /// write_model with a variable over a declared dimension produces valid bytes.
    #[test]
    fn write_variable_with_dimension_produces_bytes() {
        let mut model = NetcdfModel::default();
        model
            .root
            .dimensions
            .push(crate::dimension::NetcdfDimension::new(String::from("x"), 4));
        model.root.variables.push(
            crate::variable::NetcdfVariable::new(
                String::from("depth"),
                f32_dt(),
                vec![String::from("x")],
            )
            .with_shape(Shape::fixed(&[4])),
        );
        let bytes = NetcdfWriter::new()
            .write_model(&model)
            .expect("write_model must succeed with variable");
        assert!(!bytes.is_empty());
    }

    /// write_model with a scalar variable (rank 0) produces valid bytes.
    #[test]
    fn write_scalar_variable_produces_bytes() {
        let mut model = NetcdfModel::default();
        model.root.variables.push(
            crate::variable::NetcdfVariable::new(
                String::from("global_constant"),
                i32_le_dt(),
                vec![],
            )
            .with_shape(Shape::scalar()),
        );
        let bytes = NetcdfWriter::new()
            .write_model(&model)
            .expect("write_model must succeed for scalar variable");
        assert!(!bytes.is_empty());
    }
}
