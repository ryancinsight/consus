#![cfg_attr(not(feature = "std"), no_std)]

//! # consus-netcdf
//!
//! Authoritative netCDF-4 model and HDF5 semantic mapping layer.
//!
//! ## Scope
//!
//! This crate defines the canonical in-memory model for netCDF concepts:
//! - dimensions
//! - variables
//! - groups
//! - file-level models
//! - HDF5 mapping descriptors
//!
//! The implementation is a thin semantic layer over `consus-core` and
//! `consus-hdf5` conventions. It does not duplicate storage logic.
//!
//! ## Invariants
//!
//! - Dimension names are unique within a group scope.
//! - Variable dimension order is stable and rank-preserving.
//! - Coordinate variables are rank-1 variables whose name matches their single dimension name.
//! - Unlimited dimensions are represented explicitly and remain growable.
//! - HDF5 mapping descriptors are derived from the canonical netCDF model.
//!
//! ## Module Structure
//!
//! ```text
//! consus-netcdf
//! ├── conventions/     # netCDF and CF constants + validation
//! ├── dimension/       # dimension descriptor
//! ├── variable/        # variable descriptor
//! └── model/           # group/file model + HDF5 mappings
//! ```
//!
//! ## Status
//!
//! This is the compileable authoritative crate root for the netCDF model.

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod conventions;
pub mod dimension;
pub mod model;
pub mod variable;

pub use conventions::{
    is_cf_attribute_name, is_dimension_scale_marker, is_recognized_convention, is_valid_name,
    root_group_name, AXIS_ATTR, BOUNDS_ATTR, CELL_METHODS_ATTR, COORDINATES_ATTR,
    CONVENTIONS_ATTR, DIMENSION_SCALE_CLASS, DIMENSION_SCALE_VALUE, FILL_VALUE_ATTR,
    GRID_MAPPING_ATTR, LONG_NAME_ATTR, NETCDF_CONVENTIONS_VALUE, ROOT_GROUP_NAME,
    STANDARD_NAME_ATTR, UNITS_ATTR,
};

pub use dimension::NetcdfDimension;
pub use model::{
    Hdf5DimensionMapping, Hdf5GroupMapping, Hdf5VariableMapping, NetcdfGroup, NetcdfModel,
};
pub use variable::NetcdfVariable;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_exports_dimension_scale_marker() {
        assert!(is_dimension_scale_marker(
            DIMENSION_SCALE_CLASS,
            DIMENSION_SCALE_VALUE
        ));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn root_model_constructs() {
        let model = NetcdfModel::default();
        assert_eq!(model.root.name, ROOT_GROUP_NAME);
    }
}
