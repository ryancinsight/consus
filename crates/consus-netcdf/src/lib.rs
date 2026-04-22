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
#[cfg(feature = "std")]
pub mod hdf5;
pub mod model;
pub mod variable;

#[cfg(feature = "std")]
pub use hdf5::extract_group;
#[cfg(feature = "alloc")]
pub use hdf5::is_dimension_scale;
#[cfg(feature = "std")]
pub use hdf5::variable::read_variable_bytes;

pub use conventions::{
    ADD_OFFSET_ATTR, ANCILLARY_VARIABLES_ATTR, AXIS_ATTR, BOUNDS_ATTR, CALENDAR_ATTR,
    CELL_METHODS_ATTR, COMPRESS_ATTR, CONVENTIONS_ATTR, COORDINATES_ATTR, DIMENSION_SCALE_CLASS,
    DIMENSION_SCALE_VALUE, FILL_VALUE_ATTR, FLAG_MASKS_ATTR, FLAG_MEANINGS_ATTR, FLAG_VALUES_ATTR,
    FORMULA_TERMS_ATTR, GRID_MAPPING_ATTR, LONG_NAME_ATTR, MISSING_VALUE_ATTR,
    NETCDF_CONVENTIONS_VALUE, POSITIVE_ATTR, ROOT_GROUP_NAME, SCALE_FACTOR_ATTR,
    STANDARD_NAME_ATTR, UNITS_ATTR, VALID_MAX_ATTR, VALID_MIN_ATTR, VALID_RANGE_ATTR,
    is_cf_attribute_name, is_dimension_scale_marker, is_recognized_convention, is_valid_name,
    root_group_name,
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
