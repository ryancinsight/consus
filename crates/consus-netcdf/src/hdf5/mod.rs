//! HDF5 read layer for netCDF-4 semantic model extraction.
//!
//! ## Architecture
//!
//! This module bridges the low-level HDF5 object model exposed by
//! `consus-hdf5` to the canonical netCDF-4 semantic model in this crate.
//!
//! ### Traversal contract
//!
//! 1. A dataset is a dimension scale iff its `CLASS` attribute equals
//!    `"DIMENSION_SCALE"` (per HDF5 Dimension Scales convention).
//! 2. Variables reference only declared dimensions.
//! 3. The root group path is `/`.
//!
//! ## Module Hierarchy
//!
//! ```text
//! hdf5/
//! +-- dimension_scale/  # Detect and extract dimension scale datasets
//! +-- variable/         # Extract NetcdfVariable from HDF5 datasets
//! +-- group/            # Traverse HDF5 group hierarchy -> NetcdfGroup
//! ```

#[cfg(feature = "alloc")]
pub mod dimension_scale;
#[cfg(feature = "std")]
pub mod group;
#[cfg(feature = "alloc")]
pub mod variable;
#[cfg(feature = "std")]
pub mod write;

#[cfg(feature = "alloc")]
pub use dimension_scale::is_dimension_scale;
#[cfg(feature = "std")]
pub use group::{extract_group, read_model};
#[cfg(feature = "std")]
pub use variable::read_variable_bytes;
