//! netCDF-4 variable representation.
//!
//! ## Specification
//!
//! A netCDF variable is a named N-dimensional array associated with
//! a set of dimensions. Variables map directly to HDF5 datasets.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::datatype::Datatype;

/// A netCDF-4 variable.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct NetcdfVariable {
    /// Variable name.
    pub name: String,
    /// Data type.
    pub datatype: Datatype,
    /// Ordered list of dimension names.
    pub dimensions: Vec<String>,
}
