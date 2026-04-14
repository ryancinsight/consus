//! netCDF-4 dimension representation.
//!
//! ## Specification
//!
//! A netCDF dimension has a name and a length. It may be unlimited
//! (growable). In netCDF-4, dimensions are mapped to HDF5 dimension
//! scales.

#[cfg(feature = "alloc")]
use alloc::string::String;

/// A netCDF-4 dimension.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetcdfDimension {
    /// Dimension name.
    pub name: String,
    /// Current size.
    pub size: usize,
    /// Whether this dimension is unlimited.
    pub unlimited: bool,
}
