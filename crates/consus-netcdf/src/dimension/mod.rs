//! netCDF-4 dimension representation.
//!
//! ## Specification
//!
//! A netCDF dimension has a name and a current length. It may be unlimited,
//! which means its current length can grow over time. netCDF-4 maps dimensions
//! to HDF5 dimension scales.
//!
//! ## Invariants
//!
//! - `size` is the current length of the dimension.
//! - `unlimited == true` means the dimension may grow, but `size` remains the
//!   authoritative current extent.
//! - `name` is the stable identifier used by variables and coordinate metadata.

#[cfg(feature = "alloc")]
use alloc::string::String;

use consus_core::{Error, Result};

/// A netCDF-4 dimension.
///
/// This is the canonical in-crate representation used by the netCDF layer.
/// It is intentionally minimal so that format-specific metadata can be mapped
/// through the HDF5 layer without duplicating state.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetcdfDimension {
    /// Dimension name.
    pub name: String,
    /// Current size of the dimension.
    pub size: usize,
    /// Whether this dimension is unlimited.
    pub unlimited: bool,
}

#[cfg(feature = "alloc")]
impl NetcdfDimension {
    /// Create a fixed-size dimension.
    #[must_use]
    pub fn new(name: String, size: usize) -> Self {
        Self {
            name,
            size,
            unlimited: false,
        }
    }

    /// Create an unlimited dimension.
    #[must_use]
    pub fn unlimited(name: String, size: usize) -> Self {
        Self {
            name,
            size,
            unlimited: true,
        }
    }

    /// Whether the dimension can grow.
    #[must_use]
    pub fn is_unlimited(&self) -> bool {
        self.unlimited
    }

    /// Current length of the dimension.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the dimension has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Validate the dimension against netCDF rules.
    ///
    /// A valid dimension:
    /// - has a non-empty name
    /// - has a defined current size
    /// - may be unlimited or fixed, with size unchanged by validation
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from("dimension name must not be empty"),
            });
        }

        Ok(())
    }

    /// Resize the dimension.
    ///
    /// Fixed dimensions reject resizing to a different size.
    pub fn resize(&mut self, new_size: usize) -> Result<()> {
        if !self.unlimited && self.size != new_size {
            return Err(Error::ShapeError {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "cannot resize fixed dimension '{}' from {} to {}",
                    self.name,
                    self.size,
                    new_size
                ),
            });
        }
        self.size = new_size;
        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl core::fmt::Display for NetcdfDimension {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.unlimited {
            write!(f, "{} = {} (unlimited)", self.name, self.size)
        } else {
            write!(f, "{} = {}", self.name, self.size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn fixed_dimension_construction() {
        let dim = NetcdfDimension::new(alloc::string::String::from("time"), 12);
        assert_eq!(dim.name, "time");
        assert_eq!(dim.size, 12);
        assert!(!dim.unlimited);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn unlimited_dimension_construction() {
        let dim = NetcdfDimension::unlimited(alloc::string::String::from("record"), 7);
        assert_eq!(dim.name, "record");
        assert_eq!(dim.size, 7);
        assert!(dim.unlimited);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn fixed_dimension_resize_rejects_change() {
        let mut dim = NetcdfDimension::new(alloc::string::String::from("x"), 4);
        let err = dim.resize(5).unwrap_err();
        match err {
            consus_core::Error::ShapeError { .. } => {}
            other => panic!("expected ShapeError, got {other}"),
        }
        assert_eq!(dim.size, 4);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn unlimited_dimension_resize_allows_change() {
        let mut dim = NetcdfDimension::unlimited(alloc::string::String::from("y"), 2);
        dim.resize(9).unwrap();
        assert_eq!(dim.size, 9);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn validate_rejects_empty_name() {
        let dim = NetcdfDimension::new(alloc::string::String::from(""), 1);
        let err = dim.validate().unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got {other}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn validate_accepts_named_dimension() {
        let dim = NetcdfDimension::new(alloc::string::String::from("latitude"), 180);
        dim.validate().unwrap();
        assert_eq!(dim.len(), 180);
        assert!(!dim.is_empty());
    }
}
