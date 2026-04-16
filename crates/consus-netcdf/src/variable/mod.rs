//! netCDF-4 variable representation.
//!
//! ## Specification
//!
//! A netCDF variable is a named N-dimensional array with:
//! - an element datatype
//! - an ordered list of dimension names
//! - optional resolved shape
//! - optional fill value
//! - optional compression policy
//! - coordinate-variable and unlimited-dimension flags
//!
//! netCDF-4 variables map to HDF5 datasets with domain conventions.
//! This module provides the canonical variable descriptor used by the
//! netCDF crate; it does not perform file I/O itself.
//!
//! ## Invariants
//!
//! - `name` is non-empty.
//! - `dimensions` preserves declaration order.
//! - `shape.rank() == dimensions.len()` when `shape` is present.
//! - `fill_value`, when present, must encode a value compatible with
//!   `datatype`.
//! - Coordinate variables have rank 1 and their single dimension name
//!   matches the variable name.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::{Compression, Datatype, Error, Result, Shape};

/// A netCDF-4 variable descriptor.
///
/// This type captures the stable semantic model of a variable without
/// binding to a physical backend. The backend mapping is:
///
/// - netCDF variable -> HDF5 dataset
/// - dimension names -> HDF5 dimension scales
/// - unlimited dimension -> chunked, growable HDF5 dimension
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfVariable {
    /// Variable name.
    pub name: String,
    /// Variable datatype.
    pub datatype: Datatype,
    /// Ordered dimension names.
    pub dimensions: Vec<String>,
    /// Optional resolved shape.
    pub shape: Option<Shape>,
    /// Optional fill value encoded in backend-specific raw bytes.
    pub fill_value: Option<Vec<u8>>,
    /// Optional compression policy.
    pub compression: Option<Compression>,
    /// Whether the variable is a coordinate variable.
    pub coordinate_variable: bool,
    /// Whether the variable may grow along one or more unlimited dimensions.
    pub unlimited: bool,
}

#[cfg(feature = "alloc")]
impl NetcdfVariable {
    /// Create a variable descriptor with the required fields.
    #[must_use]
    pub fn new(name: String, datatype: Datatype, dimensions: Vec<String>) -> Self {
        Self {
            name,
            datatype,
            dimensions,
            shape: None,
            fill_value: None,
            compression: None,
            coordinate_variable: false,
            unlimited: false,
        }
    }

    /// Attach a resolved shape.
    #[must_use]
    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Attach a raw fill value.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: Vec<u8>) -> Self {
        self.fill_value = Some(fill_value);
        self
    }

    /// Attach a compression policy.
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Mark the variable as a coordinate variable.
    #[must_use]
    pub fn coordinate_variable(mut self, value: bool) -> Self {
        self.coordinate_variable = value;
        self
    }

    /// Mark the variable as unlimited.
    #[must_use]
    pub fn unlimited(mut self, value: bool) -> Self {
        self.unlimited = value;
        self
    }

    /// Returns `true` if the variable is scalar.
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }

    /// Returns the declared rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("variable name must not be empty"),
            });
        }

        if self.coordinate_variable {
            if self.dimensions.len() != 1 || self.dimensions[0] != self.name {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: String::from(
                        "coordinate variables must be rank-1 and match their dimension name",
                    ),
                });
            }
        }

        if let Some(shape) = &self.shape {
            if shape.rank() != self.dimensions.len() {
                return Err(Error::ShapeError {
                    #[cfg(feature = "alloc")]
                    message: String::from("variable rank does not match the referenced dimensions"),
                });
            }
        }

        if self.unlimited && self.dimensions.is_empty() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("unlimited variables must reference at least one dimension"),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn variable_construction_works() {
        let variable = NetcdfVariable::new(
            String::from("temperature"),
            Datatype::Boolean,
            vec![String::from("time"), String::from("station")],
        )
        .with_compression(Compression::None)
        .unlimited(true);

        assert_eq!(variable.name, "temperature");
        assert_eq!(variable.rank(), 2);
        assert!(variable.unlimited);
        assert!(!variable.is_scalar());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn validate_rejects_empty_name() {
        let variable =
            NetcdfVariable::new(String::from(""), Datatype::Boolean, vec![String::from("x")]);
        let err = variable.validate().unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got {other}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn validate_rejects_coordinate_mismatch() {
        let variable = NetcdfVariable::new(
            String::from("time"),
            Datatype::Boolean,
            vec![String::from("x")],
        )
        .coordinate_variable(true);

        let err = variable.validate().unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got {other}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn validate_accepts_coordinate_variable() {
        let variable = NetcdfVariable::new(
            String::from("time"),
            Datatype::Boolean,
            vec![String::from("time")],
        )
        .coordinate_variable(true);

        variable.validate().unwrap();
        assert!(variable.coordinate_variable);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn validate_rejects_shape_mismatch() {
        let variable = NetcdfVariable::new(
            String::from("x"),
            Datatype::Boolean,
            vec![String::from("x"), String::from("y")],
        )
        .with_shape(Shape::fixed(&[3]));

        let err = variable.validate().unwrap_err();
        match err {
            Error::ShapeError { .. } => {}
            other => panic!("expected ShapeError, got {other}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn validate_rejects_unlimited_scalar() {
        let variable = NetcdfVariable::new(String::from("scalar"), Datatype::Boolean, Vec::new())
            .unlimited(true);

        let err = variable.validate().unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got {other}"),
        }
    }
}
