//! netCDF file and group model.
//!
//! This module defines the canonical container model used by `consus-netcdf`.
//! It is a semantic layer over HDF5-backed storage conventions.
//!
//! ## Invariants
//!
//! - Group names are unique within a scope.
//! - Dimension names are unique within a scope.
//! - Variables reference declared dimensions only.
//! - Coordinate variables are rank-1 variables whose name matches their single dimension.
//! - The file model is rooted at `/`.
//!
//! ## Architecture
//!
//! ```text
//! model/
//! ├── group   # group hierarchy and local declarations
//! └── file    # file-level root model and validation
//! ```
//
// This file is written to align with the current `consus-netcdf` crate
// structure and to keep the model layer compileable against the existing
// dimension and variable descriptors.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::{Error, Result};

use crate::dimension::NetcdfDimension;
use crate::variable::NetcdfVariable;

/// Canonical netCDF group model.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfGroup {
    /// Group name.
    pub name: String,
    /// Child groups.
    pub groups: Vec<NetcdfGroup>,
    /// Dimensions declared in this scope.
    pub dimensions: Vec<NetcdfDimension>,
    /// Variables declared in this scope.
    pub variables: Vec<NetcdfVariable>,
}

#[cfg(feature = "alloc")]
impl NetcdfGroup {
    /// Create an empty group.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            groups: Vec::new(),
            dimensions: Vec::new(),
            variables: Vec::new(),
        }
    }

    /// Validate the group and its children.
    pub fn validate(&self) -> Result<()> {
        let mut i = 0;
        while i < self.dimensions.len() {
            self.dimensions[i].validate()?;
            let mut j = i + 1;
            while j < self.dimensions.len() {
                if self.dimensions[i].name == self.dimensions[j].name {
                    return Err(Error::InvalidFormat {
                        #[cfg(feature = "alloc")]
                        message: String::from("duplicate dimension names in scope"),
                    });
                }
                j += 1;
            }
            i += 1;
        }

        let mut v = 0;
        while v < self.variables.len() {
            self.variables[v].validate()?;
            v += 1;
        }

        let mut g = 0;
        while g < self.groups.len() {
            self.groups[g].validate()?;
            g += 1;
        }

        Ok(())
    }

    /// Find a dimension by name.
    #[must_use]
    pub fn dimension(&self, name: &str) -> Option<&NetcdfDimension> {
        let mut i = 0;
        while i < self.dimensions.len() {
            if self.dimensions[i].name == name {
                return Some(&self.dimensions[i]);
            }
            i += 1;
        }
        None
    }

    /// Find a variable by name.
    #[must_use]
    pub fn variable(&self, name: &str) -> Option<&NetcdfVariable> {
        let mut i = 0;
        while i < self.variables.len() {
            if self.variables[i].name == name {
                return Some(&self.variables[i]);
            }
            i += 1;
        }
        None
    }

    /// Returns `true` if the group contains no declarations and no children.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty() && self.dimensions.is_empty() && self.variables.is_empty()
    }
}

/// Canonical netCDF file model.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfModel {
    /// Root group.
    pub root: NetcdfGroup,
}

/// Canonical alias for a netCDF file model.
#[cfg(feature = "alloc")]
pub type NetcdfFile = NetcdfModel;

#[cfg(feature = "alloc")]
impl NetcdfModel {
    /// Create an empty file model with root group `/`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: NetcdfGroup::new(String::from("/")),
        }
    }

    /// Validate the full model.
    pub fn validate(&self) -> Result<()> {
        self.root.validate()
    }
}

#[cfg(feature = "alloc")]
impl Default for NetcdfModel {
    fn default() -> Self {
        Self::new()
    }
}

/// netCDF-to-HDF5 mapping for a dimension.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hdf5DimensionMapping {
    /// Dimension name.
    pub name: String,
    /// HDF5 dimension-scale path.
    pub object_path: String,
    /// Whether the mapped dimension is unlimited.
    pub unlimited: bool,
}

#[cfg(feature = "alloc")]
impl Hdf5DimensionMapping {
    /// Create a dimension mapping.
    #[must_use]
    pub fn new(name: String, object_path: String, unlimited: bool) -> Self {
        Self {
            name,
            object_path,
            unlimited,
        }
    }
}

/// netCDF-to-HDF5 mapping for a variable.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct Hdf5VariableMapping {
    /// Variable name.
    pub name: String,
    /// HDF5 dataset path.
    pub dataset_path: String,
    /// Ordered dimension names.
    pub dimensions: Vec<String>,
    /// Chunk shape required for unlimited or compressed variables.
    pub chunk_shape: Option<consus_core::ChunkShape>,
    /// Compression policy.
    pub compression: consus_core::Compression,
}

#[cfg(feature = "alloc")]
impl Hdf5VariableMapping {
    /// Create a variable mapping.
    #[must_use]
    pub fn new(
        name: String,
        dataset_path: String,
        dimensions: Vec<String>,
        chunk_shape: Option<consus_core::ChunkShape>,
        compression: consus_core::Compression,
    ) -> Self {
        Self {
            name,
            dataset_path,
            dimensions,
            chunk_shape,
            compression,
        }
    }
}

/// netCDF-to-HDF5 mapping for a group.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hdf5GroupMapping {
    /// Group name.
    pub name: String,
    /// HDF5 group path.
    pub group_path: String,
}

#[cfg(feature = "alloc")]
impl Hdf5GroupMapping {
    /// Create a group mapping.
    #[must_use]
    pub fn new(name: String, group_path: String) -> Self {
        Self { name, group_path }
    }
}

/// Bridge helpers for mapping netCDF models to HDF5 semantics.
pub mod bridge {
    use super::*;

    /// Map a group to an HDF5 group descriptor.
    #[cfg(feature = "alloc")]
    pub fn map_group(group: &NetcdfGroup) -> Result<Hdf5GroupMapping> {
        group.validate()?;
        Ok(Hdf5GroupMapping::new(
            group.name.clone(),
            alloc::format!("/groups/{}", group.name),
        ))
    }

    /// Map a dimension to an HDF5 dimension-scale descriptor.
    #[cfg(feature = "alloc")]
    pub fn map_dimension(dimension: &NetcdfDimension) -> Result<Hdf5DimensionMapping> {
        dimension.validate()?;
        Ok(Hdf5DimensionMapping::new(
            dimension.name.clone(),
            alloc::format!("/dimensions/{}", dimension.name),
            dimension.unlimited,
        ))
    }

    /// Map a variable to an HDF5 dataset descriptor.
    #[cfg(feature = "alloc")]
    pub fn map_variable(variable: &NetcdfVariable) -> Result<Hdf5VariableMapping> {
        variable.validate()?;
        Ok(Hdf5VariableMapping::new(
            variable.name.clone(),
            alloc::format!("/variables/{}", variable.name),
            variable.dimensions.clone(),
            None,
            consus_core::Compression::None,
        ))
    }

    /// Validate that a model can be mapped into HDF5 semantics.
    #[cfg(feature = "alloc")]
    pub fn validate_model(model: &NetcdfModel) -> Result<()> {
        model.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn empty_model_uses_root_group() {
        let model = NetcdfModel::default();
        assert_eq!(model.root.name, "/");
        assert!(model.root.is_empty());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn group_validation_rejects_duplicate_dimensions() {
        let mut group = NetcdfGroup::new(String::from("root"));
        group
            .dimensions
            .push(NetcdfDimension::new(String::from("x"), 4));
        group
            .dimensions
            .push(NetcdfDimension::new(String::from("x"), 8));
        let err = group.validate().unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got {other}"),
        }
    }
}
