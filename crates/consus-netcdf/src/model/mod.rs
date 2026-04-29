//! netCDF file and group model.
//!
//! This module defines the canonical container model used by `consus-netcdf`.
//! It is a semantic layer over HDF5-backed storage conventions.
//!
//! ## Invariants
//!
//! - Group names are unique within a scope.
//! - Dimension names are unique within a scope.
//! - Variables reference dimensions declared in their own scope or any ancestor scope.
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

/// A user-defined type declared in a netCDF-4 enhanced model group.
///
/// Corresponds to an HDF5 committed (named) datatype child of the group.
///
/// ## netCDF-4 enhanced model
///
/// NUG §2.5: user-defined types (compound, variable-length, opaque, enum)
/// are stored as HDF5 named datatypes linked from the enclosing group.
/// `extract_group` classifies them via `NodeType::NamedDatatype` and reads
/// the canonical `Datatype` via `Hdf5File::named_datatype_at`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfUserType {
    /// Type name (dataset name in the HDF5 group).
    pub name: String,
    /// Canonical datatype representation.
    pub datatype: consus_core::Datatype,
}

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
    /// Group-level attributes (e.g. Conventions, title, history).
    pub attributes: Vec<(String, consus_core::AttributeValue)>,
    /// User-defined types declared in this scope (netCDF-4 enhanced model).
    pub user_types: Vec<NetcdfUserType>,
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
            attributes: Vec::new(),
            user_types: Vec::new(),
        }
    }

    /// Validate the group and its children.
    pub fn validate(&self) -> Result<()> {
        self.validate_with_ancestors(&[] as &[&NetcdfDimension])
    }

    pub(crate) fn validate_with_ancestors(
        &self,
        ancestor_dimensions: &[&NetcdfDimension],
    ) -> Result<()> {
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

        let mut visible_dimensions: Vec<&NetcdfDimension> =
            Vec::with_capacity(ancestor_dimensions.len() + self.dimensions.len());
        let mut ancestor_index = 0;
        while ancestor_index < ancestor_dimensions.len() {
            visible_dimensions.push(ancestor_dimensions[ancestor_index]);
            ancestor_index += 1;
        }

        let mut local_index = 0;
        while local_index < self.dimensions.len() {
            visible_dimensions.push(&self.dimensions[local_index]);
            local_index += 1;
        }

        let mut v = 0;
        while v < self.variables.len() {
            self.variables[v].validate()?;

            let mut d = 0;
            while d < self.variables[v].dimensions.len() {
                if self
                    .resolve_dimension_in_scope(
                        self.variables[v].dimensions[d].as_str(),
                        &visible_dimensions,
                    )
                    .is_none()
                {
                    return Err(Error::InvalidFormat {
                        #[cfg(feature = "alloc")]
                        message: String::from(
                            "variable references a dimension that is not visible in the current scope chain",
                        ),
                    });
                }
                d += 1;
            }

            v += 1;
        }

        let mut g = 0;
        while g < self.groups.len() {
            self.groups[g].validate_with_ancestors(&visible_dimensions)?;
            g += 1;
        }

        Ok(())
    }

    fn resolve_dimension_in_scope<'a>(
        &self,
        name: &str,
        visible_dimensions: &'a [&'a NetcdfDimension],
    ) -> Option<&'a NetcdfDimension> {
        let mut i = visible_dimensions.len();
        while i > 0 {
            i -= 1;
            if visible_dimensions[i].name == name {
                return Some(visible_dimensions[i]);
            }
        }
        None
    }

    /// Find a dimension by name in the current scope chain.
    ///
    /// Local dimensions shadow dimensions declared in ancestor groups.
    #[must_use]
    pub fn resolve_dimension<'a>(
        &'a self,
        name: &str,
        ancestors: &[&'a NetcdfGroup],
    ) -> Option<&'a NetcdfDimension> {
        let mut local_index = 0;
        while local_index < self.dimensions.len() {
            if self.dimensions[local_index].name == name {
                return Some(&self.dimensions[local_index]);
            }
            local_index += 1;
        }

        let mut ancestor_index = ancestors.len();
        while ancestor_index > 0 {
            ancestor_index -= 1;
            let ancestor = ancestors[ancestor_index];
            let mut dimension_index = 0;
            while dimension_index < ancestor.dimensions.len() {
                if ancestor.dimensions[dimension_index].name == name {
                    return Some(&ancestor.dimensions[dimension_index]);
                }
                dimension_index += 1;
            }
        }

        None
    }

    /// Find a dimension by name in the local scope.
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

    /// Find a dimension by name in the current scope chain.
    #[must_use]
    pub fn dimension_in_scope<'a>(
        &'a self,
        name: &str,
        ancestor_dimensions: &[&'a NetcdfDimension],
    ) -> Option<&'a NetcdfDimension> {
        let mut i = 0;
        while i < self.dimensions.len() {
            if self.dimensions[i].name == name {
                return Some(&self.dimensions[i]);
            }
            i += 1;
        }

        let mut j = ancestor_dimensions.len();
        while j > 0 {
            j -= 1;
            if ancestor_dimensions[j].name == name {
                return Some(ancestor_dimensions[j]);
            }
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

    /// Find a child group by name.
    #[must_use]
    pub fn group(&self, name: &str) -> Option<&NetcdfGroup> {
        let mut i = 0;
        while i < self.groups.len() {
            if self.groups[i].name == name {
                return Some(&self.groups[i]);
            }
            i += 1;
        }
        None
    }

    /// Returns `true` if the group contains no declarations, no attributes, and no children.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
            && self.dimensions.is_empty()
            && self.variables.is_empty()
            && self.attributes.is_empty()
            && self.user_types.is_empty()
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

    #[cfg(feature = "alloc")]
    #[test]
    fn group_validation_accepts_ancestor_scoped_dimension_reference() {
        let mut root = NetcdfGroup::new(String::from("/"));
        root.dimensions
            .push(NetcdfDimension::new(String::from("time"), 4));

        let mut child = NetcdfGroup::new(String::from("observations"));
        child.variables.push(NetcdfVariable::new(
            String::from("temperature"),
            consus_core::Datatype::Boolean,
            vec![String::from("time")],
        ));

        root.groups.push(child);

        root.validate().unwrap();
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn group_validation_rejects_missing_ancestor_scoped_dimension_reference() {
        let mut root = NetcdfGroup::new(String::from("/"));

        let mut child = NetcdfGroup::new(String::from("observations"));
        child.variables.push(NetcdfVariable::new(
            String::from("temperature"),
            consus_core::Datatype::Boolean,
            vec![String::from("time")],
        ));

        root.groups.push(child);

        let err = root.validate().unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got {other}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn dimension_in_scope_prefers_local_shadowing_dimension() {
        let mut root = NetcdfGroup::new(String::from("/"));
        root.dimensions
            .push(NetcdfDimension::new(String::from("time"), 4));

        let mut child = NetcdfGroup::new(String::from("observations"));
        child
            .dimensions
            .push(NetcdfDimension::new(String::from("time"), 8));

        let ancestor_dimensions = vec![&root.dimensions[0]];
        let resolved = child
            .dimension_in_scope("time", &ancestor_dimensions)
            .expect("dimension must resolve");

        assert_eq!(resolved.size, 8);
        assert!(!resolved.unlimited);
    }
}
