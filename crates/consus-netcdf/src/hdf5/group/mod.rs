//! HDF5 group traversal and netCDF model extraction.
//!
//! ## Spec
//!
//! Mapping from an HDF5 group at `group_addr` to a `NetcdfGroup`:
//!
//! 1. List direct children with `file.list_group_at(group_addr)`.
//! 2. Classify each child with `file.node_type_at(child_addr)`.
//! 3. **Dataset** children:
//!    - Load attributes with `file.attributes_at(child_addr)`.
//!    - If `is_dimension_scale(&attrs)` -> create a `NetcdfDimension`.
//!    - Otherwise -> create a `NetcdfVariable` via `build_variable`.
//! 4. **Group** children: recurse with `extract_group`.
//! 5. **NamedDatatype** children: skipped (no netCDF equivalent).
//!
//! ## Invariants
//!
//! - Dimension size is `shape.current_dims()[0]` for rank >= 1, or
//!   `shape.num_elements()` for rank 0 (scalar dimension scale).
//! - Dimension name is resolved via `dimension_name_from_attrs` with the
//!   dataset name as fallback.
//! - Variable dimension names are resolved from `DIMENSION_LIST` object
//!   references when present and valid; otherwise they fall back to
//!   synthetic (`"d0"`, `"d1"`, ...) names.
//! - Group and dataset attributes are decoded and preserved when possible.
//! - Unlimited HDF5 extents are propagated into the netCDF dimension model.
//! - Errors on any individual child are propagated immediately.

#[cfg(feature = "alloc")]
use alloc::{collections::BTreeMap, format, string::String, vec::Vec};

use consus_core::{Datatype, NodeType, ReferenceType, Result};
use consus_hdf5::attribute::{Hdf5Attribute, decode_attribute_value};
use consus_io::ReadAt;

use crate::dimension::NetcdfDimension;
use crate::model::NetcdfGroup;

use super::dimension_scale::{dimension_name_from_attrs, is_dimension_scale};
use super::variable::build_variable;

#[cfg(feature = "alloc")]
fn synthetic_dimension_names(rank: usize) -> Vec<String> {
    (0..rank).map(|i| format!("d{i}")).collect()
}

#[cfg(feature = "alloc")]
fn dimension_names_from_dimension_list(
    attrs: &[Hdf5Attribute],
    rank: usize,
    dimension_scale_names: &BTreeMap<u64, String>,
) -> Option<Vec<String>> {
    let attr = attrs.iter().find(|attr| attr.name == "DIMENSION_LIST")?;
    if attr.shape.rank() > 1 {
        return None;
    }
    if attr.datatype != Datatype::Reference(ReferenceType::Object) {
        return None;
    }

    let expected_len = rank.checked_mul(8)?;
    if attr.raw_data.len() < expected_len {
        return None;
    }

    let mut names = Vec::with_capacity(rank);
    for axis in 0..rank {
        let start = axis * 8;
        let end = start + 8;
        let address = u64::from_le_bytes(attr.raw_data[start..end].try_into().ok()?);
        let name = dimension_scale_names.get(&address)?.clone();
        names.push(name);
    }

    Some(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use consus_core::{Datatype, ReferenceType, Shape};

    fn make_dimension_list_attr(addresses: &[u64]) -> Hdf5Attribute {
        let raw_data: Vec<u8> = addresses
            .iter()
            .flat_map(|address| address.to_le_bytes())
            .collect();

        Hdf5Attribute {
            name: String::from("DIMENSION_LIST"),
            datatype: Datatype::Reference(ReferenceType::Object),
            shape: Shape::fixed(&[addresses.len()]),
            raw_data,
            name_encoding: 0,
            creation_order: None,
        }
    }

    #[test]
    fn dimension_names_from_dimension_list_resolves_axis_order() {
        let attrs = vec![make_dimension_list_attr(&[0x200, 0x100])];
        let mut mapping = BTreeMap::new();
        mapping.insert(0x100, String::from("x"));
        mapping.insert(0x200, String::from("time"));

        assert_eq!(
            dimension_names_from_dimension_list(&attrs, 2, &mapping),
            Some(vec![String::from("time"), String::from("x")])
        );
    }

    #[test]
    fn dimension_names_from_dimension_list_rejects_missing_reference() {
        let attrs = vec![make_dimension_list_attr(&[0x200, 0x999])];
        let mut mapping = BTreeMap::new();
        mapping.insert(0x100, String::from("x"));
        mapping.insert(0x200, String::from("time"));

        assert_eq!(
            dimension_names_from_dimension_list(&attrs, 2, &mapping),
            None
        );
    }

    #[test]
    fn dimension_names_from_dimension_list_falls_back_on_short_payload() {
        let attrs = vec![Hdf5Attribute {
            name: String::from("DIMENSION_LIST"),
            datatype: Datatype::Reference(ReferenceType::Object),
            shape: Shape::fixed(&[2]),
            raw_data: vec![1, 2, 3, 4],
            name_encoding: 0,
            creation_order: None,
        }];
        let mapping = BTreeMap::new();

        assert_eq!(
            dimension_names_from_dimension_list(&attrs, 2, &mapping),
            None
        );
    }
}

/// Extract a `NetcdfGroup` from the HDF5 group at `group_addr`.
///
/// ## Parameters
///
/// - `file`: open HDF5 file. `R` must satisfy `ReadAt + Sync` so that the
///   per-child `attributes_at`, `node_type_at`, `dataset_at`, and
///   `fill_value_at` methods are available.
/// - `group_name`: the label for this group (e.g. `"/"` for root, or the
///   child name passed from a parent `extract_group` call).
/// - `group_addr`: HDF5 object header address of the group.
///
/// ## Postconditions
///
/// The returned `NetcdfGroup` is populated with:
/// - `dimensions` from dimension-scale datasets.
/// - `variables` from non-dimension-scale datasets.
/// - `groups` from nested HDF5 groups (recursive).
///
/// ## Errors
///
/// Propagates any `consus_core::Error` from the `Hdf5File` API.
#[cfg(feature = "std")]
pub fn extract_group<R>(
    file: &consus_hdf5::file::Hdf5File<R>,
    group_name: String,
    group_addr: u64,
) -> Result<NetcdfGroup>
where
    R: ReadAt + Sync,
{
    let mut group = NetcdfGroup::new(group_name);

    if let Ok(group_attrs) = file.attributes_at(group_addr) {
        for attr in &group_attrs {
            if let Ok(value) = decode_attribute_value(&attr.raw_data, &attr.datatype, &attr.shape) {
                group.attributes.push((attr.name.clone(), value));
            }
        }
    }

    // list_group_at falls back to v1 when v2 returns empty; v2-format
    // groups with no children have no SYMBOL_TABLE message, producing
    // InvalidFormat("v1 group missing symbol table message").  This is
    // structurally correct for an empty group written by the v2 writer;
    // treat it as an empty child list rather than propagating the error.
    let children = match file.list_group_at(group_addr) {
        Ok(c) => c,
        Err(consus_core::Error::InvalidFormat { message })
            if message.contains("v1 group missing symbol table") =>
        {
            alloc::vec![]
        }
        Err(e) => return Err(e),
    };

    let mut dimension_scale_names: BTreeMap<u64, String> = BTreeMap::new();
    let mut variable_children: Vec<(String, u64, Vec<Hdf5Attribute>)> = Vec::new();

    for (child_name, child_addr, _link_type) in children {
        let node_type = file.node_type_at(child_addr)?;
        match node_type {
            NodeType::Dataset => {
                let attrs = file.attributes_at(child_addr)?;
                if is_dimension_scale(&attrs) {
                    let dim_name = dimension_name_from_attrs(&attrs, &child_name);
                    let dataset = file.dataset_at(child_addr)?;
                    let size = if dataset.shape.rank() >= 1 {
                        dataset.shape.current_dims()[0]
                    } else {
                        dataset.shape.num_elements()
                    };
                    let dimension = if dataset
                        .shape
                        .extents()
                        .iter()
                        .any(|extent| extent.is_unlimited())
                    {
                        NetcdfDimension::unlimited(dim_name.clone(), size)
                    } else {
                        NetcdfDimension::new(dim_name.clone(), size)
                    };
                    dimension_scale_names.insert(child_addr, dim_name);
                    group.dimensions.push(dimension);
                } else {
                    variable_children.push((child_name, child_addr, attrs));
                }
            }
            NodeType::Group => {
                let child_group = extract_group(file, child_name, child_addr)?;
                group.groups.push(child_group);
            }
            NodeType::NamedDatatype => {
                // Named datatypes have no netCDF semantic equivalent; skip.
            }
        }
    }

    for (child_name, child_addr, attrs) in variable_children {
        let dataset = file.dataset_at(child_addr)?;
        let fill_bytes = file.fill_value_at(child_addr)?;
        let dim_names = dimension_names_from_dimension_list(
            &attrs,
            dataset.shape.rank(),
            &dimension_scale_names,
        )
        .unwrap_or_else(|| synthetic_dimension_names(dataset.shape.rank()));

        let var = build_variable(
            child_name,
            &dataset,
            fill_bytes,
            dim_names,
            &attrs,
            Some(child_addr),
        );
        group.variables.push(var);
    }

    Ok(group)
}
