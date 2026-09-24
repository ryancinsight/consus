//! Child dataset and group specifications for builder composition.

#[cfg(feature = "alloc")]
use super::DatasetCreationProps;
#[cfg(feature = "alloc")]
use consus_core::{Datatype, Shape};

// ---------------------------------------------------------------------------
// Nested group authoring
// ---------------------------------------------------------------------------

/// Specification for a child dataset to be authored inside a nested group.
///
/// Only [`DatasetLayout::Contiguous`][crate::property_list::DatasetLayout::Contiguous] is supported for `dcpl`. Compact and
/// Chunked layouts require additional writer parameters not available in this
/// context.
#[cfg(feature = "alloc")]
pub struct ChildDatasetSpec<'a> {
    /// Dataset name within the parent group.
    pub name: &'a str,
    /// Element datatype.
    pub datatype: &'a Datatype,
    /// Dataset shape.
    pub shape: &'a Shape,
    /// Raw data bytes in dataset storage order.
    pub raw_data: &'a [u8],
    /// Dataset creation properties. Only `DatasetLayout::Contiguous` is supported.
    pub dcpl: DatasetCreationProps,
    /// Attribute messages attached to this dataset.
    ///
    /// Each entry is `(attribute_name, datatype, shape, raw_data)`.
    pub attributes: &'a [(&'a str, &'a Datatype, &'a Shape, &'a [u8])],
}

/// Specification for a child sub-group to be authored inside a parent group.
///
/// Supports arbitrary nesting depth: [`sub_groups`][ChildGroupSpec::sub_groups]
/// can contain further `ChildGroupSpec` values, written recursively.
///
/// Only [`DatasetLayout::Contiguous`][crate::property_list::DatasetLayout::Contiguous] is supported for dataset children.
#[cfg(feature = "alloc")]
pub struct ChildGroupSpec<'a> {
    /// Name of this group within its parent group.
    pub name: &'a str,
    /// Attribute messages attached to this group's object header.
    ///
    /// Each entry is `(attribute_name, datatype, shape, raw_data)`.
    pub attributes: &'a [(&'a str, &'a Datatype, &'a Shape, &'a [u8])],
    /// Dataset children of this group.
    pub datasets: &'a [ChildDatasetSpec<'a>],
    /// Sub-group children of this group (written recursively before this group).
    pub sub_groups: &'a [ChildGroupSpec<'a>],
}
