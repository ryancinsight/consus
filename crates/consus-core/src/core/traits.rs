//! Abstract storage traits defining the hierarchical container model.
//!
//! ## Dependency Inversion Principle
//!
//! All format-specific crates (HDF5, Zarr, netCDF-4, Parquet) implement
//! these traits. Higher-level code depends on these abstractions, never
//! on concrete format types. This enables:
//!
//! - Format-agnostic algorithms over any compliant backend.
//! - Testing with in-memory or mock backends.
//! - Runtime format selection by file inspection.
//!
//! ## Object Safety
//!
//! All traits in this module are object-safe (`dyn Trait` compatible).
//! This is verified by compile-time unit tests.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! Node (base identity)
//! ├── HasAttributes (attribute access mixin)
//! ├── GroupRead: Node + HasAttributes
//! ├── DatasetRead: Node + HasAttributes
//! ├── GroupWrite: GroupRead
//! ├── DatasetWrite: DatasetRead
//! ├── LinkRead
//! ├── FileRead: HasAttributes
//! ├── FileWrite: FileRead
//! └── SelectionOps (standalone)
//! ```

use super::error::Result;
use crate::types::{ChunkShape, Compression, Datatype, LinkType, NodeType, Selection, Shape};

// ---------------------------------------------------------------------------
// Node identity
// ---------------------------------------------------------------------------

/// A node in the hierarchical storage model.
///
/// Every addressable object (group, dataset, named datatype) is a node
/// with a local name, an absolute path, and a type discriminant.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait Node {
    /// Leaf name of this node (e.g., `"temperature"`).
    fn name(&self) -> &str;

    /// Absolute path within the container
    /// (e.g., `"/simulations/run_001/temperature"`).
    fn path(&self) -> &str;

    /// Discriminant indicating the node kind.
    fn node_type(&self) -> NodeType;
}

// ---------------------------------------------------------------------------
// Attribute access
// ---------------------------------------------------------------------------

/// Read access to attributes attached to a node.
///
/// Attributes are small named metadata values. This trait provides a
/// `no_std`-compatible interface using caller-provided buffers and
/// visitor callbacks instead of heap-allocated return types.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait HasAttributes {
    /// Number of attributes attached to this object.
    fn num_attributes(&self) -> Result<usize>;

    /// Whether an attribute with the given name exists.
    fn has_attribute(&self, name: &str) -> Result<bool>;

    /// Datatype of the named attribute.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if the attribute does not exist.
    fn attribute_datatype(&self, name: &str) -> Result<Datatype>;

    /// Read raw bytes of the named attribute into `buf`.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// - `Error::NotFound` if the attribute does not exist.
    /// - `Error::BufferTooSmall` if `buf` is smaller than the
    ///   attribute's byte size.
    fn read_attribute_raw(&self, name: &str, buf: &mut [u8]) -> Result<usize>;

    /// Visit each attribute name.
    ///
    /// The visitor returns `true` to continue iteration, `false` to
    /// stop early.
    fn for_each_attribute(&self, visitor: &mut dyn FnMut(&str) -> bool) -> Result<()>;
}

/// Write access to attributes on a node.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait AttributeWrite: HasAttributes {
    /// Write (create or overwrite) an attribute.
    ///
    /// `data` contains the raw bytes of the attribute value encoded
    /// according to `datatype`.
    fn write_attribute_raw(&mut self, name: &str, datatype: &Datatype, data: &[u8]) -> Result<()>;

    /// Delete a named attribute.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if the attribute does not exist.
    fn delete_attribute(&mut self, name: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Group
// ---------------------------------------------------------------------------

/// Read access to a group node in the storage hierarchy.
///
/// A group is a container that holds other groups, datasets, and
/// named datatypes as children, connected by links.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait GroupRead: Node + HasAttributes {
    /// Number of direct child links.
    fn num_children(&self) -> Result<usize>;

    /// Whether a child with the given name exists.
    fn contains(&self, name: &str) -> Result<bool>;

    /// Node type of the named child.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if the child does not exist.
    fn child_node_type(&self, name: &str) -> Result<NodeType>;

    /// Visit each direct child.
    ///
    /// The visitor receives `(name, node_type)` and returns `true`
    /// to continue, `false` to stop early.
    fn for_each_child(&self, visitor: &mut dyn FnMut(&str, NodeType) -> bool) -> Result<()>;
}

/// Write access to a group node.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait GroupWrite: GroupRead {
    /// Create a child group with the given name.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidFormat` if a child with this name
    /// already exists.
    fn create_group(&mut self, name: &str) -> Result<()>;

    /// Remove a child link by name.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if the link does not exist.
    fn remove_child(&mut self, name: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// Read access to a dataset in the storage hierarchy.
///
/// A dataset is an N-dimensional typed array with optional chunking
/// and compression.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait DatasetRead: Node + HasAttributes {
    /// Canonical datatype of the dataset's elements.
    fn datatype(&self) -> &Datatype;

    /// Shape of the dataset.
    fn shape(&self) -> &Shape;

    /// Chunk shape, if the dataset uses chunked storage.
    ///
    /// Returns `None` for contiguous (non-chunked) datasets.
    fn chunk_shape(&self) -> Option<&ChunkShape>;

    /// Compression configuration applied to this dataset.
    fn compression(&self) -> Compression;

    /// Read raw element bytes for the given selection into `buf`.
    ///
    /// # Contract
    ///
    /// - `buf.len() >= selection.num_elements(shape) * datatype().element_size()`
    /// - Selection must be within the dataset's dataspace bounds.
    ///
    /// Returns the number of bytes written to `buf`.
    ///
    /// # Errors
    ///
    /// - `Error::SelectionOutOfBounds` if selection exceeds dataspace.
    /// - `Error::BufferTooSmall` if `buf` is insufficient.
    fn read_raw(&self, selection: &Selection, buf: &mut [u8]) -> Result<usize>;
}

/// Write access to a dataset.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait DatasetWrite: DatasetRead {
    /// Write raw element bytes for the given selection from `data`.
    ///
    /// # Contract
    ///
    /// - `data.len() >= selection.num_elements(shape) * datatype().element_size()`
    /// - Selection must be within the dataset's dataspace bounds.
    ///
    /// # Errors
    ///
    /// - `Error::SelectionOutOfBounds` if selection exceeds dataspace.
    /// - `Error::ReadOnly` if the dataset is not writable.
    fn write_raw(&mut self, selection: &Selection, data: &[u8]) -> Result<()>;

    /// Resize the dataset along its unlimited dimensions.
    ///
    /// # Errors
    ///
    /// - `Error::ShapeError` if the new shape is incompatible (e.g.,
    ///   resizing a fixed dimension, or rank mismatch).
    fn resize(&mut self, new_dims: &[usize]) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------

/// Read access to a link in the hierarchy.
///
/// Links connect parent groups to child nodes. A link has a name
/// (within the parent), a type, and optionally a target path.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait LinkRead {
    /// Name of this link within its parent group.
    fn name(&self) -> &str;

    /// Classification of this link.
    fn link_type(&self) -> LinkType;

    /// Target path for soft and external links.
    ///
    /// Returns `None` for hard links (the target is the object itself).
    fn target_path(&self) -> Option<&str>;
}

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// Read access to a storage container (file).
///
/// This is the entry point for format-agnostic traversal of any
/// hierarchical storage backend. Implementations are provided by
/// format-specific crates (`consus-hdf5`, `consus-zarr`, etc.).
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait FileRead: HasAttributes {
    /// Format identifier string (e.g., `"hdf5"`, `"zarr"`, `"netcdf4"`).
    fn format(&self) -> &str;

    /// Whether the given absolute path exists in the hierarchy.
    fn exists(&self, path: &str) -> Result<bool>;

    /// Node type at the given absolute path.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if the path does not exist.
    fn node_type_at(&self, path: &str) -> Result<NodeType>;

    /// Number of direct children of the group at the given path.
    ///
    /// # Errors
    ///
    /// - `Error::NotFound` if the path does not exist.
    /// - `Error::InvalidFormat` if the node is not a group.
    fn num_children_at(&self, path: &str) -> Result<usize>;

    /// Datatype of the dataset at the given path.
    ///
    /// # Errors
    ///
    /// - `Error::NotFound` if the path does not exist.
    /// - `Error::InvalidFormat` if the node is not a dataset.
    fn dataset_datatype(&self, path: &str) -> Result<Datatype>;

    /// Shape of the dataset at the given path.
    ///
    /// # Errors
    ///
    /// - `Error::NotFound` if the path does not exist.
    /// - `Error::InvalidFormat` if the node is not a dataset.
    fn dataset_shape(&self, path: &str) -> Result<Shape>;

    /// Read raw bytes from a dataset at the given path.
    ///
    /// Returns the number of bytes written to `buf`.
    fn read_dataset_raw(&self, path: &str, selection: &Selection, buf: &mut [u8]) -> Result<usize>;
}

/// Write access to a storage container (file).
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait FileWrite: FileRead {
    /// Flush all pending writes to the underlying storage.
    fn flush(&mut self) -> Result<()>;

    /// Create a group at the given absolute path.
    ///
    /// Parent groups are created as needed (recursive mkdir semantics).
    fn create_group(&mut self, path: &str) -> Result<()>;

    /// Write raw bytes to a dataset at the given path.
    fn write_dataset_raw(&mut self, path: &str, selection: &Selection, data: &[u8]) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Selection operations
// ---------------------------------------------------------------------------

/// Operations on a dataspace selection.
///
/// Provides a uniform interface for computing properties of any
/// selection variant.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait SelectionOps {
    /// Rank (number of dimensions) of this selection.
    ///
    /// For `All` and `None`, the rank is context-dependent;
    /// implementations may return 0 when the rank cannot be
    /// determined without a dataspace shape.
    fn rank(&self) -> usize;

    /// Total number of selected elements for a given dataspace shape.
    fn num_elements(&self, shape: &Shape) -> usize;

    /// Whether this selection is valid for the given dataspace shape.
    ///
    /// A selection is valid iff all selected indices fall within bounds.
    fn is_valid_for(&self, shape: &Shape) -> bool;
}

// ---------------------------------------------------------------------------
// Object-safety compile-time tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `Node` is object-safe.
    #[test]
    fn node_is_object_safe() {
        fn _assert(_: &dyn Node) {}
    }

    /// `HasAttributes` is object-safe.
    #[test]
    fn has_attributes_is_object_safe() {
        fn _assert(_: &dyn HasAttributes) {}
    }

    /// `AttributeWrite` is object-safe.
    #[test]
    fn attribute_write_is_object_safe() {
        fn _assert(_: &dyn AttributeWrite) {}
    }

    /// `GroupRead` is object-safe.
    #[test]
    fn group_read_is_object_safe() {
        fn _assert(_: &dyn GroupRead) {}
    }

    /// `GroupWrite` is object-safe.
    #[test]
    fn group_write_is_object_safe() {
        fn _assert(_: &dyn GroupWrite) {}
    }

    /// `DatasetRead` is object-safe.
    #[test]
    fn dataset_read_is_object_safe() {
        fn _assert(_: &dyn DatasetRead) {}
    }

    /// `DatasetWrite` is object-safe.
    #[test]
    fn dataset_write_is_object_safe() {
        fn _assert(_: &dyn DatasetWrite) {}
    }

    /// `LinkRead` is object-safe.
    #[test]
    fn link_read_is_object_safe() {
        fn _assert(_: &dyn LinkRead) {}
    }

    /// `FileRead` is object-safe.
    #[test]
    fn file_read_is_object_safe() {
        fn _assert(_: &dyn FileRead) {}
    }

    /// `FileWrite` is object-safe.
    #[test]
    fn file_write_is_object_safe() {
        fn _assert(_: &dyn FileWrite) {}
    }

    /// `SelectionOps` is object-safe.
    #[test]
    fn selection_ops_is_object_safe() {
        fn _assert(_: &dyn SelectionOps) {}
    }
}
