//! Storage model traits defining the abstract hierarchical container.
//!
//! ## Architecture
//!
//! The storage model separates the *logical* hierarchy (groups, datasets,
//! attributes) from the *physical* format (HDF5 B-trees, Zarr directory
//! layout, etc.). Format backends implement these traits.
//!
//! ### Dependency Inversion Principle
//!
//! Higher-level code depends on these traits, not on concrete format types.
//! This enables:
//! - Format-agnostic algorithms operating on any compliant backend
//! - Testing with in-memory backends
//! - Runtime format selection based on file inspection

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use crate::datatype::Datatype;
#[cfg(feature = "alloc")]
use crate::dimension::{ChunkShape, Layout, Shape};
#[cfg(feature = "alloc")]
use crate::error::Result;
#[cfg(feature = "alloc")]
use crate::metadata::Attribute;
#[cfg(feature = "alloc")]
use crate::selection::Selection;

/// A node in the storage hierarchy.
///
/// Every object in a Consus container is either a group (directory-like)
/// or a dataset (N-dimensional array).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// A group containing other groups and datasets.
    Group,
    /// An N-dimensional array with a datatype.
    Dataset,
}

/// Compression configuration for a dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum Compression {
    /// No compression.
    None,
    /// Deflate/zlib.
    Deflate { level: u32 },
    /// Zstandard.
    Zstd { level: i32 },
    /// LZ4.
    Lz4,
    /// Gzip (compatible with deflate).
    Gzip { level: u32 },
}

impl Default for Compression {
    fn default() -> Self {
        Compression::None
    }
}

/// Configuration for creating a new dataset.
///
/// Builder pattern is provided by format-specific crates;
/// this struct carries the resolved configuration.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Name of the dataset.
    pub name: String,
    /// Datatype of elements.
    pub datatype: Datatype,
    /// Shape of the dataset.
    pub shape: Shape,
    /// Optional chunk shape (None = contiguous storage).
    pub chunk_shape: Option<ChunkShape>,
    /// Compression configuration.
    pub compression: Compression,
    /// Memory layout order.
    pub layout: Layout,
    /// Fill value as raw bytes (datatype-dependent).
    pub fill_value: Option<Vec<u8>>,
}

/// Read access to an object in the storage hierarchy.
///
/// This trait is object-safe and format-agnostic.
#[cfg(feature = "alloc")]
pub trait StorageNode {
    /// Name of this node (not the full path).
    fn name(&self) -> &str;

    /// Full path within the container.
    fn path(&self) -> &str;

    /// Type of this node.
    fn node_type(&self) -> NodeType;
}

/// Read access to a group in the storage hierarchy.
#[cfg(feature = "alloc")]
pub trait GroupRead: StorageNode {
    /// List child node names.
    fn children(&self) -> Result<Vec<String>>;

    /// Number of child nodes.
    fn num_children(&self) -> Result<usize>;

    /// List attribute names.
    fn attribute_names(&self) -> Result<Vec<String>>;

    /// Read an attribute by name.
    fn attribute(&self, name: &str) -> Result<Attribute>;
}

/// Read access to a dataset.
#[cfg(feature = "alloc")]
pub trait DatasetRead: StorageNode {
    /// Datatype of elements in this dataset.
    fn datatype(&self) -> &Datatype;

    /// Shape of this dataset.
    fn shape(&self) -> &Shape;

    /// Chunk shape, if chunked.
    fn chunk_shape(&self) -> Option<&ChunkShape>;

    /// Read raw bytes from a selection into a caller-provided buffer.
    ///
    /// # Contract
    ///
    /// - `buf.len()` must be ≥ `selection.num_elements() * datatype().element_size()`
    /// - Selection must be within the dataset's dataspace bounds
    fn read_raw(&self, selection: &Selection, buf: &mut [u8]) -> Result<usize>;

    /// Read the entire dataset as raw bytes.
    fn read_all_raw(&self) -> Result<Vec<u8>> {
        let size = self
            .datatype()
            .element_size()
            .unwrap_or(0)
            .checked_mul(self.shape().num_elements())
            .unwrap_or(0);
        let mut buf = alloc::vec![0u8; size];
        self.read_raw(&Selection::All, &mut buf)?;
        Ok(buf)
    }

    /// List attribute names.
    fn attribute_names(&self) -> Result<Vec<String>>;

    /// Read an attribute by name.
    fn attribute(&self, name: &str) -> Result<Attribute>;
}
