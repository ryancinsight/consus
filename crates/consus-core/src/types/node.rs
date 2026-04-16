//! Node classification, link types, and storage configuration.
//!
//! ## Specification
//!
//! Every object in a Consus container is either a group (directory-like),
//! a dataset (N-dimensional typed array), or a named datatype. Links
//! connect nodes in the hierarchy.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use super::datatype::Datatype;
#[cfg(feature = "alloc")]
use super::dimension::{ChunkShape, Layout, Shape};

/// Classification of a node in the storage hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// A group containing other groups and datasets.
    Group,
    /// An N-dimensional typed array.
    Dataset,
    /// A committed (named) datatype (HDF5-specific).
    NamedDatatype,
}

/// Classification of a link in the hierarchy.
///
/// ## Semantics
///
/// - `Hard`: direct reference to an object by address; participates in
///   reference counting for object lifetime.
/// - `Soft`: symbolic path within the same container; may be dangling.
/// - `External`: symbolic path targeting a different file/container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkType {
    /// Hard link (direct object reference).
    Hard,
    /// Soft (symbolic) link to a path within the same container.
    Soft,
    /// External link to a path in a different file/container.
    External,
}

/// Compression algorithm configuration for dataset storage.
///
/// Compression is applied per-chunk for chunked datasets.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Compression {
    /// No compression.
    #[default]
    None,
    /// Deflate (zlib) compression.
    Deflate {
        /// Compression level (0-9).
        level: u32,
    },
    /// Zstandard compression.
    Zstd {
        /// Compression level (negative values enable fast mode).
        level: i32,
    },
    /// LZ4 block compression.
    Lz4,
    /// Gzip compression (wire-compatible with deflate).
    Gzip {
        /// Compression level (0-9).
        level: u32,
    },
}

/// Configuration for creating a new dataset.
///
/// This struct carries the resolved configuration for dataset creation.
/// Builder-pattern construction is provided by format-specific crates.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Dataset name.
    pub name: String,
    /// Element datatype.
    pub datatype: Datatype,
    /// Dataset shape.
    pub shape: Shape,
    /// Chunk shape (`None` for contiguous storage).
    pub chunk_shape: Option<ChunkShape>,
    /// Compression configuration.
    pub compression: Compression,
    /// Memory layout order.
    pub layout: Layout,
    /// Fill value as raw bytes (datatype-dependent encoding).
    pub fill_value: Option<Vec<u8>>,
}
