//! Zarr array and group metadata.
//!
//! ## Zarr v2 `.zarray` Schema
//!
//! ```json
//! {
//!     "zarr_format": 2,
//!     "shape": [1000, 1000],
//!     "chunks": [100, 100],
//!     "dtype": "<f8",
//!     "compressor": {"id": "zlib", "level": 1},
//!     "fill_value": 0,
//!     "order": "C",
//!     "filters": null
//! }
//! ```
//!
//! ## Zarr v3 `zarr.json` Schema
//!
//! ```json
//! {
//!     "zarr_format": 3,
//!     "node_type": "array",
//!     "shape": [1000, 1000],
//!     "data_type": "float64",
//!     "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100, 100]}},
//!     "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
//!     "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "gzip", "configuration": {"level": 1}}],
//!     "fill_value": 0
//! }
//! ```

/// Zarr format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrVersion {
    V2,
    V3,
}

/// Zarr array metadata (format-agnostic representation).
///
/// Covers both v2 and v3 array metadata after parsing.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ArrayMetadata {
    /// Zarr format version.
    pub version: ZarrVersion,
    /// Array shape (dimension extents).
    pub shape: alloc::vec::Vec<usize>,
    /// Chunk shape.
    pub chunks: alloc::vec::Vec<usize>,
    /// Data type string (NumPy-style for v2, named for v3).
    pub dtype: alloc::string::String,
    /// Fill value as a string representation.
    pub fill_value: alloc::string::String,
    /// Memory order ("C" or "F").
    pub order: char,
}
