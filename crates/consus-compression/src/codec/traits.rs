//! Codec trait, compression level, and codec identifiers.
//!
//! ## Contract
//!
//! - `compress` produces output that `decompress` can invert exactly.
//! - `decompress` must validate the compressed stream and return
//!   `Error::CompressionError` on malformed input rather than producing garbage.
//! - Buffer sizing: callers provide output buffers. Implementations write into
//!   the buffer and return the number of bytes written.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::Result;

/// Compression level hint.
///
/// Codecs interpret this value according to their own scale.
/// Out-of-range values are clamped to the codec's valid range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionLevel(pub i32);

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel(6) // typical default for deflate
    }
}

/// A compression/decompression codec.
///
/// ## Contract
///
/// - `compress` produces output that `decompress` can invert exactly.
/// - `decompress` must validate the compressed stream and return `Error::CompressionError`
///   on malformed input rather than producing garbage.
pub trait Codec: Send + Sync {
    /// Human-readable name of this codec (e.g., "deflate", "zstd").
    fn name(&self) -> &str;

    /// HDF5 filter ID, if applicable.
    fn hdf5_filter_id(&self) -> Option<u16>;

    /// Compress `input` and return the compressed bytes.
    ///
    /// # Errors
    ///
    /// Returns `Error::CompressionError` on codec failure.
    #[cfg(feature = "alloc")]
    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>>;

    /// Decompress `input` and return the decompressed bytes.
    ///
    /// # Errors
    ///
    /// Returns `Error::CompressionError` on malformed input.
    #[cfg(feature = "alloc")]
    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>>;
}

/// Identifies a codec by either its HDF5 filter ID or a string name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CodecId {
    /// HDF5 filter identifier (e.g., 1 = deflate, 32004 = lz4, 32015 = zstd).
    FilterId(u16),
    /// String identifier (e.g., "blosc", "gzip" — used by Zarr).
    #[cfg(feature = "alloc")]
    Name(alloc::string::String),
}
