//! # consus-compression
//!
//! Compression codec registry for the Consus scientific storage library.
//!
//! ## Architecture
//!
//! This crate provides a trait-based codec abstraction and a runtime registry
//! that maps codec identifiers to implementations. Format backends delegate
//! compression/decompression through this abstraction rather than depending
//! on codec crates directly.
//!
//! ### Design
//!
//! - `Codec` trait: defines compress/decompress with explicit buffer contracts.
//! - `CodecRegistry`: maps codec identifiers (u16 filter IDs for HDF5, string
//!   names for Zarr) to `Codec` implementations.
//! - Feature-gated backends: each compression algorithm is behind a cargo feature.
//!
//! ### Invariant
//!
//! For any codec `C` and input `data`:
//!   `C.decompress(C.compress(data)?) == data`

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::error::Result;

pub mod codecs;
pub mod registry;

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
/// - Buffer sizing: callers provide output buffers. Implementations write into
///   the buffer and return the number of bytes written.
pub trait Codec: Send + Sync {
    /// Human-readable name of this codec (e.g., "deflate", "zstd").
    fn name(&self) -> &str;

    /// HDF5 filter ID, if applicable.
    fn hdf5_filter_id(&self) -> Option<u16>;

    /// Compress `input` into `output`.
    ///
    /// Returns the number of bytes written to `output`.
    ///
    /// # Errors
    ///
    /// Returns `Error::BufferTooSmall` if `output` is insufficient.
    /// Returns `Error::CompressionError` on codec failure.
    #[cfg(feature = "alloc")]
    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>>;

    /// Decompress `input` into `output`.
    ///
    /// Returns the number of bytes written to `output`.
    ///
    /// # Errors
    ///
    /// Returns `Error::BufferTooSmall` if `output` is insufficient.
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
