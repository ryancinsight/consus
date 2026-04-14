//! Individual codec implementations.
//!
//! Each sub-module is feature-gated. Only codecs whose features are enabled
//! are compiled.

#[cfg(feature = "deflate")]
pub mod deflate;

#[cfg(feature = "zstd")]
pub mod zstd_codec;

#[cfg(feature = "lz4")]
pub mod lz4_codec;
