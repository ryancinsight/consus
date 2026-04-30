//! Codec trait definition and individual codec implementations.
//!
//! Each codec sub-module is feature-gated. Only codecs whose features
//! are enabled are compiled.

pub mod traits;

#[cfg(feature = "deflate")]
pub mod deflate;

#[cfg(all(feature = "gzip", feature = "std"))]
pub mod gzip;

#[cfg(feature = "zstd")]
pub mod zstd;

#[cfg(feature = "lz4")]
pub mod lz4;

#[cfg(feature = "szip")]
pub mod szip;

#[cfg(feature = "blosc")]
pub mod blosc;
