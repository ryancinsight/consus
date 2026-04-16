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
//! ## Module Hierarchy
//!
//! ```text
//! consus-compression
//! ├── checksum/        # Checksum algorithms (CRC-32, Fletcher-32, lookup3)
//! │   ├── traits       # Checksum trait
//! │   ├── crc32        # CRC-32 (IEEE 802.3)
//! │   ├── fletcher32   # Fletcher-32 (HDF5 filter ID 3)
//! │   └── lookup3      # Jenkins lookup3 (HDF5 v2 metadata checksums)
//! ├── codec/           # Codec trait and implementations
//! │   ├── traits       # Codec trait, CompressionLevel, CodecId
//! │   ├── deflate      # Deflate/zlib codec (feature-gated)
//! │   ├── gzip         # Gzip codec, Zarr-specific (feature-gated)
//! │   ├── zstd         # Zstandard codec (feature-gated)
//! │   ├── lz4          # LZ4 block codec (feature-gated)
//! │   ├── szip         # Szip/Rice entropy coding, HDF5 filter 4 (feature-gated)
//! │   └── blosc        # Blosc meta-compressor container, HDF5 filter 32001 (feature-gated)
//! ├── endian/          # Byte-order utilities
//! │   └── conversion   # Multi-byte integer read/write, byte-swap
//! ├── pipeline/        # Filter pipeline (alloc-gated)
//! │   ├── traits       # Filter, FilterDirection
//! │   ├── shuffle      # Byte shuffle/unshuffle (HDF5 filter ID 2)
//! │   ├── nbit         # N-bit packing/unpacking (HDF5 filter ID 5)
//! │   └── executor     # Pipeline execution engine
//! ├── chunking/        # Chunk indexing and iteration for N-dimensional storage
//! │   ├── index        # Chunk coordinate ↔ linear index conversion
//! │   └── iterator     # Iterate chunk ranges over a dataspace
//! ├── registry/        # Runtime codec lookup
//! └── serialization/   # Low-level binary serialization primitives
//!     └── primitives   # LEB128, NUL-terminated strings, alignment
//! ```
//!
//! ### Invariant
//!
//! For any codec `C` and input `data`:
//!   `C.decompress(C.compress(data)?) == data`

#![cfg_attr(not(feature = "std"), no_std)]

// NOTE: `extern crate alloc` is unconditional because the workspace
// dependency on consus-core does not set `default-features = false`.
// consus-core's defaults enable `std → alloc`, so `Error::InvalidFormat`
// always carries `message: String`. Every crate that constructs errors
// therefore requires alloc.
extern crate alloc;

pub mod checksum;
pub mod chunking;
pub mod codec;
pub mod endian;
#[cfg(feature = "alloc")]
pub mod pipeline;
pub mod registry;
pub mod serialization;

// Re-export primary types at crate root.
pub use checksum::{Checksum, Crc32, Fletcher32, Lookup3};
#[cfg(feature = "alloc")]
pub use chunking::ChunkIterator;
pub use codec::traits::{Codec, CodecId, CompressionLevel};
#[cfg(feature = "alloc")]
pub use pipeline::{Filter, FilterDirection, FilterPipeline, NbitFilter, ShuffleFilter};
#[cfg(feature = "alloc")]
pub use registry::{CodecRegistry, CompressionRegistry, DefaultCodecRegistry};
