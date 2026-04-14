//! # Consus
//!
//! Pure-Rust, `no_std`-compatible, memory-safe hierarchical and array-oriented
//! storage formats for scientific computing.
//!
//! ## Overview
//!
//! Consus provides a unified API for reading and writing scientific data across
//! multiple storage formats:
//!
//! | Format | Feature | Status |
//! |--------|---------|--------|
//! | HDF5 | `hdf5` | Phase 1 (active) |
//! | Zarr v2/v3 | `zarr` | Phase 2 |
//! | netCDF-4 | `netcdf` | Phase 2 |
//! | Apache Parquet | `parquet` | Phase 3 |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use consus::core::{
//!     datatype::{ByteOrder, Datatype},
//!     dimension::Shape,
//!     storage::Compression,
//! };
//!
//! // Core types are available without any format backend
//! let shape = Shape::fixed(&[100, 200]);
//! assert_eq!(shape.rank(), 2);
//! assert_eq!(shape.num_elements(), 20_000);
//!
//! let dtype = Datatype::Float {
//!     bits: core::num::NonZeroUsize::new(64).unwrap(),
//!     byte_order: ByteOrder::LittleEndian,
//! };
//! assert_eq!(dtype.element_size(), Some(8));
//! ```
//!
//! ## Architecture
//!
//! ```text
//! consus (this crate — facade)
//! ├── consus-core        → re-exported as consus::core
//! ├── consus-io          → re-exported as consus::io
//! ├── consus-compression → re-exported as consus::compression
//! ├── consus-hdf5        → re-exported as consus::hdf5 (feature = "hdf5")
//! ├── consus-zarr        → re-exported as consus::zarr (feature = "zarr")
//! ├── consus-netcdf      → re-exported as consus::netcdf (feature = "netcdf")
//! └── consus-parquet     → re-exported as consus::parquet (feature = "parquet")
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | ✓ | Enable `std::io` integration |
//! | `hdf5` | ✓ | HDF5 format backend |
//! | `deflate` | ✓ | Deflate/zlib compression |
//! | `zarr` | | Zarr v2/v3 format backend |
//! | `netcdf` | | netCDF-4 format backend |
//! | `parquet` | | Apache Parquet interop |
//! | `zstd` | | Zstandard compression |
//! | `lz4` | | LZ4 compression |
//! | `async-io` | | Async I/O traits (requires tokio) |
//! | `alloc` | | `no_std` with allocator support |
//!
//! ## Minimum Supported Rust Version
//!
//! 1.85.0 (edition 2024)

#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../../../README.md")]

/// Core types, traits, and error definitions.
///
/// Available in all configurations including `no_std`.
pub use consus_core as core;

/// Sync and async I/O abstractions.
pub use consus_io as io;

/// Compression codec registry.
pub use consus_compression as compression;

/// HDF5 format backend.
#[cfg(feature = "hdf5")]
pub use consus_hdf5 as hdf5;

/// Zarr v2/v3 format backend.
#[cfg(feature = "zarr")]
pub use consus_zarr as zarr;

/// netCDF-4 format backend.
#[cfg(feature = "netcdf")]
pub use consus_netcdf as netcdf;

/// Apache Parquet interop layer.
#[cfg(feature = "parquet")]
pub use consus_parquet as parquet;

// Re-export commonly used types at crate root for convenience.
pub use consus_core::datatype::Datatype;
pub use consus_core::dimension::Shape;
pub use consus_core::error::{Error, Result};
pub use consus_core::storage::Compression;
