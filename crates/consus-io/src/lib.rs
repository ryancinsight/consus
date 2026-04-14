//! # consus-io
//!
//! Sync and async I/O abstractions for the Consus storage library.
//!
//! ## Architecture
//!
//! This crate decouples format logic from physical I/O by defining
//! position-aware read/write traits. Format backends operate on these
//! traits rather than `std::fs::File` directly, enabling:
//!
//! - In-memory buffers for testing
//! - Memory-mapped I/O
//! - Object store backends (S3, GCS) via async adapters
//! - `no_std` environments with custom I/O
//!
//! ### Trait Hierarchy
//!
//! ```text
//! ReadAt ──┐
//!          ├── RandomAccess (read + write + seek)
//! WriteAt ─┘
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod cursor;
pub mod source;

#[cfg(feature = "async-io")]
pub mod async_source;
