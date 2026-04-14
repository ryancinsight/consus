//! # consus-zarr
//!
//! Pure-Rust implementation of the Zarr storage format (v2 and v3).
//!
//! ## Zarr Specification
//!
//! - **Zarr v2**: <https://zarr.readthedocs.io/en/stable/spec/v2.html>
//! - **Zarr v3**: <https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html>
//!
//! ### Data Model
//!
//! Zarr stores N-dimensional arrays as collections of compressed chunks.
//! Each array lives in a directory (or key prefix in object stores) with:
//! - `.zarray` (v2) or `zarr.json` (v3): array metadata (shape, chunks, dtype, compressor)
//! - `.zattrs` (v2) or inline in `zarr.json` (v3): user attributes
//! - Chunk files named by index (e.g., `0.0.0` for v2, `c/0/0/0` for v3)
//!
//! ### Design
//!
//! Zarr is structurally simpler than HDF5: metadata is JSON, storage is
//! file-per-chunk. This crate handles:
//! - Metadata parsing/generation
//! - Chunk coordinate → storage key mapping
//! - Codec pipeline (compression, filters, endian conversion)
//! - Directory store, Zip store, and object store backends
//!
//! ## Status
//!
//! Phase 2 — structural skeleton. Full implementation follows HDF5 completion.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod chunk;
pub mod metadata;
pub mod store;
