//! # consus-core
//!
//! Core types, traits, and error definitions for the Consus scientific storage library.
//!
//! This crate is `no_std`-compatible by default. Enable the `std` feature for
//! `std::io` integration and `std::error::Error` implementations.
//!
//! ## Architecture
//!
//! The core crate defines the abstract storage model shared across all format
//! backends (HDF5, Zarr, netCDF-4, Parquet). Format-specific crates depend on
//! `consus-core` and implement the traits defined here.
//!
//! ### Invariants
//!
//! - All types in this crate are `Send + Sync` when their contents are.
//! - Datatype representations are canonical: two equivalent logical types
//!   produce identical `Datatype` values regardless of source format.
//! - Dimension ordering follows row-major (C) convention by default;
//!   column-major (Fortran) is explicitly represented via `Layout`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod datatype;
pub mod dimension;
pub mod error;
pub mod metadata;
pub mod selection;
pub mod storage;

pub use error::{Error, Result};
