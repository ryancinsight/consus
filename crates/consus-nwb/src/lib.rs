//! Neurodata Without Borders (NWB) file format support.
//!
//! NWB is a data standard for neurophysiology built on HDF5. This crate
//! provides read and write support for NWB 2.x files using the
//! [`consus-hdf5`](../consus_hdf5/index.html) backend.
//!
//! ## Architecture
//!
//! ```text
//! consus-nwb
//!   ├── conventions  — NWB namespace and neurodata type resolution
//!   ├── file         — NWBFile top-level container
//!   ├── group        — NWBGroup traversal and extraction
//!   ├── io           — NWB-specific I/O helpers
//!   ├── metadata     — NWBFile metadata model (session, subject, lab)
//!   ├── model        — Core NWB semantic model (TimeSeries, ElectrodeTable, etc.)
//!   ├── namespace    — NWB namespace registry and type system
//!   ├── storage      — HDF5-backed storage layer
//!   ├── validation   — Schema conformance and constraint checking
//!   └── version      — NWB version detection and compatibility
//! ```
//!
//! ## Status
//!
//! Foundation scaffold. Module structure reflects planned domain decomposition.
//! No public API is stable yet.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod conventions;
#[cfg(feature = "alloc")]
pub mod file;
#[cfg(feature = "alloc")]
pub mod group;
#[cfg(feature = "alloc")]
pub mod io;
#[cfg(feature = "alloc")]
pub mod metadata;
#[cfg(feature = "alloc")]
pub mod model;
#[cfg(feature = "alloc")]
pub mod namespace;
#[cfg(feature = "alloc")]
pub mod storage;
#[cfg(feature = "alloc")]
pub mod validation;
pub mod version;

#[cfg(feature = "alloc")]
pub use model::electrode::{ElectrodeRow, ElectrodeTable};
#[cfg(feature = "alloc")]
pub use model::units::UnitsTable;
#[cfg(feature = "alloc")]
pub use namespace::NwbNamespaceSpec;
#[cfg(feature = "alloc")]
pub use namespace::NwbTypeSpec;
#[cfg(feature = "alloc")]
pub use validation::{ConformanceViolation, NwbConformanceReport};
