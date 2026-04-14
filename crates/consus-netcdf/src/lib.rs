//! # consus-netcdf
//!
//! Pure-Rust implementation of the netCDF-4 data model.
//!
//! ## Specification
//!
//! netCDF-4 is built on HDF5 with domain-specific conventions:
//! - **Classic model**: dimensions, variables, attributes, unlimited dimensions
//! - **Enhanced model**: groups, user-defined types, multiple unlimited dimensions
//! - **CF conventions**: coordinate variables, cell methods, standard names
//!
//! Reference: <https://www.unidata.ucar.edu/software/netcdf/docs/file_format_specifications.html>
//!
//! ### netCDF-4 ↔ HDF5 Mapping
//!
//! | netCDF-4 concept | HDF5 representation |
//! |------------------|---------------------|
//! | Dimension | Dataset with `CLASS=DIMENSION_SCALE` attribute |
//! | Variable | Dataset |
//! | Group | Group |
//! | Unlimited dimension | HDF5 unlimited dimension + chunked storage |
//! | Attribute | HDF5 attribute |
//!
//! ## Status
//!
//! Phase 2 — skeleton. Depends on `consus-hdf5` for file parsing.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod conventions;
pub mod dimension;
pub mod variable;
