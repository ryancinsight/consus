//! FITS ASCII and binary table descriptors and raw row access.
//!
//! ## Scope
//!
//! This module is the authoritative table-domain boundary for `consus-fits`.
//! It defines:
//! - ASCII table descriptors derived from FITS header keywords
//! - binary table descriptors derived from FITS header keywords
//! - row/column metadata for standard table extensions
//! - raw row-oriented access over FITS data-unit spans
//!
//! ## FITS invariants
//!
//! For standard table extensions:
//! - `XTENSION = 'TABLE'` denotes an ASCII table
//! - `XTENSION = 'BINTABLE'` denotes a binary table
//! - `NAXIS1` is the row length in bytes
//! - `NAXIS2` is the number of rows
//! - `TFIELDS` is the number of columns
//! - the logical payload size is `NAXIS1 * NAXIS2`
//!
//! FITS stores table payloads in row-major record order. The data unit is padded
//! with zero bytes to the next 2880-byte boundary.
//!
//! ## Architectural role
//!
//! This module depends on the authoritative `header`, `types`, and
//! `datastructure` modules. It does not duplicate FITS keyword parsing or
//! blocking math.

mod data;
pub(crate) mod decode;
mod parse;
mod types;

#[cfg(feature = "alloc")]
pub use data::FitsTableData;
#[cfg(feature = "alloc")]
pub use decode::FitsColumnValue;
#[cfg(feature = "alloc")]
pub use types::{
    FitsAsciiTableDescriptor, FitsBinaryTableDescriptor, FitsTableColumn, FitsTableDescriptor,
};
