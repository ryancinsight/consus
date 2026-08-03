//! FITS image metadata extraction and raw image access.
//!
//! ## Scope
//!
//! This module is the authoritative image-domain boundary for `consus-fits`.
//! It defines:
//! - multi-dimensional FITS image descriptors derived from header keywords
//! - optional physical-value scaling metadata (`BSCALE`, `BZERO`, `BLANK`)
//! - random-groups detection metadata
//! - raw image byte access over FITS data-unit spans
//!
//! ## FITS invariants
//!
//! For a standard image HDU:
//! - `BITPIX` defines the stored element representation
//! - `NAXIS` defines the rank
//! - `NAXISn` for `n ∈ [1, NAXIS]` define axis extents
//! - the logical payload size is `product(NAXISn) * element_size(BITPIX)`
//!
//! For random groups:
//! - `GROUPS = T`
//! - `NAXIS1 = 0`
//! - `GCOUNT >= 1`
//! - `PCOUNT >= 0`
//! - the logical payload size is
//!   `(PCOUNT + product(NAXIS2..NAXISn)) * GCOUNT * element_size(BITPIX)`
//!
//! FITS stores multi-byte numeric values in big-endian order.
//!
//! ## Architectural role
//!
//! This module depends on the authoritative `header`, `types`, and
//! `datastructure` modules. It does not duplicate FITS keyword parsing or
//! blocking math.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod types;

#[cfg(feature = "alloc")]
pub use types::{FitsImageData, FitsImageDescriptor};
pub use types::{FitsImageScaling, FitsRandomGroups};

#[cfg(all(test, feature = "alloc"))]
mod tests;
