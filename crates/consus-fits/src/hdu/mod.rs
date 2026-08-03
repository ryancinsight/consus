//! FITS Header/Data Unit descriptors and sequencing.
//!
//! ## Scope
//!
//! This module is the authoritative HDU-domain boundary for `consus-fits`.
//! It defines:
//! - HDU indexing and ordered sequencing
//! - primary and extension HDU classification
//! - header/data-unit coupling
//! - image/table payload descriptor attachment
//!
//! ## FITS invariants
//!
//! A FITS file is an ordered sequence of HDUs.
//! - The first HDU is the primary HDU.
//! - Subsequent HDUs are extension HDUs.
//! - Each HDU owns exactly one header block and one data-unit span.
//! - HDU kind is derived from authoritative header semantics.
//!
//! ## Architectural role
//!
//! This module depends on the authoritative `header`, `types`,
//! `datastructure`, `image`, and `table` modules. It does not duplicate
//! FITS keyword parsing, image metadata extraction, table metadata extraction,
//! or 2880-byte blocking math.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod descriptor;
mod index;
mod kind;
mod payload;
mod sequence;
mod support;

#[cfg(feature = "alloc")]
pub use descriptor::FitsHdu;
pub use index::FitsHduIndex;
pub use kind::FitsHduKind;
#[cfg(feature = "alloc")]
pub use payload::FitsHduPayload;
#[cfg(feature = "alloc")]
pub use sequence::FitsHduSequence;

#[cfg(all(test, feature = "alloc"))]
mod tests;
