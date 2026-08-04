//! FITS file wrapper with HDU scanning and `consus-core` trait integration.
//!
//! ## Scope
//!
//! This module is the authoritative file-domain boundary for `consus-fits`.
//! It defines:
//! - FITS file scanning over positioned I/O
//! - ordered HDU indexing and traversal
//! - synthetic path mapping from FITS HDUs to `consus-core` file traits
//! - raw dataset reads and writes for image and table HDUs
//!
//! ## Architectural mapping
//!
//! `consus-core` models hierarchical containers with groups and datasets,
//! while FITS is an ordered sequence of HDUs. This module uses the minimal
//! deterministic mapping:
//! - `/` => synthetic root group
//! - `/PRIMARY` => primary HDU dataset
//! - `/HDU/{n}` => HDU dataset at zero-based ordinal `n`
//!
//! Header cards remain available through the concrete `FitsFile` API and are
//! not projected into the `consus-core` node model.
//!
//! ## Invariants
//!
//! - HDU scan order matches on-disk order.
//! - The first HDU is primary.
//! - Header and data extents remain 2880-byte aligned.
//! - Dataset paths resolve only to HDU payloads.
//! - Image HDUs expose canonical numeric datatypes.
//! - Table HDUs expose row-wise opaque records.
//!
//! ## Current write semantics
//!
//! - Existing HDU payloads may be overwritten in-place.
//! - Structural mutation of the HDU sequence is not implemented.
//! - `create_group` is unsupported except for `/`.
//! - Partial image selections are not implemented.
//! - Table selections support `All`, `None`, and contiguous 1-D row hyperslabs.

mod read;
pub(crate) mod types;
mod write;

pub use types::{FITS_FORMAT_NAME, FitsFile};
