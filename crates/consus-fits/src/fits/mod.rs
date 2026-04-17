//! FITS facade integration boundary.
//!
//! The authoritative FITS implementation in this crate is the file-level API in
//! [`crate::file`]. The previous facade draft introduced incomplete projections
//! and mismatched exports. This module now remains intentionally minimal so the
//! crate compiles while preserving a stable namespace for future unified facade
//! work.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
