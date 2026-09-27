//! Low-level binary reading helpers for HDF5 format parsing.
//!
//! Re-export shim over [`consus_core::decode`], the single home for
//! fixed-width byte-order decoding. These read offset and length fields of
//! variable size as specified by the superblock's `offset_size` and
//! `length_size`.

pub use consus_core::{read_length, read_offset};
