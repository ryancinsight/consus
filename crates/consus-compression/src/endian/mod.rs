//! Byte-order utilities for multi-byte integer reading and writing.
//!
//! This module is the SSOT for all endian conversion in Consus.
//! No other crate may duplicate these implementations.
//! (consus-hdf5's `primitives.rs` should be replaced by imports from here.)

pub mod conversion;

pub use conversion::{
    read_length, read_offset, read_uint_be, read_uint_le, swap_bytes, write_uint_be, write_uint_le,
};
