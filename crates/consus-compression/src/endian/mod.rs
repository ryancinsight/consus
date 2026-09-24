//! Byte-order utilities for multi-byte integer reading and writing.
//!
//! The implementations now live in [`consus_core::decode`], the single home
//! for fixed-width byte-order decoding across Consus. This module re-exports
//! them so the historical `consus_compression::endian` path keeps working.

pub mod conversion;

pub use conversion::{
    read_length, read_offset, read_uint_be, read_uint_le, swap_bytes, write_uint_be, write_uint_le,
};
