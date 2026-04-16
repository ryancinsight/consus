//! Low-level binary reading helpers for HDF5 format parsing.
//!
//! These functions read offset and length fields of variable size
//! as specified by the superblock's `offset_size` and `length_size`.

use byteorder::{ByteOrder, LittleEndian};

/// Read a file offset of `size` bytes (little-endian) from a buffer.
///
/// Supports 2, 4, and 8 byte offsets as per HDF5 spec.
pub fn read_offset(buf: &[u8], size: usize) -> u64 {
    match size {
        2 => LittleEndian::read_u16(buf) as u64,
        4 => LittleEndian::read_u32(buf) as u64,
        8 => LittleEndian::read_u64(buf),
        _ => 0, // invalid; caught during validation
    }
}

/// Read a file length of `size` bytes (little-endian) from a buffer.
///
/// Identical encoding to offsets, but semantically distinct.
pub fn read_length(buf: &[u8], size: usize) -> u64 {
    read_offset(buf, size)
}
