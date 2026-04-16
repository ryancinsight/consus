//! HDF5 parsing context and address utilities.
//!
//! The parsing context carries superblock-derived structural parameters
//! (offset size, length size) needed by all binary parsers in the crate.

use crate::primitives::{read_length, read_offset};

/// Parsing context derived from the superblock.
///
/// Carries the variable-width field sizes needed to parse HDF5 structures.
/// Passed by reference to all parsing functions.
#[derive(Debug, Clone, Copy)]
pub struct ParseContext {
    /// Size of file offsets in bytes (2, 4, or 8).
    pub offset_size: u8,
    /// Size of file lengths in bytes (2, 4, or 8).
    pub length_size: u8,
}

impl ParseContext {
    /// Create a new parsing context.
    pub const fn new(offset_size: u8, length_size: u8) -> Self {
        Self {
            offset_size,
            length_size,
        }
    }

    /// Read a file offset from a buffer using this context's offset size.
    pub fn read_offset(&self, buf: &[u8]) -> u64 {
        read_offset(buf, self.offset_size as usize)
    }

    /// Read a file length from a buffer using this context's length size.
    pub fn read_length(&self, buf: &[u8]) -> u64 {
        read_length(buf, self.length_size as usize)
    }

    /// Offset size as usize for buffer arithmetic.
    pub const fn offset_bytes(&self) -> usize {
        self.offset_size as usize
    }

    /// Length size as usize for buffer arithmetic.
    pub const fn length_bytes(&self) -> usize {
        self.length_size as usize
    }
}
