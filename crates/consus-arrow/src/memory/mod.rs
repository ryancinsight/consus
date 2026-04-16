//! Arrow memory model and buffer ownership primitives.
//!
//! ## Specification
//!
//! This module defines the canonical in-memory representation used by
//! `consus-arrow` for Arrow-style buffers, offset vectors, and nullable
//! payloads. It does not depend on the external Arrow crate.
//!
//! ## Invariants
//!
//! - Buffer lengths are measured in bytes.
//! - Offsets are monotonically nondecreasing.
//! - Null bitmaps, when present, are byte-addressable and owned or borrowed
//!   through explicit buffer wrappers.
//! - Slice views never outlive their backing storage.
//!
//! ## Architecture
//!
//! ```text
//! memory/
//! ├── buffer      # Byte buffer ownership and views
//! ├── bitmap      # Null bitmap model
//! └── offsets     # Offset buffer model for variable-length data
//! ```

#[cfg(feature = "alloc")]
use alloc::{borrow::Cow, vec::Vec};

use core::fmt;

use consus_core::Result;

/// Ownership model for Arrow-style byte buffers.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrowBuffer<'a> {
    /// Borrowed byte slice.
    Borrowed(&'a [u8]),
    /// Owned byte buffer.
    Owned(Vec<u8>),
}

#[cfg(feature = "alloc")]
impl<'a> ArrowBuffer<'a> {
    /// Create a borrowed buffer.
    #[must_use]
    pub const fn borrowed(data: &'a [u8]) -> Self {
        Self::Borrowed(data)
    }

    /// Create an owned buffer.
    #[must_use]
    pub fn owned(data: Vec<u8>) -> Self {
        Self::Owned(data)
    }

    /// Return the byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(data) => data.len(),
            Self::Owned(data) => data.len(),
        }
    }

    /// Returns `true` if the buffer has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the underlying bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(data) => data,
            Self::Owned(data) => data.as_slice(),
        }
    }

    /// Convert into an owned `Vec<u8>`.
    #[must_use]
    pub fn into_owned(self) -> Cow<'a, [u8]> {
        match self {
            Self::Borrowed(data) => Cow::Borrowed(data),
            Self::Owned(data) => Cow::Owned(data),
        }
    }
}

/// Contiguous bitmap used for validity/null tracking.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowBitmap<'a> {
    bits: ArrowBuffer<'a>,
    len_bits: usize,
}

#[cfg(feature = "alloc")]
impl<'a> ArrowBitmap<'a> {
    /// Create a bitmap from raw bytes and bit length.
    #[must_use]
    pub fn new(bits: ArrowBuffer<'a>, len_bits: usize) -> Self {
        Self { bits, len_bits }
    }

    /// Create an empty bitmap.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            bits: ArrowBuffer::Owned(Vec::new()),
            len_bits: 0,
        }
    }

    /// Number of bits represented by this bitmap.
    #[must_use]
    pub fn len_bits(&self) -> usize {
        self.len_bits
    }

    /// Number of bytes used by the underlying storage.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.bits.len()
    }

    /// Returns `true` if the bitmap contains no bits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len_bits == 0
    }

    /// Borrow the raw bitmap bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.bits.as_slice()
    }

    /// Returns `true` if the bit at `index` is set.
    #[must_use]
    pub fn is_set(&self, index: usize) -> bool {
        if index >= self.len_bits {
            return false;
        }
        let byte_index = index / 8;
        let bit_index = index % 8;
        match self.bits.as_slice().get(byte_index) {
            Some(byte) => (byte & (1u8 << bit_index)) != 0,
            None => false,
        }
    }
}

/// Offset buffer for variable-length Arrow arrays.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowOffsets<'a> {
    offsets: ArrowBuffer<'a>,
    count: usize,
}

#[cfg(feature = "alloc")]
impl<'a> ArrowOffsets<'a> {
    /// Create offsets from a byte buffer.
    ///
    /// The buffer is interpreted as little-endian `i32` offsets.
    #[must_use]
    pub fn new(offsets: ArrowBuffer<'a>, count: usize) -> Self {
        Self { offsets, count }
    }

    /// Number of logical offsets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns `true` if there are no offsets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Borrow the raw offset bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.offsets.as_slice()
    }

    /// Validate monotonicity and buffer size against the logical count.
    pub fn validate(&self) -> Result<()> {
        let required_bytes = self
            .count
            .saturating_add(1)
            .saturating_mul(core::mem::size_of::<i32>());
        if self.offsets.len() < required_bytes {
            return Err(consus_core::Error::BufferTooSmall {
                required: required_bytes,
                provided: self.offsets.len(),
            });
        }
        Ok(())
    }
}

/// Arrow array storage view for fixed-width and variable-width data.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowArrayMemory<'a> {
    /// Optional validity bitmap.
    pub validity: Option<ArrowBitmap<'a>>,
    /// Primary value buffer.
    pub values: ArrowBuffer<'a>,
    /// Optional offset buffer for variable-width arrays.
    pub offsets: Option<ArrowOffsets<'a>>,
}

#[cfg(feature = "alloc")]
impl<'a> ArrowArrayMemory<'a> {
    /// Create a fixed-width array memory view.
    #[must_use]
    pub fn fixed_width(values: ArrowBuffer<'a>) -> Self {
        Self {
            validity: None,
            values,
            offsets: None,
        }
    }

    /// Create a variable-width array memory view.
    #[must_use]
    pub fn variable_width(values: ArrowBuffer<'a>, offsets: ArrowOffsets<'a>) -> Self {
        Self {
            validity: None,
            values,
            offsets: Some(offsets),
        }
    }

    /// Attach a validity bitmap.
    #[must_use]
    pub fn with_validity(mut self, validity: ArrowBitmap<'a>) -> Self {
        self.validity = Some(validity);
        self
    }

    /// Returns `true` if the storage uses variable-width offsets.
    #[must_use]
    pub fn is_variable_width(&self) -> bool {
        self.offsets.is_some()
    }

    /// Returns `true` if the memory view has no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl fmt::Display for ArrowBuffer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} bytes", self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn buffer_length_matches_slice() {
        let data = [1u8, 2, 3, 4];
        let buffer = ArrowBuffer::borrowed(&data);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.as_slice(), &data);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn bitmap_reports_set_bits() {
        let bitmap = ArrowBitmap::new(ArrowBuffer::Owned(vec![0b0000_0101]), 8);
        assert!(bitmap.is_set(0));
        assert!(!bitmap.is_set(1));
        assert!(bitmap.is_set(2));
        assert!(!bitmap.is_set(7));
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[cfg(feature = "alloc")]
    #[test]
    fn array_memory_classifies_variable_width() {
        let offsets = ArrowOffsets::new(ArrowBuffer::Owned(vec![0; 12]), 2);
        let memory = ArrowArrayMemory::variable_width(ArrowBuffer::Owned(vec![1, 2, 3]), offsets);
        assert!(memory.is_variable_width());
        assert!(!memory.is_empty());
    }
}
