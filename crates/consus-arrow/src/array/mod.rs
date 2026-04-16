/// Arrow array model.
///
/// This module defines a Rust-native columnar array representation for the
/// `consus-arrow` crate. It is intentionally independent of any external Arrow
/// implementation so it can serve as a canonical bridge layer for Consus.
///
/// ## Specification
///
/// A columnar array is a typed sequence of values with optional nulls.
/// The model is split into:
/// - physical representation (`ArrayData`)
/// - logical array wrappers (`ArrowArray`)
/// - validity bitmap semantics (`ValidityBitmap`)
/// - slicing and projection helpers
///
/// ## Invariants
///
/// - `len` is the logical row count.
/// - `null_count <= len`.
/// - `values.len() >= len * element_width` for fixed-width arrays.
/// - Variable-width arrays keep offsets monotonic and length-consistent.
/// - Validity and value buffers are independent concerns.
///
/// This file is the authoritative array layer for the crate.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use core::fmt;

/// Null bitmap for an array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidityBitmap {
    /// Packed validity bits, least-significant bit first.
    #[cfg(feature = "alloc")]
    bits: Vec<u8>,
    /// Logical number of values represented by the bitmap.
    len: usize,
}

impl ValidityBitmap {
    /// Create an all-valid bitmap for `len` values.
    #[must_use]
    pub fn all_valid(len: usize) -> Self {
        let byte_len = len.div_ceil(8);
        Self {
            #[cfg(feature = "alloc")]
            bits: vec![0xFF; byte_len],
            len,
        }
    }

    /// Create an empty bitmap.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            #[cfg(feature = "alloc")]
            bits: Vec::new(),
            len: 0,
        }
    }

    /// Return the logical length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitmap has no logical values.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the value at `index` is valid.
    #[must_use]
    pub fn is_valid(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }

        #[cfg(feature = "alloc")]
        {
            let byte = self.bits[index / 8];
            let mask = 1u8 << (index % 8);
            return byte & mask != 0;
        }

        #[cfg(not(feature = "alloc"))]
        {
            let _ = index;
            false
        }
    }
}

impl fmt::Display for ValidityBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValidityBitmap(len={})", self.len)
    }
}

/// Fixed-width or variable-width array payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayData {
    /// Fixed-width values with a contiguous byte buffer.
    FixedWidth {
        /// Number of logical values.
        len: usize,
        /// Width of each element in bytes.
        element_width: usize,
        /// Raw value bytes.
        #[cfg(feature = "alloc")]
        values: Vec<u8>,
        /// Optional validity bitmap.
        validity: Option<ValidityBitmap>,
    },
    /// Variable-width UTF-8 or binary values.
    VariableWidth {
        /// Number of logical values.
        len: usize,
        /// Offsets into the values buffer. Length is `len + 1`.
        #[cfg(feature = "alloc")]
        offsets: Vec<usize>,
        /// Raw concatenated payload bytes.
        #[cfg(feature = "alloc")]
        values: Vec<u8>,
        /// Optional validity bitmap.
        validity: Option<ValidityBitmap>,
    },
}

impl ArrayData {
    /// Return the logical length.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::FixedWidth { len, .. } | Self::VariableWidth { len, .. } => *len,
        }
    }

    /// Returns `true` if the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of nulls if a validity bitmap exists.
    #[must_use]
    pub fn null_count(&self) -> usize {
        let len = self.len();
        match self.validity() {
            Some(bitmap) => (0..len).filter(|&i| !bitmap.is_valid(i)).count(),
            None => 0,
        }
    }

    /// Borrow the validity bitmap.
    #[must_use]
    pub fn validity(&self) -> Option<&ValidityBitmap> {
        match self {
            Self::FixedWidth { validity, .. } | Self::VariableWidth { validity, .. } => {
                validity.as_ref()
            }
        }
    }

    /// Returns `true` if all values are valid.
    #[must_use]
    pub fn is_all_valid(&self) -> bool {
        self.null_count() == 0
    }
}

/// Arrow-style array wrapper that pairs physical buffers with a schema type.
///
/// This is the canonical array abstraction used by `consus-arrow`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowArray {
    /// Logical array data.
    pub data: ArrayData,
}

impl ArrowArray {
    /// Create a new array wrapper.
    #[must_use]
    pub fn new(data: ArrayData) -> Self {
        Self { data }
    }

    /// Logical length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of null values.
    #[must_use]
    pub fn null_count(&self) -> usize {
        self.data.null_count()
    }

    /// Returns `true` if all values are valid.
    #[must_use]
    pub fn is_all_valid(&self) -> bool {
        self.data.is_all_valid()
    }

    /// Slice the array by logical row range.
    ///
    /// This method preserves the logical model and validates boundaries.
    #[must_use]
    pub fn slice(&self, offset: usize, length: usize) -> Option<Self> {
        if offset.checked_add(length)? > self.len() {
            return None;
        }

        match &self.data {
            ArrayData::FixedWidth {
                element_width,
                validity,
                #[cfg(feature = "alloc")]
                values,
                ..
            } => {
                let start = offset * *element_width;
                let end = start + length * *element_width;

                #[cfg(feature = "alloc")]
                {
                    let sliced = values.get(start..end)?.to_vec();
                    Some(Self::new(ArrayData::FixedWidth {
                        len: length,
                        element_width: *element_width,
                        values: sliced,
                        validity: validity.clone(),
                    }))
                }

                #[cfg(not(feature = "alloc"))]
                {
                    let _ = (start, end);
                    let _ = validity;
                    None
                }
            }
            ArrayData::VariableWidth {
                validity,
                #[cfg(feature = "alloc")]
                offsets,
                #[cfg(feature = "alloc")]
                values,
                ..
            } => {
                #[cfg(feature = "alloc")]
                {
                    let start = *offsets.get(offset)?;
                    let end = *offsets.get(offset + length)?;
                    let sliced_values = values.get(start..end)?.to_vec();

                    let mut sliced_offsets = Vec::with_capacity(length + 1);
                    for idx in offset..=offset + length {
                        sliced_offsets.push(offsets[idx] - start);
                    }

                    Some(Self::new(ArrayData::VariableWidth {
                        len: length,
                        offsets: sliced_offsets,
                        values: sliced_values,
                        validity: validity.clone(),
                    }))
                }

                #[cfg(not(feature = "alloc"))]
                {
                    let _ = validity;
                    None
                }
            }
        }
    }
}

impl fmt::Display for ArrowArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.data {
            ArrayData::FixedWidth {
                len, element_width, ..
            } => write!(
                f,
                "ArrowArray::FixedWidth(len={}, element_width={})",
                len, element_width
            ),
            ArrayData::VariableWidth { len, .. } => {
                write!(f, "ArrowArray::VariableWidth(len={})", len)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validity_bitmap_reports_length() {
        let bitmap = ValidityBitmap::all_valid(10);
        assert_eq!(bitmap.len(), 10);
        assert!(!bitmap.is_empty());
    }

    #[test]
    fn fixed_width_array_reports_length() {
        #[cfg(feature = "alloc")]
        let array = ArrowArray::new(ArrayData::FixedWidth {
            len: 4,
            element_width: 8,
            values: vec![0; 32],
            validity: Some(ValidityBitmap::all_valid(4)),
        });

        #[cfg(feature = "alloc")]
        {
            assert_eq!(array.len(), 4);
            assert_eq!(array.null_count(), 0);
            assert!(array.is_all_valid());
        }
    }

    #[test]
    fn variable_width_slice_is_consistent() {
        #[cfg(feature = "alloc")]
        let array = ArrowArray::new(ArrayData::VariableWidth {
            len: 2,
            offsets: vec![0, 3, 6],
            values: b"abcdef".to_vec(),
            validity: None,
        });

        #[cfg(feature = "alloc")]
        {
            let sliced = array.slice(1, 1).expect("slice must succeed");
            assert_eq!(sliced.len(), 1);
            assert!(sliced.is_all_valid());
        }
    }
}
