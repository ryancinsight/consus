pub mod materialize;
#[cfg(feature = "alloc")]
pub use materialize::column_values_to_arrow;

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

#[cfg(feature = "alloc")]
use crate::memory::{ArrowBitmap, ArrowBuffer, ArrowOffsets};

/// Fixed-width or variable-width array payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayData<'a> {
    /// Fixed-width values with a contiguous byte buffer.
    FixedWidth {
        /// Number of logical values.
        len: usize,
        /// Width of each element in bytes.
        element_width: usize,
        /// Raw value bytes.
        #[cfg(feature = "alloc")]
        values: ArrowBuffer<'a>,
        /// Optional validity bitmap.
        #[cfg(feature = "alloc")]
        validity: Option<ArrowBitmap<'a>>,
        #[cfg(not(feature = "alloc"))]
        validity: Option<()>,
    },
    /// Variable-width UTF-8 or binary values.
    VariableWidth {
        /// Number of logical values.
        len: usize,
        /// Offsets into the values buffer. Length is `len + 1`.
        #[cfg(feature = "alloc")]
        offsets: ArrowOffsets<'a>,
        /// Raw concatenated payload bytes.
        #[cfg(feature = "alloc")]
        values: ArrowBuffer<'a>,
        /// Optional validity bitmap.
        #[cfg(feature = "alloc")]
        validity: Option<ArrowBitmap<'a>>,
        #[cfg(not(feature = "alloc"))]
        validity: Option<()>,
    },
}

impl<'a> ArrayData<'a> {
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
        #[cfg(feature = "alloc")]
        {
            let len = self.len();
            match self.validity() {
                Some(bitmap) => (0..len).filter(|&i| !bitmap.is_set(i)).count(),
                None => 0,
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            0
        }
    }

    /// Borrow the validity bitmap.
    #[must_use]
    #[cfg(feature = "alloc")]
    pub fn validity(&self) -> Option<&ArrowBitmap<'a>> {
        match self {
            Self::FixedWidth { validity, .. } | Self::VariableWidth { validity, .. } => {
                validity.as_ref()
            }
        }
    }

    /// Borrow the validity bitmap.
    #[must_use]
    #[cfg(not(feature = "alloc"))]
    pub fn validity(&self) -> Option<&()> {
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
pub struct ArrowArray<'a> {
    /// Logical array data.
    pub data: ArrayData<'a>,
}

impl<'a> ArrowArray<'a> {
    /// Create a new array wrapper.
    #[must_use]
    pub fn new(data: ArrayData<'a>) -> Self {
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
                    let sliced = values.as_slice().get(start..end)?;
                    Some(Self::new(ArrayData::FixedWidth {
                        len: length,
                        element_width: *element_width,
                        values: ArrowBuffer::owned(sliced.to_vec()), // Simplified fallback: could preserve Cow via borrowed if lifetime allows, but slicing might require new owned wrapper if the original was owned and we return an independent slice. Since ArrowBuffer is an enum, we could actually use `.into_owned()` or clone. Let's just clone. Wait! If `values` is Borrowed, we can return Borrowed.
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
                    // For ArrowOffsets, we read little-endian i32 values.
                    let offset_bytes = offsets.as_slice();
                    
                    let get_i32 = |idx: usize| -> Option<usize> {
                        let start = idx * 4;
                        let bytes = offset_bytes.get(start..start + 4)?;
                        Some(i32::from_le_bytes(bytes.try_into().ok()?) as usize)
                    };
                    
                    let start = get_i32(offset)?;
                    let end = get_i32(offset + length)?;
                    let sliced_values = values.as_slice().get(start..end)?;

                    let mut sliced_offsets_bytes = Vec::with_capacity((length + 1) * 4);
                    for idx in offset..=offset + length {
                        let val = get_i32(idx)?;
                        sliced_offsets_bytes.extend_from_slice(&(i32::try_from(val - start).ok()?).to_le_bytes());
                    }

                    Some(Self::new(ArrayData::VariableWidth {
                        len: length,
                        offsets: ArrowOffsets::new(ArrowBuffer::owned(sliced_offsets_bytes), length),
                        values: ArrowBuffer::owned(sliced_values.to_vec()),
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

impl<'a> fmt::Display for ArrowArray<'a> {
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
        let bitmap = ArrowBitmap::new(ArrowBuffer::owned(vec![0xFF, 0xFF]), 10);
        assert_eq!(bitmap.len_bits(), 10);
        assert!(!bitmap.is_empty());
    }

    #[test]
    fn fixed_width_array_reports_length() {
        #[cfg(feature = "alloc")]
        let array = ArrowArray::new(ArrayData::FixedWidth {
            len: 4,
            element_width: 8,
            values: ArrowBuffer::owned(vec![0; 32]),
            validity: Some(ArrowBitmap::new(ArrowBuffer::owned(vec![0xFF]), 4)),
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
            offsets: ArrowOffsets::new(ArrowBuffer::owned(vec![0, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0]), 2), // 0, 3, 6 as i32 LE
            values: ArrowBuffer::owned(b"abcdef".to_vec()),
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
