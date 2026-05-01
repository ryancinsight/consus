//! Materialization bridge: Parquet `ColumnValues` → canonical `ArrowArray`.
//!
//! ## Specification
//!
//! `column_values_to_arrow` converts a decoded Parquet column (`ColumnValues`)
//! into the canonical `ArrowArray` type used by `consus-arrow`.
//!
//! ### Physical-type mapping
//!
//! | `ColumnValues` variant                  | `ArrayData` kind | element_width | byte order |
//! |-----------------------------------------|------------------|---------------|------------|
//! | `Boolean(Vec<bool>)`                    | `FixedWidth`     | 1             | N/A        |
//! | `Int32(Vec<i32>)`                       | `FixedWidth`     | 4             | LE         |
//! | `Int64(Vec<i64>)`                       | `FixedWidth`     | 8             | LE         |
//! | `Int96(Vec<[u8; 12]>)`                  | `FixedWidth`     | 12            | raw        |
//! | `Float(Vec<f32>)`                       | `FixedWidth`     | 4             | LE         |
//! | `Double(Vec<f64>)`                      | `FixedWidth`     | 8             | LE         |
//! | `ByteArray(Vec<Vec<u8>>)`               | `VariableWidth`  | —             | raw        |
//! | `FixedLenByteArray { fixed_len, .. }`   | `FixedWidth`     | fixed_len     | raw        |
//!
//! ### Invariants
//!
//! - Boolean: stored as 0x00 (false) or 0x01 (true), one byte per element.
//! - Fixed-width numerics: little-endian byte order (Arrow memory-format convention).
//! - Int96: raw 12 bytes preserved as decoded (Parquet INT96 timestamp encoding).
//! - VariableWidth offsets: `offsets.len() == values.len() + 1`; strictly monotone.
//! - `values_bytes.len() == len * element_width` for all FixedWidth arrays.
//! - No validity bitmap attached; required Parquet columns carry no nulls.
//!
//! ### Zero-copy path
//!
//! On little-endian architectures with the `zerocopy` feature enabled, fixed-width
//! numeric types (Int32, Int64, Float, Double) are materialized via
//! [`zerocopy::IntoBytes::as_bytes`], which reinterprets the native-memory slice
//! directly as `&[u8]` without element-by-element byte swapping.
//!
//! ## Correctness invariant
//!
//! Arrow memory format requires little-endian byte order. On LE targets the native
//! in-memory representation of `i32`, `i64`, `f32`, and `f64` is already LE, so
//! `zerocopy::IntoBytes::as_bytes()` yields the correct byte sequence directly.
//! On BE targets, element-by-element `to_le_bytes()` is required; the zerocopy
//! path is therefore `#[cfg(target_endian = "little")]`-gated.
//!
//! The fast path is active when **both** conditions hold at compile time:
//! - feature `zerocopy` is enabled, AND
//! - `target_endian = "little"`

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_parquet::ColumnValues;

use super::{ArrayData, ArrowArray};
#[cfg(feature = "alloc")]
use crate::memory::ArrowBuffer;

/// Convert a slice of [`zerocopy::IntoBytes`] primitives to a `Vec<u8>` via a
/// single bulk `memcpy`.
///
/// On little-endian architectures, the native memory layout of `i32`, `i64`,
/// `f32`, and `f64` is identical to their Arrow LE representation, so
/// `IntoBytes::as_bytes` is a zero-cost reinterpretation followed by one
/// `to_vec` allocation.  The element-by-element `to_le_bytes` path is never
/// invoked on this code path.
///
/// This function is only compiled when both `feature = "zerocopy"` and
/// `target_endian = "little"` hold.
#[cfg(all(feature = "alloc", feature = "zerocopy", target_endian = "little"))]
fn fixed_to_le_bytes_fast<'a, T: zerocopy::IntoBytes + zerocopy::Immutable>(slice: &'a [T]) -> ArrowBuffer<'a> {
    use zerocopy::IntoBytes;
    ArrowBuffer::Borrowed(slice.as_bytes())
}

/// Convert decoded Parquet column values into a canonical [`ArrowArray`].
///
/// The conversion is exact and allocation-minimal: each output buffer is
/// sized precisely to the input data with a single `Vec::with_capacity` call
/// (element-by-element path) or a single `to_vec` call (zerocopy path).
/// No validity bitmap is attached because `ColumnValues` carries no null
/// information; callers that require nullable semantics must attach a
/// [`super::ValidityBitmap`] after conversion.
#[cfg(feature = "alloc")]
#[must_use]
pub fn column_values_to_arrow<'a>(values: &'a ColumnValues) -> ArrowArray<'a> {
    match values {
        ColumnValues::Boolean(bools) => {
            let bytes: Vec<u8> = bools.iter().map(|&b| u8::from(b)).collect();
            ArrowArray::new(ArrayData::FixedWidth {
                len: bools.len(),
                element_width: 1,
                values: ArrowBuffer::owned(bytes),
                validity: None,
            })
        }

        ColumnValues::Int32(ints) => {
            #[cfg(all(feature = "zerocopy", target_endian = "little"))]
            let bytes = fixed_to_le_bytes_fast(ints.as_slice());
            #[cfg(not(all(feature = "zerocopy", target_endian = "little")))]
            let bytes = {
                let mut b = Vec::with_capacity(ints.len() * 4);
                for &v in ints {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                ArrowBuffer::owned(b)
            };
            ArrowArray::new(ArrayData::FixedWidth {
                len: ints.len(),
                element_width: 4,
                values: bytes,
                validity: None,
            })
        }

        ColumnValues::Int64(ints) => {
            #[cfg(all(feature = "zerocopy", target_endian = "little"))]
            let bytes = fixed_to_le_bytes_fast(ints.as_slice());
            #[cfg(not(all(feature = "zerocopy", target_endian = "little")))]
            let bytes = {
                let mut b = Vec::with_capacity(ints.len() * 8);
                for &v in ints {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                ArrowBuffer::owned(b)
            };
            ArrowArray::new(ArrayData::FixedWidth {
                len: ints.len(),
                element_width: 8,
                values: bytes,
                validity: None,
            })
        }

        ColumnValues::Int96(raw) => {
            let mut bytes = Vec::with_capacity(raw.len() * 12);
            for arr in raw {
                bytes.extend_from_slice(arr.as_ref());
            }
            ArrowArray::new(ArrayData::FixedWidth {
                len: raw.len(),
                element_width: 12,
                values: ArrowBuffer::owned(bytes),
                validity: None,
            })
        }

        ColumnValues::Float(floats) => {
            #[cfg(all(feature = "zerocopy", target_endian = "little"))]
            let bytes = fixed_to_le_bytes_fast(floats.as_slice());
            #[cfg(not(all(feature = "zerocopy", target_endian = "little")))]
            let bytes = {
                let mut b = Vec::with_capacity(floats.len() * 4);
                for &v in floats {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                ArrowBuffer::owned(b)
            };
            ArrowArray::new(ArrayData::FixedWidth {
                len: floats.len(),
                element_width: 4,
                values: bytes,
                validity: None,
            })
        }

        ColumnValues::Double(doubles) => {
            #[cfg(all(feature = "zerocopy", target_endian = "little"))]
            let bytes = fixed_to_le_bytes_fast(doubles.as_slice());
            #[cfg(not(all(feature = "zerocopy", target_endian = "little")))]
            let bytes = {
                let mut b = Vec::with_capacity(doubles.len() * 8);
                for &v in doubles {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                ArrowBuffer::owned(b)
            };
            ArrowArray::new(ArrayData::FixedWidth {
                len: doubles.len(),
                element_width: 8,
                values: bytes,
                validity: None,
            })
        }

        ColumnValues::ByteArray(bufs) => {
            let total: usize = bufs.iter().map(|b| b.len()).sum();
            let mut offsets_bytes: Vec<u8> = Vec::with_capacity((bufs.len() + 1) * 4);
            let mut payload: Vec<u8> = Vec::with_capacity(total);
            offsets_bytes.extend_from_slice(&0i32.to_le_bytes());
            for buf in bufs {
                payload.extend_from_slice(buf);
                let current_offset = i32::try_from(payload.len()).expect("size exceeds i32");
                offsets_bytes.extend_from_slice(&current_offset.to_le_bytes());
            }
            ArrowArray::new(ArrayData::VariableWidth {
                len: bufs.len(),
                offsets: crate::memory::ArrowOffsets::new(ArrowBuffer::owned(offsets_bytes), bufs.len()),
                values: ArrowBuffer::owned(payload),
                validity: None,
            })
        }

        ColumnValues::FixedLenByteArray {
            fixed_len,
            values: bufs,
        } => {
            let mut bytes = Vec::with_capacity(bufs.len() * fixed_len);
            for buf in bufs {
                bytes.extend_from_slice(buf);
            }
            ArrowArray::new(ArrayData::FixedWidth {
                len: bufs.len(),
                element_width: *fixed_len,
                values: ArrowBuffer::owned(bytes),
                validity: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArrayData;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn fixed_bytes<'a>(array: &'a ArrowArray<'a>) -> (&'a usize, &'a usize, &'a [u8]) {
        match &array.data {
            ArrayData::FixedWidth {
                len,
                element_width,
                values,
                ..
            } => (len, element_width, values.as_slice()),
            _ => panic!("expected FixedWidth, got VariableWidth"),
        }
    }

    fn var_parts<'a>(array: &'a ArrowArray<'a>) -> (&'a usize, &'a [u8], &'a [u8]) {
        match &array.data {
            ArrayData::VariableWidth {
                len,
                offsets,
                values,
                ..
            } => (len, offsets.as_slice(), values.as_slice()),
            _ => panic!("expected VariableWidth, got FixedWidth"),
        }
    }

    // ── boolean ──────────────────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn boolean_false_and_true_map_to_zero_and_one() {
        let cv = ColumnValues::Boolean(alloc::vec![false, true, false, true]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 4);
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 4);
        assert_eq!(*width, 1);
        assert_eq!(bytes, &[0u8, 1, 0, 1]);
        assert!(array.is_all_valid());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn empty_boolean_array_produces_empty_fixed_width() {
        let cv = ColumnValues::Boolean(alloc::vec![]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 0);
        assert!(array.is_empty());
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 0);
        assert_eq!(*width, 1);
        assert!(bytes.is_empty());
    }

    // ── int32 ─────────────────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn int32_three_values_stored_little_endian() {
        let cv = ColumnValues::Int32(alloc::vec![1i32, -1i32, 0x0102_0304i32]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 3);
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 3);
        assert_eq!(*width, 4);
        assert_eq!(bytes.len(), *len * *width);
        assert_eq!(&bytes[0..4], &1i32.to_le_bytes());
        assert_eq!(&bytes[4..8], &(-1i32).to_le_bytes());
        assert_eq!(&bytes[8..12], &0x0102_0304i32.to_le_bytes());
    }

    // ── int64 ─────────────────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn int64_two_values_stored_little_endian() {
        let cv = ColumnValues::Int64(alloc::vec![i64::MAX, i64::MIN]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 2);
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 2);
        assert_eq!(*width, 8);
        assert_eq!(bytes.len(), *len * *width);
        assert_eq!(&bytes[0..8], &i64::MAX.to_le_bytes());
        assert_eq!(&bytes[8..16], &i64::MIN.to_le_bytes());
    }

    // ── int96 ─────────────────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn int96_one_value_raw_twelve_bytes_preserved() {
        let raw: [u8; 12] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C,
        ];
        let cv = ColumnValues::Int96(alloc::vec![raw]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 1);
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 1);
        assert_eq!(*width, 12);
        assert_eq!(bytes, raw.as_ref());
    }

    // ── float32 ───────────────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn float32_two_values_stored_little_endian() {
        let cv = ColumnValues::Float(alloc::vec![1.5f32, -2.5f32]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 2);
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 2);
        assert_eq!(*width, 4);
        assert_eq!(bytes.len(), *len * *width);
        assert_eq!(&bytes[0..4], &1.5f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &(-2.5f32).to_le_bytes());
    }

    // ── float64 ───────────────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn double_two_values_stored_little_endian() {
        let cv = ColumnValues::Double(alloc::vec![1.0f64, f64::INFINITY]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 2);
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 2);
        assert_eq!(*width, 8);
        assert_eq!(bytes.len(), *len * *width);
        assert_eq!(&bytes[0..8], &1.0f64.to_le_bytes());
        assert_eq!(&bytes[8..16], &f64::INFINITY.to_le_bytes());
    }

    // ── byte array ────────────────────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn byte_array_two_entries_variable_width_offsets_and_payload() {
        let v0 = alloc::vec![0x61u8, 0x62]; // b"ab"
        let v1 = alloc::vec![0x63u8, 0x64, 0x65]; // b"cde"
        let cv = ColumnValues::ByteArray(alloc::vec![v0, v1]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 2);
        let (len, offsets, payload) = var_parts(&array);
        assert_eq!(*len, 2);
        // offsets: [0, 2, 5] encoded as i32 LE
        assert_eq!(offsets, &[0u8, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0]);
        assert_eq!(payload, &[0x61u8, 0x62, 0x63, 0x64, 0x65]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn empty_byte_array_column_produces_singleton_offset() {
        let cv = ColumnValues::ByteArray(alloc::vec![]);
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 0);
        assert!(array.is_empty());
        let (len, offsets, payload) = var_parts(&array);
        assert_eq!(*len, 0);
        assert_eq!(offsets, &[0u8, 0, 0, 0]);
        assert!(payload.is_empty());
    }

    // ── fixed-length byte array ───────────────────────────────────────────

    #[cfg(feature = "alloc")]
    #[test]
    fn fixed_len_byte_array_two_values_concatenated() {
        let v0 = alloc::vec![0xAAu8, 0xBB, 0xCC];
        let v1 = alloc::vec![0xDDu8, 0xEE, 0xFF];
        let cv = ColumnValues::FixedLenByteArray {
            fixed_len: 3,
            values: alloc::vec![v0, v1],
        };
        let array = column_values_to_arrow(&cv);
        assert_eq!(array.len(), 2);
        let (len, width, bytes) = fixed_bytes(&array);
        assert_eq!(*len, 2);
        assert_eq!(*width, 3);
        assert_eq!(bytes.len(), *len * *width);
        assert_eq!(bytes, &[0xAAu8, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    // ── zerocopy agreement tests ──────────────────────────────────────────
    //
    // These tests verify that the zerocopy fast path produces byte-for-byte
    // identical output to the element-by-element `to_le_bytes()` reference
    // path. Both paths are evaluated indirectly: the test calls
    // `column_values_to_arrow` (which uses the fast path when the feature is
    // active) and compares the result against a locally computed reference
    // built with `to_le_bytes()`.
    //
    // Correctness foundation: on LE targets, `IntoBytes::as_bytes` for `[i32]`
    // and `[f64]` produces the same bytes as iterating `to_le_bytes()` on each
    // element because the native memory representation equals the LE wire
    // representation.

    #[cfg(all(feature = "alloc", feature = "zerocopy", target_endian = "little"))]
    #[test]
    fn zerocopy_i32_agrees_with_element_loop() {
        // Covers positive, negative, MAX, and MIN to exercise all sign-extension
        // and bit-pattern boundary cases.
        let ints = alloc::vec![1i32, -1, i32::MAX, i32::MIN];
        let cv = ColumnValues::Int32(ints.clone());
        let arr = column_values_to_arrow(&cv);
        // Reference: element-by-element LE encoding.
        let mut expected = Vec::with_capacity(ints.len() * 4);
        for &v in &ints {
            expected.extend_from_slice(&v.to_le_bytes());
        }
        let bytes = match &arr.data {
            ArrayData::FixedWidth {
                len,
                element_width,
                values,
                ..
            } => {
                assert_eq!(*len, ints.len());
                assert_eq!(*element_width, 4);
                assert_eq!(values.len(), ints.len() * 4);
                values.as_slice()
            }
            _ => panic!("expected FixedWidth"),
        };
        assert_eq!(bytes, expected.as_slice());
    }

    #[cfg(all(feature = "alloc", feature = "zerocopy", target_endian = "little"))]
    #[test]
    fn zerocopy_f64_agrees_with_element_loop() {
        // Covers normal, negative, and non-finite values to exercise all
        // IEEE 754 bit-pattern classes that appear in real data.
        let doubles = alloc::vec![1.5f64, -0.25, f64::INFINITY, f64::NEG_INFINITY];
        let cv = ColumnValues::Double(doubles.clone());
        let arr = column_values_to_arrow(&cv);
        // Reference: element-by-element LE encoding.
        let mut expected = Vec::with_capacity(doubles.len() * 8);
        for &v in &doubles {
            expected.extend_from_slice(&v.to_le_bytes());
        }
        let bytes = match &arr.data {
            ArrayData::FixedWidth {
                len,
                element_width,
                values,
                ..
            } => {
                assert_eq!(*len, doubles.len());
                assert_eq!(*element_width, 8);
                assert_eq!(values.len(), doubles.len() * 8);
                values.as_slice()
            }
            _ => panic!("expected FixedWidth"),
        };
        assert_eq!(bytes, expected.as_slice());
    }
}
