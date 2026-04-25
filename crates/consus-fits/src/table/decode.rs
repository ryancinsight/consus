//! FITS table column value decoding.
//!
//! Authoritative decoder for FITS table column cell values.
//! - `FitsColumnValue`: decoded value for one column cell.
//! - `decode_binary_column`: decode one column from a binary table row buffer.
//! - `decode_ascii_column`: decode one column from an ASCII table row buffer.
//!
//! ## FITS Standard Reference
//!
//! Binary table encodings per FITS Standard 4.0 section 7.3.
//! - All multi-byte integers and floats are stored big-endian.
//! - Logical: b'T' = true, b'F' / 0x00 = false, other nonzero = true.
//! - Char: repeat ASCII bytes form one fixed-length string.
//! - Bit: repeat bits in ceil(repeat/8) bytes, MSB-first.
//!
//! ASCII table encodings per FITS Standard 4.0 section 7.2.
//! - D exponent notation uses 'D' in place of 'E'.
//!
//! ## Invariants
//!
//! For `decode_binary_column`: `row.len() >= col.col_offset() + col.byte_width()`
//! For `decode_ascii_column`: `row.len() >= col.col_offset() + col.byte_width()`

use consus_core::{Error, Result};

use crate::types::{BinaryFormatCode, binary_format_element_size, parse_binary_format};

use super::FitsTableColumn;

/// Decoded value for a single FITS table column cell.
///
/// Variants correspond to FITS binary table format codes (FITS Standard 4.0 section 7.3)
/// and ASCII table format categories (section 7.2). Multi-byte numeric values are decoded
/// to native byte order from the big-endian FITS file representation.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum FitsColumnValue {
    /// `L`: FITS logical. b'T' = true; b'F' or 0x00 = false; other nonzero = true.
    Logical(bool),
    /// `X`: Bit array. `bits` is the repeat count; `bytes` = ceil(bits/8) bytes, MSB-first.
    Bits { bits: usize, bytes: alloc::vec::Vec<u8> },
    /// `B`: Unsigned byte.
    UInt8(u8),
    /// `I`: Big-endian signed 16-bit integer.
    Int16(i16),
    /// `J`: Big-endian signed 32-bit integer.
    Int32(i32),
    /// `K`: Big-endian signed 64-bit integer.
    Int64(i64),
    /// `A` / ASCII string: fixed-length ASCII string. Trailing spaces stripped.
    Chars(alloc::string::String),
    /// `E`: Big-endian IEEE 754 f32.
    Float32(f32),
    /// `D`: Big-endian IEEE 754 f64.
    Float64(f64),
    /// `C`: Big-endian complex f32. 8 bytes = [real:f32 be][imag:f32 be].
    Complex32 { real: f32, imag: f32 },
    /// `M`: Big-endian complex f64. 16 bytes = [real:f64 be][imag:f64 be].
    Complex64 { real: f64, imag: f64 },
    /// `P`: 32-bit heap descriptor. 8 bytes = [count:i32 be][offset:i32 be].
    Descriptor32 { count: i32, offset: i32 },
    /// `Q`: 64-bit heap descriptor. 16 bytes = [count:i64 be][offset:i64 be].
    Descriptor64 { count: i64, offset: i64 },
    /// Array of repeat > 1 scalar values (L, B, I, J, K, E, D, C, M).
    /// Not used for A (Chars) or X (Bits).
    Array(alloc::vec::Vec<FitsColumnValue>),
}
/// Decode one column cell from a binary table row buffer.
///
/// Extracts `row[col.col_offset()..col.col_offset() + col.byte_width()]`
/// and decodes per FITS Standard 4.0 section 7.3.
///
/// ## Errors
/// - `BufferTooSmall` if `row` is shorter than `col_offset + byte_width`.
/// - `InvalidFormat` if `column.format()` is not a valid TFORM string.
#[cfg(feature = "alloc")]
pub fn decode_binary_column(row: &[u8], column: &FitsTableColumn) -> Result<FitsColumnValue> {
    let col_offset = column.col_offset();
    let byte_width = column.byte_width();
    let end = col_offset.checked_add(byte_width).ok_or(Error::Overflow)?;
    let cell = row.get(col_offset..end).ok_or(Error::BufferTooSmall {
        required: end,
        provided: row.len(),
    })?;
    let (repeat, code) = parse_binary_format(column.format())?;

    // Bit and Char have special repeat semantics not handled by the scalar path.
    match code {
        BinaryFormatCode::Bit => {
            return Ok(FitsColumnValue::Bits {
                bits: repeat,
                bytes: cell.to_vec(),
            });
        }
        BinaryFormatCode::Char => {
            // All `repeat` chars form one string; trailing spaces stripped.
            let raw = core::str::from_utf8(cell).unwrap_or("");
            let s = raw.trim_end_matches(|c: char| c == ' ').to_owned();
            return Ok(FitsColumnValue::Chars(s));
        }
        _ => {}
    }

    let elem_size = binary_format_element_size(code);
    if repeat == 1 {
        decode_scalar_binary(code, cell)
    } else {
        let mut values = alloc::vec::Vec::with_capacity(repeat);
        for i in 0..repeat {
            let start = i.checked_mul(elem_size).ok_or(Error::Overflow)?;
            let stop = start.checked_add(elem_size).ok_or(Error::Overflow)?;
            let slice = cell.get(start..stop).ok_or(Error::BufferTooSmall {
                required: stop,
                provided: cell.len(),
            })?;
            values.push(decode_scalar_binary(code, slice)?);
        }
        Ok(FitsColumnValue::Array(values))
    }
}
/// Decode one column cell from an ASCII table row buffer.
///
/// Extracts `row[col.col_offset()..col.col_offset() + col.byte_width()]`,
/// interprets as ASCII, trims whitespace, and parses per the column's
/// FITS Standard 4.0 section 7.2 format code.
///
/// ## Errors
/// - `BufferTooSmall` if `row` is shorter than `col_offset + byte_width`.
/// - `InvalidFormat` if the field content cannot be parsed for numeric types.
/// - `UnsupportedFeature` for unrecognised format codes.
#[cfg(feature = "alloc")]
pub fn decode_ascii_column(row: &[u8], column: &FitsTableColumn) -> Result<FitsColumnValue> {
    let col_offset = column.col_offset();
    let byte_width = column.byte_width();
    let end = col_offset.checked_add(byte_width).ok_or(Error::Overflow)?;
    let cell = row.get(col_offset..end).ok_or(Error::BufferTooSmall {
        required: end,
        provided: row.len(),
    })?;
    let field = core::str::from_utf8(cell).map_err(|_| Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: "ASCII table column contains non-UTF-8 bytes".into(),
    })?;
    let trimmed = field.trim();

    // Identify format code as first ASCII-alphabetic char in TFORMAn.
    let code_char = column
        .format()
        .chars()
        .find(|c| c.is_ascii_alphabetic())
        .unwrap_or('A')
        .to_ascii_uppercase();

    match code_char {
        'A' => Ok(FitsColumnValue::Chars(trimmed.to_owned())),
        'I' => {
            let v = trimmed.parse::<i64>().map_err(|_| Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("cannot parse ASCII integer field: '{trimmed}'"),
            })?;
            Ok(FitsColumnValue::Int64(v))
        }
        'F' | 'E' => {
            let v = trimmed.parse::<f64>().map_err(|_| Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("cannot parse ASCII real field: '{trimmed}'"),
            })?;
            Ok(FitsColumnValue::Float64(v))
        }
        'D' => {
            // FITS Fortran D-notation: replace D/d with E before parsing.
            let normalized = {
                let mut s = alloc::string::String::with_capacity(trimmed.len());
                for c in trimmed.chars() {
                    if c == 'D' || c == 'd' {
                        s.push('E');
                    } else {
                        s.push(c);
                    }
                }
                s
            };
            let v = normalized.parse::<f64>().map_err(|_| Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("cannot parse ASCII D-notation field: '{trimmed}'"),
            })?;
            Ok(FitsColumnValue::Float64(v))
        }
        other => Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: alloc::format!("ASCII TFORM format code '{other}'"),
        }),
    }
}
/// Decode exactly one scalar element for the given `BinaryFormatCode`.
///
/// `bytes` must be `binary_format_element_size(code)` bytes.
/// Callers must NOT pass `Bit` or `Char` -- those are handled in `decode_binary_column`.
#[cfg(feature = "alloc")]
fn decode_scalar_binary(code: BinaryFormatCode, bytes: &[u8]) -> Result<FitsColumnValue> {
    match code {
        BinaryFormatCode::Logical => {
            let b = *bytes.first().ok_or(Error::BufferTooSmall {
                required: 1,
                provided: 0,
            })?;
            // FITS: b'T' (0x54) = true, b'F' (0x46) = false, 0x00 = undefined/false.
            Ok(FitsColumnValue::Logical(b != 0 && b != b'F'))
        }
        BinaryFormatCode::UnsignedByte => {
            let b = *bytes.first().ok_or(Error::BufferTooSmall {
                required: 1,
                provided: 0,
            })?;
            Ok(FitsColumnValue::UInt8(b))
        }
        BinaryFormatCode::Int16 => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| Error::BufferTooSmall {
                required: 2,
                provided: bytes.len(),
            })?;
            Ok(FitsColumnValue::Int16(i16::from_be_bytes(arr)))
        }
        BinaryFormatCode::Int32 => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| Error::BufferTooSmall {
                required: 4,
                provided: bytes.len(),
            })?;
            Ok(FitsColumnValue::Int32(i32::from_be_bytes(arr)))
        }
        BinaryFormatCode::Int64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| Error::BufferTooSmall {
                required: 8,
                provided: bytes.len(),
            })?;
            Ok(FitsColumnValue::Int64(i64::from_be_bytes(arr)))
        }
        BinaryFormatCode::Float32 => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| Error::BufferTooSmall {
                required: 4,
                provided: bytes.len(),
            })?;
            Ok(FitsColumnValue::Float32(f32::from_be_bytes(arr)))
        }
        BinaryFormatCode::Float64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| Error::BufferTooSmall {
                required: 8,
                provided: bytes.len(),
            })?;
            Ok(FitsColumnValue::Float64(f64::from_be_bytes(arr)))
        }
        BinaryFormatCode::Complex32 => {
            if bytes.len() < 8 {
                return Err(Error::BufferTooSmall { required: 8, provided: bytes.len() });
            }
            let real = f32::from_be_bytes(bytes[0..4].try_into().unwrap());
            let imag = f32::from_be_bytes(bytes[4..8].try_into().unwrap());
            Ok(FitsColumnValue::Complex32 { real, imag })
        }
        BinaryFormatCode::Complex64 => {
            if bytes.len() < 16 {
                return Err(Error::BufferTooSmall { required: 16, provided: bytes.len() });
            }
            let real = f64::from_be_bytes(bytes[0..8].try_into().unwrap());
            let imag = f64::from_be_bytes(bytes[8..16].try_into().unwrap());
            Ok(FitsColumnValue::Complex64 { real, imag })
        }
        BinaryFormatCode::Descriptor32 => {
            if bytes.len() < 8 {
                return Err(Error::BufferTooSmall { required: 8, provided: bytes.len() });
            }
            let count = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
            let offset = i32::from_be_bytes(bytes[4..8].try_into().unwrap());
            Ok(FitsColumnValue::Descriptor32 { count, offset })
        }
        BinaryFormatCode::Descriptor64 => {
            if bytes.len() < 16 {
                return Err(Error::BufferTooSmall { required: 16, provided: bytes.len() });
            }
            let count = i64::from_be_bytes(bytes[0..8].try_into().unwrap());
            let offset = i64::from_be_bytes(bytes[8..16].try_into().unwrap());
            Ok(FitsColumnValue::Descriptor64 { count, offset })
        }
        BinaryFormatCode::Bit | BinaryFormatCode::Char => Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: "Bit/Char must be decoded via decode_binary_column".into(),
        }),
    }
}
#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::Datatype;

    fn col(format: &str, byte_width: usize, col_offset: usize) -> FitsTableColumn {
        FitsTableColumn::new(
            1,
            None,
            format.to_owned(),
            Datatype::Boolean,
            byte_width,
            col_offset,
            None,
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn binary_logical_t_is_true() {
        let row = vec![b'T'];
        assert_eq!(decode_binary_column(&row, &col("1L", 1, 0)).unwrap(), FitsColumnValue::Logical(true));
    }

    #[test]
    fn binary_logical_f_is_false() {
        let row = vec![b'F'];
        assert_eq!(decode_binary_column(&row, &col("1L", 1, 0)).unwrap(), FitsColumnValue::Logical(false));
    }

    #[test]
    fn binary_logical_zero_is_false() {
        let row = vec![0x00u8];
        assert_eq!(decode_binary_column(&row, &col("1L", 1, 0)).unwrap(), FitsColumnValue::Logical(false));
    }

    #[test]
    fn binary_bit_array_8bits() {
        let row = vec![0b1010_1010u8];
        assert_eq!(
            decode_binary_column(&row, &col("8X", 1, 0)).unwrap(),
            FitsColumnValue::Bits { bits: 8, bytes: vec![0b1010_1010u8] }
        );
    }

    #[test]
    fn binary_bit_array_partial() {
        let row = vec![0b1110_0000u8];
        assert_eq!(
            decode_binary_column(&row, &col("3X", 1, 0)).unwrap(),
            FitsColumnValue::Bits { bits: 3, bytes: vec![0b1110_0000u8] }
        );
    }

    #[test]
    fn binary_uint8() {
        let row = vec![42u8];
        assert_eq!(decode_binary_column(&row, &col("1B", 1, 0)).unwrap(), FitsColumnValue::UInt8(42));
    }

    #[test]
    fn binary_int16() {
        let row = 1000_i16.to_be_bytes().to_vec();
        assert_eq!(decode_binary_column(&row, &col("1I", 2, 0)).unwrap(), FitsColumnValue::Int16(1000));
    }

    #[test]
    fn binary_int16_negative() {
        let row = (-1_i16).to_be_bytes().to_vec();
        assert_eq!(decode_binary_column(&row, &col("1I", 2, 0)).unwrap(), FitsColumnValue::Int16(-1));
    }

    #[test]
    fn binary_int32() {
        let row = 70_000_i32.to_be_bytes().to_vec();
        assert_eq!(decode_binary_column(&row, &col("1J", 4, 0)).unwrap(), FitsColumnValue::Int32(70_000));
    }

    #[test]
    fn binary_int64() {
        let row = 1_000_000_000_i64.to_be_bytes().to_vec();
        assert_eq!(decode_binary_column(&row, &col("1K", 8, 0)).unwrap(), FitsColumnValue::Int64(1_000_000_000));
    }
    #[test]
    fn binary_char_three_bytes() {
        let row = b"ABC".to_vec();
        assert_eq!(decode_binary_column(&row, &col("3A", 3, 0)).unwrap(), FitsColumnValue::Chars("ABC".to_owned()));
    }

    #[test]
    fn binary_char_trailing_spaces_stripped() {
        let row = b"HI  ".to_vec();
        assert_eq!(decode_binary_column(&row, &col("4A", 4, 0)).unwrap(), FitsColumnValue::Chars("HI".to_owned()));
    }

    #[test]
    fn binary_float32() {
        let row = 1.5_f32.to_be_bytes().to_vec();
        assert_eq!(decode_binary_column(&row, &col("1E", 4, 0)).unwrap(), FitsColumnValue::Float32(1.5));
    }

    #[test]
    fn binary_float64() {
        let row = 3.14_f64.to_be_bytes().to_vec();
        assert_eq!(decode_binary_column(&row, &col("1D", 8, 0)).unwrap(), FitsColumnValue::Float64(3.14));
    }

    #[test]
    fn binary_complex32() {
        let mut row = alloc::vec::Vec::new();
        row.extend_from_slice(&1.0_f32.to_be_bytes());
        row.extend_from_slice(&2.0_f32.to_be_bytes());
        assert_eq!(
            decode_binary_column(&row, &col("1C", 8, 0)).unwrap(),
            FitsColumnValue::Complex32 { real: 1.0, imag: 2.0 }
        );
    }

    #[test]
    fn binary_complex64() {
        let mut row = alloc::vec::Vec::new();
        row.extend_from_slice(&3.14_f64.to_be_bytes());
        row.extend_from_slice(&1.41_f64.to_be_bytes());
        assert_eq!(
            decode_binary_column(&row, &col("1M", 16, 0)).unwrap(),
            FitsColumnValue::Complex64 { real: 3.14, imag: 1.41 }
        );
    }

    #[test]
    fn binary_descriptor32() {
        let mut row = alloc::vec::Vec::new();
        row.extend_from_slice(&3_i32.to_be_bytes());
        row.extend_from_slice(&100_i32.to_be_bytes());
        assert_eq!(
            decode_binary_column(&row, &col("1P", 8, 0)).unwrap(),
            FitsColumnValue::Descriptor32 { count: 3, offset: 100 }
        );
    }

    #[test]
    fn binary_descriptor64() {
        let mut row = alloc::vec::Vec::new();
        row.extend_from_slice(&5_i64.to_be_bytes());
        row.extend_from_slice(&200_i64.to_be_bytes());
        assert_eq!(
            decode_binary_column(&row, &col("1Q", 16, 0)).unwrap(),
            FitsColumnValue::Descriptor64 { count: 5, offset: 200 }
        );
    }
    #[test]
    fn binary_array_of_int32() {
        let mut row = alloc::vec::Vec::new();
        row.extend_from_slice(&10_i32.to_be_bytes());
        row.extend_from_slice(&20_i32.to_be_bytes());
        row.extend_from_slice(&30_i32.to_be_bytes());
        assert_eq!(
            decode_binary_column(&row, &col("3J", 12, 0)).unwrap(),
            FitsColumnValue::Array(vec![
                FitsColumnValue::Int32(10),
                FitsColumnValue::Int32(20),
                FitsColumnValue::Int32(30),
            ])
        );
    }

    #[test]
    fn binary_array_of_float32() {
        let mut row = alloc::vec::Vec::new();
        row.extend_from_slice(&1.0_f32.to_be_bytes());
        row.extend_from_slice(&(-1.0_f32).to_be_bytes());
        assert_eq!(
            decode_binary_column(&row, &col("2E", 8, 0)).unwrap(),
            FitsColumnValue::Array(vec![
                FitsColumnValue::Float32(1.0),
                FitsColumnValue::Float32(-1.0),
            ])
        );
    }

    #[test]
    fn binary_array_of_logical() {
        let row = vec![b'T', b'F', b'T'];
        assert_eq!(
            decode_binary_column(&row, &col("3L", 3, 0)).unwrap(),
            FitsColumnValue::Array(vec![
                FitsColumnValue::Logical(true),
                FitsColumnValue::Logical(false),
                FitsColumnValue::Logical(true),
            ])
        );
    }

    #[test]
    fn binary_nonzero_col_offset() {
        let mut row = vec![0u8; 4];
        row.extend_from_slice(&42_i32.to_be_bytes());
        assert_eq!(decode_binary_column(&row, &col("1J", 4, 4)).unwrap(), FitsColumnValue::Int32(42));
    }

    #[test]
    fn binary_buffer_too_small() {
        let row = vec![0u8; 2];
        let err = decode_binary_column(&row, &col("1J", 4, 0)).unwrap_err();
        assert!(matches!(err, Error::BufferTooSmall { required: 4, provided: 2 }));
    }

    #[test]
    fn binary_offset_past_end() {
        let row = vec![0u8; 4];
        let err = decode_binary_column(&row, &col("1J", 4, 4)).unwrap_err();
        assert!(matches!(err, Error::BufferTooSmall { .. }));
    }
    #[test]
    fn ascii_string_field() {
        let row = b"Hello   ".to_vec();
        assert_eq!(decode_ascii_column(&row, &col("A8", 8, 0)).unwrap(), FitsColumnValue::Chars("Hello".to_owned()));
    }

    #[test]
    fn ascii_string_all_spaces() {
        let row = b"    ".to_vec();
        assert_eq!(decode_ascii_column(&row, &col("A4", 4, 0)).unwrap(), FitsColumnValue::Chars("".to_owned()));
    }

    #[test]
    fn ascii_integer_field() {
        let row = b"       123".to_vec();
        assert_eq!(decode_ascii_column(&row, &col("I10", 10, 0)).unwrap(), FitsColumnValue::Int64(123));
    }

    #[test]
    fn ascii_integer_negative() {
        let row = b"      -456".to_vec();
        assert_eq!(decode_ascii_column(&row, &col("I10", 10, 0)).unwrap(), FitsColumnValue::Int64(-456));
    }

    #[test]
    fn ascii_float_f_format() {
        let row = b"  3.141590".to_vec();
        let val = decode_ascii_column(&row, &col("F10.5", 10, 0)).unwrap();
        if let FitsColumnValue::Float64(v) = val {
            assert!((v - 3.14159).abs() < 1e-6, "expected ~3.14159, got {v}");
        } else {
            panic!("expected Float64");
        }
    }

    #[test]
    fn ascii_scientific_e_format() {
        let row = b"  1.5000000E+02".to_vec();
        assert_eq!(decode_ascii_column(&row, &col("E16.7", 15, 0)).unwrap(), FitsColumnValue::Float64(150.0));
    }

    #[test]
    fn ascii_fortran_d_notation() {
        let row = b"  1.5000000D+02".to_vec();
        assert_eq!(decode_ascii_column(&row, &col("D16.7", 15, 0)).unwrap(), FitsColumnValue::Float64(150.0));
    }

    #[test]
    fn ascii_nonzero_col_offset() {
        let mut row = b"SKIP".to_vec();
        row.extend_from_slice(b"World   ");
        assert_eq!(decode_ascii_column(&row, &col("A8", 8, 4)).unwrap(), FitsColumnValue::Chars("World".to_owned()));
    }

    #[test]
    fn ascii_buffer_too_small() {
        let row = b"Hi".to_vec();
        let err = decode_ascii_column(&row, &col("A8", 8, 0)).unwrap_err();
        assert!(matches!(err, Error::BufferTooSmall { required: 8, provided: 2 }));
    }

    #[test]
    fn ascii_integer_invalid() {
        let row = b"    notnum".to_vec();
        assert!(matches!(decode_ascii_column(&row, &col("I10", 10, 0)).unwrap_err(), Error::InvalidFormat { .. }));
    }

    #[test]
    fn ascii_float_invalid() {
        let row = b"    notnum".to_vec();
        assert!(matches!(decode_ascii_column(&row, &col("F10.3", 10, 0)).unwrap_err(), Error::InvalidFormat { .. }));
    }

    #[test]
    fn ascii_unsupported_format_code() {
        let row = b"  1234.0  ".to_vec();
        assert!(matches!(decode_ascii_column(&row, &col("G10.4", 10, 0)).unwrap_err(), Error::UnsupportedFeature { .. }));
    }
}
