//! FITS binary table TFORM format code parsing and canonical Datatype mapping.
//!
//! ## Specification
//!
//! FITS Standard 4.0 defines binary table column format codes in the `TFORMn`
//! keywords. Each TFORM value has the form `rTa` where `r` is an optional
//! repeat count (defaults to 1) and `T` is a single-character type code.
//!
//! ## Format Codes
//!
//! | Code | Meaning | Element bytes |
//! |------|---------|---------------|
//! | `L` | Logical (bool) | 1 |
//! | `X` | Bit (unsigned, 1-byte units) | 1 |
//! | `B` | Unsigned byte | 1 |
//! | `I` | 16-bit integer | 2 |
//! | `J` | 32-bit integer | 4 |
//! | `K` | 64-bit integer | 8 |
//! | `A` | Character (fixed-length ASCII) | 1 |
//! | `E` | 32-bit IEEE float | 4 |
//! | `D` | 64-bit IEEE float | 8 |
//! | `C` | 32-bit complex | 8 |
//! | `M` | 64-bit complex | 16 |
//! | `P` | 32-bit array descriptor | 8 |
//! | `Q` | 64-bit array descriptor | 16 |
//!
//! ## Invariants
//!
//! - FITS stores all multi-byte numeric values in big-endian byte order.
//! - Every valid `TFORM` string maps to exactly one canonical `Datatype`.
//! - Invalid format codes produce `Error::InvalidFormat`.
//! - Repeat count defaults to 1 when absent.
//! - `A` type uses repeat count as string length, not array dimension.
//! - Scalar types with repeat > 1 wrap in `Datatype::Array`.
//! - `Compound` and `Array` variants require `alloc` feature.

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String, vec};

use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype, Error, Result, StringEncoding};

#[cfg(feature = "alloc")]
use consus_core::CompoundField;

/// FITS Standard 4.0 binary table column type code.
///
/// Each variant corresponds to exactly one TFORM format character.
/// FITS stores all multi-byte numeric values in big-endian byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryFormatCode {
    /// `L` — Logical (bool, 1 byte; 0 = false, nonzero = true).
    Logical,
    /// `X` — Bit array (unsigned, stored in 1-byte units).
    Bit,
    /// `B` — Unsigned byte (u8).
    UnsignedByte,
    /// `I` — 16-bit signed integer (i16, big-endian).
    Int16,
    /// `J` — 32-bit signed integer (i32, big-endian).
    Int32,
    /// `K` — 64-bit signed integer (i64, big-endian).
    Int64,
    /// `A` — Character (fixed-length ASCII string).
    Char,
    /// `E` — 32-bit IEEE 754 float (f32, big-endian).
    Float32,
    /// `D` — 64-bit IEEE 754 float (f64, big-endian).
    Float64,
    /// `C` — 32-bit complex (2 × f32, big-endian).
    Complex32,
    /// `M` — 64-bit complex (2 × f64, big-endian).
    Complex64,
    /// `P` — 32-bit array descriptor (count: i32 + offset: i32).
    Descriptor32,
    /// `Q` — 64-bit array descriptor (count: i64 + offset: i64).
    Descriptor64,
}

impl BinaryFormatCode {
    /// Return the single-character FITS format code for this variant.
    pub const fn to_char(self) -> char {
        match self {
            Self::Logical => 'L',
            Self::Bit => 'X',
            Self::UnsignedByte => 'B',
            Self::Int16 => 'I',
            Self::Int32 => 'J',
            Self::Int64 => 'K',
            Self::Char => 'A',
            Self::Float32 => 'E',
            Self::Float64 => 'D',
            Self::Complex32 => 'C',
            Self::Complex64 => 'M',
            Self::Descriptor32 => 'P',
            Self::Descriptor64 => 'Q',
        }
    }
}

/// Parse a FITS binary table TFORM string into (repeat_count, format_code).
///
/// ## Specification
///
/// A TFORM value has the form `rT` where:
/// - `r` is an optional decimal repeat count (defaults to 1 if absent)
/// - `T` is a single-character format code from the FITS Standard 4.0
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if the format code is not a recognized
/// FITS binary table type code.
///
/// ## Examples
///
/// - `"1J"` → `(1, BinaryFormatCode::Int32)`
/// - `"20A"` → `(20, BinaryFormatCode::Char)`
/// - `"J"` → `(1, BinaryFormatCode::Int32)` (repeat defaults to 1)
pub fn parse_binary_format(tform: &str) -> Result<(usize, BinaryFormatCode)> {
    let trimmed = tform.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!("empty TFORM string"),
        });
    }

    // Split into leading digits (repeat count) and trailing type code character.
    let split_pos = trimmed.char_indices().rfind(|(_, c)| c.is_ascii_digit());

    let (digits, code_char) = match split_pos {
        Some((i, _)) if i + 1 < trimmed.len() => {
            // Last digit is at index i; type code starts at i + 1.
            let (d, c) = trimmed.split_at(i + 1);
            (d, c.chars().next().unwrap())
        }
        _ => {
            // No digits, or digit is the last character (treated as type code).
            let code = trimmed.chars().last().unwrap();
            ("1", code)
        }
    };

    let repeat: usize = digits.parse().map_err(|_| Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: alloc::format!("invalid repeat count in TFORM: {tform}"),
    })?;

    let code = match code_char {
        'L' => BinaryFormatCode::Logical,
        'X' => BinaryFormatCode::Bit,
        'B' => BinaryFormatCode::UnsignedByte,
        'I' => BinaryFormatCode::Int16,
        'J' => BinaryFormatCode::Int32,
        'K' => BinaryFormatCode::Int64,
        'A' => BinaryFormatCode::Char,
        'E' => BinaryFormatCode::Float32,
        'D' => BinaryFormatCode::Float64,
        'C' => BinaryFormatCode::Complex32,
        'M' => BinaryFormatCode::Complex64,
        'P' => BinaryFormatCode::Descriptor32,
        'Q' => BinaryFormatCode::Descriptor64,
        _ => {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "invalid FITS binary table format code: '{code_char}' in TFORM '{tform}'"
                ),
            });
        }
    };

    Ok((repeat, code))
}

/// Return the byte size of one element for the given format code.
///
/// ## Derivation
///
/// From FITS Standard 4.0 Table 7:
///
/// | Code | Bytes |
/// |------|-------|
/// | L | 1 |
/// | X | 1 |
/// | B | 1 |
/// | I | 2 |
/// | J | 4 |
/// | K | 8 |
/// | A | 1 |
/// | E | 4 |
/// | D | 8 |
/// | C | 8 |
/// | M | 16 |
/// | P | 8 |
/// | Q | 16 |
pub const fn binary_format_element_size(code: BinaryFormatCode) -> usize {
    match code {
        BinaryFormatCode::Logical => 1,
        BinaryFormatCode::Bit => 1,
        BinaryFormatCode::UnsignedByte => 1,
        BinaryFormatCode::Int16 => 2,
        BinaryFormatCode::Int32 => 4,
        BinaryFormatCode::Int64 => 8,
        BinaryFormatCode::Char => 1,
        BinaryFormatCode::Float32 => 4,
        BinaryFormatCode::Float64 => 8,
        BinaryFormatCode::Complex32 => 8,
        BinaryFormatCode::Complex64 => 16,
        BinaryFormatCode::Descriptor32 => 8,
        BinaryFormatCode::Descriptor64 => 16,
    }
}

/// Map a `BinaryFormatCode` and repeat count to a canonical `Datatype`.
///
/// ## Mapping Specification
///
/// | Code | Datatype variant |
/// |------|-----------------|
/// | `L` | `Datatype::Boolean` |
/// | `X` | `Datatype::Opaque { size: repeat, tag: Some("FITS_bit") }` |
/// | `B` | `Datatype::Integer { bits: 8, byte_order: BigEndian, signed: false }` |
/// | `I` | `Datatype::Integer { bits: 16, byte_order: BigEndian, signed: true }` |
/// | `J` | `Datatype::Integer { bits: 32, byte_order: BigEndian, signed: true }` |
/// | `K` | `Datatype::Integer { bits: 64, byte_order: BigEndian, signed: true }` |
/// | `A` | `Datatype::FixedString { length: repeat, encoding: Ascii }` |
/// | `E` | `Datatype::Float { bits: 32, byte_order: BigEndian }` |
/// | `D` | `Datatype::Float { bits: 64, byte_order: BigEndian }` |
/// | `C` | `Datatype::Complex { component_bits: 32, byte_order: BigEndian }` |
/// | `M` | `Datatype::Complex { component_bits: 64, byte_order: BigEndian }` |
/// | `P` | `Datatype::Compound { fields: [count(i32), offset(i32)], size: 8 }` |
/// | `Q` | `Datatype::Compound { fields: [count(i64), offset(i64)], size: 16 }` |
///
/// For `A`, the repeat count is the string length (not an array dimension).
/// For all other scalar types with `repeat > 1`, the result wraps in
/// `Datatype::Array { base, dims: [repeat] }`.
/// For `repeat == 1`, the scalar type is returned directly.
#[allow(clippy::result_large_err)]
pub fn binary_format_to_datatype(code: BinaryFormatCode, repeat: usize) -> Datatype {
    match code {
        BinaryFormatCode::Logical => Datatype::Boolean,
        BinaryFormatCode::Bit => Datatype::Opaque {
            size: repeat,
            #[cfg(feature = "alloc")]
            tag: Some(String::from("FITS_bit")),
        },
        BinaryFormatCode::UnsignedByte => Datatype::Integer {
            bits: nonzero(8),
            byte_order: ByteOrder::BigEndian,
            signed: false,
        },
        BinaryFormatCode::Int16 => Datatype::Integer {
            bits: nonzero(16),
            byte_order: ByteOrder::BigEndian,
            signed: true,
        },
        BinaryFormatCode::Int32 => Datatype::Integer {
            bits: nonzero(32),
            byte_order: ByteOrder::BigEndian,
            signed: true,
        },
        BinaryFormatCode::Int64 => Datatype::Integer {
            bits: nonzero(64),
            byte_order: ByteOrder::BigEndian,
            signed: true,
        },
        BinaryFormatCode::Char => Datatype::FixedString {
            length: repeat,
            encoding: StringEncoding::Ascii,
        },
        BinaryFormatCode::Float32 => Datatype::Float {
            bits: nonzero(32),
            byte_order: ByteOrder::BigEndian,
        },
        BinaryFormatCode::Float64 => Datatype::Float {
            bits: nonzero(64),
            byte_order: ByteOrder::BigEndian,
        },
        BinaryFormatCode::Complex32 => Datatype::Complex {
            component_bits: nonzero(32),
            byte_order: ByteOrder::BigEndian,
        },
        BinaryFormatCode::Complex64 => Datatype::Complex {
            component_bits: nonzero(64),
            byte_order: ByteOrder::BigEndian,
        },
        BinaryFormatCode::Descriptor32 => Datatype::Compound {
            fields: vec![
                CompoundField {
                    name: String::from("count"),
                    datatype: Datatype::Integer {
                        bits: nonzero(32),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    },
                    offset: 0,
                },
                CompoundField {
                    name: String::from("offset"),
                    datatype: Datatype::Integer {
                        bits: nonzero(32),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    },
                    offset: 4,
                },
            ],
            size: 8,
        },
        BinaryFormatCode::Descriptor64 => Datatype::Compound {
            fields: vec![
                CompoundField {
                    name: String::from("count"),
                    datatype: Datatype::Integer {
                        bits: nonzero(64),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    },
                    offset: 0,
                },
                CompoundField {
                    name: String::from("offset"),
                    datatype: Datatype::Integer {
                        bits: nonzero(64),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    },
                    offset: 8,
                },
            ],
            size: 16,
        },
    }
}

/// High-level function: parse a TFORM string and produce the canonical `Datatype`.
///
/// ## Semantics
///
/// - `A` type: repeat count becomes `FixedString { length: repeat }`.
/// - `X` type (bit): repeat count becomes `Opaque { size: repeat }`.
/// - `P` / `Q` types (descriptors): repeat count is not applied as array
///   dimension since descriptors are self-describing structures.
/// - All other scalar types with `repeat > 1`: wraps in
///   `Datatype::Array { base, dims: [repeat] }`.
/// - `repeat == 1`: returns the scalar type directly.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` for unrecognized format codes.
pub fn tform_to_datatype(tform: &str) -> Result<Datatype> {
    let (repeat, code) = parse_binary_format(tform)?;

    // Types that consume the repeat count internally (not as array dimension).
    let datatype = match code {
        // Character: repeat is the string length.
        BinaryFormatCode::Char => binary_format_to_datatype(code, repeat),
        // Bit: repeat is the opaque blob size in bytes.
        BinaryFormatCode::Bit => binary_format_to_datatype(code, repeat),
        // Descriptors: structural types, repeat not used as array dim.
        BinaryFormatCode::Descriptor32 | BinaryFormatCode::Descriptor64 => {
            binary_format_to_datatype(code, repeat)
        }
        // All other types: wrap in Array if repeat > 1.
        _ => {
            let base = binary_format_to_datatype(code, 1);
            if repeat > 1 {
                Datatype::Array {
                    base: Box::new(base),
                    dims: vec![repeat],
                }
            } else {
                base
            }
        }
    };

    Ok(datatype)
}

/// Construct a `NonZeroUsize` guaranteed nonzero by construction.
///
/// ## Safety Contract
///
/// Callers must pass a nonzero `bits` value. Used only with FITS-mandated
/// bit widths (8, 16, 32, 64) which are all nonzero by specification.
const fn nonzero(bits: usize) -> NonZeroUsize {
    match NonZeroUsize::new(bits) {
        Some(v) => v,
        None => panic!("FITS format code mapping requires non-zero bit widths"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // parse_binary_format tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_binary_format_all_codes() {
        let cases: &[(&str, usize, BinaryFormatCode)] = &[
            ("1L", 1, BinaryFormatCode::Logical),
            ("1X", 1, BinaryFormatCode::Bit),
            ("1B", 1, BinaryFormatCode::UnsignedByte),
            ("1I", 1, BinaryFormatCode::Int16),
            ("1J", 1, BinaryFormatCode::Int32),
            ("1K", 1, BinaryFormatCode::Int64),
            ("20A", 20, BinaryFormatCode::Char),
            ("1E", 1, BinaryFormatCode::Float32),
            ("1D", 1, BinaryFormatCode::Float64),
            ("1C", 1, BinaryFormatCode::Complex32),
            ("1M", 1, BinaryFormatCode::Complex64),
            ("1P", 1, BinaryFormatCode::Descriptor32),
            ("1Q", 1, BinaryFormatCode::Descriptor64),
        ];
        for (tform, expected_repeat, expected_code) in cases {
            let (repeat, code) = parse_binary_format(tform)
                .unwrap_or_else(|e| panic!("parse_binary_format(\"{tform}\") failed: {e:?}"));
            assert_eq!(
                repeat, *expected_repeat,
                "repeat count for TFORM \"{tform}\""
            );
            assert_eq!(code, *expected_code, "format code for TFORM \"{tform}\"");
        }
    }

    #[test]
    fn parse_binary_format_default_repeat() {
        let (repeat, code) = parse_binary_format("J").unwrap();
        assert_eq!(repeat, 1);
        assert_eq!(code, BinaryFormatCode::Int32);
    }

    #[test]
    fn parse_binary_format_multi_repeat() {
        let (repeat, code) = parse_binary_format("100E").unwrap();
        assert_eq!(repeat, 100);
        assert_eq!(code, BinaryFormatCode::Float32);
    }

    #[test]
    fn parse_binary_format_rejects_invalid_code() {
        let result = parse_binary_format("Z");
        assert!(
            result.is_err(),
            "expected error for invalid format code 'Z'"
        );
        match result.unwrap_err() {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // tform_to_datatype tests
    // -----------------------------------------------------------------------

    #[test]
    fn tform_to_datatype_scalar_types() {
        // 1J → Integer(32, BigEndian, signed)
        let dt = tform_to_datatype("1J").unwrap();
        assert_eq!(
            dt,
            Datatype::Integer {
                bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );

        // 1D → Float(64, BigEndian)
        let dt = tform_to_datatype("1D").unwrap();
        assert_eq!(
            dt,
            Datatype::Float {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );

        // 1C → Complex(32, BigEndian)
        let dt = tform_to_datatype("1C").unwrap();
        assert_eq!(
            dt,
            Datatype::Complex {
                component_bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );
    }

    #[test]
    fn tform_to_datatype_string_type() {
        let dt = tform_to_datatype("20A").unwrap();
        assert_eq!(
            dt,
            Datatype::FixedString {
                length: 20,
                encoding: StringEncoding::Ascii,
            }
        );
    }

    #[test]
    fn tform_to_datatype_array_type() {
        // 5J → Array { base: Integer(32, BigEndian, signed), dims: [5] }
        let dt = tform_to_datatype("5J").unwrap();
        match &dt {
            Datatype::Array { base, dims } => {
                assert_eq!(
                    &**base,
                    &Datatype::Integer {
                        bits: NonZeroUsize::new(32).unwrap(),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    }
                );
                assert_eq!(dims, &[5usize]);
            }
            _ => panic!("expected Array variant, got: {dt:?}"),
        }
    }

    #[test]
    fn tform_to_datatype_logical() {
        let dt = tform_to_datatype("1L").unwrap();
        assert_eq!(dt, Datatype::Boolean);
    }

    #[test]
    fn tform_to_datatype_complex_float32() {
        let dt = tform_to_datatype("1C").unwrap();
        assert_eq!(
            dt,
            Datatype::Complex {
                component_bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );
    }

    #[test]
    fn tform_to_datatype_complex_float64() {
        let dt = tform_to_datatype("1M").unwrap();
        assert_eq!(
            dt,
            Datatype::Complex {
                component_bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );
    }

    #[test]
    fn tform_to_datatype_bit_opaque() {
        let dt = tform_to_datatype("1X").unwrap();
        match &dt {
            Datatype::Opaque { size, tag } => {
                assert_eq!(*size, 1);
                assert_eq!(tag.as_deref(), Some("FITS_bit"));
            }
            _ => panic!("expected Opaque variant, got: {dt:?}"),
        }
    }

    #[test]
    fn tform_to_datatype_unsigned_byte() {
        let dt = tform_to_datatype("1B").unwrap();
        assert_eq!(
            dt,
            Datatype::Integer {
                bits: NonZeroUsize::new(8).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: false,
            }
        );
    }

    #[test]
    fn tform_to_datatype_int16() {
        let dt = tform_to_datatype("1I").unwrap();
        assert_eq!(
            dt,
            Datatype::Integer {
                bits: NonZeroUsize::new(16).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );
    }

    #[test]
    fn tform_to_datatype_int64() {
        let dt = tform_to_datatype("1K").unwrap();
        assert_eq!(
            dt,
            Datatype::Integer {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );
    }

    #[test]
    fn tform_to_datatype_float32() {
        let dt = tform_to_datatype("1E").unwrap();
        assert_eq!(
            dt,
            Datatype::Float {
                bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );
    }

    #[test]
    fn tform_to_datatype_descriptor32() {
        let dt = tform_to_datatype("1P").unwrap();
        match &dt {
            Datatype::Compound { fields, size } => {
                assert_eq!(*size, 8);
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "count");
                assert_eq!(fields[0].offset, 0);
                assert_eq!(
                    fields[0].datatype,
                    Datatype::Integer {
                        bits: NonZeroUsize::new(32).unwrap(),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    }
                );
                assert_eq!(fields[1].name, "offset");
                assert_eq!(fields[1].offset, 4);
                assert_eq!(
                    fields[1].datatype,
                    Datatype::Integer {
                        bits: NonZeroUsize::new(32).unwrap(),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    }
                );
            }
            _ => panic!("expected Compound variant for P descriptor, got: {dt:?}"),
        }
    }

    #[test]
    fn tform_to_datatype_descriptor64() {
        let dt = tform_to_datatype("1Q").unwrap();
        match &dt {
            Datatype::Compound { fields, size } => {
                assert_eq!(*size, 16);
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "count");
                assert_eq!(fields[0].offset, 0);
                assert_eq!(
                    fields[0].datatype,
                    Datatype::Integer {
                        bits: NonZeroUsize::new(64).unwrap(),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    }
                );
                assert_eq!(fields[1].name, "offset");
                assert_eq!(fields[1].offset, 8);
                assert_eq!(
                    fields[1].datatype,
                    Datatype::Integer {
                        bits: NonZeroUsize::new(64).unwrap(),
                        byte_order: ByteOrder::BigEndian,
                        signed: true,
                    }
                );
            }
            _ => panic!("expected Compound variant for Q descriptor, got: {dt:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // binary_format_element_size tests
    // -----------------------------------------------------------------------

    #[test]
    fn element_size_matches_standard() {
        let cases: &[(BinaryFormatCode, usize)] = &[
            (BinaryFormatCode::Logical, 1),
            (BinaryFormatCode::Bit, 1),
            (BinaryFormatCode::UnsignedByte, 1),
            (BinaryFormatCode::Int16, 2),
            (BinaryFormatCode::Int32, 4),
            (BinaryFormatCode::Int64, 8),
            (BinaryFormatCode::Char, 1),
            (BinaryFormatCode::Float32, 4),
            (BinaryFormatCode::Float64, 8),
            (BinaryFormatCode::Complex32, 8),
            (BinaryFormatCode::Complex64, 16),
            (BinaryFormatCode::Descriptor32, 8),
            (BinaryFormatCode::Descriptor64, 16),
        ];
        for (code, expected_size) in cases {
            let actual = binary_format_element_size(*code);
            assert_eq!(
                actual,
                *expected_size,
                "element size for {:?} ({})",
                code,
                code.to_char()
            );
        }
    }
}
