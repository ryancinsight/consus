//! HDF5 datatype <-> canonical datatype mapping.
//!
//! ## Specification
//!
//! HDF5 datatypes are encoded in header messages (type 0x0003).
//! The encoding begins with a 4-byte class+version+flags field
//! followed by a 4-byte size field, then class-specific properties.
//!
//! ## Datatype Class Coverage
//!
//! | Class | Value | Mapping function | Canonical `Datatype` |
//! |-------|-------|------------------|----------------------|
//! | Fixed-point | 0 | `map_fixed_point` | `Integer` |
//! | Floating-point | 1 | `map_floating_point` | `Float` |
//! | Time | 2 | *(deprecated, rejected)* | — |
//! | String | 3 | `map_string` | `FixedString` / `VariableString` |
//! | Bitfield | 4 | `map_bitfield` | `Opaque` |
//! | Opaque | 5 | `map_opaque` | `Opaque` |
//! | Compound | 6 | `map_compound` | `Compound` |
//! | Reference | 7 | `map_reference` | `Reference` |
//! | Enum | 8 | `map_enum` | `Enum` |
//! | Variable-length | 9 | `map_variable_length` | `VarLen` |
//! | Array | 10 | `map_array` | `Array` |
//!
//! ## Invariants
//!
//! - Every HDF5 datatype class has exactly one mapping function.
//! - Byte order is preserved from HDF5 flags.
//! - Fixed-length strings carry their byte length; variable-length strings
//!   are unbounded.
//! - Compound field offsets are computed from element sizes when not
//!   provided by the caller.
//! - Enum members are populated by the parser; `map_enum` produces the
//!   structural envelope with empty members.
//! - All multi-byte numeric types use the byte order extracted from HDF5
//!   class bit fields.

pub mod classes;
#[cfg(feature = "alloc")]
pub mod compound;

use consus_core::{ByteOrder, Datatype, StringEncoding};

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String, vec::Vec};
#[cfg(feature = "alloc")]
use consus_core::CompoundField;

/// Extract the datatype class from the first 4 bytes of a datatype message.
///
/// Bits 0-3 of byte 0 contain the class.
pub fn datatype_class(header_byte: u8) -> u8 {
    header_byte & 0x0F
}

/// Extract the byte order from a fixed-point or floating-point datatype.
///
/// Bit 0 of the class bit field (bits 8-31 of the 4-byte header):
/// 0 = little-endian, 1 = big-endian.
pub fn byte_order_from_flags(flags_byte: u8) -> ByteOrder {
    if flags_byte & 0x01 == 0 {
        ByteOrder::LittleEndian
    } else {
        ByteOrder::BigEndian
    }
}

/// Extract the character set encoding from HDF5 string datatype flags.
///
/// Bits 0-3 of the class bit field byte 4 encode the character set:
/// - 0 → ASCII
/// - 1 → UTF-8
/// - Other → ASCII (conservative default per HDF5 specification).
///
/// ## Reference
///
/// HDF5 File Format Specification Version 3.0, Section 4.3:
/// "String Datatype Properties — Character Set"
pub fn charset_from_flags(flags_byte: u8) -> StringEncoding {
    match flags_byte & 0x0F {
        0 => StringEncoding::Ascii,
        1 => StringEncoding::Utf8,
        _ => StringEncoding::Ascii,
    }
}

/// Map an HDF5 fixed-point (integer) datatype to canonical form.
///
/// ## Parameters
///
/// - `size_bytes`: total size in bytes (1, 2, 4, or 8).
/// - `flags`: class bit field byte. Bit 0 = byte order, bit 3 = signedness.
pub fn map_fixed_point(size_bytes: usize, flags: u8) -> Datatype {
    let signed = (flags & 0x08) != 0;
    let byte_order = byte_order_from_flags(flags);
    let bits = core::num::NonZeroUsize::new(size_bytes * 8).expect("HDF5 integer size must be > 0");
    Datatype::Integer {
        bits,
        byte_order,
        signed,
    }
}

/// Map an HDF5 floating-point datatype to canonical form.
///
/// ## Parameters
///
/// - `size_bytes`: total size in bytes (4 or 8).
/// - `flags`: class bit field byte. Bit 0 = byte order.
pub fn map_floating_point(size_bytes: usize, flags: u8) -> Datatype {
    let byte_order = byte_order_from_flags(flags);
    let bits = core::num::NonZeroUsize::new(size_bytes * 8).expect("HDF5 float size must be > 0");
    Datatype::Float { bits, byte_order }
}

/// Map an HDF5 string datatype to canonical form.
///
/// ## Parameters
///
/// - `size_bytes`: total size in bytes of the string storage.
/// - `flags`: class bit field byte.
///   - Bit 6: 0 = fixed-length, 1 = variable-length.
///   - Bits 0-3 (via `charset_from_flags`): character set encoding.
///
/// ## Derivation
///
/// - Fixed-length: `Datatype::FixedString { length, encoding }`.
///   `length = size_bytes` for fixed storage.
/// - Variable-length: `Datatype::VariableString { encoding }`.
///   `size_bytes` is the heap descriptor size, not the string length.
pub fn map_string(size_bytes: usize, flags: u8) -> Datatype {
    let encoding = charset_from_flags(flags);
    let is_variable = (flags & 0x40) != 0;
    if is_variable {
        Datatype::VariableString { encoding }
    } else {
        Datatype::FixedString {
            length: size_bytes,
            encoding,
        }
    }
}

/// Map an HDF5 bitfield datatype to canonical form.
///
/// ## Derivation
///
/// HDF5 bitfield types are packed bit sequences with no arithmetic
/// semantics. They map to `Datatype::Opaque` with the tag
/// `"HDF5_bitfield"` to distinguish them from generic opaque blobs.
///
/// ## Parameters
///
/// - `size_bytes`: total size in bytes of the bitfield storage.
/// - `flags`: class bit field byte. Bit 0 = byte order (preserved for
///   future interop but not encoded in `Opaque`).
pub fn map_bitfield(size_bytes: usize, _flags: u8) -> Datatype {
    Datatype::Opaque {
        size: size_bytes,
        #[cfg(feature = "alloc")]
        tag: Some(String::from("HDF5_bitfield")),
    }
}

/// Map an HDF5 opaque datatype to canonical form.
///
/// ## Parameters
///
/// - `size_bytes`: total size in bytes of the opaque blob.
/// - `tag`: optional application-defined tag from the HDF5 opaque
///   datatype description.
pub fn map_opaque(size_bytes: usize, tag: Option<&str>) -> Datatype {
    Datatype::Opaque {
        size: size_bytes,
        #[cfg(feature = "alloc")]
        tag: tag.map(String::from),
    }
}

/// Map an HDF5 compound datatype to canonical form.
///
/// ## Parameters
///
/// - `fields`: ordered compound member descriptors.
/// - `size`: total compound size in bytes (may include padding).
///
/// ## Invariant
///
/// `size >= sum(field element sizes)`. Fields are non-overlapping.
/// The caller is responsible for ensuring the offset invariant.
#[cfg(feature = "alloc")]
pub fn map_compound(fields: Vec<CompoundField>, size: usize) -> Datatype {
    Datatype::Compound { fields, size }
}

/// Map an HDF5 reference datatype to canonical form.
///
/// ## Parameters
///
/// - `size_bytes`: reference size in bytes.
/// - `ref_type`: the reference class (Object or Region).
///
/// ## Default Behavior
///
/// If `ref_type` is `None`:
/// - `size_bytes == 8` → `ReferenceType::Object` (HDF5 default for
///   standard object references).
/// - Otherwise → `ReferenceType::Region` (conservative for non-standard
///   sizes, which typically indicate region references).
pub fn map_reference(size_bytes: usize, ref_type: Option<consus_core::ReferenceType>) -> Datatype {
    let resolved = ref_type.unwrap_or(if size_bytes == 8 {
        consus_core::ReferenceType::Object
    } else {
        consus_core::ReferenceType::Region
    });
    Datatype::Reference(resolved)
}

/// Map an HDF5 enumeration datatype to canonical form.
///
/// ## Parameters
///
/// - `base`: the base integer datatype of the enumeration.
///
/// ## Note
///
/// The returned `Datatype::Enum` has an empty `members` vector.
/// The parser is responsible for populating enumeration members
/// after extracting them from the HDF5 datatype message.
/// This function produces the structural envelope only.
#[cfg(feature = "alloc")]
pub fn map_enum(base: &Datatype) -> Datatype {
    Datatype::Enum {
        base: Box::new(base.clone()),
        members: Vec::new(),
    }
}

/// Map an HDF5 variable-length datatype to canonical form.
///
/// ## Parameters
///
/// - `base`: the element datatype of the variable-length sequence.
///
/// ## Derivation
///
/// HDF5 variable-length types are sequences of a base type with
/// runtime-determined length. They map directly to `Datatype::VarLen`.
#[cfg(feature = "alloc")]
pub fn map_variable_length(base: &Datatype) -> Datatype {
    Datatype::VarLen {
        base: Box::new(base.clone()),
    }
}

/// Map an HDF5 array datatype to canonical form.
///
/// ## Parameters
///
/// - `base`: the element datatype of the array.
/// - `dims`: array dimensions (each must be > 0).
///
/// ## Invariant
///
/// All `dims[i] > 0`. The total element count is `∏ dims[i]`.
#[cfg(feature = "alloc")]
pub fn map_array(base: &Datatype, dims: &[usize]) -> Datatype {
    Datatype::Array {
        base: Box::new(base.clone()),
        dims: dims.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroUsize;

    fn nz(v: usize) -> NonZeroUsize {
        NonZeroUsize::new(v).expect("nonzero")
    }

    #[test]
    fn map_fixed_point_signed_32bit_little_endian() {
        let dt = map_fixed_point(4, 0x00);
        assert_eq!(
            dt,
            Datatype::Integer {
                bits: nz(32),
                byte_order: ByteOrder::LittleEndian,
                signed: false,
            }
        );
    }

    #[test]
    fn map_fixed_point_signed_64bit_big_endian() {
        let dt = map_fixed_point(8, 0x09);
        assert_eq!(
            dt,
            Datatype::Integer {
                bits: nz(64),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );
    }

    #[test]
    fn map_floating_point_32bit_little_endian() {
        let dt = map_floating_point(4, 0x00);
        assert_eq!(
            dt,
            Datatype::Float {
                bits: nz(32),
                byte_order: ByteOrder::LittleEndian,
            }
        );
    }

    #[test]
    fn map_floating_point_64bit_big_endian() {
        let dt = map_floating_point(8, 0x01);
        assert_eq!(
            dt,
            Datatype::Float {
                bits: nz(64),
                byte_order: ByteOrder::BigEndian,
            }
        );
    }

    #[test]
    fn map_string_fixed_ascii() {
        let dt = map_string(10, 0x00);
        assert_eq!(
            dt,
            Datatype::FixedString {
                length: 10,
                encoding: StringEncoding::Ascii,
            }
        );
    }

    #[test]
    fn map_string_fixed_utf8() {
        let dt = map_string(20, 0x01);
        assert_eq!(
            dt,
            Datatype::FixedString {
                length: 20,
                encoding: StringEncoding::Utf8,
            }
        );
    }

    #[test]
    fn map_string_variable_ascii() {
        let dt = map_string(16, 0x40);
        assert_eq!(
            dt,
            Datatype::VariableString {
                encoding: StringEncoding::Ascii,
            }
        );
    }

    #[test]
    fn map_string_variable_utf8() {
        let dt = map_string(16, 0x41);
        assert_eq!(
            dt,
            Datatype::VariableString {
                encoding: StringEncoding::Utf8,
            }
        );
    }

    #[test]
    fn map_bitfield_to_opaque() {
        let dt = map_bitfield(2, 0x00);
        assert_eq!(
            dt,
            Datatype::Opaque {
                size: 2,
                #[cfg(feature = "alloc")]
                tag: Some(String::from("HDF5_bitfield")),
            }
        );
    }

    #[test]
    fn map_opaque_with_tag() {
        let dt = map_opaque(8, Some("SIMPLE"));
        assert_eq!(
            dt,
            Datatype::Opaque {
                size: 8,
                #[cfg(feature = "alloc")]
                tag: Some(String::from("SIMPLE")),
            }
        );
    }

    #[test]
    fn map_opaque_without_tag() {
        let dt = map_opaque(4, None);
        assert_eq!(
            dt,
            Datatype::Opaque {
                size: 4,
                #[cfg(feature = "alloc")]
                tag: None,
            }
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn map_compound_with_fields() {
        let fields = vec![
            CompoundField {
                name: String::from("x"),
                datatype: Datatype::Integer {
                    bits: nz(32),
                    byte_order: ByteOrder::LittleEndian,
                    signed: true,
                },
                offset: 0,
            },
            CompoundField {
                name: String::from("y"),
                datatype: Datatype::Float {
                    bits: nz(64),
                    byte_order: ByteOrder::LittleEndian,
                },
                offset: 8,
            },
        ];
        let dt = map_compound(fields, 16);
        match dt {
            Datatype::Compound { fields, size } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[1].name, "y");
                assert_eq!(size, 16);
            }
            other => panic!("expected Compound, got {other:?}"),
        }
    }

    #[test]
    fn map_reference_object_default() {
        let dt = map_reference(8, None);
        assert_eq!(dt, Datatype::Reference(consus_core::ReferenceType::Object));
    }

    #[test]
    fn map_reference_region_default_non_standard_size() {
        let dt = map_reference(16, None);
        assert_eq!(dt, Datatype::Reference(consus_core::ReferenceType::Region));
    }

    #[test]
    fn map_reference_explicit_object() {
        let dt = map_reference(8, Some(consus_core::ReferenceType::Object));
        assert_eq!(dt, Datatype::Reference(consus_core::ReferenceType::Object));
    }

    #[test]
    fn map_reference_explicit_region() {
        let dt = map_reference(8, Some(consus_core::ReferenceType::Region));
        assert_eq!(dt, Datatype::Reference(consus_core::ReferenceType::Region));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn map_enum_from_integer_base() {
        let base = Datatype::Integer {
            bits: nz(32),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let dt = map_enum(&base);
        match dt {
            Datatype::Enum { base, members } => {
                assert_eq!(
                    *base,
                    Datatype::Integer {
                        bits: nz(32),
                        byte_order: ByteOrder::LittleEndian,
                        signed: true,
                    }
                );
                assert!(members.is_empty());
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn map_variable_length_from_float() {
        let base = Datatype::Float {
            bits: nz(64),
            byte_order: ByteOrder::LittleEndian,
        };
        let dt = map_variable_length(&base);
        match dt {
            Datatype::VarLen { base } => {
                assert_eq!(
                    *base,
                    Datatype::Float {
                        bits: nz(64),
                        byte_order: ByteOrder::LittleEndian,
                    }
                );
            }
            other => panic!("expected VarLen, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn map_array_2d_from_integer() {
        let base = Datatype::Integer {
            bits: nz(32),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        let dt = map_array(&base, &[3, 4]);
        match dt {
            Datatype::Array { base, dims } => {
                assert_eq!(
                    *base,
                    Datatype::Integer {
                        bits: nz(32),
                        byte_order: ByteOrder::LittleEndian,
                        signed: true,
                    }
                );
                assert_eq!(dims, vec![3, 4]);
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn charset_from_flags_ascii() {
        assert_eq!(charset_from_flags(0x00), StringEncoding::Ascii);
    }

    #[test]
    fn charset_from_flags_utf8() {
        assert_eq!(charset_from_flags(0x01), StringEncoding::Utf8);
    }

    #[test]
    fn charset_from_flags_unknown_defaults_ascii() {
        assert_eq!(charset_from_flags(0x05), StringEncoding::Ascii);
    }

    #[test]
    fn datatype_class_extracts_low_nibble() {
        assert_eq!(datatype_class(0x00), 0);
        assert_eq!(datatype_class(0x0F), 0x0F);
        assert_eq!(datatype_class(0x10), 0x00);
        assert_eq!(datatype_class(0x3A), 0x0A);
    }

    #[test]
    fn byte_order_from_flags_little_endian() {
        assert_eq!(byte_order_from_flags(0x00), ByteOrder::LittleEndian);
        assert_eq!(byte_order_from_flags(0x08), ByteOrder::LittleEndian);
    }

    #[test]
    fn byte_order_from_flags_big_endian() {
        assert_eq!(byte_order_from_flags(0x01), ByteOrder::BigEndian);
        assert_eq!(byte_order_from_flags(0x09), ByteOrder::BigEndian);
    }
}
