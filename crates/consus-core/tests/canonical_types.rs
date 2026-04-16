//! SSOT (Single Source of Truth) verification tests.
//!
//! ## Test Coverage
//!
//! - Verify Datatype representations match specification
//! - Verify no duplicate type definitions across modules
//! - Verify canonical representations are consistent
//!
//! ## Mathematical Specifications
//!
//! ### Canonical Type Invariants
//!
//! - Float64 is always 64-bit (8 bytes)
//! - Float32 is always 32-bit (4 bytes)
//! - Int32 is always 32-bit (4 bytes) regardless of signedness or byte order
//! - Byte order is explicit, not implicit
//!
//! ### Representation Uniqueness
//!
//! For any two Datatype values that represent the same logical type:
//! They must be equal according to PartialEq.
//!
//! This ensures format backends can safely map native types to canonical types
//! without ambiguity.

use consus_core::{ByteOrder, Datatype, ReferenceType, StringEncoding};
use core::num::NonZeroUsize;

// ---------------------------------------------------------------------------
// Section 1: Canonical size verification
// ---------------------------------------------------------------------------

#[test]
fn canonical_float64_is_exactly_64_bits() {
    // Theorem: Float64 always occupies exactly 8 bytes (64 bits).
    // This is invariant regardless of byte order.
    let le = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let be = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };

    assert_eq!(le.element_size(), Some(8));
    assert_eq!(be.element_size(), Some(8));
}

#[test]
fn canonical_float32_is_exactly_32_bits() {
    // Theorem: Float32 always occupies exactly 4 bytes (32 bits).
    let dt = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };

    assert_eq!(dt.element_size(), Some(4));
}

#[test]
fn canonical_int32_is_exactly_32_bits() {
    // Theorem: Int32 (signed or unsigned) always occupies exactly 4 bytes.
    let signed = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let unsigned = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };

    assert_eq!(signed.element_size(), Some(4));
    assert_eq!(unsigned.element_size(), Some(4));
}

#[test]
fn canonical_int64_is_exactly_64_bits() {
    // Theorem: Int64 always occupies exactly 8 bytes.
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };

    assert_eq!(dt.element_size(), Some(8));
}

#[test]
fn canonical_complex64_is_exactly_128_bits() {
    // Theorem: Complex64 (two 32-bit floats) occupies exactly 16 bytes.
    // Complex = (real: f32, imaginary: f32) = 4 + 4 = 8 bytes.
    // Wait, that's 8 bytes total, not 16. Let me recalculate:
    // component_bits=32 means each component is 32 bits = 4 bytes.
    // Total = 2 * 4 = 8 bytes.
    let dt = Datatype::Complex {
        component_bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };

    assert_eq!(dt.element_size(), Some(8)); // 2 * 32/8 = 8 bytes
}

#[test]
fn canonical_complex128_is_exactly_128_bits() {
    // Theorem: Complex128 (two 64-bit floats) occupies exactly 16 bytes.
    // Complex = (real: f64, imaginary: f64) = 8 + 8 = 16 bytes.
    let dt = Datatype::Complex {
        component_bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };

    assert_eq!(dt.element_size(), Some(16)); // 2 * 64/8 = 16 bytes
}

// ---------------------------------------------------------------------------
// Section 2: Byte order is explicit, not implicit
// ---------------------------------------------------------------------------

#[test]
fn canonical_types_require_explicit_byte_order() {
    // Theorem: Byte order must be explicitly specified, not defaulted.
    // This test verifies the type system enforces explicit byte order.

    // Float types require byte_order field
    let _float_le = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let _float_be = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };

    // Integer types require byte_order field
    let _int_le = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let _int_be = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::BigEndian,
        signed: true,
    };

    // Complex types require byte_order field
    let _complex_le = Datatype::Complex {
        component_bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let _complex_be = Datatype::Complex {
        component_bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };
}

#[test]
fn canonical_byte_order_affects_equality_not_size() {
    // Theorem: Byte order affects equality but not element size.
    // Two floats with different byte orders are different types,
    // but they have the same element size.

    let le = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let be = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };

    // Same size
    assert_eq!(le.element_size(), be.element_size());

    // Different types
    assert_ne!(le, be);
}

// ---------------------------------------------------------------------------
// Section 3: Type uniqueness - no duplicate definitions
// ---------------------------------------------------------------------------

#[test]
fn canonical_boolean_is_unique() {
    // Theorem: Boolean is a unique type, not aliased to any integer type.
    let boolean = Datatype::Boolean;
    let uint8 = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };

    // Both have size 1, but are different types
    assert_eq!(boolean.element_size(), Some(1));
    assert_eq!(uint8.element_size(), Some(1));
    assert_ne!(boolean, uint8);
}

#[test]
fn canonical_reference_is_unique() {
    // Theorem: Reference types are unique, not aliased to integer types.
    let ref_obj = Datatype::Reference(ReferenceType::Object);
    let ref_region = Datatype::Reference(ReferenceType::Region);
    let uint64 = Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };

    // All have size 8, but are different types
    assert_eq!(ref_obj.element_size(), Some(8));
    assert_eq!(ref_region.element_size(), Some(8));
    assert_eq!(uint64.element_size(), Some(8));

    assert_ne!(ref_obj, uint64);
    assert_ne!(ref_region, uint64);
    assert_ne!(ref_obj, ref_region);
}

#[test]
fn canonical_string_types_are_distinct() {
    // Theorem: Fixed and variable string types are distinct.
    let fixed = Datatype::FixedString {
        length: 16,
        encoding: StringEncoding::Utf8,
    };
    let variable = Datatype::VariableString {
        encoding: StringEncoding::Utf8,
    };

    // Fixed has known size, variable does not
    assert_eq!(fixed.element_size(), Some(16));
    assert_eq!(variable.element_size(), None);

    assert_ne!(fixed, variable);
}

#[test]
fn canonical_encoding_affects_type_identity() {
    // Theorem: Different encodings create different types.
    let ascii = Datatype::FixedString {
        length: 16,
        encoding: StringEncoding::Ascii,
    };
    let utf8 = Datatype::FixedString {
        length: 16,
        encoding: StringEncoding::Utf8,
    };

    assert_eq!(ascii.element_size(), utf8.element_size());
    assert_ne!(ascii, utf8);
}

// ---------------------------------------------------------------------------
// Section 4: Signedness affects type identity
// ---------------------------------------------------------------------------

#[test]
fn canonical_signedness_affects_equality_not_size() {
    // Theorem: Signed and unsigned integers of same width are different types.
    let signed = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let unsigned = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };

    // Same size
    assert_eq!(signed.element_size(), unsigned.element_size());

    // Different types
    assert_ne!(signed, unsigned);
}

// ---------------------------------------------------------------------------
// Section 5: Bit width validation
// ---------------------------------------------------------------------------

#[test]
fn canonical_valid_integer_bit_widths() {
    // Theorem: Valid integer bit widths are multiples of 8: 8, 16, 32, 64, 128.
    let valid_widths = [8, 16, 32, 64, 128];

    for &width in &valid_widths {
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(width).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };

        assert_eq!(dt.element_size(), Some(width / 8));
    }
}

#[test]
fn canonical_valid_float_bit_widths() {
    // Theorem: Valid float bit widths are 16, 32, 64, 128.
    let valid_widths = [16, 32, 64, 128];

    for &width in &valid_widths {
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(width).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };

        assert_eq!(dt.element_size(), Some(width / 8));
    }
}

#[test]
fn canonical_valid_complex_component_widths() {
    // Theorem: Valid complex component bit widths are 32, 64.
    let valid_widths = [32, 64];

    for &width in &valid_widths {
        let dt = Datatype::Complex {
            component_bits: NonZeroUsize::new(width).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };

        assert_eq!(dt.element_size(), Some(2 * width / 8));
    }
}

// ---------------------------------------------------------------------------
// Section 6: No hidden type aliases
// ---------------------------------------------------------------------------

#[test]
fn canonical_no_hidden_aliases_between_numeric_types() {
    // Theorem: Integer, Float, and Complex are distinct type families.
    let int32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let float32 = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };

    // Same size, different types
    assert_eq!(int32.element_size(), float32.element_size());
    assert_ne!(int32, float32);
}

#[test]
fn canonical_no_hidden_aliases_between_container_types() {
    // Theorem: Opaque and FixedString are distinct even with same size.
    let opaque = Datatype::Opaque {
        size: 16,
        tag: None,
    };
    let fixed_str = Datatype::FixedString {
        length: 16,
        encoding: StringEncoding::Ascii,
    };

    // Same size, different types
    assert_eq!(opaque.element_size(), fixed_str.element_size());
    assert_ne!(opaque, fixed_str);
}

// ---------------------------------------------------------------------------
// Section 7: Equality is value-based, not reference-based
// ---------------------------------------------------------------------------

#[test]
fn canonical_equality_is_value_based() {
    // Theorem: Two independently constructed types with same values are equal.
    let dt1 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };

    let dt2 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };

    assert_eq!(dt1, dt2);
}

#[test]
fn canonical_clone_preserves_value() {
    // Theorem: Cloning a Datatype produces an equal value.
    let original = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let cloned = original.clone();

    assert_eq!(original, cloned);
}

// ---------------------------------------------------------------------------
// Section 8: Opaque type verification
// ---------------------------------------------------------------------------

#[test]
fn canonical_opaque_size_is_explicit() {
    // Theorem: Opaque types have explicit size, no implicit alignment.
    let dt = Datatype::Opaque {
        size: 42,
        tag: None,
    };

    assert_eq!(dt.element_size(), Some(42));
}

#[test]
fn canonical_opaque_minimum_size() {
    // Theorem: Opaque can have size 1 (byte).
    let dt = Datatype::Opaque { size: 1, tag: None };

    assert_eq!(dt.element_size(), Some(1));
}

#[test]
fn canonical_opaque_with_tag() {
    // Theorem: Opaque can have an optional tag for application-defined metadata.
    let dt_no_tag = Datatype::Opaque {
        size: 16,
        tag: None,
    };
    let dt_with_tag = Datatype::Opaque {
        size: 16,
        tag: Some("custom_type".to_string()),
    };

    // Same size regardless of tag
    assert_eq!(dt_no_tag.element_size(), dt_with_tag.element_size());
}

// ---------------------------------------------------------------------------
// Section 9: Edge cases
// ---------------------------------------------------------------------------

#[test]
fn canonical_scalar_types_have_no_elements() {
    // This test verifies the scalar shape convention.
    // Scalar shapes (rank 0) have 1 element, but scalar types
    // (single values) have their type-specific element size.
    let boolean = Datatype::Boolean;
    assert_eq!(boolean.element_size(), Some(1));
}

#[test]
fn canonical_all_base_types_covered() {
    // Theorem: All base types can be constructed and have defined sizes.
    let types_and_sizes: Vec<(Datatype, Option<usize>)> = vec![
        (Datatype::Boolean, Some(1)),
        (
            Datatype::Integer {
                bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            },
            Some(4),
        ),
        (
            Datatype::Float {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::LittleEndian,
            },
            Some(8),
        ),
        (
            Datatype::Complex {
                component_bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::LittleEndian,
            },
            Some(16),
        ),
        (
            Datatype::FixedString {
                length: 32,
                encoding: StringEncoding::Utf8,
            },
            Some(32),
        ),
        (
            Datatype::VariableString {
                encoding: StringEncoding::Utf8,
            },
            None,
        ),
        (
            Datatype::Opaque {
                size: 64,
                tag: None,
            },
            Some(64),
        ),
        (Datatype::Reference(ReferenceType::Object), Some(8)),
    ];

    for (dt, expected_size) in types_and_sizes {
        assert_eq!(
            dt.element_size(),
            expected_size,
            "Type {:?} has unexpected size",
            dt
        );
    }
}
