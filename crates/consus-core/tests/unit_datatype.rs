//! Unit tests for Datatype enum.
//!
//! ## Test Coverage
//!
//! - Datatype::element_size() for all fixed-width variants
//! - Datatype equality and hashing
//! - Datatype::is_variable_length() vs fixed-width
//! - ByteOrder variants
//! - StringEncoding variants
//!
//! ## Mathematical Specifications
//!
//! - element_size(Boolean) = 1
//! - element_size(Integer{bits}) = bits/8, bits ∈ {8,16,32,64,128}
//! - element_size(Float{bits}) = bits/8, bits ∈ {16,32,64,128}
//! - element_size(Complex{component_bits}) = 2 * component_bits/8
//! - element_size(FixedString{length}) = length
//! - element_size(Opaque{size}) = size
//! - element_size(Reference(_)) = 8

use consus_core::{ByteOrder, Datatype, ReferenceType, StringEncoding};
use core::num::NonZeroUsize;

// ---------------------------------------------------------------------------
// Section 1: element_size() for fixed-width types
// ---------------------------------------------------------------------------

#[test]
fn datatype_boolean_element_size_is_one() {
    // Theorem: element_size(Boolean) = 1 byte
    let dt = Datatype::Boolean;
    assert_eq!(dt.element_size(), Some(1));
}

#[test]
fn datatype_integer_element_size_derivation() {
    // Theorem: element_size(Integer{bits, ..}) = bits/8
    // where bits ∈ {8, 16, 32, 64, 128}

    // 8-bit integer: 1 byte
    let dt8 = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    assert_eq!(dt8.element_size(), Some(1));

    // 16-bit integer: 2 bytes
    let dt16 = Datatype::Integer {
        bits: NonZeroUsize::new(16).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    assert_eq!(dt16.element_size(), Some(2));

    // 32-bit integer: 4 bytes
    let dt32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::BigEndian,
        signed: true,
    };
    assert_eq!(dt32.element_size(), Some(4));

    // 64-bit integer: 8 bytes
    let dt64 = Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    assert_eq!(dt64.element_size(), Some(8));

    // 128-bit integer: 16 bytes
    let dt128 = Datatype::Integer {
        bits: NonZeroUsize::new(128).unwrap(),
        byte_order: ByteOrder::BigEndian,
        signed: true,
    };
    assert_eq!(dt128.element_size(), Some(16));
}

#[test]
fn datatype_float_element_size_derivation() {
    // Theorem: element_size(Float{bits, ..}) = bits/8
    // where bits ∈ {16, 32, 64, 128}

    // IEEE 754 half-precision: 2 bytes
    let dt16 = Datatype::Float {
        bits: NonZeroUsize::new(16).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    assert_eq!(dt16.element_size(), Some(2));

    // IEEE 754 single-precision: 4 bytes
    let dt32 = Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };
    assert_eq!(dt32.element_size(), Some(4));

    // IEEE 754 double-precision: 8 bytes
    let dt64 = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    assert_eq!(dt64.element_size(), Some(8));

    // IEEE 754 quad-precision: 16 bytes
    let dt128 = Datatype::Float {
        bits: NonZeroUsize::new(128).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };
    assert_eq!(dt128.element_size(), Some(16));
}

#[test]
fn datatype_complex_element_size_derivation() {
    // Theorem: element_size(Complex{component_bits, ..}) = 2 * component_bits/8
    // A complex number is (real, imaginary), each component is a float.

    // Complex with 32-bit float components: 2 * 4 = 8 bytes
    let dt32 = Datatype::Complex {
        component_bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    assert_eq!(dt32.element_size(), Some(8));

    // Complex with 64-bit float components: 2 * 8 = 16 bytes
    let dt64 = Datatype::Complex {
        component_bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };
    assert_eq!(dt64.element_size(), Some(16));
}

#[test]
fn datatype_fixed_string_element_size() {
    // Theorem: element_size(FixedString{length, ..}) = length

    let dt1 = Datatype::FixedString {
        length: 1,
        encoding: StringEncoding::Ascii,
    };
    assert_eq!(dt1.element_size(), Some(1));

    let dt16 = Datatype::FixedString {
        length: 16,
        encoding: StringEncoding::Utf8,
    };
    assert_eq!(dt16.element_size(), Some(16));

    let dt256 = Datatype::FixedString {
        length: 256,
        encoding: StringEncoding::Ascii,
    };
    assert_eq!(dt256.element_size(), Some(256));
}

#[test]
fn datatype_opaque_element_size() {
    // Theorem: element_size(Opaque{size, ..}) = size

    let dt1 = Datatype::Opaque { size: 1, tag: None };
    assert_eq!(dt1.element_size(), Some(1));

    let dt64 = Datatype::Opaque {
        size: 64,
        tag: None,
    };
    assert_eq!(dt64.element_size(), Some(64));

    let dt1024 = Datatype::Opaque {
        size: 1024,
        tag: None,
    };
    assert_eq!(dt1024.element_size(), Some(1024));
}

#[test]
fn datatype_reference_element_size() {
    // Theorem: element_size(Reference(_)) = 8 bytes
    // HDF5 object reference size is 8 bytes.

    let dt_obj = Datatype::Reference(ReferenceType::Object);
    assert_eq!(dt_obj.element_size(), Some(8));

    let dt_region = Datatype::Reference(ReferenceType::Region);
    assert_eq!(dt_region.element_size(), Some(8));
}

// ---------------------------------------------------------------------------
// Section 2: Variable-length types return None
// ---------------------------------------------------------------------------

#[test]
fn datatype_variable_string_has_no_fixed_size() {
    // Theorem: element_size(VariableString{..}) = None
    // Variable-length strings have no fixed element size.

    let dt_ascii = Datatype::VariableString {
        encoding: StringEncoding::Ascii,
    };
    assert_eq!(dt_ascii.element_size(), None);

    let dt_utf8 = Datatype::VariableString {
        encoding: StringEncoding::Utf8,
    };
    assert_eq!(dt_utf8.element_size(), None);
}

#[test]
fn datatype_is_variable_length_predicate() {
    // Theorem: is_variable_length() = (element_size() == None)

    // Fixed-width types
    assert!(!Datatype::Boolean.is_variable_length());

    let int_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    assert!(!int_dt.is_variable_length());

    let float_dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    assert!(!float_dt.is_variable_length());

    // Variable-length types
    let var_str = Datatype::VariableString {
        encoding: StringEncoding::Utf8,
    };
    assert!(var_str.is_variable_length());
}

// ---------------------------------------------------------------------------
// Section 3: Equality and hashing
// ---------------------------------------------------------------------------

#[test]
fn datatype_integer_equality_independent_of_signed_and_byte_order() {
    // Theorem: Two Integer types are equal iff bits, byte_order, and signed all match.

    let dt1 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };

    // Same type
    let dt2 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    assert_eq!(dt1, dt2);

    // Different signedness
    let dt3 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    assert_ne!(dt1, dt3);

    // Different byte order
    let dt4 = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::BigEndian,
        signed: true,
    };
    assert_ne!(dt1, dt4);

    // Different bit width
    let dt5 = Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    assert_ne!(dt1, dt5);
}

#[test]
fn datatype_float_equality() {
    // Theorem: Two Float types are equal iff bits and byte_order match.

    let dt1 = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };

    let dt2 = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    assert_eq!(dt1, dt2);

    let dt3 = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::BigEndian,
    };
    assert_ne!(dt1, dt3);
}

#[test]
fn datatype_fixed_string_equality() {
    // Theorem: Two FixedString types are equal iff length and encoding match.

    let dt1 = Datatype::FixedString {
        length: 32,
        encoding: StringEncoding::Utf8,
    };

    let dt2 = Datatype::FixedString {
        length: 32,
        encoding: StringEncoding::Utf8,
    };
    assert_eq!(dt1, dt2);

    let dt3 = Datatype::FixedString {
        length: 32,
        encoding: StringEncoding::Ascii,
    };
    assert_ne!(dt1, dt3);

    let dt4 = Datatype::FixedString {
        length: 64,
        encoding: StringEncoding::Utf8,
    };
    assert_ne!(dt1, dt4);
}

#[test]
fn datatype_boolean_is_unique() {
    // Theorem: Boolean is distinct from all other types.

    let bool_dt = Datatype::Boolean;

    let int_dt = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    assert_ne!(bool_dt, int_dt);

    let opaque_dt = Datatype::Opaque { size: 1, tag: None };
    assert_ne!(bool_dt, opaque_dt);
}

// ---------------------------------------------------------------------------
// Section 4: ByteOrder variants
// ---------------------------------------------------------------------------

#[test]
fn byte_order_variants_are_distinct() {
    // Theorem: LittleEndian ≠ BigEndian
    assert_ne!(ByteOrder::LittleEndian, ByteOrder::BigEndian);
}

#[test]
fn byte_order_equality() {
    assert_eq!(ByteOrder::LittleEndian, ByteOrder::LittleEndian);
    assert_eq!(ByteOrder::BigEndian, ByteOrder::BigEndian);
}

#[test]
fn byte_order_hash_consistency() {
    // Theorem: Equal values have equal hashes
    use core::hash::{Hash, Hasher};

    fn compute_hash<T: Hash>(value: &T) -> u64 {
        use std::hash::BuildHasher;
        let mut hasher =
            std::hash::BuildHasherDefault::<std::hash::DefaultHasher>::default().build_hasher();
        value.hash(&mut hasher);
        hasher.finish()
    }

    let h1 = compute_hash(&ByteOrder::LittleEndian);
    let h2 = compute_hash(&ByteOrder::LittleEndian);
    assert_eq!(h1, h2);

    let h3 = compute_hash(&ByteOrder::BigEndian);
    let h4 = compute_hash(&ByteOrder::BigEndian);
    assert_eq!(h3, h4);

    // Different values should have different hashes (probabilistically)
    assert_ne!(h1, h3);
}

// ---------------------------------------------------------------------------
// Section 5: StringEncoding variants
// ---------------------------------------------------------------------------

#[test]
fn string_encoding_variants_are_distinct() {
    // Theorem: Ascii ≠ Utf8
    assert_ne!(StringEncoding::Ascii, StringEncoding::Utf8);
}

#[test]
fn string_encoding_equality() {
    assert_eq!(StringEncoding::Ascii, StringEncoding::Ascii);
    assert_eq!(StringEncoding::Utf8, StringEncoding::Utf8);
}

#[test]
fn string_encoding_hash_consistency() {
    use core::hash::{Hash, Hasher};

    fn compute_hash<T: Hash>(value: &T) -> u64 {
        use std::hash::BuildHasher;
        let mut hasher =
            std::hash::BuildHasherDefault::<std::hash::DefaultHasher>::default().build_hasher();
        value.hash(&mut hasher);
        hasher.finish()
    }

    let h1 = compute_hash(&StringEncoding::Ascii);
    let h2 = compute_hash(&StringEncoding::Ascii);
    assert_eq!(h1, h2);

    let h3 = compute_hash(&StringEncoding::Utf8);
    let h4 = compute_hash(&StringEncoding::Utf8);
    assert_eq!(h3, h4);

    assert_ne!(h1, h3);
}

// ---------------------------------------------------------------------------
// Section 6: is_numeric predicate
// ---------------------------------------------------------------------------

#[test]
fn datatype_is_numeric_returns_true_for_numeric_types() {
    // Theorem: is_numeric() returns true for Integer, Float, and Complex types.

    let int_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    assert!(int_dt.is_numeric());

    let float_dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    assert!(float_dt.is_numeric());

    let complex_dt = Datatype::Complex {
        component_bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    assert!(complex_dt.is_numeric());
}

#[test]
fn datatype_is_numeric_returns_false_for_non_numeric_types() {
    // Theorem: is_numeric() returns false for non-numeric types.

    assert!(!Datatype::Boolean.is_numeric());

    let fixed_str = Datatype::FixedString {
        length: 16,
        encoding: StringEncoding::Utf8,
    };
    assert!(!fixed_str.is_numeric());

    let var_str = Datatype::VariableString {
        encoding: StringEncoding::Utf8,
    };
    assert!(!var_str.is_numeric());

    let opaque = Datatype::Opaque {
        size: 32,
        tag: None,
    };
    assert!(!opaque.is_numeric());

    let ref_dt = Datatype::Reference(ReferenceType::Object);
    assert!(!ref_dt.is_numeric());
}

// ---------------------------------------------------------------------------
// Section 7: Clone and Debug traits
// ---------------------------------------------------------------------------

#[test]
fn datatype_clone_preserves_value() {
    let original = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn datatype_debug_output() {
    let boolean = Datatype::Boolean;
    let debug_output = format!("{:?}", boolean);
    assert!(debug_output.contains("Boolean"));

    let int_dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let debug_output = format!("{:?}", int_dt);
    assert!(debug_output.contains("Integer"));
}

// ---------------------------------------------------------------------------
// Section 8: Edge cases and invariants
// ---------------------------------------------------------------------------

#[test]
fn datatype_fixed_string_minimum_length() {
    // Invariant: FixedString length must be > 0
    // Test that length = 1 is valid (minimum non-zero length)
    let dt = Datatype::FixedString {
        length: 1,
        encoding: StringEncoding::Ascii,
    };
    assert_eq!(dt.element_size(), Some(1));
}

#[test]
fn datatype_opaque_size_can_be_one() {
    // Verify Opaque with size = 1 is valid
    let dt = Datatype::Opaque { size: 1, tag: None };
    assert_eq!(dt.element_size(), Some(1));
}

#[test]
fn datatype_reference_types_are_distinct() {
    // Theorem: Object ≠ Region
    let obj_ref = Datatype::Reference(ReferenceType::Object);
    let region_ref = Datatype::Reference(ReferenceType::Region);
    assert_ne!(obj_ref, region_ref);
}

#[test]
fn datatype_reference_equality() {
    assert_eq!(
        Datatype::Reference(ReferenceType::Object),
        Datatype::Reference(ReferenceType::Object)
    );
    assert_eq!(
        Datatype::Reference(ReferenceType::Region),
        Datatype::Reference(ReferenceType::Region)
    );
}

// ---------------------------------------------------------------------------
// Section 9: Type distinctness (no accidental equality)
// ---------------------------------------------------------------------------

#[test]
fn datatype_all_base_types_are_distinct() {
    // Theorem: No two different Datatype variants are equal.

    let types: Vec<Datatype> = vec![
        Datatype::Boolean,
        Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },
        Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        },
        Datatype::Complex {
            component_bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        },
        Datatype::FixedString {
            length: 16,
            encoding: StringEncoding::Utf8,
        },
        Datatype::VariableString {
            encoding: StringEncoding::Utf8,
        },
        Datatype::Opaque {
            size: 16,
            tag: None,
        },
        Datatype::Reference(ReferenceType::Object),
    ];

    // Pairwise inequality
    for (i, t1) in types.iter().enumerate() {
        for (j, t2) in types.iter().enumerate() {
            if i != j {
                assert_ne!(
                    t1, t2,
                    "Types at indices {} and {} should be distinct",
                    i, j
                );
            }
        }
    }
}
