//! Comprehensive proptest property tests for consus-core types.
//!
//! ## Coverage
//!
//! - `Datatype::element_size()` for all variants (Boolean, Integer, Float, Complex,
//!   FixedString, VariableString, Opaque, Compound, Array, Enum, VarLen, Reference)
//! - `Datatype::is_variable_length()` / `is_numeric()` consistency with `element_size()`
//! - `Selection::num_elements()` and `is_valid_for_shape()` for All, None, Hyperslab, Points
//! - `Shape` / `Extent` / `ChunkShape` invariants (rank, element count, constructors)
//!
//! ## Mathematical Specifications
//!
//! ### Datatype::element_size() Contract
//!
//! For fixed-size types, `element_size()` returns `Some(n)` where `n > 0`.
//! For variable-length types, `element_size()` returns `None`.
//!
//! | Variant              | Formula                                    |
//! |----------------------|---------------------------------------------|
//! | Boolean              | 1                                           |
//! | Integer { bits }     | bits / 8                                    |
//! | Float { bits }       | bits / 8                                    |
//! | Complex { cb }       | 2 * cb / 8                                  |
//! | FixedString { len }  | len                                         |
//! | Opaque { size }      | size                                        |
//! | Compound { size }    | size                                        |
//! | Array { base, dims } | base.element_size() * ∏(dims)              |
//! | Enum { base }        | base.element_size()                         |
//! | VariableString       | None                                        |
//! | VarLen               | None                                        |
//! | Reference(_)         | Some(8)                                     |
//!
//! ### Shape Invariants
//!
//! - `num_elements() = ∏_{i=0}^{rank-1} extents[i].current_size()`
//! - `num_elements() = 1` for rank-0 (scalar) shapes (empty product convention)
//!
//! ### Selection Invariants
//!
//! - `Selection::All.num_elements(shape) == shape.num_elements()`
//! - `Selection::None.num_elements(shape) == 0`
//! - `Hyperslab.num_elements() = ∏_i count[i] × block[i]`
//! - `is_valid_for_shape` iff rank matches and every dim fits within bounds

#![allow(unused_imports)]

use consus_core::{
    ByteOrder, ChunkShape, CompoundField, Datatype, EnumMember, Extent, Hyperslab, HyperslabDim,
    Layout, PointSelection, ReferenceType, Selection, Shape, StringEncoding,
};
use core::num::NonZeroUsize;
use proptest::prelude::*;
use smallvec::SmallVec;
use std::boxed::Box;
use std::format;
use std::string::String;

// ===========================================================================
// Section 1: Primitive strategies
// ===========================================================================

fn byte_order_strategy() -> BoxedStrategy<ByteOrder> {
    prop_oneof![Just(ByteOrder::LittleEndian), Just(ByteOrder::BigEndian)].boxed()
}

fn string_encoding_strategy() -> BoxedStrategy<StringEncoding> {
    prop_oneof![Just(StringEncoding::Ascii), Just(StringEncoding::Utf8)].boxed()
}

fn reference_type_strategy() -> BoxedStrategy<ReferenceType> {
    prop_oneof![Just(ReferenceType::Object), Just(ReferenceType::Region)].boxed()
}

fn layout_strategy() -> BoxedStrategy<Layout> {
    prop_oneof![Just(Layout::RowMajor), Just(Layout::ColumnMajor)].boxed()
}

/// Valid integer bit widths per the representation invariant: {8, 16, 32, 64, 128}.
fn integer_bits_strategy() -> BoxedStrategy<usize> {
    prop_oneof![
        Just(8usize),
        Just(16usize),
        Just(32usize),
        Just(64usize),
        Just(128usize),
    ]
    .boxed()
}

/// Valid float bit widths per the representation invariant: {16, 32, 64, 128}.
fn float_bits_strategy() -> BoxedStrategy<usize> {
    prop_oneof![Just(16usize), Just(32usize), Just(64usize), Just(128usize),].boxed()
}

/// Valid complex component bit widths per the representation invariant: {32, 64}.
fn complex_bits_strategy() -> BoxedStrategy<usize> {
    prop_oneof![Just(32usize), Just(64usize)].boxed()
}

fn extent_strategy() -> BoxedStrategy<Extent> {
    // Keep sizes bounded to prevent overflow in product calculations.
    (any::<bool>(), 0usize..=200usize)
        .prop_map(|(fixed, size)| {
            if fixed {
                Extent::Fixed(size)
            } else {
                Extent::Unlimited { current: size }
            }
        })
        .boxed()
}

/// Strategy for non-zero usize values within a bounded range.
fn _nonzero_size_strategy(max: usize) -> BoxedStrategy<NonZeroUsize> {
    (1usize..=max)
        .prop_map(|n| NonZeroUsize::new(n).unwrap())
        .boxed()
}

// ===========================================================================
// Section 2: Datatype strategies
// ===========================================================================

/// Strategy for simple (non-recursive) Datatype variants.
fn simple_datatype_strategy() -> BoxedStrategy<Datatype> {
    prop_oneof![
        // Boolean
        Just(Datatype::Boolean),
        // Integer
        (
            integer_bits_strategy(),
            byte_order_strategy(),
            any::<bool>()
        )
            .prop_map(|(bits, byte_order, signed)| Datatype::Integer {
                bits: NonZeroUsize::new(bits).unwrap(),
                byte_order,
                signed,
            }),
        // Float
        (float_bits_strategy(), byte_order_strategy()).prop_map(|(bits, byte_order)| {
            Datatype::Float {
                bits: NonZeroUsize::new(bits).unwrap(),
                byte_order,
            }
        }),
        // Complex
        (complex_bits_strategy(), byte_order_strategy()).prop_map(
            |(component_bits, byte_order)| Datatype::Complex {
                component_bits: NonZeroUsize::new(component_bits).unwrap(),
                byte_order,
            }
        ),
        // FixedString
        (1usize..=1024, string_encoding_strategy())
            .prop_map(|(length, encoding)| Datatype::FixedString { length, encoding }),
        // VariableString
        string_encoding_strategy().prop_map(|encoding| Datatype::VariableString { encoding }),
        // Opaque
        (1usize..=4096).prop_map(|size| Datatype::Opaque { size, tag: None }),
        // Reference
        reference_type_strategy().prop_map(|r| Datatype::Reference(r)),
    ]
    .boxed()
}

/// Strategy for fixed-size Datatype variants (element_size() returns Some).
fn fixed_size_datatype_strategy() -> BoxedStrategy<Datatype> {
    prop_oneof![
        Just(Datatype::Boolean),
        (
            integer_bits_strategy(),
            byte_order_strategy(),
            any::<bool>()
        )
            .prop_map(|(bits, byte_order, signed)| Datatype::Integer {
                bits: NonZeroUsize::new(bits).unwrap(),
                byte_order,
                signed,
            }),
        (float_bits_strategy(), byte_order_strategy()).prop_map(|(bits, byte_order)| {
            Datatype::Float {
                bits: NonZeroUsize::new(bits).unwrap(),
                byte_order,
            }
        }),
        (complex_bits_strategy(), byte_order_strategy()).prop_map(
            |(component_bits, byte_order)| Datatype::Complex {
                component_bits: NonZeroUsize::new(component_bits).unwrap(),
                byte_order,
            }
        ),
        (1usize..=1024, string_encoding_strategy())
            .prop_map(|(length, encoding)| Datatype::FixedString { length, encoding }),
        (1usize..=4096).prop_map(|size| Datatype::Opaque { size, tag: None }),
        reference_type_strategy().prop_map(|r| Datatype::Reference(r)),
    ]
    .boxed()
}

/// Strategy for variable-length Datatype variants (element_size() returns None).
fn variable_datatype_strategy() -> BoxedStrategy<Datatype> {
    prop_oneof![
        string_encoding_strategy().prop_map(|encoding| Datatype::VariableString { encoding }),
    ]
    .boxed()
}

// ===========================================================================
// Section 3: Datatype::element_size() property tests
// ===========================================================================

proptest! {
    // ----- Boolean: element_size() == 1 -----

    #[test]
    fn prop_datatype_boolean_element_size_is_one(_unused: u8) {
        let dt = Datatype::Boolean;
        prop_assert_eq!(dt.element_size(), Some(1));
        prop_assert!(!dt.is_variable_length());
    }

    // ----- Integer: element_size() == bits / 8 -----

    #[test]
    fn prop_datatype_integer_element_size(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
        signed in any::<bool>(),
    ) {
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed,
        };
        let expected = bits / 8;
        prop_assert_eq!(dt.element_size(), Some(expected));
        prop_assert!(!dt.is_variable_length());
        prop_assert!(dt.is_numeric());

        // Size does not depend on signedness or byte order
        let dt_other_sign = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed: !signed,
        };
        prop_assert_eq!(dt.element_size(), dt_other_sign.element_size());

        let dt_other_order = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order: match byte_order {
                ByteOrder::LittleEndian => ByteOrder::BigEndian,
                ByteOrder::BigEndian => ByteOrder::LittleEndian,
            },
            signed,
        };
        prop_assert_eq!(dt.element_size(), dt_other_order.element_size());
    }

    // ----- Float: element_size() == bits / 8 -----

    #[test]
    fn prop_datatype_float_element_size(
        bits in float_bits_strategy(),
        byte_order in byte_order_strategy(),
    ) {
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
        };
        let expected = bits / 8;
        prop_assert_eq!(dt.element_size(), Some(expected));
        prop_assert!(!dt.is_variable_length());
        prop_assert!(dt.is_numeric());

        // Size is independent of byte order
        let dt_other_order = Datatype::Float {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order: match byte_order {
                ByteOrder::LittleEndian => ByteOrder::BigEndian,
                ByteOrder::BigEndian => ByteOrder::LittleEndian,
            },
        };
        prop_assert_eq!(dt.element_size(), dt_other_order.element_size());
    }

    // ----- Complex: element_size() == 2 * component_bits / 8 -----

    #[test]
    fn prop_datatype_complex_element_size(
        component_bits in complex_bits_strategy(),
        byte_order in byte_order_strategy(),
    ) {
        let dt = Datatype::Complex {
            component_bits: NonZeroUsize::new(component_bits).unwrap(),
            byte_order,
        };
        let expected = 2 * component_bits / 8;
        prop_assert_eq!(dt.element_size(), Some(expected));
        prop_assert!(!dt.is_variable_length());
        prop_assert!(dt.is_numeric());

        // Complex is exactly twice the size of its component float
        let float_dt = Datatype::Float {
            bits: NonZeroUsize::new(component_bits).unwrap(),
            byte_order,
        };
        prop_assert_eq!(
            dt.element_size().unwrap(),
            2 * float_dt.element_size().unwrap()
        );
    }

    // ----- FixedString: element_size() == length -----

    #[test]
    fn prop_datatype_fixed_string_element_size(
        length in 1usize..=4096,
        encoding in string_encoding_strategy(),
    ) {
        let dt = Datatype::FixedString { length, encoding };
        prop_assert_eq!(dt.element_size(), Some(length));
        prop_assert!(!dt.is_variable_length());
        prop_assert!(!dt.is_numeric());

        // Size is independent of encoding
        let other_encoding = match encoding {
            StringEncoding::Ascii => StringEncoding::Utf8,
            StringEncoding::Utf8 => StringEncoding::Ascii,
        };
        let dt_other_enc = Datatype::FixedString {
            length,
            encoding: other_encoding,
        };
        prop_assert_eq!(dt.element_size(), dt_other_enc.element_size());
    }

    // ----- VariableString: element_size() == None -----

    #[test]
    fn prop_datatype_variable_string_element_size_none(
        encoding in string_encoding_strategy(),
    ) {
        let dt = Datatype::VariableString { encoding };
        prop_assert_eq!(dt.element_size(), None);
        prop_assert!(dt.is_variable_length());
        prop_assert!(!dt.is_numeric());
    }

    // ----- Opaque: element_size() == size -----

    #[test]
    fn prop_datatype_opaque_element_size(size in 1usize..=8192) {
        let dt = Datatype::Opaque { size, tag: None };
        prop_assert_eq!(dt.element_size(), Some(size));
        prop_assert!(!dt.is_variable_length());
        prop_assert!(!dt.is_numeric());
    }

    // ----- Reference: element_size() == 8 -----

    #[test]
    fn prop_datatype_reference_element_size(ref_type in reference_type_strategy()) {
        let dt = Datatype::Reference(ref_type);
        prop_assert_eq!(dt.element_size(), Some(8));
        prop_assert!(!dt.is_variable_length());
        prop_assert!(!dt.is_numeric());

        // Both reference classes yield the same size
        let other = Datatype::Reference(match ref_type {
            ReferenceType::Object => ReferenceType::Region,
            ReferenceType::Region => ReferenceType::Object,
        });
        prop_assert_eq!(dt.element_size(), other.element_size());
    }

    // ----- Compound: element_size() == size field -----

    #[test]
    fn prop_datatype_compound_element_size(
        field_count in 1usize..=5,
        field_size in 1usize..=256,
        total_size in 1usize..=4096,
    ) {
        // total_size must be >= sum of field sizes to satisfy the invariant.
        // We construct a compound where each field is a Boolean (1 byte)
        // at successive offsets, then the compound size equals total_size.
        let fields: Vec<CompoundField> = (0..field_count)
            .map(|i| CompoundField {
                name: format!("f{}", i),
                datatype: Datatype::Boolean, // 1 byte each
                offset: i,
            })
            .collect();
        let min_size = field_count;
        let size = total_size.max(min_size);
        let dt = Datatype::Compound { fields, size };
        prop_assert_eq!(dt.element_size(), Some(size));
        prop_assert!(!dt.is_variable_length());
        prop_assert!(!dt.is_numeric());

        // Suppress unused-variable warning for field_size
        let _ = field_size;
    }

    // ----- Array: element_size() == base.element_size() * product(dims) -----

    #[test]
    fn prop_datatype_array_element_size(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
        signed in any::<bool>(),
        d1 in 1usize..=10,
        d2 in 1usize..=10,
        d3 in 1usize..=10,
    ) {
        let base = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed,
        };
        let base_size = base.element_size().unwrap();
        let dims = vec![d1, d2, d3];
        let expected = base_size * d1 * d2 * d3;
        let dt = Datatype::Array {
            base: Box::new(base),
            dims: dims.clone(),
        };
        prop_assert_eq!(dt.element_size(), Some(expected));
        prop_assert!(!dt.is_variable_length());

        // Array of a variable-length type is variable-length
        let var_base = Datatype::VariableString {
            encoding: StringEncoding::Utf8,
        };
        let var_array = Datatype::Array {
            base: Box::new(var_base),
            dims,
        };
        prop_assert_eq!(var_array.element_size(), None);
        prop_assert!(var_array.is_variable_length());
    }

    // ----- Array with single dimension: element_size() == base_size * dim -----

    #[test]
    fn prop_datatype_array_single_dim(
        length in 1usize..=1024,
        dim in 1usize..=100,
        encoding in string_encoding_strategy(),
    ) {
        let base = Datatype::FixedString { length, encoding };
        let base_size = base.element_size().unwrap();
        let dt = Datatype::Array {
            base: Box::new(base),
            dims: vec![dim],
        };
        prop_assert_eq!(dt.element_size(), Some(base_size * dim));
    }

    // ----- Enum: element_size() == base.element_size() -----

    #[test]
    fn prop_datatype_enum_element_size(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
    ) {
        let base = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed: false,
        };
        let expected = base.element_size();
        let members = vec![
            EnumMember { name: String::from("A"), value: 0 },
            EnumMember { name: String::from("B"), value: 1 },
        ];
        let dt = Datatype::Enum {
            base: Box::new(base),
            members,
        };
        prop_assert_eq!(dt.element_size(), expected);
        prop_assert!(!dt.is_variable_length());
    }

    // ----- Enum with empty members still reports base size -----

    #[test]
    fn prop_datatype_enum_empty_members(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
    ) {
        let base = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed: true,
        };
        let expected = base.element_size();
        let dt = Datatype::Enum {
            base: Box::new(base.clone()),
            members: vec![],
        };
        prop_assert_eq!(dt.element_size(), expected);
    }

    // ----- VarLen: element_size() == None -----

    #[test]
    fn prop_datatype_varlen_element_size_none(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
    ) {
        let base = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed: true,
        };
        let dt = Datatype::VarLen {
            base: Box::new(base),
        };
        prop_assert_eq!(dt.element_size(), None);
        prop_assert!(dt.is_variable_length());
        prop_assert!(!dt.is_numeric());
    }

    // ----- Cross-variant: fixed-size types never return None -----

    #[test]
    fn prop_datatype_fixed_size_always_some(dt in fixed_size_datatype_strategy()) {
        let size = dt.element_size();
        prop_assert!(size.is_some(), "fixed-size datatype returned None: {:?}", dt);
        prop_assert!(size.unwrap() > 0, "fixed-size datatype returned zero: {:?}", dt);
        prop_assert!(!dt.is_variable_length());
    }

    // ----- Cross-variant: variable-length types always return None -----

    #[test]
    fn prop_datatype_variable_always_none(dt in variable_datatype_strategy()) {
        prop_assert_eq!(dt.element_size(), None, "variable datatype returned Some: {:?}", dt);
        prop_assert!(dt.is_variable_length());
    }

    // ----- is_variable_length is consistent with element_size() -----

    #[test]
    fn prop_datatype_is_variable_length_consistent(dt in simple_datatype_strategy()) {
        let size = dt.element_size();
        let is_var = dt.is_variable_length();
        prop_assert_eq!(is_var, size.is_none(),
            "is_variable_length={} but element_size={:?} for {:?}",
            is_var, size, dt);
    }

    // ----- is_numeric is true only for Integer, Float, Complex -----

    #[test]
    fn prop_datatype_is_numeric_exclusive(dt in simple_datatype_strategy()) {
        let matches_variant = matches!(dt, Datatype::Integer { .. } | Datatype::Float { .. } | Datatype::Complex { .. });
        prop_assert_eq!(dt.is_numeric(), matches_variant,
            "is_numeric={} but matches_variant={} for {:?}",
            dt.is_numeric(), matches_variant, dt);
    }

    // ----- Integer element_size is deterministic across all valid bit widths -----

    #[test]
    fn prop_datatype_integer_all_bit_widths(
        byte_order in byte_order_strategy(),
        signed in any::<bool>(),
    ) {
        for &bits in &[8usize, 16, 32, 64, 128] {
            let dt = Datatype::Integer {
                bits: NonZeroUsize::new(bits).unwrap(),
                byte_order,
                signed,
            };
            prop_assert_eq!(dt.element_size(), Some(bits / 8));
        }
    }

    // ----- Float element_size for all valid bit widths -----

    #[test]
    fn prop_datatype_float_all_bit_widths(byte_order in byte_order_strategy()) {
        for &bits in &[16usize, 32, 64, 128] {
            let dt = Datatype::Float {
                bits: NonZeroUsize::new(bits).unwrap(),
                byte_order,
            };
            prop_assert_eq!(dt.element_size(), Some(bits / 8));
        }
    }

    // ----- Complex element_size for all valid component bit widths -----

    #[test]
    fn prop_datatype_complex_all_bit_widths(byte_order in byte_order_strategy()) {
        for &cb in &[32usize, 64] {
            let dt = Datatype::Complex {
                component_bits: NonZeroUsize::new(cb).unwrap(),
                byte_order,
            };
            prop_assert_eq!(dt.element_size(), Some(2 * cb / 8));
        }
    }
}

// ===========================================================================
// Section 4: Shape / Extent property tests
// ===========================================================================

proptest! {
    // ----- num_elements == product of current_size values -----

    #[test]
    fn prop_shape_num_elements_is_product(
        extents in prop::collection::vec(extent_strategy(), 0..=6),
    ) {
        let shape = Shape::new(&extents);
        let expected: usize = if extents.is_empty() {
            1 // empty product convention
        } else {
            extents.iter().map(|e| e.current_size()).product()
        };
        prop_assert_eq!(shape.num_elements(), expected);
    }

    // ----- rank == number of extents -----

    #[test]
    fn prop_shape_rank_equals_extent_count(
        extents in prop::collection::vec(extent_strategy(), 0..=8),
    ) {
        let shape = Shape::new(&extents);
        prop_assert_eq!(shape.rank(), extents.len());
    }

    // ----- is_scalar iff rank == 0 -----

    #[test]
    fn prop_shape_is_scalar_iff_rank_zero(
        extents in prop::collection::vec(extent_strategy(), 0..=8),
    ) {
        let shape = Shape::new(&extents);
        prop_assert_eq!(shape.is_scalar(), shape.rank() == 0);
    }

    // ----- Scalar shape always has 1 element -----

    #[test]
    fn prop_shape_scalar_one_element(_unused: u8) {
        let shape = Shape::scalar();
        prop_assert_eq!(shape.num_elements(), 1);
        prop_assert_eq!(shape.rank(), 0);
        prop_assert!(shape.is_scalar());
        prop_assert!(!shape.has_unlimited());
    }

    // ----- has_unlimited matches extent inspection -----

    #[test]
    fn prop_shape_has_unlimited_consistent(
        extents in prop::collection::vec(extent_strategy(), 0..=8),
    ) {
        let shape = Shape::new(&extents);
        let any_unlimited = extents.iter().any(|e| e.is_unlimited());
        prop_assert_eq!(shape.has_unlimited(), any_unlimited);
    }

    // ----- Shape::fixed creates all-Fixed extents -----

    #[test]
    fn prop_shape_fixed_all_extents_fixed(
        dims in prop::collection::vec(0usize..=1000, 0..=8),
    ) {
        let shape = Shape::fixed(&dims);
        prop_assert_eq!(shape.rank(), dims.len());
        for (i, extent) in shape.extents().iter().enumerate() {
            prop_assert!(extent.is_fixed());
            prop_assert!(!extent.is_unlimited());
            prop_assert_eq!(extent.current_size(), dims[i]);
        }
    }

    // ----- current_dims matches extents -----

    #[test]
    fn prop_shape_current_dims_matches_extents(
        extents in prop::collection::vec(extent_strategy(), 0..=8),
    ) {
        let shape = Shape::new(&extents);
        let current_dims = shape.current_dims();
        for (i, extent) in shape.extents().iter().enumerate() {
            prop_assert_eq!(current_dims[i], extent.current_size());
        }
    }

    // ----- Single-dimension shape invariant -----

    #[test]
    fn prop_shape_single_dim(size in 0usize..=100_000) {
        let shape = Shape::fixed(&[size]);
        prop_assert_eq!(shape.num_elements(), size);
        prop_assert_eq!(shape.rank(), 1);
        prop_assert!(!shape.has_unlimited());
    }

    // ----- Two-dimension shape invariant -----

    #[test]
    fn prop_shape_two_dims(d1 in 0usize..=1000, d2 in 0usize..=1000) {
        let shape = Shape::fixed(&[d1, d2]);
        prop_assert_eq!(shape.num_elements(), d1 * d2);
        prop_assert_eq!(shape.rank(), 2);
    }

    // ----- Three-dimension shape invariant -----

    #[test]
    fn prop_shape_three_dims(
        d1 in 0usize..=1000,
        d2 in 0usize..=1000,
        d3 in 0usize..=1000,
    ) {
        let shape = Shape::fixed(&[d1, d2, d3]);
        prop_assert_eq!(shape.num_elements(), d1 * d2 * d3);
    }

    // ----- Extent::Fixed invariants -----

    #[test]
    fn prop_extent_fixed(n in 0usize..=100_000) {
        let e = Extent::Fixed(n);
        prop_assert_eq!(e.current_size(), n);
        prop_assert!(e.is_fixed());
        prop_assert!(!e.is_unlimited());
    }

    // ----- Extent::Unlimited invariants -----

    #[test]
    fn prop_extent_unlimited(current in 0usize..=100_000) {
        let e = Extent::Unlimited { current };
        prop_assert_eq!(e.current_size(), current);
        prop_assert!(!e.is_fixed());
        prop_assert!(e.is_unlimited());
    }

    // ----- Shape with zero-size dimension has zero elements -----

    #[test]
    fn prop_shape_zero_dim_yields_zero(
        other_dims in prop::collection::vec(1usize..=100, 1..=4),
    ) {
        let mut all_dims = other_dims.clone();
        all_dims.push(0);
        let shape = Shape::fixed(&all_dims);
        prop_assert_eq!(shape.num_elements(), 0);
    }

    // ----- num_elements invariant with mixed Fixed/Unlimited extents -----

    #[test]
    fn prop_shape_mixed_extents_product(
        fixed_dims in prop::collection::vec(1usize..=50, 0..=3),
        unlimited_currents in prop::collection::vec(1usize..=50, 0..=3),
    ) {
        let mut extents: Vec<Extent> = fixed_dims.iter().map(|&d| Extent::Fixed(d)).collect();
        extents.extend(unlimited_currents.iter().map(|&c| Extent::Unlimited { current: c }));
        let shape = Shape::new(&extents);
        let expected: usize = extents.iter().map(|e| e.current_size()).product();
        prop_assert_eq!(shape.num_elements(), expected);
        prop_assert_eq!(shape.has_unlimited(), !unlimited_currents.is_empty());
    }
}

// ===========================================================================
// Section 5: ChunkShape property tests
// ===========================================================================

proptest! {
    // ----- ChunkShape::new rejects zero dimensions -----

    #[test]
    fn prop_chunk_shape_rejects_zero_dims(
        dims in prop::collection::vec(0usize..=10, 1..=4),
    ) {
        let has_zero = dims.contains(&0);
        let chunk = ChunkShape::new(&dims);
        prop_assert_eq!(chunk.is_none(), has_zero,
            "ChunkShape::new({:?}) = {:?}, expected None when zero present",
            dims, chunk);
    }

    // ----- ChunkShape::new accepts all-positive dimensions -----

    #[test]
    fn prop_chunk_shape_accepts_positive_dims(
        dims in prop::collection::vec(1usize..=100, 1..=4),
    ) {
        let chunk = ChunkShape::new(&dims);
        prop_assert!(chunk.is_some());
        let c = chunk.unwrap();
        prop_assert_eq!(c.rank(), dims.len());
        prop_assert_eq!(c.dims(), dims.as_slice());
    }

    // ----- num_chunks uses ceiling division -----

    #[test]
    fn prop_chunk_shape_num_chunks_ceiling_div(
        shape_dim in 1usize..=1000,
        chunk_dim in 1usize..=100,
    ) {
        let shape = Shape::fixed(&[shape_dim]);
        let chunk = ChunkShape::new(&[chunk_dim]).unwrap();
        let nc = chunk.num_chunks(&shape);
        let expected = shape_dim.div_ceil(chunk_dim);
        prop_assert_eq!(nc[0], expected);
    }

    // ----- total_chunks == product of per-dimension chunk counts -----

    #[test]
    fn prop_chunk_shape_total_chunks_product(
        d1 in 1usize..=100,
        d2 in 1usize..=100,
        c1 in 1usize..=20,
        c2 in 1usize..=20,
    ) {
        let shape = Shape::fixed(&[d1, d2]);
        let chunk = ChunkShape::new(&[c1, c2]).unwrap();
        let total = chunk.total_chunks(&shape);
        let expected = d1.div_ceil(c1) * d2.div_ceil(c2);
        prop_assert_eq!(total, expected);
    }

    // ----- Chunk rank equals dimension count -----

    #[test]
    fn prop_chunk_shape_rank(dims in prop::collection::vec(1usize..=50, 1..=5)) {
        let chunk = ChunkShape::new(&dims).unwrap();
        prop_assert_eq!(chunk.rank(), dims.len());
    }

    // ----- total_chunks >= 1 for non-empty shapes -----

    #[test]
    fn prop_chunk_shape_total_chunks_at_least_one(
        shape_dims in prop::collection::vec(1usize..=500, 1..=3),
        chunk_dims in prop::collection::vec(1usize..=100, 1..=3),
    ) {
        // ranks must match
        let rank = shape_dims.len().min(chunk_dims.len());
        let shape = Shape::fixed(&shape_dims[..rank]);
        let chunk = ChunkShape::new(&chunk_dims[..rank]).unwrap();
        let total = chunk.total_chunks(&shape);
        prop_assert!(total >= 1);
    }
}

// ===========================================================================
// Section 6: HyperslabDim property tests
// ===========================================================================

proptest! {
    // ----- num_elements == count * block -----

    #[test]
    fn prop_hyperslab_dim_num_elements(
        start in 0usize..=1000,
        stride in 1usize..=100,
        count in 0usize..=100,
        block in 1usize..=50,
    ) {
        let dim = HyperslabDim { start, stride, count, block };
        prop_assert_eq!(dim.num_elements(), count * block);
    }

    // ----- max_index formula: start + (count-1)*stride + block - 1 -----

    #[test]
    fn prop_hyperslab_dim_max_index_formula(
        start in 0usize..=500,
        stride in 1usize..=50,
        count in 1usize..=50,
        block in 1usize..=50,
    ) {
        let dim = HyperslabDim { start, stride, count, block };
        let expected = start + (count - 1) * stride + block - 1;
        prop_assert_eq!(dim.max_index(), Some(expected));
    }

    // ----- max_index is None when count == 0 -----

    #[test]
    fn prop_hyperslab_dim_max_index_none_when_count_zero(
        start in 0usize..=1000,
        stride in 1usize..=100,
        block in 1usize..=100,
    ) {
        let dim = HyperslabDim { start, stride, count: 0, block };
        prop_assert_eq!(dim.max_index(), None);
        prop_assert_eq!(dim.num_elements(), 0);
    }

    // ----- is_valid_for_extent: valid iff max_index < extent (or count == 0) -----

    #[test]
    fn prop_hyperslab_dim_validity(
        start in 0usize..=100,
        stride in 1usize..=20,
        count in 0usize..=20,
        block in 1usize..=20,
        extent in 1usize..=200,
    ) {
        let dim = HyperslabDim { start, stride, count, block };
        let is_valid = dim.is_valid_for_extent(extent);
        let expected = match dim.max_index() {
            None => true, // count == 0
            Some(max) => max < extent,
        };
        prop_assert_eq!(is_valid, expected,
            "is_valid_for_extent({}) = {}, expected {} for dim {:?}",
            extent, is_valid, expected, dim);
    }

    // ----- range constructor: stride=1, block=1 -----

    #[test]
    fn prop_hyperslab_dim_range(
        start in 0usize..=500,
        count in 0usize..=500,
    ) {
        let dim = HyperslabDim::range(start, count);
        prop_assert_eq!(dim.start, start);
        prop_assert_eq!(dim.stride, 1);
        prop_assert_eq!(dim.count, count);
        prop_assert_eq!(dim.block, 1);
        prop_assert_eq!(dim.num_elements(), count);
    }

    // ----- Zero-count hyperslab dim is always valid -----

    #[test]
    fn prop_hyperslab_dim_zero_count_always_valid(extent in 0usize..=1000) {
        let dim = HyperslabDim { start: 0, stride: 1, count: 0, block: 1 };
        prop_assert!(dim.is_valid_for_extent(extent));
    }

    // ----- Contiguous selection: max_index == start + count - 1 (when count > 0) -----

    #[test]
    fn prop_hyperslab_dim_contiguous_max_index(
        start in 0usize..=500,
        count in 1usize..=500,
    ) {
        let dim = HyperslabDim::range(start, count);
        prop_assert_eq!(dim.max_index(), Some(start + count - 1));
    }
}

// ===========================================================================
// Section 7: Hyperslab (multi-dim) property tests
// ===========================================================================

proptest! {
    // ----- num_elements == product of per-dim num_elements -----

    #[test]
    fn prop_hyperslab_num_elements_product(
        starts in prop::collection::vec(0usize..=50, 1..=4),
        strides in prop::collection::vec(1usize..=10, 1..=4),
        counts in prop::collection::vec(0usize..=10, 1..=4),
        blocks in prop::collection::vec(1usize..=10, 1..=4),
    ) {
        let rank = starts.len().min(strides.len()).min(counts.len()).min(blocks.len());
        let dims: Vec<HyperslabDim> = (0..rank).map(|i| HyperslabDim {
            start: starts[i],
            stride: strides[i],
            count: counts[i],
            block: blocks[i],
        }).collect();
        let slab = Hyperslab::new(&dims);
        let expected: usize = dims.iter().map(|d| d.num_elements()).product();
        prop_assert_eq!(slab.num_elements(), expected);
        prop_assert_eq!(slab.rank(), rank);
    }

    // ----- contiguous constructor: stride=1, block=1 per dim -----

    #[test]
    fn prop_hyperslab_contiguous(
        starts in prop::collection::vec(0usize..=50, 1..=4),
        counts in prop::collection::vec(0usize..=50, 1..=4),
    ) {
        let rank = starts.len().min(counts.len());
        let slab = Hyperslab::contiguous(&starts[..rank], &counts[..rank]);
        for (i, dim) in slab.dims.iter().enumerate() {
            prop_assert_eq!(dim.start, starts[i]);
            prop_assert_eq!(dim.stride, 1);
            prop_assert_eq!(dim.count, counts[i]);
            prop_assert_eq!(dim.block, 1);
        }
        let expected: usize = counts[..rank].iter().product();
        prop_assert_eq!(slab.num_elements(), expected);
    }

    // ----- is_valid_for_shape: rank mismatch => invalid -----

    #[test]
    fn prop_hyperslab_validity_rank_mismatch(
        shape_dims in prop::collection::vec(1usize..=50, 1..=4),
        slab_dims in prop::collection::vec(1usize..=10, 1..=4),
    ) {
        let shape = Shape::fixed(&shape_dims);
        let shape_rank = shape.rank();
        let slab_rank = slab_dims.len();
        if shape_rank != slab_rank {
            let dims: Vec<HyperslabDim> = slab_dims.iter().map(|&c| {
                HyperslabDim::range(0, c)
            }).collect();
            let slab = Hyperslab::new(&dims);
            prop_assert!(!slab.is_valid_for_shape(&shape));
        }
    }

    // ----- Well-formed in-bounds hyperslab is valid -----

    #[test]
    fn prop_hyperslab_in_bounds_is_valid(
        shape_dims in prop::collection::vec(10usize..=200, 1..=4),
        counts in prop::collection::vec(1usize..=5, 1..=4),
    ) {
        let rank = shape_dims.len().min(counts.len());
        let dims: Vec<HyperslabDim> = (0..rank).map(|i| {
            HyperslabDim::range(0, counts[i].min(shape_dims[i]))
        }).collect();
        let shape = Shape::fixed(&shape_dims[..rank]);
        let slab = Hyperslab::new(&dims);
        prop_assert!(slab.is_valid_for_shape(&shape),
            "expected valid: shape={:?}, slab={:?}", &shape_dims[..rank], dims);
    }

    // ----- Out-of-bounds hyperslab is invalid -----

    #[test]
    fn prop_hyperslab_out_of_bounds_is_invalid(
        shape_dim in 1usize..=50,
        start in 1usize..=100,
    ) {
        // If start >= shape_dim, any non-zero count is invalid
        let shape = Shape::fixed(&[shape_dim]);
        let dim = HyperslabDim::range(start, 1);
        let slab = Hyperslab::new(&[dim]);
        if start >= shape_dim {
            prop_assert!(!slab.is_valid_for_shape(&shape));
        }
    }
}

// ===========================================================================
// Section 8: PointSelection property tests
// ===========================================================================

proptest! {
    // ----- num_points == coords.len() / rank -----

    #[test]
    fn prop_point_selection_num_points(
        rank in 1usize..=5,
        num_points in 0usize..=20,
    ) {
        let coords: Vec<usize> = (0..rank * num_points).collect();
        let ps = PointSelection {
            rank,
            coords: coords.clone().into(),
        };
        prop_assert_eq!(ps.num_points(), num_points);
    }

    // ----- rank-0 point selection has 0 points -----

    #[test]
    fn prop_point_selection_rank_zero(_unused: u8) {
        let ps = PointSelection {
            rank: 0,
            coords: SmallVec::<[usize; 64]>::new(),
        };
        prop_assert_eq!(ps.num_points(), 0);
    }

    // ----- In-bounds point selection is valid -----

    #[test]
    fn prop_point_selection_in_bounds_valid(
        shape_dims in prop::collection::vec(10usize..=100, 1..=3),
    ) {
        let rank = shape_dims.len();
        // Generate a point with all coords = 0 (always in bounds for shape_dim >= 1)
        let coords: Vec<usize> = vec![0; rank];
        let ps = PointSelection {
            rank,
            coords: coords.clone().into(),
        };
        let shape = Shape::fixed(&shape_dims);
        prop_assert!(ps.is_valid_for_shape(&shape));
    }

    // ----- Out-of-bounds point selection is invalid -----

    #[test]
    fn prop_point_selection_out_of_bounds_invalid(
        shape_dims in prop::collection::vec(1usize..=10, 1..=3),
    ) {
        let rank = shape_dims.len();
        // Generate a point with one coord = shape_dim (out of bounds)
        let mut coords = vec![0usize; rank];
        coords[0] = shape_dims[0]; // exactly at boundary => out of bounds
        let ps = PointSelection {
            rank,
            coords: coords.clone().into(),
        };
        let shape = Shape::fixed(&shape_dims);
        prop_assert!(!ps.is_valid_for_shape(&shape));
    }

    // ----- Rank mismatch makes point selection invalid -----

    #[test]
    fn prop_point_selection_rank_mismatch_invalid(
        shape_dims in prop::collection::vec(1usize..=10, 1..=3),
        wrong_rank in 4usize..=6,
    ) {
        let shape = Shape::fixed(&shape_dims);
        let ps = PointSelection {
            rank: wrong_rank,
            coords: vec![0usize; wrong_rank].into(),
        };
        prop_assert!(!ps.is_valid_for_shape(&shape));
    }

    // ----- Boundary-coord point selection is valid -----

    #[test]
    fn prop_point_selection_boundary_valid(
        shape_dims in prop::collection::vec(1usize..=100, 1..=3),
    ) {
        let rank = shape_dims.len();
        // Point at (shape_dim[i]-1, ...) is the last valid index
        let coords: Vec<usize> = shape_dims.iter().map(|&d| d - 1).collect();
        let ps = PointSelection {
            rank,
            coords: coords.clone().into(),
        };
        let shape = Shape::fixed(&shape_dims);
        prop_assert!(ps.is_valid_for_shape(&shape));
    }
}

// ===========================================================================
// Section 9: Selection enum property tests
// ===========================================================================

proptest! {
    // ----- Selection::All is always valid -----

    #[test]
    fn prop_selection_all_always_valid(
        extents in prop::collection::vec(extent_strategy(), 0..=6),
    ) {
        let shape = Shape::new(&extents);
        prop_assert!(Selection::All.is_valid_for_shape(&shape));
    }

    // ----- Selection::None is always valid -----

    #[test]
    fn prop_selection_none_always_valid(
        extents in prop::collection::vec(extent_strategy(), 0..=6),
    ) {
        let shape = Shape::new(&extents);
        prop_assert!(Selection::None.is_valid_for_shape(&shape));
    }

    // ----- Selection::All count == shape.num_elements() -----

    #[test]
    fn prop_selection_all_num_elements(
        extents in prop::collection::vec(extent_strategy(), 0..=6),
    ) {
        let shape = Shape::new(&extents);
        prop_assert_eq!(
            Selection::All.num_elements(&shape),
            shape.num_elements()
        );
    }

    // ----- Selection::None count == 0 -----

    #[test]
    fn prop_selection_none_num_elements(
        extents in prop::collection::vec(extent_strategy(), 0..=6),
    ) {
        let shape = Shape::new(&extents);
        prop_assert_eq!(Selection::None.num_elements(&shape), 0);
    }

    // ----- Selection::Hyperslab num_elements matches product formula -----

    #[test]
    fn prop_selection_hyperslab_num_elements(
        shape_dims in prop::collection::vec(10usize..=100, 1..=4),
        counts in prop::collection::vec(1usize..=5, 1..=4),
        blocks in prop::collection::vec(1usize..=3, 1..=4),
    ) {
        let rank = shape_dims.len().min(counts.len()).min(blocks.len());
        let shape = Shape::fixed(&shape_dims[..rank]);
        let dims: Vec<HyperslabDim> = (0..rank).map(|i| {
            let count = counts[i].min(shape_dims[i]);
            HyperslabDim { start: 0, stride: 1, count, block: blocks[i] }
        }).collect();
        let slab = Hyperslab::new(&dims);
        let sel = Selection::Hyperslab(slab);
        let expected: usize = dims.iter().map(|d| d.num_elements()).product();
        prop_assert_eq!(sel.num_elements(&shape), expected);
    }

    // ----- Selection::Points num_elements matches num_points -----

    #[test]
    fn prop_selection_points_num_elements(
        rank in 1usize..=4,
        num_points in 0usize..=10,
    ) {
        let coords: Vec<usize> = vec![0; rank * num_points];
        let ps = PointSelection {
            rank,
            coords: coords.clone().into(),
        };
        let shape = Shape::fixed(&vec![100; rank]);
        let sel = Selection::Points(ps);
        prop_assert_eq!(sel.num_elements(&shape), num_points);
    }

    // ----- Valid in-bounds Hyperslab selection is valid -----

    #[test]
    fn prop_selection_hyperslab_in_bounds_valid(
        shape_dims in prop::collection::vec(10usize..=100, 1..=4),
        counts in prop::collection::vec(1usize..=5, 1..=4),
    ) {
        let rank = shape_dims.len().min(counts.len());
        let shape = Shape::fixed(&shape_dims[..rank]);
        let dims: Vec<HyperslabDim> = (0..rank).map(|i| {
            HyperslabDim::range(0, counts[i].min(shape_dims[i]))
        }).collect();
        let slab = Hyperslab::new(&dims);
        let sel = Selection::Hyperslab(slab);
        prop_assert!(sel.is_valid_for_shape(&shape));
    }

    // ----- Selection::Hyperslab with zero count is valid and has 0 elements -----

    #[test]
    fn prop_selection_hyperslab_zero_count(
        shape_dims in prop::collection::vec(1usize..=100, 1..=4),
    ) {
        let rank = shape_dims.len();
        let shape = Shape::fixed(&shape_dims);
        let dims: Vec<HyperslabDim> = (0..rank).map(|_| {
            HyperslabDim { start: 0, stride: 1, count: 0, block: 1 }
        }).collect();
        let slab = Hyperslab::new(&dims);
        let sel = Selection::Hyperslab(slab);
        prop_assert!(sel.is_valid_for_shape(&shape));
        prop_assert_eq!(sel.num_elements(&shape), 0);
    }

    // ----- Selection::Hyperslab num_elements <= shape.num_elements() when valid -----

    #[test]
    fn prop_selection_hyperslab_bounded(
        shape_dims in prop::collection::vec(10usize..=100, 1..=4),
        counts in prop::collection::vec(1usize..=5, 1..=4),
    ) {
        let rank = shape_dims.len().min(counts.len());
        let shape = Shape::fixed(&shape_dims[..rank]);
        let dims: Vec<HyperslabDim> = (0..rank).map(|i| {
            HyperslabDim::range(0, counts[i].min(shape_dims[i]))
        }).collect();
        let slab = Hyperslab::new(&dims);
        let sel = Selection::Hyperslab(slab);
        if sel.is_valid_for_shape(&shape) {
            prop_assert!(sel.num_elements(&shape) <= shape.num_elements(),
                "selection count {} exceeds shape count {}",
                sel.num_elements(&shape), shape.num_elements());
        }
    }
}

// ===========================================================================
// Section 10: Layout property tests
// ===========================================================================

proptest! {
    // ----- Default layout is RowMajor -----

    #[test]
    fn prop_layout_default_is_row_major(_unused: u8) {
        prop_assert_eq!(Layout::default(), Layout::RowMajor);
    }

    // ----- Layout round-trip through all variants -----

    #[test]
    fn prop_layout_variants_are_distinct(layout in layout_strategy()) {
        match layout {
            Layout::RowMajor => prop_assert_ne!(Layout::RowMajor, Layout::ColumnMajor),
            Layout::ColumnMajor => prop_assert_ne!(Layout::ColumnMajor, Layout::RowMajor),
        }
    }
}

// ===========================================================================
// Section 11: Cross-cutting invariants
// ===========================================================================

proptest! {
    // ----- Compound with padding: element_size >= sum of field sizes -----

    #[test]
    fn prop_datatype_compound_size_geq_field_sum(
        n_fields in 1usize..=8,
        padding in 0usize..=64,
    ) {
        let fields: Vec<CompoundField> = (0..n_fields)
            .map(|i| CompoundField {
                name: format!("field_{}", i),
                datatype: Datatype::Boolean, // 1 byte each
                offset: i,
            })
            .collect();
        let field_sum: usize = fields.len(); // each Boolean = 1 byte
        let size = field_sum + padding;
        let dt = Datatype::Compound { fields, size };
        prop_assert_eq!(dt.element_size(), Some(size));
        prop_assert!(dt.element_size().unwrap() >= field_sum,
            "compound size {} < field sum {}", size, field_sum);
    }

    // ----- Nested array: element_size == inner_base_size * product of all dims -----

    #[test]
    fn prop_datatype_nested_array_size(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
        d1 in 1usize..=5,
        d2 in 1usize..=5,
        d3 in 1usize..=5,
    ) {
        let base = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed: false,
        };
        let base_size = base.element_size().unwrap();

        // inner array: base × [d1, d2]
        let inner = Datatype::Array {
            base: Box::new(base),
            dims: vec![d1, d2],
        };
        let inner_size = inner.element_size().unwrap();
        prop_assert_eq!(inner_size, base_size * d1 * d2);

        // outer array: inner × [d3]
        let outer = Datatype::Array {
            base: Box::new(inner),
            dims: vec![d3],
        };
        let outer_size = outer.element_size().unwrap();
        prop_assert_eq!(outer_size, base_size * d1 * d2 * d3);
    }

    // ----- Enum element_size is independent of members -----

    #[test]
    fn prop_datatype_enum_size_independent_of_members(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
        n_members in 0usize..=10,
    ) {
        let base = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed: false,
        };
        let base_size = base.element_size().unwrap();
        let members: Vec<EnumMember> = (0..n_members)
            .map(|i| EnumMember {
                name: format!("m{}", i),
                value: i as i64,
            })
            .collect();
        let dt = Datatype::Enum {
            base: Box::new(base),
            members,
        };
        prop_assert_eq!(dt.element_size(), Some(base_size));
    }

    // ----- VarLen always reports None regardless of base -----

    #[test]
    fn prop_datatype_varlen_none_for_any_base(dt in fixed_size_datatype_strategy()) {
        let varlen = Datatype::VarLen {
            base: Box::new(dt.clone()),
        };
        prop_assert_eq!(varlen.element_size(), None);
        prop_assert!(varlen.is_variable_length());
    }

    // ----- Shape equality: same extents => equal -----

    #[test]
    fn prop_shape_equality_same_extents(
        dims in prop::collection::vec(1usize..=100, 1..=4),
    ) {
        let s1 = Shape::fixed(&dims);
        let s2 = Shape::fixed(&dims);
        prop_assert_eq!(s1, s2);
    }

    // ----- ChunkShape total_chunks * chunk_volume >= shape_volume -----

    #[test]
    fn prop_chunk_shape_covers_shape(
        d1 in 1usize..=100,
        d2 in 1usize..=100,
        c1 in 1usize..=50,
        c2 in 1usize..=50,
    ) {
        let shape = Shape::fixed(&[d1, d2]);
        let chunk = ChunkShape::new(&[c1, c2]).unwrap();
        let total_chunks = chunk.total_chunks(&shape);
        let chunk_volume: usize = chunk.dims().iter().product();
        let shape_volume = shape.num_elements();
        // total_chunks * chunk_volume >= shape_volume (chunks may overhang)
        prop_assert!(total_chunks * chunk_volume >= shape_volume,
            "total_chunks={} * chunk_volume={} = {} < shape_volume={}",
            total_chunks, chunk_volume, total_chunks * chunk_volume, shape_volume);
    }
}

// ===========================================================================
// Section 12: Edge-case determinism tests
// ===========================================================================

proptest! {
    // ----- element_size() is deterministic: same input => same output -----

    #[test]
    fn prop_datatype_element_size_deterministic(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
        signed in any::<bool>(),
    ) {
        let dt1 = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed,
        };
        let dt2 = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed,
        };
        prop_assert_eq!(dt1.element_size(), dt2.element_size());
    }

    // ----- Multiple calls to element_size() are idempotent -----

    #[test]
    fn prop_datatype_element_size_idempotent(dt in simple_datatype_strategy()) {
        let s1 = dt.element_size();
        let s2 = dt.element_size();
        prop_assert_eq!(s1, s2);
    }

    // ----- Shape::new then fixed produce different representations for same sizes -----

    #[test]
    fn prop_shape_new_vs_fixed_extents(
        dims in prop::collection::vec(1usize..=100, 1..=4),
    ) {
        let fixed_shape = Shape::fixed(&dims);
        let extents: Vec<Extent> = dims.iter().map(|&d| Extent::Fixed(d)).collect();
        let new_shape = Shape::new(&extents);
        // Same current_dims, same rank, same num_elements
        prop_assert_eq!(fixed_shape.num_elements(), new_shape.num_elements());
        prop_assert_eq!(fixed_shape.rank(), new_shape.rank());
        prop_assert_eq!(fixed_shape.current_dims(), new_shape.current_dims());
    }
}

// ===========================================================================
// Section 13: Focused value-semantic assertions for all Datatype variants
// ===========================================================================

#[test]
fn test_datatype_boolean_exact_value() {
    assert_eq!(Datatype::Boolean.element_size(), Some(1));
    assert!(!Datatype::Boolean.is_variable_length());
    assert!(!Datatype::Boolean.is_numeric());
}

#[test]
fn test_datatype_integer_exact_values_all_widths() {
    for (bits, expected_bytes) in [(8, 1), (16, 2), (32, 4), (64, 8), (128, 16)] {
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        assert_eq!(
            dt.element_size(),
            Some(expected_bytes),
            "Integer({}) expected {} bytes, got {:?}",
            bits,
            expected_bytes,
            dt.element_size()
        );
    }
}

#[test]
fn test_datatype_float_exact_values_all_widths() {
    for (bits, expected_bytes) in [(16, 2), (32, 4), (64, 8), (128, 16)] {
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order: ByteOrder::BigEndian,
        };
        assert_eq!(
            dt.element_size(),
            Some(expected_bytes),
            "Float({}) expected {} bytes, got {:?}",
            bits,
            expected_bytes,
            dt.element_size()
        );
    }
}

#[test]
fn test_datatype_complex_exact_values_all_widths() {
    for (cb, expected_bytes) in [(32, 8), (64, 16)] {
        let dt = Datatype::Complex {
            component_bits: NonZeroUsize::new(cb).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        assert_eq!(
            dt.element_size(),
            Some(expected_bytes),
            "Complex(component_bits={}) expected {} bytes, got {:?}",
            cb,
            expected_bytes,
            dt.element_size()
        );
    }
}

#[test]
fn test_datatype_fixed_string_exact_values() {
    for length in [1, 10, 256, 1024] {
        let dt = Datatype::FixedString {
            length,
            encoding: StringEncoding::Utf8,
        };
        assert_eq!(dt.element_size(), Some(length));
    }
}

#[test]
fn test_datatype_variable_string_exact_value() {
    let dt = Datatype::VariableString {
        encoding: StringEncoding::Ascii,
    };
    assert_eq!(dt.element_size(), None);
    assert!(dt.is_variable_length());
}

#[test]
fn test_datatype_opaque_exact_values() {
    for size in [1, 16, 256, 4096] {
        let dt = Datatype::Opaque { size, tag: None };
        assert_eq!(dt.element_size(), Some(size));
    }
}

#[test]
fn test_datatype_compound_exact_value() {
    let fields = vec![
        CompoundField {
            name: "x".into(),
            datatype: Datatype::Boolean,
            offset: 0,
        },
        CompoundField {
            name: "y".into(),
            datatype: Datatype::Boolean,
            offset: 1,
        },
    ];
    let dt = Datatype::Compound { fields, size: 4 };
    assert_eq!(dt.element_size(), Some(4)); // includes 2 bytes padding
}

#[test]
fn test_datatype_array_exact_value() {
    // 3-element array of Float64 (8 bytes each) = 24 bytes
    let base = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let dt = Datatype::Array {
        base: Box::new(base),
        dims: vec![3],
    };
    assert_eq!(dt.element_size(), Some(24));
}

#[test]
fn test_datatype_array_2d_exact_value() {
    // 4×5 array of Int32 (4 bytes each) = 80 bytes
    let base = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::BigEndian,
        signed: true,
    };
    let dt = Datatype::Array {
        base: Box::new(base),
        dims: vec![4, 5],
    };
    assert_eq!(dt.element_size(), Some(80));
}

#[test]
fn test_datatype_enum_exact_value() {
    let base = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let dt = Datatype::Enum {
        base: Box::new(base),
        members: vec![
            EnumMember {
                name: "Red".into(),
                value: 0,
            },
            EnumMember {
                name: "Green".into(),
                value: 1,
            },
            EnumMember {
                name: "Blue".into(),
                value: 2,
            },
        ],
    };
    assert_eq!(dt.element_size(), Some(4));
}

#[test]
fn test_datatype_varlen_exact_value() {
    let base = Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let dt = Datatype::VarLen {
        base: Box::new(base),
    };
    assert_eq!(dt.element_size(), None);
    assert!(dt.is_variable_length());
}

#[test]
fn test_datatype_reference_exact_value() {
    assert_eq!(
        Datatype::Reference(ReferenceType::Object).element_size(),
        Some(8)
    );
    assert_eq!(
        Datatype::Reference(ReferenceType::Region).element_size(),
        Some(8)
    );
}

// ===========================================================================
// Section 14: Focused value-semantic assertions for Selection/Shape
// ===========================================================================

#[test]
fn test_selection_all_exact_count() {
    let shape = Shape::fixed(&[3, 4, 5]);
    assert_eq!(Selection::All.num_elements(&shape), 60);
}

#[test]
fn test_selection_none_exact_count() {
    let shape = Shape::fixed(&[3, 4, 5]);
    assert_eq!(Selection::None.num_elements(&shape), 0);
}

#[test]
fn test_selection_hyperslab_contiguous_exact_count() {
    let shape = Shape::fixed(&[10, 20]);
    let slab = Hyperslab::contiguous(&[2, 3], &[4, 5]);
    let sel = Selection::Hyperslab(slab);
    assert_eq!(sel.num_elements(&shape), 20); // 4 × 5
    assert!(sel.is_valid_for_shape(&shape));
}

#[test]
fn test_selection_hyperslab_strided_exact_count() {
    let shape = Shape::fixed(&[100, 100]);
    let slab = Hyperslab::new(&[
        HyperslabDim {
            start: 0,
            stride: 2,
            count: 5,
            block: 3,
        },
        HyperslabDim {
            start: 10,
            stride: 5,
            count: 4,
            block: 2,
        },
    ]);
    let sel = Selection::Hyperslab(slab);
    assert_eq!(sel.num_elements(&shape), 120); // (5*3) × (4*2) = 15 × 8 = 120
    assert!(sel.is_valid_for_shape(&shape));
}

#[test]
fn test_selection_points_exact_count() {
    let shape = Shape::fixed(&[10, 10]);
    let ps = PointSelection {
        rank: 2,
        coords: SmallVec::<[usize; 64]>::from_slice(&[0, 0, 5, 5, 9, 9]),
    };
    let sel = Selection::Points(ps);
    assert_eq!(sel.num_elements(&shape), 3);
    assert!(sel.is_valid_for_shape(&shape));
}

#[test]
fn test_shape_scalar_exact() {
    let s = Shape::scalar();
    assert_eq!(s.num_elements(), 1);
    assert_eq!(s.rank(), 0);
    assert!(s.is_scalar());
}

#[test]
fn test_shape_fixed_exact() {
    let s = Shape::fixed(&[7, 13]);
    assert_eq!(s.num_elements(), 91);
    assert_eq!(s.rank(), 2);
}

#[test]
fn test_chunk_shape_exact() {
    let shape = Shape::fixed(&[100, 200]);
    let chunk = ChunkShape::new(&[32, 64]).unwrap();
    assert_eq!(chunk.num_chunks(&shape)[0], 4); // ceil(100/32) = 4
    assert_eq!(chunk.num_chunks(&shape)[1], 4); // ceil(200/64) = 4
    assert_eq!(chunk.total_chunks(&shape), 16);
}

#[test]
fn test_extent_fixed_vs_unlimited() {
    let f = Extent::Fixed(42);
    let u = Extent::Unlimited { current: 42 };
    assert_eq!(f.current_size(), u.current_size());
    assert!(f.is_fixed());
    assert!(!f.is_unlimited());
    assert!(!u.is_fixed());
    assert!(u.is_unlimited());
}
