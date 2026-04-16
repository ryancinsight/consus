//! Property-based tests using proptest.
//!
//! ## Test Coverage
//!
//! - Arbitrary Datatype generation
//! - Shape generation with edge cases
//! - Selection invariants (hyperslab within bounds)

use consus_core::{
    ByteOrder, Datatype, Extent, HyperslabDim, ReferenceType, Selection, Shape, StringEncoding,
};
use core::num::NonZeroUsize;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Section 1: Strategy functions for generating types
// ---------------------------------------------------------------------------

fn byte_order_strategy() -> BoxedStrategy<ByteOrder> {
    prop_oneof![Just(ByteOrder::LittleEndian), Just(ByteOrder::BigEndian)].boxed()
}

fn string_encoding_strategy() -> BoxedStrategy<StringEncoding> {
    prop_oneof![Just(StringEncoding::Ascii), Just(StringEncoding::Utf8)].boxed()
}

fn reference_type_strategy() -> BoxedStrategy<ReferenceType> {
    prop_oneof![Just(ReferenceType::Object), Just(ReferenceType::Region)].boxed()
}

fn integer_bits_strategy() -> BoxedStrategy<usize> {
    prop_oneof![
        Just(8usize),
        Just(16usize),
        Just(32usize),
        Just(64usize),
        Just(128usize)
    ]
    .boxed()
}

fn float_bits_strategy() -> BoxedStrategy<usize> {
    prop_oneof![Just(16usize), Just(32usize), Just(64usize), Just(128usize)].boxed()
}

fn complex_bits_strategy() -> BoxedStrategy<usize> {
    prop_oneof![Just(32usize), Just(64usize)].boxed()
}

fn extent_strategy() -> BoxedStrategy<Extent> {
    // Use smaller size range to avoid overflow in product calculations
    (any::<bool>(), 1usize..=100usize)
        .prop_map(|(fixed, size)| {
            if fixed {
                Extent::Fixed(size)
            } else {
                Extent::Unlimited { current: size }
            }
        })
        .boxed()
}

// ---------------------------------------------------------------------------
// Section 2: Property tests for Datatype
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_datatype_element_size_consistency(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
        signed in any::<bool>()
    ) {
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed,
        };
        let size = dt.element_size();
        prop_assert!(size.is_some());
        prop_assert_eq!(size.unwrap(), bits / 8);
    }

    #[test]
    fn prop_datatype_float_size(
        bits in float_bits_strategy(),
        byte_order in byte_order_strategy()
    ) {
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
        };
        prop_assert_eq!(dt.element_size(), Some(bits / 8));
    }

    #[test]
    fn prop_datatype_complex_size(
        component_bits in complex_bits_strategy(),
        byte_order in byte_order_strategy()
    ) {
        let dt = Datatype::Complex {
            component_bits: NonZeroUsize::new(component_bits).unwrap(),
            byte_order,
        };
        prop_assert_eq!(dt.element_size(), Some(2 * component_bits / 8));
    }

    #[test]
    fn prop_datatype_fixed_string_size(
        length in 1usize..=1024usize,
        encoding in string_encoding_strategy()
    ) {
        let dt = Datatype::FixedString { length, encoding };
        prop_assert_eq!(dt.element_size(), Some(length));
        prop_assert!(!dt.is_variable_length());
    }

    #[test]
    fn prop_datatype_variable_string_no_size(encoding in string_encoding_strategy()) {
        let dt = Datatype::VariableString { encoding };
        prop_assert_eq!(dt.element_size(), None);
        prop_assert!(dt.is_variable_length());
    }

    #[test]
    fn prop_datatype_opaque_size(size in 1usize..=1024usize) {
        let dt = Datatype::Opaque { size, tag: None };
        prop_assert_eq!(dt.element_size(), Some(size));
        prop_assert!(!dt.is_variable_length());
    }

    #[test]
    fn prop_datatype_reference_size(ref_type in reference_type_strategy()) {
        let dt = Datatype::Reference(ref_type);
        prop_assert_eq!(dt.element_size(), Some(8));
        prop_assert!(!dt.is_variable_length());
    }

    #[test]
    fn prop_datatype_integer_is_numeric(
        bits in integer_bits_strategy(),
        byte_order in byte_order_strategy(),
        signed in any::<bool>()
    ) {
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
            signed,
        };
        prop_assert!(dt.is_numeric());
    }

    #[test]
    fn prop_datatype_float_is_numeric(
        bits in float_bits_strategy(),
        byte_order in byte_order_strategy()
    ) {
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(bits).unwrap(),
            byte_order,
        };
        prop_assert!(dt.is_numeric());
    }

    #[test]
    fn prop_datatype_complex_is_numeric(
        component_bits in complex_bits_strategy(),
        byte_order in byte_order_strategy()
    ) {
        let dt = Datatype::Complex {
            component_bits: NonZeroUsize::new(component_bits).unwrap(),
            byte_order,
        };
        prop_assert!(dt.is_numeric());
    }
}

// ---------------------------------------------------------------------------
// Section 3: Property tests for Shape
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_shape_num_elements_product(extents in prop::collection::vec(extent_strategy(), 0..=8)) {
        let shape = Shape::new(&extents);
        let expected: usize = if extents.is_empty() {
            1
        } else {
            extents.iter().map(|e| e.current_size()).product()
        };
        prop_assert_eq!(shape.num_elements(), expected);
    }

    #[test]
    fn prop_shape_rank_matches_extents(extents in prop::collection::vec(extent_strategy(), 0..=8)) {
        let shape = Shape::new(&extents);
        prop_assert_eq!(shape.rank(), extents.len());
    }

    #[test]
    fn prop_shape_has_unlimited_consistency(
        extents in prop::collection::vec(extent_strategy(), 0..=8)
    ) {
        let shape = Shape::new(&extents);
        let expected = extents.iter().any(|e| e.is_unlimited());
        prop_assert_eq!(shape.has_unlimited(), expected);
    }

    #[test]
    fn prop_shape_fixed_dimensions(dims in prop::collection::vec(1usize..=1000usize, 1..=8)) {
        let shape = Shape::fixed(&dims);
        for extent in shape.extents() {
            prop_assert!(extent.is_fixed());
            prop_assert!(!extent.is_unlimited());
        }
    }

    #[test]
    fn prop_shape_single_dimension(size in 1usize..=1_000_000usize) {
        let shape = Shape::fixed(&[size]);
        prop_assert_eq!(shape.num_elements(), size);
        prop_assert_eq!(shape.rank(), 1);
        prop_assert!(!shape.has_unlimited());
    }

    #[test]
    fn prop_shape_three_dimensions(
        d1 in 1usize..=1000usize,
        d2 in 1usize..=1000usize,
        d3 in 1usize..=1000usize
    ) {
        let shape = Shape::fixed(&[d1, d2, d3]);
        prop_assert_eq!(shape.num_elements(), d1 * d2 * d3);
    }
}

// ---------------------------------------------------------------------------
// Section 4: Property tests for Hyperslab (using unit tests)
// ---------------------------------------------------------------------------

#[test]
fn test_hyperslab_dim_num_elements_formula() {
    // Theorem: num_elements() = count × block
    for count in [0, 1, 10, 100] {
        for block in [1, 5, 10] {
            let dim = HyperslabDim {
                start: 0,
                stride: 1,
                count,
                block,
            };
            assert_eq!(dim.num_elements(), count * block);
        }
    }
}

#[test]
fn test_hyperslab_dim_max_index_formula() {
    // Theorem: max_index = start + (count - 1) * stride + block - 1
    let cases = [
        (0, 1, 10, 1, Some(9)),  // start=0, count=10, stride=1, block=1
        (5, 2, 3, 1, Some(9)),   // start=5, count=3, stride=2, block=1
        (0, 10, 2, 3, Some(12)), // start=0, count=2, stride=10, block=3
    ];
    for (start, stride, count, block, expected) in cases {
        let dim = HyperslabDim {
            start,
            stride,
            count,
            block,
        };
        assert_eq!(dim.max_index(), expected);
    }
}

#[test]
fn test_hyperslab_dim_zero_count() {
    let dim = HyperslabDim {
        start: 0,
        stride: 1,
        count: 0,
        block: 1,
    };
    assert_eq!(dim.max_index(), None);
    assert_eq!(dim.num_elements(), 0);
}

#[test]
fn test_hyperslab_dim_validity() {
    // Valid cases
    assert!(HyperslabDim::range(0, 10).is_valid_for_extent(10));
    assert!(HyperslabDim::range(0, 10).is_valid_for_extent(20));
    // Invalid cases
    assert!(!HyperslabDim::range(5, 10).is_valid_for_extent(10));
}

// ---------------------------------------------------------------------------
// Section 5: Property tests for Selection
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_selection_all_valid(extents in prop::collection::vec(extent_strategy(), 0..=8)) {
        let shape = Shape::new(&extents);
        prop_assert!(Selection::All.is_valid_for_shape(&shape));
    }

    #[test]
    fn prop_selection_none_valid(extents in prop::collection::vec(extent_strategy(), 0..=8)) {
        let shape = Shape::new(&extents);
        prop_assert!(Selection::None.is_valid_for_shape(&shape));
    }

    #[test]
    fn prop_selection_all_count(extents in prop::collection::vec(extent_strategy(), 0..=8)) {
        let shape = Shape::new(&extents);
        prop_assert_eq!(Selection::All.num_elements(&shape), shape.num_elements());
    }

    #[test]
    fn prop_selection_none_count(extents in prop::collection::vec(extent_strategy(), 0..=8)) {
        let shape = Shape::new(&extents);
        prop_assert_eq!(Selection::None.num_elements(&shape), 0);
    }
}
