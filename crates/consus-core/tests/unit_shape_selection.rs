//! Unit tests for Shape and Selection types.
//!
//! ## Test Coverage
//!
//! - Shape::fixed() vs Shape::scalar()
//! - Shape::num_elements() calculation
//! - Extent::Fixed vs Extent::Unlimited
//! - Selection::Hyperslab indexing
//! - Selection::PointSelection validation
//! - HyperslabDim stride/offset/count validation
//!
//! ## Mathematical Specifications
//!
//! ### Shape Invariants
//!
//! - `num_elements() = ∏_{i=0}^{rank-1} extents[i].current_size()`
//! - `num_elements() = 1` for rank-0 (scalar) shapes (empty product convention)
//!
//! ### Hyperslab Definition
//!
//! Selected indices along dimension `i`:
//! `{ start[i] + n × stride[i] + b : n ∈ [0, count[i]), b ∈ [0, block[i]) }`
//!
//! Total selected elements = `∏_i count[i] × block[i]`

use consus_core::{Extent, Hyperslab, HyperslabDim, PointSelection, Selection, Shape};

// ---------------------------------------------------------------------------
// Section 1: Shape::fixed() construction
// ---------------------------------------------------------------------------

#[test]
fn shape_fixed_creates_fixed_extents() {
    // Theorem: Shape::fixed(dims) creates Extent::Fixed for each dimension.
    let shape = Shape::fixed(&[10, 20, 30]);

    assert_eq!(shape.rank(), 3);

    let extents = shape.extents();
    assert!(extents[0].is_fixed());
    assert!(extents[1].is_fixed());
    assert!(extents[2].is_fixed());

    assert_eq!(extents[0].current_size(), 10);
    assert_eq!(extents[1].current_size(), 20);
    assert_eq!(extents[2].current_size(), 30);
}

#[test]
fn shape_fixed_single_dimension() {
    // 1D array with fixed dimension.
    let shape = Shape::fixed(&[100]);

    assert_eq!(shape.rank(), 1);
    assert_eq!(shape.num_elements(), 100);
    assert!(!shape.has_unlimited());
}

#[test]
fn shape_fixed_empty_dimensions() {
    // Empty dimensions creates scalar shape.
    let shape = Shape::fixed(&[]);

    assert_eq!(shape.rank(), 0);
    assert!(shape.is_scalar());
}

#[test]
fn shape_fixed_preserves_dimension_order() {
    // Theorem: Dimensions are stored in the order provided.
    let shape = Shape::fixed(&[3, 5, 7, 11]);

    assert_eq!(shape.current_dims().as_slice(), &[3, 5, 7, 11]);
}

// ---------------------------------------------------------------------------
// Section 2: Shape::scalar() (rank-0)
// ---------------------------------------------------------------------------

#[test]
fn shape_scalar_has_rank_zero() {
    // Theorem: Scalar shape has rank 0.
    let shape = Shape::scalar();

    assert_eq!(shape.rank(), 0);
    assert!(shape.is_scalar());
}

#[test]
fn shape_scalar_has_one_element() {
    // Theorem: Scalar shape has exactly 1 element (empty product convention).
    // Mathematical: num_elements() = ∏_{i=0}^{-1} ... = 1 (empty product).
    let shape = Shape::scalar();

    assert_eq!(shape.num_elements(), 1);
}

#[test]
fn shape_scalar_has_no_unlimited_dimensions() {
    let shape = Shape::scalar();
    assert!(!shape.has_unlimited());
}

#[test]
fn shape_scalar_current_dims_is_empty() {
    let shape = Shape::scalar();
    assert!(shape.current_dims().is_empty());
}

// ---------------------------------------------------------------------------
// Section 3: Shape::num_elements() calculation
// ---------------------------------------------------------------------------

#[test]
fn shape_num_elements_is_product_of_dimensions() {
    // Theorem: num_elements() = ∏ dimensions[i]

    // 2D: 10 × 20 = 200
    let shape2d = Shape::fixed(&[10, 20]);
    assert_eq!(shape2d.num_elements(), 200);

    // 3D: 3 × 4 × 5 = 60
    let shape3d = Shape::fixed(&[3, 4, 5]);
    assert_eq!(shape3d.num_elements(), 60);

    // 4D: 2 × 3 × 4 × 5 = 120
    let shape4d = Shape::fixed(&[2, 3, 4, 5]);
    assert_eq!(shape4d.num_elements(), 120);
}

#[test]
fn shape_num_elements_with_dimension_size_one() {
    // Dimension with size 1 doesn't change the product.
    let shape = Shape::fixed(&[10, 1, 20]);
    assert_eq!(shape.num_elements(), 200);
}

#[test]
fn shape_num_elements_single_dimension() {
    let shape = Shape::fixed(&[1000]);
    assert_eq!(shape.num_elements(), 1000);
}

#[test]
fn shape_num_elements_large_dimensions() {
    // Test with larger values to verify no overflow in simple cases.
    let shape = Shape::fixed(&[1024, 1024]);
    assert_eq!(shape.num_elements(), 1_048_576);
}

// ---------------------------------------------------------------------------
// Section 4: Extent variants
// ---------------------------------------------------------------------------

#[test]
fn extent_fixed_current_size() {
    // Theorem: Extent::Fixed(n).current_size() = n
    let extent = Extent::Fixed(42);
    assert_eq!(extent.current_size(), 42);
}

#[test]
fn extent_fixed_is_not_unlimited() {
    let extent = Extent::Fixed(100);

    assert!(extent.is_fixed());
    assert!(!extent.is_unlimited());
}

#[test]
fn extent_unlimited_current_size() {
    // Theorem: Extent::Unlimited{current}.current_size() = current
    let extent = Extent::Unlimited { current: 50 };
    assert_eq!(extent.current_size(), 50);
}

#[test]
fn extent_unlimited_is_unlimited() {
    let extent = Extent::Unlimited { current: 100 };

    assert!(!extent.is_fixed());
    assert!(extent.is_unlimited());
}

#[test]
fn extent_fixed_equality() {
    assert_eq!(Extent::Fixed(10), Extent::Fixed(10));
    assert_ne!(Extent::Fixed(10), Extent::Fixed(20));
}

#[test]
fn extent_unlimited_equality() {
    assert_eq!(
        Extent::Unlimited { current: 10 },
        Extent::Unlimited { current: 10 }
    );
    assert_ne!(
        Extent::Unlimited { current: 10 },
        Extent::Unlimited { current: 20 }
    );
}

#[test]
fn extent_fixed_vs_unlimited_are_distinct() {
    // Fixed and Unlimited with same current size are still different.
    let fixed = Extent::Fixed(10);
    let unlimited = Extent::Unlimited { current: 10 };
    assert_ne!(fixed, unlimited);
}

// ---------------------------------------------------------------------------
// Section 5: Shape with mixed fixed/unlimited dimensions
// ---------------------------------------------------------------------------

#[test]
fn shape_new_with_mixed_extents() {
    let extents = vec![
        Extent::Fixed(10),
        Extent::Unlimited { current: 5 },
        Extent::Fixed(20),
    ];
    let shape = Shape::new(&extents);

    assert_eq!(shape.rank(), 3);
    assert!(shape.has_unlimited());
    assert_eq!(shape.num_elements(), 10 * 5 * 20);
}

#[test]
fn shape_has_unlimited_true_when_any_dimension_unlimited() {
    let extents = vec![
        Extent::Fixed(10),
        Extent::Fixed(20),
        Extent::Unlimited { current: 5 },
    ];
    let shape = Shape::new(&extents);
    assert!(shape.has_unlimited());
}

#[test]
fn shape_has_unlimited_false_when_all_fixed() {
    let shape = Shape::fixed(&[10, 20, 30]);
    assert!(!shape.has_unlimited());
}

// ---------------------------------------------------------------------------
// Section 6: HyperslabDim basic operations
// ---------------------------------------------------------------------------

#[test]
fn hyperslab_dim_range_creates_contiguous_selection() {
    // Theorem: HyperslabDim::range(start, count) has stride=1, block=1
    let dim = HyperslabDim::range(5, 10);

    assert_eq!(dim.start, 5);
    assert_eq!(dim.stride, 1);
    assert_eq!(dim.count, 10);
    assert_eq!(dim.block, 1);
}

#[test]
fn hyperslab_dim_num_elements() {
    // Theorem: num_elements() = count × block

    // Contiguous: count=10, block=1 → 10 elements
    let contiguous = HyperslabDim::range(0, 10);
    assert_eq!(contiguous.num_elements(), 10);

    // Strided blocks: count=5, block=3 → 15 elements
    let strided = HyperslabDim {
        start: 0,
        stride: 4,
        count: 5,
        block: 3,
    };
    assert_eq!(strided.num_elements(), 15);
}

#[test]
fn hyperslab_dim_num_elements_with_zero_count() {
    // count=0 selects nothing.
    let dim = HyperslabDim {
        start: 0,
        stride: 1,
        count: 0,
        block: 5,
    };
    assert_eq!(dim.num_elements(), 0);
}

#[test]
fn hyperslab_dim_max_index_calculation() {
    // Theorem: max_index = start + (count - 1) × stride + block - 1

    // start=5, count=3, stride=2, block=1
    // max_index = 5 + (3-1)*2 + 1 - 1 = 5 + 4 + 0 = 9
    let dim = HyperslabDim {
        start: 5,
        stride: 2,
        count: 3,
        block: 1,
    };
    assert_eq!(dim.max_index(), Some(9));

    // With block > 1: start=0, count=2, stride=10, block=3
    // max_index = 0 + (2-1)*10 + 3 - 1 = 0 + 10 + 2 = 12
    let dim2 = HyperslabDim {
        start: 0,
        stride: 10,
        count: 2,
        block: 3,
    };
    assert_eq!(dim2.max_index(), Some(12));
}

#[test]
fn hyperslab_dim_max_index_none_when_count_zero() {
    let dim = HyperslabDim {
        start: 0,
        stride: 1,
        count: 0,
        block: 1,
    };
    assert_eq!(dim.max_index(), None);
}

// ---------------------------------------------------------------------------
// Section 7: HyperslabDim validation
// ---------------------------------------------------------------------------

#[test]
fn hyperslab_dim_is_valid_for_extent_true_when_within_bounds() {
    // Selection [0..10) is valid for extent 10.
    let dim = HyperslabDim::range(0, 10);
    assert!(dim.is_valid_for_extent(10));
    assert!(dim.is_valid_for_extent(20));
    assert!(!dim.is_valid_for_extent(5));
}

#[test]
fn hyperslab_dim_is_valid_for_extent_boundary_case() {
    // Selection touching the boundary is valid.
    // start=0, count=10, stride=1, block=1 → max_index=9
    // Valid for extent=10 (indices 0-9)
    let dim = HyperslabDim::range(0, 10);
    assert!(dim.is_valid_for_extent(10));
}

#[test]
fn hyperslab_dim_is_valid_for_extent_out_of_bounds() {
    // start=5, count=10 → indices 5-14, max_index=14
    // Invalid for extent=10 (only indices 0-9 valid)
    let dim = HyperslabDim::range(5, 10);
    assert!(!dim.is_valid_for_extent(10));
    assert!(dim.is_valid_for_extent(15));
    assert!(dim.is_valid_for_extent(20));
}

#[test]
fn hyperslab_dim_is_valid_for_extent_with_stride() {
    // Strided selection: start=0, stride=3, count=3, block=1
    // Indices: 0, 3, 6 → max_index=6
    // Valid for extent ≥ 7
    let dim = HyperslabDim {
        start: 0,
        stride: 3,
        count: 3,
        block: 1,
    };
    assert!(!dim.is_valid_for_extent(6));
    assert!(dim.is_valid_for_extent(7));
    assert!(dim.is_valid_for_extent(100));
}

#[test]
fn hyperslab_dim_is_valid_for_extent_with_block() {
    // Block selection: start=0, stride=10, count=2, block=3
    // First block: 0,1,2; second block: 10,11,12
    // max_index = 12
    let dim = HyperslabDim {
        start: 0,
        stride: 10,
        count: 2,
        block: 3,
    };
    assert!(!dim.is_valid_for_extent(12));
    assert!(dim.is_valid_for_extent(13));
}

#[test]
fn hyperslab_dim_zero_count_always_valid() {
    // count=0 selects nothing, always valid.
    let dim = HyperslabDim {
        start: 0,
        stride: 1,
        count: 0,
        block: 1,
    };
    assert!(dim.is_valid_for_extent(0));
    assert!(dim.is_valid_for_extent(1));
}

// ---------------------------------------------------------------------------
// Section 8: Hyperslab operations
// ---------------------------------------------------------------------------

#[test]
fn hyperslab_new_creates_from_dims() {
    let dims = vec![HyperslabDim::range(0, 10), HyperslabDim::range(5, 20)];
    let slab = Hyperslab::new(&dims);

    assert_eq!(slab.rank(), 2);
    assert_eq!(slab.dims.len(), 2);
}

#[test]
fn hyperslab_contiguous_creates_range_per_dimension() {
    let slab = Hyperslab::contiguous(&[0, 10], &[5, 20]);

    assert_eq!(slab.rank(), 2);
    assert_eq!(slab.dims[0].start, 0);
    assert_eq!(slab.dims[0].count, 5);
    assert_eq!(slab.dims[1].start, 10);
    assert_eq!(slab.dims[1].count, 20);
}

#[test]
fn hyperslab_num_elements_is_product() {
    // 2D selection: 10 × 20 = 200 elements
    let slab = Hyperslab::contiguous(&[0, 0], &[10, 20]);
    assert_eq!(slab.num_elements(), 200);

    // 3D selection: 2 × 3 × 4 = 24 elements
    let slab3d = Hyperslab::contiguous(&[0, 0, 0], &[2, 3, 4]);
    assert_eq!(slab3d.num_elements(), 24);
}

#[test]
fn hyperslab_num_elements_with_strided_selection() {
    // Per-dimension: count × block
    // Dim 0: 5 × 2 = 10
    // Dim 1: 3 × 1 = 3
    // Total: 30
    let dims = vec![
        HyperslabDim {
            start: 0,
            stride: 4,
            count: 5,
            block: 2,
        },
        HyperslabDim {
            start: 0,
            stride: 1,
            count: 3,
            block: 1,
        },
    ];
    let slab = Hyperslab::new(&dims);
    assert_eq!(slab.num_elements(), 30);
}

#[test]
fn hyperslab_is_valid_for_shape_matching_rank() {
    let shape = Shape::fixed(&[100, 200]);
    let slab = Hyperslab::contiguous(&[0, 0], &[50, 100]);

    assert!(slab.is_valid_for_shape(&shape));
}

#[test]
fn hyperslab_is_valid_for_shape_wrong_rank() {
    let shape = Shape::fixed(&[100, 200]);
    let slab = Hyperslab::contiguous(&[0, 0, 0], &[10, 10, 10]);

    assert!(!slab.is_valid_for_shape(&shape));
}

#[test]
fn hyperslab_is_valid_for_shape_out_of_bounds() {
    let shape = Shape::fixed(&[100, 200]);
    let slab = Hyperslab::contiguous(&[50, 150], &[60, 60]); // Extends past bounds

    // Dim 0: start=50, count=60 → indices 50-109, max=109 < 100? No!
    assert!(!slab.is_valid_for_shape(&shape));
}

// ---------------------------------------------------------------------------
// Section 9: PointSelection operations
// ---------------------------------------------------------------------------

#[test]
fn point_selection_num_points_calculation() {
    // 10 points in 3D space: 30 coordinates total
    let mut coords = smallvec::SmallVec::new();
    for i in 0..10 {
        coords.push(i);
        coords.push(i * 2);
        coords.push(i * 3);
    }

    let selection = PointSelection { rank: 3, coords };

    assert_eq!(selection.num_points(), 10);
}

#[test]
fn point_selection_zero_points() {
    let selection = PointSelection {
        rank: 3,
        coords: smallvec::SmallVec::new(),
    };

    assert_eq!(selection.num_points(), 0);
}

#[test]
fn point_selection_rank_zero_edge_case() {
    // Rank-0 (scalar) point selection is a degenerate case.
    let selection = PointSelection {
        rank: 0,
        coords: smallvec::SmallVec::new(),
    };

    assert_eq!(selection.num_points(), 0);
}

#[test]
fn point_selection_is_valid_for_shape_within_bounds() {
    let shape = Shape::fixed(&[10, 20]);

    // Points: (0, 0), (5, 10), (9, 19) - all valid
    let coords = smallvec::SmallVec::from_slice(&[0, 0, 5, 10, 9, 19]);

    let selection = PointSelection { rank: 2, coords };

    assert!(selection.is_valid_for_shape(&shape));
}

#[test]
fn point_selection_is_valid_for_shape_out_of_bounds() {
    let shape = Shape::fixed(&[10, 20]);

    // Point (10, 0) is out of bounds (valid indices are 0-9)
    let coords = smallvec::SmallVec::from_slice(&[10, 0]);

    let selection = PointSelection { rank: 2, coords };

    assert!(!selection.is_valid_for_shape(&shape));
}

#[test]
fn point_selection_is_valid_for_shape_wrong_rank() {
    let shape = Shape::fixed(&[10, 20]);

    // 3D points for 2D shape
    let coords = smallvec::SmallVec::from_slice(&[1, 2, 3]);

    let selection = PointSelection { rank: 3, coords };

    assert!(!selection.is_valid_for_shape(&shape));
}

#[test]
fn point_selection_boundary_coordinates() {
    let shape = Shape::fixed(&[10, 20]);

    // Point at exact boundary (9, 19) is valid
    let coords = smallvec::SmallVec::from_slice(&[9, 19]);

    let selection = PointSelection { rank: 2, coords };

    assert!(selection.is_valid_for_shape(&shape));

    // Point just past boundary (9, 20) is invalid
    let coords2 = smallvec::SmallVec::from_slice(&[9, 20]);
    let selection2 = PointSelection {
        rank: 2,
        coords: coords2,
    };
    assert!(!selection2.is_valid_for_shape(&shape));
}

// ---------------------------------------------------------------------------
// Section 10: Selection enum operations
// ---------------------------------------------------------------------------

#[test]
fn selection_all_num_elements() {
    let shape = Shape::fixed(&[10, 20, 30]);
    let selection = Selection::All;

    assert_eq!(selection.num_elements(&shape), 10 * 20 * 30);
}

#[test]
fn selection_none_num_elements() {
    let shape = Shape::fixed(&[10, 20, 30]);
    let selection = Selection::None;

    assert_eq!(selection.num_elements(&shape), 0);
}

#[test]
fn selection_hyperslab_num_elements() {
    let shape = Shape::fixed(&[100, 200]);
    let slab = Hyperslab::contiguous(&[10, 20], &[30, 40]);
    let selection = Selection::Hyperslab(slab);

    assert_eq!(selection.num_elements(&shape), 30 * 40);
}

#[test]
fn selection_points_num_elements() {
    let shape = Shape::fixed(&[100, 200]);

    let coords = smallvec::SmallVec::from_slice(&[1, 2, 3, 4, 5, 6]);
    let points = PointSelection { rank: 2, coords };
    let selection = Selection::Points(points);

    assert_eq!(selection.num_elements(&shape), 3);
}

#[test]
fn selection_is_valid_for_shape_all() {
    let shape = Shape::fixed(&[10, 20]);
    assert!(Selection::All.is_valid_for_shape(&shape));
}

#[test]
fn selection_is_valid_for_shape_none() {
    let shape = Shape::fixed(&[10, 20]);
    assert!(Selection::None.is_valid_for_shape(&shape));
}

#[test]
fn selection_is_valid_for_shape_hyperslab() {
    let shape = Shape::fixed(&[100, 200]);

    let valid_slab = Hyperslab::contiguous(&[0, 0], &[50, 100]);
    assert!(Selection::Hyperslab(valid_slab).is_valid_for_shape(&shape));

    let invalid_slab = Hyperslab::contiguous(&[0, 0], &[200, 200]);
    assert!(!Selection::Hyperslab(invalid_slab).is_valid_for_shape(&shape));
}

#[test]
fn selection_is_valid_for_shape_points() {
    let shape = Shape::fixed(&[10, 10]);

    let valid_coords = smallvec::SmallVec::from_slice(&[0, 0, 5, 5]);
    let valid_points = PointSelection {
        rank: 2,
        coords: valid_coords,
    };
    assert!(Selection::Points(valid_points).is_valid_for_shape(&shape));

    let invalid_coords = smallvec::SmallVec::from_slice(&[10, 0]);
    let invalid_points = PointSelection {
        rank: 2,
        coords: invalid_coords,
    };
    assert!(!Selection::Points(invalid_points).is_valid_for_shape(&shape));
}

// ---------------------------------------------------------------------------
// Section 11: Edge cases and invariants
// ---------------------------------------------------------------------------

#[test]
fn shape_fixed_dimension_size_one() {
    // Dimension size 1 is valid.
    let shape = Shape::fixed(&[1, 1, 1]);
    assert_eq!(shape.num_elements(), 1);
}

#[test]
fn shape_fixed_large_rank() {
    // Test with higher rank (up to 8 for SmallVec inline storage).
    let shape = Shape::fixed(&[2, 3, 5, 7, 11, 13, 17, 19]);
    assert_eq!(shape.rank(), 8);

    // Verify product: 2×3×5×7×11×13×17×19 = 9699690
    assert_eq!(shape.num_elements(), 9_699_690);
}

#[test]
fn hyperslab_dim_stride_one_block_contiguous() {
    // stride=1, block=1 is standard contiguous selection.
    let dim = HyperslabDim {
        start: 10,
        stride: 1,
        count: 20,
        block: 1,
    };
    assert_eq!(dim.num_elements(), 20);
}

#[test]
fn hyperslab_dim_large_block() {
    // Block larger than stride creates overlapping selection.
    // This is semantically valid in the model.
    let dim = HyperslabDim {
        start: 0,
        stride: 2,
        count: 3,
        block: 5,
    };

    // Indices:
    // Block 0: 0,1,2,3,4
    // Block 1: 2,3,4,5,6
    // Block 2: 4,5,6,7,8
    // max_index = 0 + 2*2 + 5 - 1 = 8
    assert_eq!(dim.max_index(), Some(8));
    assert_eq!(dim.num_elements(), 15); // 3 × 5
}

#[test]
fn shape_equality_same_dimensions() {
    let shape1 = Shape::fixed(&[10, 20, 30]);
    let shape2 = Shape::fixed(&[10, 20, 30]);
    assert_eq!(shape1, shape2);
}

#[test]
fn shape_inequality_different_dimensions() {
    let shape1 = Shape::fixed(&[10, 20, 30]);
    let shape2 = Shape::fixed(&[10, 20, 31]);
    assert_ne!(shape1, shape2);
}

#[test]
fn shape_inequality_different_rank() {
    let shape1 = Shape::fixed(&[10, 20]);
    let shape2 = Shape::fixed(&[10, 20, 1]);
    assert_ne!(shape1, shape2);
}

#[test]
fn hyperslab_equality() {
    let slab1 = Hyperslab::contiguous(&[0, 0], &[10, 20]);
    let slab2 = Hyperslab::contiguous(&[0, 0], &[10, 20]);
    assert_eq!(slab1, slab2);

    let slab3 = Hyperslab::contiguous(&[1, 1], &[10, 20]);
    assert_ne!(slab1, slab3);
}

#[test]
fn point_selection_equality() {
    let coords1 = smallvec::SmallVec::from_slice(&[1, 2, 3, 4]);
    let coords2 = smallvec::SmallVec::from_slice(&[1, 2, 3, 4]);

    let ps1 = PointSelection {
        rank: 2,
        coords: coords1,
    };
    let ps2 = PointSelection {
        rank: 2,
        coords: coords2,
    };

    assert_eq!(ps1, ps2);
}

#[test]
fn selection_equality() {
    assert_eq!(Selection::All, Selection::All);
    assert_eq!(Selection::None, Selection::None);
    assert_ne!(Selection::All, Selection::None);
}

// ---------------------------------------------------------------------------
// Section 12: Integration tests
// ---------------------------------------------------------------------------

#[test]
fn full_shape_with_hyperslab_workflow() {
    // Create a 100x200 array
    let shape = Shape::fixed(&[100, 200]);

    // Select a 50x100 region starting at (10, 20)
    let slab = Hyperslab::contiguous(&[10, 20], &[50, 100]);
    let selection = Selection::Hyperslab(slab);

    // Verify selection
    assert!(selection.is_valid_for_shape(&shape));
    assert_eq!(selection.num_elements(&shape), 5000);

    // Verify it's a proper subset
    assert!(selection.num_elements(&shape) < shape.num_elements());
}

#[test]
fn strided_hyperslab_selection() {
    // 3D shape: 30x40x50
    let shape = Shape::fixed(&[30, 40, 50]);

    // Strided selection: every other element in each dimension
    let dims = vec![
        HyperslabDim {
            start: 0,
            stride: 2,
            count: 15,
            block: 1,
        },
        HyperslabDim {
            start: 0,
            stride: 2,
            count: 20,
            block: 1,
        },
        HyperslabDim {
            start: 0,
            stride: 2,
            count: 25,
            block: 1,
        },
    ];
    let slab = Hyperslab::new(&dims);
    let selection = Selection::Hyperslab(slab);

    assert!(selection.is_valid_for_shape(&shape));
    assert_eq!(selection.num_elements(&shape), 15 * 20 * 25);
}

#[test]
fn point_selection_scattered_points() {
    // 100x100 array
    let shape = Shape::fixed(&[100, 100]);

    // Select scattered diagonal points
    let coords: smallvec::SmallVec<[usize; 64]> = (0..10).flat_map(|i| [i * 10, i * 10]).collect();

    let points = PointSelection { rank: 2, coords };
    let selection = Selection::Points(points);

    assert!(selection.is_valid_for_shape(&shape));
    assert_eq!(selection.num_elements(&shape), 10);
}
