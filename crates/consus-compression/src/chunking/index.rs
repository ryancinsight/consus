//! Chunk coordinate ↔ linear index conversion.
//!
//! All conversions use row-major (C) order: the last dimension varies fastest.
//!
//! ## Theorems
//!
//! **Theorem 1** (Bijectivity): For a chunk grid of shape `G = [G₀, ..., Gₙ₋₁]`,
//! the mapping [`chunk_coord_to_linear`] is a bijection from
//! `{(c₀,...,cₙ₋₁) : 0 ≤ cᵢ < Gᵢ}` → `{0, ..., ∏Gᵢ - 1}`.
//!
//! **Proof**: This is the standard row-major linearization:
//!
//! ```text
//!   linear = Σᵢ cᵢ · ∏(j>i) Gⱼ
//! ```
//!
//! which is a bijection by the mixed-radix representation theorem.
//! Each coordinate `cᵢ` occupies a unique positional value weighted by
//! the product of all subsequent dimension extents, forming a complete
//! mixed-radix numeral system. The representation is unique because
//! `0 ≤ cᵢ < Gᵢ` for all `i`, satisfying the digit-bound constraint
//! of the mixed-radix system.
//!
//! **Theorem 2** (Round-trip):
//!
//! ```text
//!   linear_to_chunk_coord(chunk_coord_to_linear(coord, grid), grid) == coord
//!   chunk_coord_to_linear(linear_to_chunk_coord(linear, grid), grid) == linear
//! ```
//!
//! **Proof**: Follows directly from Theorem 1 (bijectivity). The inverse
//! of the mixed-radix encoding is the iterated quotient-remainder
//! decomposition implemented in [`linear_to_chunk_coord`].
//!
//! **Theorem 3** (Ceiling division correctness):
//!
//! ```text
//!   ⌈a / b⌉ = (a + b - 1) / b   for a ≥ 0, b > 0 (integer division)
//! ```
//!
//! **Proof**: Let `a = q·b + r` where `0 ≤ r < b`.
//! - If `r = 0`: `⌈a/b⌉ = q` and `(a + b - 1)/b = (q·b + b - 1)/b = q + (b-1)/b = q`. ✓
//! - If `r > 0`: `⌈a/b⌉ = q + 1` and `(a + b - 1)/b = (q·b + r + b - 1)/b = q + (r + b - 1)/b`.
//!   Since `1 ≤ r ≤ b-1`, we have `b ≤ r + b - 1 ≤ 2b - 2`, so the floor is `q + 1`. ✓
//!
//! We use `usize::div_ceil` (stable since Rust 1.73) which implements this correctly
//! and handles `a = 0` (yielding 0).

/// Compute the chunk grid shape from dataset and chunk dimensions.
///
/// For each axis `i`:
///
/// ```text
///   grid[i] = ⌈dataset_shape[i] / chunk_shape[i]⌉
/// ```
///
/// This is the number of chunks needed to cover every element along that axis.
/// A dataset dimension of zero yields a grid dimension of zero (empty axis).
///
/// # Errors
///
/// Returns [`consus_core::Error::ShapeError`] if:
/// - `dataset_shape` and `chunk_shape` have different lengths (rank mismatch).
/// - Any `chunk_shape[i]` is zero (division by zero is undefined).
#[cfg(feature = "alloc")]
pub fn chunk_grid_shape(
    dataset_shape: &[usize],
    chunk_shape: &[usize],
) -> consus_core::Result<alloc::vec::Vec<usize>> {
    let ndim = dataset_shape.len();
    let mut output = alloc::vec![0usize; ndim];
    let written = chunk_grid_shape_fixed(dataset_shape, chunk_shape, &mut output)?;
    debug_assert_eq!(written, ndim);
    Ok(output)
}

/// Compute chunk grid shape into a caller-provided output slice.
///
/// This is the `no_std`, no-alloc variant of [`chunk_grid_shape`].
/// The caller supplies `output` which must have length ≥ `dataset_shape.len()`.
///
/// Returns the number of dimensions written (equal to `dataset_shape.len()` on success).
///
/// # Errors
///
/// Returns [`consus_core::Error::ShapeError`] if:
/// - `dataset_shape.len() != chunk_shape.len()` (rank mismatch).
/// - Any `chunk_shape[i] == 0` (undefined division).
/// - `output.len() < dataset_shape.len()` (output buffer too small).
pub fn chunk_grid_shape_fixed(
    dataset_shape: &[usize],
    chunk_shape: &[usize],
    output: &mut [usize],
) -> consus_core::Result<usize> {
    let ndim = dataset_shape.len();

    if chunk_shape.len() != ndim {
        return Err(consus_core::Error::ShapeError {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "rank mismatch: dataset_shape has {} dimensions, chunk_shape has {}",
                ndim,
                chunk_shape.len()
            ),
        });
    }

    if output.len() < ndim {
        return Err(consus_core::Error::ShapeError {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "output buffer too small: need {} slots, got {}",
                ndim,
                output.len()
            ),
        });
    }

    for i in 0..ndim {
        if chunk_shape[i] == 0 {
            return Err(consus_core::Error::ShapeError {
                #[cfg(feature = "alloc")]
                message: alloc::format!("chunk_shape[{}] is zero", i),
            });
        }
        output[i] = dataset_shape[i].div_ceil(chunk_shape[i]);
    }

    Ok(ndim)
}

/// Total number of chunks in the grid.
///
/// ```text
///   total = ∏ᵢ grid_shape[i]
/// ```
///
/// For a 0-dimensional (scalar) dataset, `grid_shape` is empty and the product
/// is the empty product, which equals 1 by convention. This is correct: a
/// scalar dataset consists of exactly one chunk containing the single element.
///
/// # Errors
///
/// Returns [`consus_core::Error::Overflow`] if the product of grid dimensions
/// overflows `usize`.
pub fn total_chunks(grid_shape: &[usize]) -> consus_core::Result<usize> {
    let mut total: usize = 1;
    for &g in grid_shape {
        total = total.checked_mul(g).ok_or(consus_core::Error::Overflow)?;
    }
    Ok(total)
}

/// Convert an N-dimensional chunk coordinate to a linear (row-major) index.
///
/// Implements the mixed-radix linearization:
///
/// ```text
///   linear = Σᵢ coord[i] · ∏(j>i) grid_shape[j]
/// ```
///
/// Computed via Horner's method to avoid explicit stride computation:
///
/// ```text
///   linear = coord[0]
///   for i in 1..n:
///       linear = linear * grid_shape[i] + coord[i]
/// ```
///
/// For 0-dimensional data (`coord` and `grid_shape` both empty), returns 0
/// (the single scalar chunk has linear index 0).
///
/// # Panics
///
/// Panics if `coord.len() != grid_shape.len()`.
pub fn chunk_coord_to_linear(coord: &[usize], grid_shape: &[usize]) -> usize {
    assert_eq!(
        coord.len(),
        grid_shape.len(),
        "coord length {} != grid_shape length {}",
        coord.len(),
        grid_shape.len()
    );

    let mut linear: usize = 0;
    for i in 0..coord.len() {
        linear = linear * grid_shape[i] + coord[i];
    }
    linear
}

/// Convert a linear index to an N-dimensional chunk coordinate (row-major).
///
/// Implements the iterated quotient-remainder decomposition:
///
/// ```text
///   for i in (0..n).rev():
///       coord[i] = remaining % grid_shape[i]
///       remaining = remaining / grid_shape[i]
/// ```
///
/// This is the inverse of [`chunk_coord_to_linear`] (Theorem 2).
///
/// For 0-dimensional data (`grid_shape` and `coord_out` both empty), no
/// writes are performed. The linear index 0 maps to the empty coordinate.
///
/// # Panics
///
/// Panics if `coord_out.len() != grid_shape.len()`.
pub fn linear_to_chunk_coord(linear: usize, grid_shape: &[usize], coord_out: &mut [usize]) {
    assert_eq!(
        coord_out.len(),
        grid_shape.len(),
        "coord_out length {} != grid_shape length {}",
        coord_out.len(),
        grid_shape.len()
    );

    let mut remaining = linear;
    for i in (0..grid_shape.len()).rev() {
        coord_out[i] = remaining % grid_shape[i];
        remaining /= grid_shape[i];
    }
}

/// Compute the element range covered by a chunk within the dataset.
///
/// For chunk at coordinate `coord` with chunk dimensions `chunk_shape` and
/// dataset dimensions `dataset_shape`, the chunk covers the axis-aligned
/// hyper-rectangle:
///
/// ```text
///   axis i: [coord[i] * chunk_shape[i],
///            min((coord[i] + 1) * chunk_shape[i], dataset_shape[i]))
/// ```
///
/// The start and count for each axis are written into `start_out` and
/// `count_out` respectively:
///
/// ```text
///   start_out[i] = coord[i] * chunk_shape[i]
///   count_out[i] = min(chunk_shape[i], dataset_shape[i] - start_out[i])
/// ```
///
/// Boundary chunks (where the dataset extent is not evenly divisible by the
/// chunk extent) will have `count_out[i] < chunk_shape[i]` on the truncated
/// axes.
///
/// # Panics
///
/// Panics if the slice lengths are not all equal to `coord.len()`.
pub fn chunk_element_range(
    coord: &[usize],
    chunk_shape: &[usize],
    dataset_shape: &[usize],
    start_out: &mut [usize],
    count_out: &mut [usize],
) {
    let ndim = coord.len();
    assert_eq!(
        chunk_shape.len(),
        ndim,
        "chunk_shape length {} != coord length {}",
        chunk_shape.len(),
        ndim
    );
    assert_eq!(
        dataset_shape.len(),
        ndim,
        "dataset_shape length {} != coord length {}",
        dataset_shape.len(),
        ndim
    );
    assert_eq!(
        start_out.len(),
        ndim,
        "start_out length {} != coord length {}",
        start_out.len(),
        ndim
    );
    assert_eq!(
        count_out.len(),
        ndim,
        "count_out length {} != coord length {}",
        count_out.len(),
        ndim
    );

    for i in 0..ndim {
        start_out[i] = coord[i] * chunk_shape[i];
        count_out[i] = core::cmp::min(
            chunk_shape[i],
            dataset_shape[i].saturating_sub(start_out[i]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // chunk_grid_shape tests (alloc-gated)
    // -----------------------------------------------------------------------

    #[cfg(feature = "alloc")]
    mod grid_shape_tests {
        use super::*;

        #[test]
        fn exact_division() {
            let grid = chunk_grid_shape(&[100, 200], &[10, 20]).unwrap();
            assert_eq!(grid, alloc::vec![10, 10]);
        }

        #[test]
        fn ceiling_division() {
            // ⌈101/10⌉ = 11, ⌈200/20⌉ = 10
            let grid = chunk_grid_shape(&[101, 200], &[10, 20]).unwrap();
            assert_eq!(grid, alloc::vec![11, 10]);
        }

        #[test]
        fn empty_dimension() {
            // A dataset with zero extent along an axis has zero chunks along that axis.
            let grid = chunk_grid_shape(&[0, 100], &[10, 10]).unwrap();
            assert_eq!(grid, alloc::vec![0, 10]);
        }

        #[test]
        fn rank_mismatch_error() {
            let result = chunk_grid_shape(&[100, 200], &[10]);
            assert!(result.is_err());
            let err = result.unwrap_err();
            match &err {
                consus_core::Error::ShapeError {
                    #[cfg(feature = "alloc")]
                    message,
                } => {
                    #[cfg(feature = "alloc")]
                    assert!(
                        message.contains("rank mismatch"),
                        "expected rank mismatch message, got: {message}"
                    );
                }
                other => panic!("expected ShapeError, got: {other:?}"),
            }
        }

        #[test]
        fn zero_chunk_dim_error() {
            let result = chunk_grid_shape(&[100, 200], &[10, 0]);
            assert!(result.is_err());
            let err = result.unwrap_err();
            match &err {
                consus_core::Error::ShapeError {
                    #[cfg(feature = "alloc")]
                    message,
                } => {
                    #[cfg(feature = "alloc")]
                    assert!(
                        message.contains("zero"),
                        "expected zero-chunk message, got: {message}"
                    );
                }
                other => panic!("expected ShapeError, got: {other:?}"),
            }
        }

        #[test]
        fn one_dimensional() {
            // ⌈1000/256⌉ = ⌈3.90625⌉ = 4
            let grid = chunk_grid_shape(&[1000], &[256]).unwrap();
            assert_eq!(grid, alloc::vec![4]);
        }

        #[test]
        fn scalar_zero_dimensional() {
            // 0-D dataset: empty slices, empty grid.
            let grid = chunk_grid_shape(&[], &[]).unwrap();
            assert_eq!(grid, alloc::vec::Vec::<usize>::new());
        }
    }

    // -----------------------------------------------------------------------
    // chunk_grid_shape_fixed tests (no-alloc compatible)
    // -----------------------------------------------------------------------

    #[test]
    fn fixed_exact_division() {
        let mut out = [0usize; 2];
        let n = chunk_grid_shape_fixed(&[100, 200], &[10, 20], &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out, [10, 10]);
    }

    #[test]
    fn fixed_ceiling_division() {
        let mut out = [0usize; 2];
        let n = chunk_grid_shape_fixed(&[101, 200], &[10, 20], &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out, [11, 10]);
    }

    #[test]
    fn fixed_output_buffer_too_small() {
        let mut out = [0usize; 1];
        let result = chunk_grid_shape_fixed(&[100, 200], &[10, 20], &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn fixed_rank_mismatch() {
        let mut out = [0usize; 2];
        let result = chunk_grid_shape_fixed(&[100, 200], &[10], &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn fixed_zero_chunk_dim() {
        let mut out = [0usize; 2];
        let result = chunk_grid_shape_fixed(&[100, 200], &[0, 10], &mut out);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // total_chunks tests
    // -----------------------------------------------------------------------

    #[test]
    fn total_chunks_2d() {
        assert_eq!(total_chunks(&[10, 10]).unwrap(), 100);
    }

    #[test]
    fn total_chunks_scalar() {
        // Empty product = 1. A scalar dataset has exactly 1 chunk.
        assert_eq!(total_chunks(&[]).unwrap(), 1);
    }

    #[test]
    fn total_chunks_with_zero_extent() {
        // If any dimension is zero, total is zero (no chunks).
        assert_eq!(total_chunks(&[10, 0, 5]).unwrap(), 0);
    }

    #[test]
    fn total_chunks_overflow() {
        // usize::MAX * 2 must overflow.
        let result = total_chunks(&[usize::MAX, 2]);
        assert!(result.is_err());
        match result.unwrap_err() {
            consus_core::Error::Overflow => {}
            other => panic!("expected Overflow, got: {other:?}"),
        }
    }

    #[test]
    fn total_chunks_3d() {
        assert_eq!(total_chunks(&[4, 5, 6]).unwrap(), 120);
    }

    // -----------------------------------------------------------------------
    // chunk_coord_to_linear tests
    // -----------------------------------------------------------------------

    #[test]
    fn linear_origin() {
        assert_eq!(chunk_coord_to_linear(&[0, 0], &[10, 10]), 0);
    }

    #[test]
    fn linear_last_dim_varies_fastest() {
        // Incrementing the last dimension by 1 increments linear by 1.
        assert_eq!(chunk_coord_to_linear(&[0, 1], &[10, 10]), 1);
    }

    #[test]
    fn linear_first_dim_stride() {
        // Incrementing the first dimension by 1 increments linear by grid_shape[1].
        assert_eq!(chunk_coord_to_linear(&[1, 0], &[10, 10]), 10);
    }

    #[test]
    fn linear_last_element() {
        assert_eq!(chunk_coord_to_linear(&[9, 9], &[10, 10]), 99);
    }

    #[test]
    fn linear_3d() {
        // coord [1, 2, 3] in grid [4, 5, 6]:
        // linear = 1 * (5*6) + 2 * 6 + 3 = 30 + 12 + 3 = 45
        assert_eq!(chunk_coord_to_linear(&[1, 2, 3], &[4, 5, 6]), 45);
    }

    #[test]
    fn linear_1d() {
        assert_eq!(chunk_coord_to_linear(&[3], &[10]), 3);
    }

    #[test]
    fn linear_0d_scalar() {
        // 0-D: empty coord, empty grid → linear index 0.
        assert_eq!(chunk_coord_to_linear(&[], &[]), 0);
    }

    // -----------------------------------------------------------------------
    // linear_to_chunk_coord tests
    // -----------------------------------------------------------------------

    #[test]
    fn inverse_origin() {
        let mut coord = [0usize; 2];
        linear_to_chunk_coord(0, &[10, 10], &mut coord);
        assert_eq!(coord, [0, 0]);
    }

    #[test]
    fn inverse_last_element() {
        let mut coord = [0usize; 2];
        linear_to_chunk_coord(99, &[10, 10], &mut coord);
        assert_eq!(coord, [9, 9]);
    }

    #[test]
    fn inverse_3d() {
        let mut coord = [0usize; 3];
        linear_to_chunk_coord(45, &[4, 5, 6], &mut coord);
        assert_eq!(coord, [1, 2, 3]);
    }

    #[test]
    fn inverse_0d_scalar() {
        let mut coord = [0usize; 0];
        linear_to_chunk_coord(0, &[], &mut coord);
        // No elements to check; the call must not panic.
    }

    // -----------------------------------------------------------------------
    // Round-trip (Theorem 2) tests
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_all_2d() {
        let grid = [10, 10];
        let total = total_chunks(&grid).unwrap();
        assert_eq!(total, 100);

        let mut coord = [0usize; 2];
        for linear in 0..total {
            linear_to_chunk_coord(linear, &grid, &mut coord);
            let recovered = chunk_coord_to_linear(&coord, &grid);
            assert_eq!(
                recovered, linear,
                "round-trip failed: linear {linear} -> coord {coord:?} -> {recovered}"
            );
        }
    }

    #[test]
    fn round_trip_coord_to_linear_to_coord() {
        let grid = [4, 5, 6];
        let mut coord_out = [0usize; 3];
        for c0 in 0..4 {
            for c1 in 0..5 {
                for c2 in 0..6 {
                    let coord = [c0, c1, c2];
                    let linear = chunk_coord_to_linear(&coord, &grid);
                    linear_to_chunk_coord(linear, &grid, &mut coord_out);
                    assert_eq!(
                        coord_out, coord,
                        "round-trip failed: coord {coord:?} -> linear {linear} -> {coord_out:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn round_trip_1d() {
        let grid = [7];
        let mut coord = [0usize; 1];
        for linear in 0..7 {
            linear_to_chunk_coord(linear, &grid, &mut coord);
            let recovered = chunk_coord_to_linear(&coord, &grid);
            assert_eq!(recovered, linear);
        }
    }

    // -----------------------------------------------------------------------
    // chunk_element_range tests
    // -----------------------------------------------------------------------

    #[test]
    fn element_range_interior_chunk() {
        // chunk [1, 2] in dataset [100, 200] with chunk shape [10, 20]
        // start = [1*10, 2*20] = [10, 40]
        // count = [min(10, 100-10), min(20, 200-40)] = [10, 20]
        let mut starts = [0usize; 2];
        let mut counts = [0usize; 2];
        chunk_element_range(&[1, 2], &[10, 20], &[100, 200], &mut starts, &mut counts);
        assert_eq!(starts, [10, 40]);
        assert_eq!(counts, [10, 20]);
    }

    #[test]
    fn element_range_boundary_chunk() {
        // dataset [101, 200], chunk [10, 20], coord [10, 9]
        // start = [10*10, 9*20] = [100, 180]
        // count = [min(10, 101-100), min(20, 200-180)] = [1, 20]
        let mut starts = [0usize; 2];
        let mut counts = [0usize; 2];
        chunk_element_range(&[10, 9], &[10, 20], &[101, 200], &mut starts, &mut counts);
        assert_eq!(starts, [100, 180]);
        assert_eq!(counts, [1, 20]);
    }

    #[test]
    fn element_range_first_chunk() {
        let mut starts = [0usize; 2];
        let mut counts = [0usize; 2];
        chunk_element_range(&[0, 0], &[10, 20], &[100, 200], &mut starts, &mut counts);
        assert_eq!(starts, [0, 0]);
        assert_eq!(counts, [10, 20]);
    }

    #[test]
    fn element_range_1d() {
        // 1D: dataset [1000], chunk [256], coord [3]
        // start = [3 * 256] = [768]
        // count = [min(256, 1000 - 768)] = [232]
        let mut starts = [0usize; 1];
        let mut counts = [0usize; 1];
        chunk_element_range(&[3], &[256], &[1000], &mut starts, &mut counts);
        assert_eq!(starts, [768]);
        assert_eq!(counts, [232]);
    }

    #[test]
    fn element_range_3d_boundary() {
        // 3D: dataset [15, 25, 35], chunk [10, 10, 10], coord [1, 2, 3]
        // start = [10, 20, 30]
        // count = [min(10, 15-10), min(10, 25-20), min(10, 35-30)] = [5, 5, 5]
        let mut starts = [0usize; 3];
        let mut counts = [0usize; 3];
        chunk_element_range(
            &[1, 2, 3],
            &[10, 10, 10],
            &[15, 25, 35],
            &mut starts,
            &mut counts,
        );
        assert_eq!(starts, [10, 20, 30]);
        assert_eq!(counts, [5, 5, 5]);
    }

    #[test]
    fn element_range_chunk_beyond_dataset_saturates() {
        // If start_out[i] >= dataset_shape[i], count should be 0 via saturating_sub.
        // This can occur with a coordinate that places the chunk entirely outside
        // the dataset bounds (degenerate case, but must not panic).
        let mut starts = [0usize; 1];
        let mut counts = [0usize; 1];
        chunk_element_range(&[5], &[10], &[10], &mut starts, &mut counts);
        // start = 50, dataset = 10, count = min(10, 10.saturating_sub(50)) = min(10, 0) = 0
        assert_eq!(starts, [50]);
        assert_eq!(counts, [0]);
    }

    // -----------------------------------------------------------------------
    // Panic tests
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "coord length")]
    fn coord_to_linear_panics_on_length_mismatch() {
        chunk_coord_to_linear(&[0, 0, 0], &[10, 10]);
    }

    #[test]
    #[should_panic(expected = "coord_out length")]
    fn linear_to_coord_panics_on_length_mismatch() {
        let mut coord = [0usize; 3];
        linear_to_chunk_coord(0, &[10, 10], &mut coord);
    }

    #[test]
    #[should_panic(expected = "chunk_shape length")]
    fn element_range_panics_on_chunk_shape_mismatch() {
        let mut starts = [0usize; 2];
        let mut counts = [0usize; 2];
        chunk_element_range(&[0, 0], &[10], &[100, 200], &mut starts, &mut counts);
    }
}
