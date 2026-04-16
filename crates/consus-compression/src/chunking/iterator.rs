//! Chunk iterator over an N-dimensional chunk grid.
//!
//! Iterates over all chunk coordinates in row-major order.
//!
//! ## Iterator Invariant
//!
//! The iterator yields exactly `total_chunks(grid_shape)` coordinates,
//! each a valid index into the chunk grid, in row-major order.
//!
//! **Proof sketch**: The iterator maintains a mixed-radix counter
//! `current ∈ [0, G₀) × [0, G₁) × ... × [0, Gₙ₋₁)` that advances by 1
//! in the last dimension on each call to [`ChunkIterator::next`], carrying
//! into preceding dimensions when a digit overflows its radix `Gᵢ`. This
//! is the standard odometer algorithm over a mixed-radix numeral system,
//! which enumerates exactly `∏ Gᵢ` distinct tuples before the carry
//! propagates past dimension 0, at which point the iterator is exhausted.
//!
//! **0-dimensional special case**: When `grid_shape` is empty (scalar dataset),
//! the product `∏ Gᵢ` over an empty index set is 1 (the empty product).
//! The iterator yields exactly one item: the empty coordinate vector `[]`.

#[cfg(feature = "alloc")]
use super::index::total_chunks;

/// Iterator over all chunk coordinates in an N-dimensional grid.
///
/// Yields coordinates in row-major (C) order: the last dimension varies fastest.
///
/// ## Example (2D grid `[2, 3]`)
///
/// Yields: `[0,0]`, `[0,1]`, `[0,2]`, `[1,0]`, `[1,1]`, `[1,2]`
///
/// ## Construction
///
/// Use [`ChunkIterator::new`] with the chunk grid shape (not the dataset shape).
/// Obtain the grid shape via [`super::index::chunk_grid_shape`] or
/// [`super::index::chunk_grid_shape_fixed`].
///
/// ## Exact size
///
/// Implements [`ExactSizeIterator`]. The reported length equals
/// [`super::index::total_chunks`] for the grid shape, provided that value
/// does not overflow `usize`. If it does overflow, the remaining count
/// saturates at `usize::MAX` and [`ExactSizeIterator::len`] is a lower bound.
#[cfg(feature = "alloc")]
pub struct ChunkIterator {
    /// Grid extents per axis.
    grid_shape: alloc::vec::Vec<usize>,
    /// Current coordinate being prepared for the next yield.
    current: alloc::vec::Vec<usize>,
    /// Number of coordinates remaining (including the one at `current`).
    remaining: usize,
    /// `true` once all coordinates have been yielded.
    exhausted: bool,
}

#[cfg(feature = "alloc")]
impl ChunkIterator {
    /// Create an iterator over all chunk coordinates in the given grid.
    ///
    /// The iterator starts at the origin `[0, 0, ..., 0]` and advances in
    /// row-major order.
    ///
    /// If any `grid_shape[i] == 0`, the grid contains zero chunks and the
    /// iterator is immediately exhausted.
    ///
    /// For a 0-dimensional grid (`grid_shape` is empty), there is exactly
    /// one chunk (the scalar), so the iterator yields one empty coordinate.
    pub fn new(grid_shape: &[usize]) -> Self {
        let any_zero = grid_shape.contains(&0);
        let remaining = if any_zero {
            0
        } else {
            // total_chunks returns Err on overflow; saturate to usize::MAX
            // in that (practically unreachable) case.
            total_chunks(grid_shape).unwrap_or(usize::MAX)
        };
        let exhausted = remaining == 0;

        Self {
            grid_shape: alloc::vec::Vec::from(grid_shape),
            current: alloc::vec![0usize; grid_shape.len()],
            remaining,
            exhausted,
        }
    }
}

#[cfg(feature = "alloc")]
impl Iterator for ChunkIterator {
    type Item = alloc::vec::Vec<usize>;

    /// Yield the next chunk coordinate in row-major order.
    ///
    /// Returns `None` once all coordinates have been enumerated.
    ///
    /// ## Algorithm (odometer increment)
    ///
    /// 1. Clone `current` as the result to yield.
    /// 2. Increment `current` by advancing the last dimension.
    /// 3. If the last dimension overflows its grid extent, reset it to 0
    ///    and carry into the preceding dimension.
    /// 4. If the carry propagates past dimension 0, mark exhausted.
    ///
    /// For the 0-dimensional case the for-loop body does not execute,
    /// `carry` remains `true`, and the iterator is exhausted after yielding
    /// the single empty coordinate.
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let result = self.current.clone();

        // Decrement remaining count.
        self.remaining = self.remaining.saturating_sub(1);

        // Odometer increment: advance the mixed-radix counter.
        let ndim = self.grid_shape.len();
        let mut carry = true;
        for i in (0..ndim).rev() {
            self.current[i] += 1;
            if self.current[i] < self.grid_shape[i] {
                carry = false;
                break;
            }
            // Overflow in this dimension: reset and carry to next.
            self.current[i] = 0;
        }

        if carry {
            // Carry propagated past dimension 0 (or ndim == 0): all
            // coordinates have been enumerated.
            self.exhausted = true;
        }

        Some(result)
    }

    /// Returns `(remaining, Some(remaining))`.
    ///
    /// Exact when the total chunk count fits in `usize`.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(feature = "alloc")]
impl ExactSizeIterator for ChunkIterator {
    fn len(&self) -> usize {
        self.remaining
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // -----------------------------------------------------------------------
    // 2D grid
    // -----------------------------------------------------------------------

    #[test]
    fn iter_2d_yields_correct_sequence() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[2, 3]).collect();
        assert_eq!(
            coords,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2],
            ]
        );
    }

    #[test]
    fn iter_2d_count_matches_total_chunks() {
        let grid = [2, 3];
        let count = ChunkIterator::new(&grid).count();
        assert_eq!(count, total_chunks(&grid).unwrap());
        assert_eq!(count, 6);
    }

    // -----------------------------------------------------------------------
    // 1D grid
    // -----------------------------------------------------------------------

    #[test]
    fn iter_1d_yields_correct_sequence() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[4]).collect();
        assert_eq!(coords, vec![vec![0], vec![1], vec![2], vec![3],]);
    }

    // -----------------------------------------------------------------------
    // 0D (scalar) grid
    // -----------------------------------------------------------------------

    #[test]
    fn iter_0d_yields_single_empty_coord() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[]).collect();
        // A scalar dataset has exactly one chunk. Its coordinate is the
        // empty vector (0 dimensions).
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0], Vec::<usize>::new());
    }

    #[test]
    fn iter_0d_count_matches_total_chunks() {
        let count = ChunkIterator::new(&[]).count();
        assert_eq!(count, total_chunks(&[]).unwrap());
        assert_eq!(count, 1);
    }

    // -----------------------------------------------------------------------
    // Empty grid (zero extent in at least one dimension)
    // -----------------------------------------------------------------------

    #[test]
    fn iter_empty_grid_first_dim_zero() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[0, 3]).collect();
        assert!(coords.is_empty());
    }

    #[test]
    fn iter_empty_grid_second_dim_zero() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[3, 0]).collect();
        assert!(coords.is_empty());
    }

    #[test]
    fn iter_empty_grid_all_zero() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[0, 0, 0]).collect();
        assert!(coords.is_empty());
    }

    // -----------------------------------------------------------------------
    // 3D grid
    // -----------------------------------------------------------------------

    #[test]
    fn iter_3d_count_and_bounds() {
        let grid = [2, 2, 2];
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&grid).collect();

        // Total count.
        assert_eq!(coords.len(), 8);
        assert_eq!(coords.len(), total_chunks(&grid).unwrap());

        // First coordinate is the origin.
        assert_eq!(coords[0], vec![0, 0, 0]);

        // Last coordinate is the grid maximum.
        assert_eq!(coords[7], vec![1, 1, 1]);
    }

    #[test]
    fn iter_3d_full_sequence() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[2, 2, 2]).collect();
        assert_eq!(
            coords,
            vec![
                vec![0, 0, 0],
                vec![0, 0, 1],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![1, 0, 0],
                vec![1, 0, 1],
                vec![1, 1, 0],
                vec![1, 1, 1],
            ]
        );
    }

    // -----------------------------------------------------------------------
    // ExactSizeIterator contract
    // -----------------------------------------------------------------------

    #[test]
    fn exact_size_len_matches_actual_count() {
        let grid = [3, 4, 5];
        let iter = ChunkIterator::new(&grid);
        let expected = total_chunks(&grid).unwrap();
        assert_eq!(iter.len(), expected);
        assert_eq!(expected, 60);

        // Verify len() decreases correctly as items are consumed.
        let mut iter = ChunkIterator::new(&grid);
        for remaining in (0..expected).rev() {
            assert_eq!(iter.len(), remaining + 1);
            let item = iter.next();
            assert!(item.is_some());
        }
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn exact_size_empty_grid() {
        let iter = ChunkIterator::new(&[0, 5]);
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.count(), 0);
    }

    #[test]
    fn exact_size_scalar() {
        let iter = ChunkIterator::new(&[]);
        assert_eq!(iter.len(), 1);
    }

    // -----------------------------------------------------------------------
    // size_hint consistency
    // -----------------------------------------------------------------------

    #[test]
    fn size_hint_matches_len() {
        let grid = [2, 3];
        let mut iter = ChunkIterator::new(&grid);
        let total = total_chunks(&grid).unwrap();

        for consumed in 0..=total {
            let expected_remaining = total - consumed;
            let (lo, hi) = iter.size_hint();
            assert_eq!(lo, expected_remaining);
            assert_eq!(hi, Some(expected_remaining));
            assert_eq!(iter.len(), expected_remaining);

            if consumed < total {
                iter.next();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Row-major ordering verification via linear index
    // -----------------------------------------------------------------------

    #[test]
    fn iter_yields_monotonically_increasing_linear_indices() {
        use super::super::index::chunk_coord_to_linear;

        let grid = [3, 4, 5];
        let mut prev_linear: Option<usize> = None;
        for coord in ChunkIterator::new(&grid) {
            let linear = chunk_coord_to_linear(&coord, &grid);
            if let Some(prev) = prev_linear {
                assert_eq!(
                    linear,
                    prev + 1,
                    "non-consecutive linear indices: {prev} -> {linear}"
                );
            } else {
                assert_eq!(linear, 0, "first linear index must be 0");
            }
            prev_linear = Some(linear);
        }
        // Last index must be total - 1.
        assert_eq!(prev_linear, Some(total_chunks(&grid).unwrap() - 1));
    }

    // -----------------------------------------------------------------------
    // Single-element grid (all dimensions == 1)
    // -----------------------------------------------------------------------

    #[test]
    fn iter_single_element_grid() {
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&[1, 1, 1]).collect();
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0], vec![0, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // Asymmetric grid
    // -----------------------------------------------------------------------

    #[test]
    fn iter_asymmetric_grid() {
        let grid = [1, 3];
        let coords: Vec<Vec<usize>> = ChunkIterator::new(&grid).collect();
        assert_eq!(coords, vec![vec![0, 0], vec![0, 1], vec![0, 2],]);
    }

    // -----------------------------------------------------------------------
    // Double-exhaust safety
    // -----------------------------------------------------------------------

    #[test]
    fn iter_returns_none_after_exhaustion() {
        let mut iter = ChunkIterator::new(&[1, 1]);
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
        // Subsequent calls must also return None (fused behavior).
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }
}
