//! Selection-to-chunk mapping for partial I/O on chunked datasets.
//!
//! ## Specification
//!
//! Given a dataset shape, chunk shape, and selection (hyperslab or points),
//! this module determines which chunks are involved and what sub-region of
//! each chunk contributes to the selection.
//!
//! ### Decomposition Algorithm
//!
//! For a contiguous hyperslab with `start[d]` and `count[d]` per dimension `d`:
//!
//! 1. Compute the chunk grid coordinate range:
//!    - `first_chunk[d] = start[d] / chunk_dims[d]`
//!    - `last_chunk[d]  = (start[d] + count[d] - 1) / chunk_dims[d]`
//!
//! 2. Enumerate all chunk coordinates in the bounding box via an odometer
//!    (mixed-radix counter, last dimension varies fastest — row-major order).
//!
//! 3. For each chunk, compute the intersection of the chunk's element range
//!    with the selection range. A chunk covers elements
//!    `[chunk_coord[d] * chunk_dims[d], (chunk_coord[d]+1) * chunk_dims[d])`.
//!
//! ### Correctness Invariant
//!
//! The union of all `ChunkSlice` output regions exactly covers the selection
//! with no overlaps and no gaps:
//!
//!   `⋃ᵢ ChunkSlice[i].output_region == Selection`
//!
//! This follows from the fact that the chunk grid partitions the dataspace
//! into disjoint tiles, and each tile's intersection with a contiguous
//! hyperslab is itself a contiguous sub-rectangle (or empty).

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::{Error, Hyperslab, HyperslabDim, Result, Selection, Shape};

/// A sub-region of a single chunk needed for a selection.
///
/// Describes both the source region within the chunk and the
/// destination region within the caller's output buffer.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ChunkSlice {
    /// N-dimensional chunk coordinate in the chunk grid.
    pub chunk_coord: Vec<usize>,
    /// Start offset within the chunk (per dimension).
    ///
    /// `chunk_start[d] ∈ [0, chunk_dims[d])`.
    pub chunk_start: Vec<usize>,
    /// Count of elements within the chunk (per dimension).
    ///
    /// `chunk_count[d] ∈ [1, chunk_dims[d]]`.
    pub chunk_count: Vec<usize>,
    /// Start offset within the output buffer (per dimension).
    ///
    /// Measured relative to the selection's origin.
    pub output_start: Vec<usize>,
    /// Count of elements in the output (per dimension).
    ///
    /// Equal to `chunk_count` for contiguous hyperslabs.
    pub output_count: Vec<usize>,
}

#[cfg(feature = "alloc")]
impl ChunkSlice {
    /// Total number of elements in this slice.
    ///
    /// `∏ᵢ chunk_count[i]`.
    pub fn num_elements(&self) -> usize {
        if self.chunk_count.is_empty() {
            1 // scalar
        } else {
            self.chunk_count.iter().product()
        }
    }
}

/// Decompose a contiguous hyperslab into per-chunk slices.
///
/// For a dataset chunked with `chunk_dims`, computes which chunks overlap
/// with the given `hyperslab` and what portion of each chunk is selected.
///
/// ## Arguments
///
/// - `hyperslab`: the selection (supports arbitrary stride/block).
/// - `chunk_dims`: chunk extent along each dimension. Must have the same
///   rank as the hyperslab.
///
/// ## Returns
///
/// A vector of [`ChunkSlice`] in row-major chunk-coordinate order.
/// Empty if the hyperslab selects zero elements.
///
/// ## Errors
///
/// - [`Error::ShapeError`] if `hyperslab.rank() != chunk_dims.len()`.
#[cfg(feature = "alloc")]
pub fn decompose_hyperslab(hyperslab: &Hyperslab, chunk_dims: &[usize]) -> Result<Vec<ChunkSlice>> {
    let rank = hyperslab.rank();
    if rank != chunk_dims.len() {
        return Err(Error::ShapeError {
            message: alloc::format!("hyperslab rank {} != chunk rank {}", rank, chunk_dims.len()),
        });
    }

    // Scalar: single chunk at origin.
    if rank == 0 {
        return Ok(vec![ChunkSlice {
            chunk_coord: vec![],
            chunk_start: vec![],
            chunk_count: vec![],
            output_start: vec![],
            output_count: vec![],
        }]);
    }

    // Check for empty selection.
    for d in 0..rank {
        if hyperslab.dims[d].count == 0 {
            return Ok(vec![]);
        }
    }

    // For simple contiguous hyperslabs (stride=1, block=1) we use a
    // fast path. For strided/blocked hyperslabs we expand to the set
    // of contiguous sub-blocks and decompose each.
    let is_simple = hyperslab.dims.iter().all(|d| d.stride == 1 && d.block == 1);

    if is_simple {
        decompose_simple_hyperslab(hyperslab, chunk_dims)
    } else {
        decompose_strided_hyperslab(hyperslab, chunk_dims)
    }
}

/// Fast path: decompose a simple contiguous hyperslab (stride=1, block=1).
#[cfg(feature = "alloc")]
fn decompose_simple_hyperslab(
    hyperslab: &Hyperslab,
    chunk_dims: &[usize],
) -> Result<Vec<ChunkSlice>> {
    let rank = hyperslab.rank();

    // Compute chunk ranges per dimension.
    let mut chunk_ranges: Vec<(usize, usize)> = Vec::with_capacity(rank);
    for d in 0..rank {
        let dim = &hyperslab.dims[d];
        let sel_start = dim.start;
        let sel_last = sel_start + dim.count - 1;
        let first_chunk = sel_start / chunk_dims[d];
        let last_chunk = sel_last / chunk_dims[d];
        chunk_ranges.push((first_chunk, last_chunk));
    }

    // Total number of chunks in the bounding box.
    let total_chunks: usize = chunk_ranges.iter().map(|(f, l)| l - f + 1).product();

    let mut result = Vec::with_capacity(total_chunks);

    // Odometer iteration over chunk coordinates (row-major order).
    let mut chunk_coord: Vec<usize> = chunk_ranges.iter().map(|(f, _)| *f).collect();
    let mut c_start = vec![0usize; rank];
    let mut c_count = vec![0usize; rank];
    let mut o_start = vec![0usize; rank];

    for _ in 0..total_chunks {
        let mut valid = true;

        for d in 0..rank {
            let dim = &hyperslab.dims[d];
            let chunk_begin = chunk_coord[d] * chunk_dims[d];
            let chunk_end = chunk_begin + chunk_dims[d];

            let sel_start = dim.start;
            let sel_end = sel_start + dim.count; // exclusive

            let inter_start = sel_start.max(chunk_begin);
            let inter_end = sel_end.min(chunk_end);

            if inter_start >= inter_end {
                valid = false;
                break;
            }

            c_start[d] = inter_start - chunk_begin;
            c_count[d] = inter_end - inter_start;
            o_start[d] = inter_start - sel_start;
        }

        if valid {
            result.push(ChunkSlice {
                chunk_coord: chunk_coord.clone(),
                chunk_start: c_start.clone(),
                chunk_count: c_count.clone(),
                output_start: o_start.clone(),
                output_count: c_count.clone(),
            });
        }

        // Advance odometer (last dimension varies fastest).
        advance_odometer(&mut chunk_coord, &chunk_ranges);
    }

    Ok(result)
}

/// Strided/blocked hyperslab decomposition.
///
/// Expands the strided selection into its constituent contiguous blocks,
/// then decomposes each block individually and adjusts output offsets.
#[cfg(feature = "alloc")]
fn decompose_strided_hyperslab(
    hyperslab: &Hyperslab,
    chunk_dims: &[usize],
) -> Result<Vec<ChunkSlice>> {
    let rank = hyperslab.rank();

    // Total number of blocks = ∏ count[d].
    let total_blocks: usize = hyperslab.dims.iter().map(|d| d.count).product();

    let mut result = Vec::new();
    let ranges: Vec<(usize, usize)> = hyperslab
        .dims
        .iter()
        .map(|d| (0, d.count.saturating_sub(1)))
        .collect();

    // Enumerate all blocks via an odometer over `count` per dimension.
    let mut block_idx = vec![0usize; rank];

    for _ in 0..total_blocks {
        // Build a simple contiguous hyperslab for this block.
        let mut block_dims = Vec::with_capacity(rank);
        let mut block_output_origin = vec![0usize; rank];

        for d in 0..rank {
            let dim = &hyperslab.dims[d];
            let block_start = dim.start + block_idx[d] * dim.stride;
            let block_count = dim.block;
            block_dims.push(HyperslabDim::range(block_start, block_count));

            // Output offset for this block within the full selection.
            // Each block occupies `block` elements, and block_idx[d] selects
            // which block we are in.
            block_output_origin[d] = block_idx[d] * dim.block;
        }

        let block_slab = Hyperslab::new(&block_dims);
        let sub_slices = decompose_simple_hyperslab(&block_slab, chunk_dims)?;
        result.reserve(sub_slices.len());

        // Adjust output offsets to account for the block's position.
        for mut s in sub_slices {
            for d in 0..rank {
                s.output_start[d] += block_output_origin[d];
            }
            result.push(s);
        }

        // Advance block odometer.
        advance_odometer(&mut block_idx, &ranges);
    }

    Ok(result)
}

/// Advance an odometer (mixed-radix counter) in row-major order
/// (last dimension varies fastest).
///
/// Each `coord[d]` cycles through `ranges[d].0 ..= ranges[d].1`.
#[cfg(feature = "alloc")]
fn advance_odometer(coord: &mut [usize], ranges: &[(usize, usize)]) {
    for d in (0..coord.len()).rev() {
        coord[d] += 1;
        if coord[d] > ranges[d].1 {
            coord[d] = ranges[d].0;
        } else {
            return;
        }
    }
}

/// Decompose a point selection into per-chunk slices.
///
/// Each selected point produces exactly one [`ChunkSlice`] with a
/// one-element region (`chunk_count` = all-ones). The output buffer
/// is treated as a 1-D sequence of values: `output_start = [point_index]`,
/// `output_count = [1]`.
///
/// ## Complexity
///
/// O(n_points) time and space. Correct for arbitrary point distributions
/// across the chunk grid.
#[cfg(feature = "alloc")]
fn decompose_points(
    pts: &consus_core::PointSelection,
    chunk_dims: &[usize],
) -> Result<Vec<ChunkSlice>> {
    use consus_core::Error;

    let rank = chunk_dims.len();
    if rank == 0 {
        return Ok(vec![]);
    }
    let n = pts.num_points();
    let mut slices = Vec::with_capacity(n);

    for point_idx in 0..n {
        let point = &pts.coords[point_idx * pts.rank..(point_idx + 1) * pts.rank];

        if point.len() != rank {
            return Err(Error::ShapeError {
                #[cfg(feature = "alloc")]
                message: alloc::format!("point rank {} != chunk dims rank {}", point.len(), rank),
            });
        }

        let mut chunk_coord = Vec::with_capacity(rank);
        let mut chunk_start = Vec::with_capacity(rank);
        for (&p, &c) in point.iter().zip(chunk_dims.iter()) {
            chunk_coord.push(p / c);
            chunk_start.push(p % c);
        }

        let chunk_count: Vec<usize> = vec![1; rank];
        // Output is a 1-D sequence: point_idx-th slot.
        let output_start: Vec<usize> = vec![point_idx];
        let output_count: Vec<usize> = vec![1];

        slices.push(ChunkSlice {
            chunk_coord,
            chunk_start,
            chunk_count,
            output_start,
            output_count,
        });
    }

    Ok(slices)
}

/// Map a [`Selection`] to chunk slices for a chunked dataset.
///
/// Dispatches on the selection variant (`All`, `Hyperslab`, `Points`, `None`).
///
/// ## Arguments
///
/// - `selection`: the dataspace selection.
/// - `dataset_shape`: current shape of the dataset.
/// - `chunk_dims`: chunk extent along each dimension.
///
/// ## Returns
///
/// Vector of [`ChunkSlice`] describing each chunk's contribution.
///
/// ## Errors
///
/// - [`Error::ShapeError`] on rank mismatches.
#[cfg(feature = "alloc")]
pub fn map_selection_to_chunks(
    selection: &Selection,
    dataset_shape: &Shape,
    chunk_dims: &[usize],
) -> Result<Vec<ChunkSlice>> {
    match selection {
        Selection::All => {
            // Convert to a full hyperslab covering the entire dataspace.
            let dims: Vec<HyperslabDim> = dataset_shape
                .current_dims()
                .iter()
                .map(|&d| HyperslabDim::range(0, d))
                .collect();
            let hyperslab = Hyperslab::new(&dims);
            decompose_hyperslab(&hyperslab, chunk_dims)
        }
        Selection::None => Ok(vec![]),
        Selection::Hyperslab(h) => decompose_hyperslab(h, chunk_dims),
        Selection::Points(pts) => decompose_points(pts, chunk_dims),
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;

    /// Single chunk fully covered by the selection.
    #[test]
    fn single_chunk_full_coverage() {
        let slab = Hyperslab::contiguous(&[0, 0], &[4, 6]);
        let chunks = decompose_hyperslab(&slab, &[4, 6]).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_coord, [0, 0]);
        assert_eq!(chunks[0].chunk_start, [0, 0]);
        assert_eq!(chunks[0].chunk_count, [4, 6]);
        assert_eq!(chunks[0].output_start, [0, 0]);
        assert_eq!(chunks[0].num_elements(), 24);
    }

    /// Selection spanning 2×2 chunks.
    #[test]
    fn four_chunks_partial() {
        // Dataset shape implied: at least 8×8
        // Chunk dims: 4×4
        // Selection: rows 2..6, cols 2..6 → 4×4 block spanning 4 chunks
        let slab = Hyperslab::contiguous(&[2, 2], &[4, 4]);
        let chunks = decompose_hyperslab(&slab, &[4, 4]).unwrap();
        assert_eq!(chunks.len(), 4);

        // Total elements across all slices must equal selection size.
        let total: usize = chunks.iter().map(|c| c.num_elements()).sum();
        assert_eq!(total, 16);

        // Verify chunk (0,0) gets rows 2..4, cols 2..4 → 2×2
        let c00 = chunks.iter().find(|c| c.chunk_coord == [0, 0]).unwrap();
        assert_eq!(c00.chunk_start, [2, 2]);
        assert_eq!(c00.chunk_count, [2, 2]);
        assert_eq!(c00.output_start, [0, 0]);

        // Verify chunk (1,1) gets rows 0..2, cols 0..2 (within chunk) → 2×2
        let c11 = chunks.iter().find(|c| c.chunk_coord == [1, 1]).unwrap();
        assert_eq!(c11.chunk_start, [0, 0]);
        assert_eq!(c11.chunk_count, [2, 2]);
        assert_eq!(c11.output_start, [2, 2]);
    }

    /// Selection that falls entirely within one chunk but not at origin.
    #[test]
    fn interior_selection_single_chunk() {
        let slab = Hyperslab::contiguous(&[5, 5], &[2, 2]);
        let chunks = decompose_hyperslab(&slab, &[10, 10]).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_coord, [0, 0]);
        assert_eq!(chunks[0].chunk_start, [5, 5]);
        assert_eq!(chunks[0].chunk_count, [2, 2]);
        assert_eq!(chunks[0].output_start, [0, 0]);
    }

    /// Empty selection produces zero slices.
    #[test]
    fn empty_selection() {
        let slab = Hyperslab::new(&[HyperslabDim {
            start: 0,
            stride: 1,
            count: 0,
            block: 1,
        }]);
        let chunks = decompose_hyperslab(&slab, &[4]).unwrap();
        assert!(chunks.is_empty());
    }

    /// Scalar (rank-0) selection.
    #[test]
    fn scalar_selection() {
        let slab = Hyperslab::new(&[]);
        let chunks = decompose_hyperslab(&slab, &[]).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].num_elements(), 1);
    }

    /// Rank mismatch returns error.
    #[test]
    fn rank_mismatch_error() {
        let slab = Hyperslab::contiguous(&[0, 0], &[4, 4]);
        let result = decompose_hyperslab(&slab, &[4, 4, 4]);
        assert!(result.is_err());
    }

    /// 1-D selection spanning three chunks.
    #[test]
    fn one_d_three_chunks() {
        let slab = Hyperslab::contiguous(&[3], &[10]);
        let chunks = decompose_hyperslab(&slab, &[5]).unwrap();
        // Chunks: 0 (3..5), 1 (0..5), 2 (0..3) → 3 chunks
        assert_eq!(chunks.len(), 3);

        let total: usize = chunks.iter().map(|c| c.num_elements()).sum();
        assert_eq!(total, 10);

        let c0 = chunks.iter().find(|c| c.chunk_coord == [0]).unwrap();
        assert_eq!(c0.chunk_start, [3]);
        assert_eq!(c0.chunk_count, [2]);

        let c1 = chunks.iter().find(|c| c.chunk_coord == [1]).unwrap();
        assert_eq!(c1.chunk_start, [0]);
        assert_eq!(c1.chunk_count, [5]);

        let c2 = chunks.iter().find(|c| c.chunk_coord == [2]).unwrap();
        assert_eq!(c2.chunk_start, [0]);
        assert_eq!(c2.chunk_count, [3]);
    }

    /// `map_selection_to_chunks` with `Selection::All`.
    #[test]
    fn map_selection_all() {
        let shape = Shape::fixed(&[8, 12]);
        let chunks = map_selection_to_chunks(&Selection::All, &shape, &[4, 4]).unwrap();
        // 2×3 = 6 chunks
        assert_eq!(chunks.len(), 6);
        let total: usize = chunks.iter().map(|c| c.num_elements()).sum();
        assert_eq!(total, 96);
    }

    /// `map_selection_to_chunks` with `Selection::None`.
    #[test]
    fn map_selection_none() {
        let shape = Shape::fixed(&[8, 8]);
        let chunks = map_selection_to_chunks(&Selection::None, &shape, &[4, 4]).unwrap();
        assert!(chunks.is_empty());
    }

    /// Strided hyperslab: every other row.
    #[test]
    fn strided_hyperslab() {
        // Select rows 0, 2, 4, 6 from an 8-row dataset, all 4 columns.
        let dims = [
            HyperslabDim {
                start: 0,
                stride: 2,
                count: 4,
                block: 1,
            },
            HyperslabDim::range(0, 4),
        ];
        let slab = Hyperslab::new(&dims);
        let chunks = decompose_hyperslab(&slab, &[4, 4]).unwrap();

        // 4 blocks, each 1×4. With chunk size 4×4, block at row 0 → chunk (0,0),
        // row 2 → chunk (0,0), row 4 → chunk (1,0), row 6 → chunk (1,0).
        // So we expect some slices.
        let total: usize = chunks.iter().map(|c| c.num_elements()).sum();
        assert_eq!(total, 16); // 4 rows × 4 cols
    }

    /// Blocked hyperslab: 2-element blocks with stride 4.
    #[test]
    fn blocked_hyperslab() {
        // 1-D: start=0, stride=4, count=2, block=2 → selects [0,1,4,5]
        let dims = [HyperslabDim {
            start: 0,
            stride: 4,
            count: 2,
            block: 2,
        }];
        let slab = Hyperslab::new(&dims);
        let chunks = decompose_hyperslab(&slab, &[3]).unwrap();

        // Elements selected: 0,1 (block 0) and 4,5 (block 1)
        let total: usize = chunks.iter().map(|c| c.num_elements()).sum();
        assert_eq!(total, 4);
    }

    /// 3-D selection for a realistic scientific dataset scenario.
    #[test]
    fn three_dimensional_selection() {
        // 10×10×10 dataset, 5×5×5 chunks
        // Select a 3×3×3 cube starting at (3,3,3)
        let slab = Hyperslab::contiguous(&[3, 3, 3], &[3, 3, 3]);
        let chunks = decompose_hyperslab(&slab, &[5, 5, 5]).unwrap();

        // The selection (3..6, 3..6, 3..6) spans chunks:
        // dim 0: chunk 0 (3..5), chunk 1 (0..1) → 2
        // dim 1: chunk 0 (3..5), chunk 1 (0..1) → 2
        // dim 2: chunk 0 (3..5), chunk 1 (0..1) → 2
        // Total: 2×2×2 = 8 chunks
        assert_eq!(chunks.len(), 8);

        let total: usize = chunks.iter().map(|c| c.num_elements()).sum();
        assert_eq!(total, 27); // 3×3×3
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn point_selection_each_in_own_chunk() {
        use consus_core::{PointSelection, Selection};
        use smallvec::SmallVec;

        // 2×2 chunks of [2,2]. Points (0,0), (2,2) → in different chunks.
        let pts = PointSelection {
            rank: 2,
            coords: SmallVec::from_vec(vec![0, 0, 2, 2]),
        };
        let sel = Selection::Points(pts);
        let chunks = map_selection_to_chunks(&sel, &Shape::fixed(&[4, 4]), &[2, 2]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_coord, vec![0, 0]);
        assert_eq!(chunks[0].chunk_start, vec![0, 0]);
        assert_eq!(chunks[0].chunk_count, vec![1, 1]);
        assert_eq!(chunks[0].output_start, vec![0]);
        assert_eq!(chunks[1].chunk_coord, vec![1, 1]);
        assert_eq!(chunks[1].chunk_start, vec![0, 0]);
        assert_eq!(chunks[1].output_start, vec![1]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn point_selection_same_chunk() {
        use consus_core::{PointSelection, Selection};
        use smallvec::SmallVec;

        // Both points in chunk (0,0) of 4×4 chunks.
        let pts = PointSelection {
            rank: 2,
            coords: SmallVec::from_vec(vec![0, 1, 1, 3]),
        };
        let sel = Selection::Points(pts);
        let chunks = map_selection_to_chunks(&sel, &Shape::fixed(&[8, 8]), &[4, 4]).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_coord, vec![0, 0]);
        assert_eq!(chunks[0].chunk_start, vec![0, 1]);
        assert_eq!(chunks[1].chunk_coord, vec![0, 0]);
        assert_eq!(chunks[1].chunk_start, vec![1, 3]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn point_selection_empty() {
        use consus_core::{PointSelection, Selection};
        use smallvec::SmallVec;

        let pts = PointSelection {
            rank: 2,
            coords: SmallVec::new(),
        };
        let sel = Selection::Points(pts);
        let chunks = map_selection_to_chunks(&sel, &Shape::fixed(&[4, 4]), &[2, 2]).unwrap();
        assert_eq!(chunks.len(), 0);
    }
}
