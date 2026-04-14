//! Hyperslab and point selections for partial I/O.
//!
//! ## Specification
//!
//! A selection identifies a subset of elements in an N-dimensional dataspace.
//! Selections are used for partial reads and writes (hyperslabs, point lists,
//! and element masks).
//!
//! ### Hyperslab definition
//!
//! A hyperslab is defined by `(start, stride, count, block)` per dimension:
//! - `start[i]`: starting index along dimension `i`
//! - `stride[i]`: step between blocks (≥ 1)
//! - `count[i]`: number of blocks
//! - `block[i]`: size of each block (≥ 1)
//!
//! Selected indices along dimension `i`:
//!   `{ start[i] + n * stride[i] + b : n ∈ [0, count[i]), b ∈ [0, block[i]) }`
//!
//! Total elements = ∏ᵢ count[i] * block[i]

use smallvec::SmallVec;

/// A selection within an N-dimensional dataspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// Select all elements.
    All,

    /// No elements selected (empty selection).
    None,

    /// Hyperslab (regular strided subarray).
    Hyperslab(Hyperslab),

    /// List of explicit point coordinates.
    Points(PointSelection),
}

/// Regular strided subarray selection.
///
/// See module-level docs for the mathematical definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyperslab {
    /// Per-dimension selection parameters.
    pub dims: SmallVec<[HyperslabDim; 8]>,
}

/// Hyperslab parameters for a single dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperslabDim {
    /// Starting index.
    pub start: usize,
    /// Stride between blocks (≥ 1).
    pub stride: usize,
    /// Number of blocks.
    pub count: usize,
    /// Size of each block (≥ 1).
    pub block: usize,
}

impl HyperslabDim {
    /// Simple contiguous range: `start..start+count`.
    pub fn range(start: usize, count: usize) -> Self {
        Self {
            start,
            stride: 1,
            count,
            block: 1,
        }
    }

    /// Number of elements selected along this dimension.
    ///
    /// `count * block`
    pub fn num_elements(&self) -> usize {
        self.count * self.block
    }
}

impl Hyperslab {
    /// Create a hyperslab from per-dimension parameters.
    pub fn new(dims: &[HyperslabDim]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    /// Create a simple contiguous sub-array selection.
    ///
    /// Equivalent to `start[i]..start[i]+count[i]` per dimension.
    pub fn contiguous(start: &[usize], count: &[usize]) -> Self {
        let dims: SmallVec<[HyperslabDim; 8]> = start
            .iter()
            .zip(count.iter())
            .map(|(&s, &c)| HyperslabDim::range(s, c))
            .collect();
        Self { dims }
    }

    /// Total number of elements in this selection.
    ///
    /// `∏ᵢ count[i] * block[i]`
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|d| d.num_elements()).product()
    }

    /// Rank of the selection.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// Point-based selection: explicit list of coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointSelection {
    /// Rank of the dataspace.
    pub rank: usize,
    /// Flat array of coordinates: `points[i*rank..(i+1)*rank]` is point `i`.
    pub coords: SmallVec<[usize; 64]>,
}

impl PointSelection {
    /// Number of selected points.
    pub fn num_points(&self) -> usize {
        if self.rank == 0 {
            0
        } else {
            self.coords.len() / self.rank
        }
    }
}
