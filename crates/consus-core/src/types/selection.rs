//! Hyperslab and point selections for partial I/O.
//!
//! ## Specification
//!
//! A selection identifies a subset of elements in an N-dimensional dataspace.
//! Selections are used for partial reads and writes.
//!
//! ### Hyperslab Definition
//!
//! A hyperslab is defined by `(start, stride, count, block)` per dimension:
//! - `start[i]`: starting index along dimension `i`
//! - `stride[i]`: step between blocks (≥ 1)
//! - `count[i]`: number of blocks
//! - `block[i]`: size of each block (≥ 1)
//!
//! Selected indices along dimension `i`:
//!   `{ start[i] + n × stride[i] + b : n ∈ [0, count[i]), b ∈ [0, block[i]) }`
//!
//! Total selected elements = `∏_i count[i] × block[i]`

use smallvec::SmallVec;

use super::dimension::Shape;

/// Hyperslab parameters for a single dimension.
///
/// ## Invariants
///
/// - `stride >= 1`
/// - `block >= 1`
/// - `count >= 0` (0 selects nothing along this dimension)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    ///
    /// Equivalent to `stride = 1, block = 1`.
    pub fn range(start: usize, count: usize) -> Self {
        Self {
            start,
            stride: 1,
            count,
            block: 1,
        }
    }

    /// Number of elements selected along this dimension: `count × block`.
    pub fn num_elements(&self) -> usize {
        self.count * self.block
    }

    /// The maximum index touched by this selection along this dimension.
    ///
    /// Returns `None` if `count == 0`.
    ///
    /// `max_index = start + (count - 1) × stride + block - 1`
    pub fn max_index(&self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        Some(self.start + (self.count - 1) * self.stride + self.block - 1)
    }

    /// Whether this selection is valid for a dimension of the given size.
    ///
    /// Valid iff `max_index() < extent` or `count == 0`.
    pub fn is_valid_for_extent(&self, extent: usize) -> bool {
        match self.max_index() {
            None => true,
            Some(max) => max < extent,
        }
    }
}

/// Regular strided subarray selection (hyperslab).
///
/// See module-level documentation for the mathematical definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyperslab {
    /// Per-dimension selection parameters.
    pub dims: SmallVec<[HyperslabDim; 8]>,
}

impl Hyperslab {
    /// Create a hyperslab from per-dimension parameters.
    pub fn new(dims: &[HyperslabDim]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    /// Create a contiguous sub-array selection.
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
    /// `∏_i count[i] × block[i]`
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            return 0;
        }
        self.dims.iter().map(|d| d.num_elements()).product()
    }

    /// Rank of the selection.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Whether this hyperslab is valid for the given shape.
    ///
    /// Valid iff `rank == shape.rank()` and each dimension's selection
    /// fits within the corresponding extent.
    pub fn is_valid_for_shape(&self, shape: &Shape) -> bool {
        if self.rank() != shape.rank() {
            return false;
        }
        self.dims
            .iter()
            .zip(shape.extents().iter())
            .all(|(d, e)| d.is_valid_for_extent(e.current_size()))
    }
}

/// Point-based coordinate selection.
///
/// An explicit list of N-dimensional point coordinates for irregular
/// subsets of a dataspace.
///
/// ## Representation
///
/// Points are stored in a flat array: `coords[i*rank..(i+1)*rank]` is point `i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointSelection {
    /// Rank of the dataspace.
    pub rank: usize,
    /// Flat coordinate array. Length must be a multiple of `rank`.
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

    /// Whether this point selection is valid for the given shape.
    ///
    /// Valid iff `rank == shape.rank()` and all coordinates are within bounds.
    pub fn is_valid_for_shape(&self, shape: &Shape) -> bool {
        if self.rank != shape.rank() {
            return false;
        }
        if self.rank == 0 {
            return true;
        }
        let dims = shape.current_dims();
        self.coords
            .chunks_exact(self.rank)
            .all(|pt| pt.iter().zip(dims.iter()).all(|(&c, &d)| c < d))
    }
}

/// A selection within an N-dimensional dataspace.
///
/// Selections specify which elements to read or write during partial I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum Selection {
    /// Select all elements in the dataspace.
    All,

    /// Empty selection (no elements).
    None,

    /// Regular strided subarray.
    Hyperslab(Hyperslab),

    /// Explicit point coordinates.
    Points(PointSelection),
}

impl Selection {
    /// Total number of selected elements for a given dataspace shape.
    ///
    /// - `All` → `shape.num_elements()`
    /// - `None` → `0`
    /// - `Hyperslab(h)` → `h.num_elements()`
    /// - `Points(p)` → `p.num_points()`
    pub fn num_elements(&self, shape: &Shape) -> usize {
        match self {
            Self::All => shape.num_elements(),
            Self::None => 0,
            Self::Hyperslab(h) => h.num_elements(),
            Self::Points(p) => p.num_points(),
        }
    }

    /// Whether this selection is valid for the given shape.
    pub fn is_valid_for_shape(&self, shape: &Shape) -> bool {
        match self {
            Self::All | Self::None => true,
            Self::Hyperslab(h) => h.is_valid_for_shape(shape),
            Self::Points(p) => p.is_valid_for_shape(shape),
        }
    }
}
