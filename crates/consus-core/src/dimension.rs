//! Dimension and shape types for N-dimensional arrays.
//!
//! ## Specification
//!
//! A shape is an ordered sequence of dimension extents. A dimension may be
//! fixed or unlimited (growable along that axis). Chunk shapes must divide
//! evenly into the dataset shape or tile the last partial chunk.
//!
//! ### Invariant
//!
//! For a shape `S` and chunk shape `C`, both of rank `R`:
//!   ∀ i ∈ [0, R): C[i] > 0 ∧ C[i] ≤ S[i] (when S[i] is finite)

use smallvec::SmallVec;

/// A dimension extent: either fixed or unlimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extent {
    /// Fixed dimension with known size.
    Fixed(usize),
    /// Unlimited (growable) dimension with current size.
    Unlimited { current: usize },
}

impl Extent {
    /// Current size of this dimension.
    pub fn current_size(&self) -> usize {
        match self {
            Extent::Fixed(n) => *n,
            Extent::Unlimited { current } => *current,
        }
    }

    /// Whether this dimension is unlimited.
    pub fn is_unlimited(&self) -> bool {
        matches!(self, Extent::Unlimited { .. })
    }
}

/// Shape of an N-dimensional array.
///
/// Uses `SmallVec` to avoid allocation for arrays up to rank 8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    extents: SmallVec<[Extent; 8]>,
}

impl Shape {
    /// Create a shape from fixed dimensions.
    pub fn fixed(dims: &[usize]) -> Self {
        Self {
            extents: dims.iter().map(|&d| Extent::Fixed(d)).collect(),
        }
    }

    /// Create a shape from explicit extents.
    pub fn new(extents: &[Extent]) -> Self {
        Self {
            extents: SmallVec::from_slice(extents),
        }
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    /// Total number of elements: product of all current dimension sizes.
    ///
    /// Returns 0 for scalar (rank-0) shapes by convention of empty product = 1,
    /// but we return 1 for rank-0 to correctly represent a scalar.
    pub fn num_elements(&self) -> usize {
        if self.extents.is_empty() {
            1 // scalar
        } else {
            self.extents.iter().map(|e| e.current_size()).product()
        }
    }

    /// Current dimension sizes as a slice-like view.
    pub fn current_dims(&self) -> SmallVec<[usize; 8]> {
        self.extents.iter().map(|e| e.current_size()).collect()
    }

    /// Access the underlying extents.
    pub fn extents(&self) -> &[Extent] {
        &self.extents
    }

    /// Whether any dimension is unlimited.
    pub fn has_unlimited(&self) -> bool {
        self.extents.iter().any(|e| e.is_unlimited())
    }
}

/// Chunk shape for chunked storage.
///
/// ## Invariant
///
/// All chunk dimensions are strictly positive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkShape {
    dims: SmallVec<[usize; 8]>,
}

impl ChunkShape {
    /// Create a chunk shape.
    ///
    /// # Errors
    ///
    /// Returns `None` if any dimension is zero.
    pub fn new(dims: &[usize]) -> Option<Self> {
        if dims.iter().any(|&d| d == 0) {
            return None;
        }
        Some(Self {
            dims: SmallVec::from_slice(dims),
        })
    }

    /// Chunk dimension sizes.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of chunks along each dimension given a dataset shape.
    ///
    /// Uses ceiling division: `⌈shape[i] / chunk[i]⌉`.
    pub fn num_chunks(&self, shape: &Shape) -> SmallVec<[usize; 8]> {
        shape
            .current_dims()
            .iter()
            .zip(self.dims.iter())
            .map(|(&s, &c)| (s + c - 1) / c)
            .collect()
    }
}

/// Memory layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C order): last index varies fastest.
    RowMajor,
    /// Column-major (Fortran order): first index varies fastest.
    ColumnMajor,
}

impl Default for Layout {
    fn default() -> Self {
        Layout::RowMajor
    }
}
