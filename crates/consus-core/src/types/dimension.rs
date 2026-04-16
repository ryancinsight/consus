//! N-dimensional array shape and chunking types.
//!
//! ## Specification
//!
//! A shape is an ordered sequence of dimension extents. Each extent is either
//! fixed (known at creation) or unlimited (growable). The rank is the number
//! of dimensions. Scalar values have rank 0.
//!
//! Chunk shapes define the tiling of a dataset into fixed-size blocks.
//!
//! ### Shape Invariants
//!
//! - `num_elements() = ∏_{i=0}^{rank-1} extents[i].current_size()`
//! - `num_elements() = 1` for rank-0 (scalar) shapes (empty product convention).
//!
//! ### Chunk Invariants
//!
//! For a chunk shape `C` of rank `R`:
//!   `∀ i ∈ [0, R): C[i] > 0`
//!
//! For a dataset shape `S` with chunk shape `C`, both of rank `R`:
//!   `num_chunks[i] = ⌈S[i] / C[i]⌉`

use smallvec::SmallVec;

/// Memory layout order for multi-dimensional arrays.
///
/// Determines the mapping from N-dimensional indices to linear memory offsets.
///
/// ## Definition
///
/// For a rank-R array with shape `(d_0, d_1, ..., d_{R-1})`:
/// - **RowMajor** (C order): `offset = ∑_{i=0}^{R-1} idx[i] × ∏_{j=i+1}^{R-1} d[j]`
///   (last index varies fastest in memory).
/// - **ColumnMajor** (Fortran order): `offset = ∑_{i=0}^{R-1} idx[i] × ∏_{j=0}^{i-1} d[j]`
///   (first index varies fastest in memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Layout {
    /// Row-major (C order): last index varies fastest.
    #[default]
    RowMajor,
    /// Column-major (Fortran order): first index varies fastest.
    ColumnMajor,
}

/// A dimension extent: either fixed or unlimited.
///
/// Fixed dimensions have a known, immutable size. Unlimited dimensions
/// can grow and have a current size that may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extent {
    /// Fixed dimension with immutable size.
    Fixed(usize),
    /// Unlimited (growable) dimension with current size.
    Unlimited {
        /// Current number of elements along this dimension.
        current: usize,
    },
}

impl Extent {
    /// Current number of elements along this dimension.
    pub fn current_size(&self) -> usize {
        match self {
            Self::Fixed(n) => *n,
            Self::Unlimited { current } => *current,
        }
    }

    /// Whether this dimension is unlimited (growable).
    pub fn is_unlimited(&self) -> bool {
        matches!(self, Self::Unlimited { .. })
    }

    /// Whether this dimension is fixed (immutable).
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }
}

/// Shape of an N-dimensional array.
///
/// Uses `SmallVec<[Extent; 8]>` to avoid heap allocation for arrays up to
/// rank 8, which covers the vast majority of scientific datasets.
///
/// ## Invariant
///
/// `num_elements()` is the product of all current dimension sizes.
/// For rank-0 (scalar), `num_elements() = 1` (empty product convention).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    extents: SmallVec<[Extent; 8]>,
}

impl Shape {
    /// Create a shape from fixed dimension sizes.
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

    /// Create a scalar (rank-0) shape.
    pub fn scalar() -> Self {
        Self {
            extents: SmallVec::new(),
        }
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    /// Whether this is a scalar (rank-0) shape.
    pub fn is_scalar(&self) -> bool {
        self.extents.is_empty()
    }

    /// Total number of elements: product of all current dimension sizes.
    ///
    /// Returns 1 for scalar (rank-0) shapes per the empty product convention.
    pub fn num_elements(&self) -> usize {
        if self.extents.is_empty() {
            1
        } else {
            self.extents.iter().map(|e| e.current_size()).product()
        }
    }

    /// Current dimension sizes.
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

/// Chunk shape for chunked array storage.
///
/// ## Invariant
///
/// All chunk dimensions are strictly positive: `∀ i: dims[i] > 0`.
///
/// ## Chunk Count Formula
///
/// For a dataset shape `S` with chunk shape `C`, both of rank `R`:
///   `num_chunks[i] = ⌈S[i] / C[i]⌉`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkShape {
    dims: SmallVec<[usize; 8]>,
}

impl ChunkShape {
    /// Create a chunk shape from dimension sizes.
    ///
    /// Returns `None` if any dimension is zero (invariant violation).
    pub fn new(dims: &[usize]) -> Option<Self> {
        if dims.contains(&0) {
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

    /// Rank of the chunk shape.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Number of chunks along each dimension for a given dataset shape.
    ///
    /// Uses ceiling division: `⌈shape[i] / chunk[i]⌉`.
    ///
    /// ## Precondition
    ///
    /// `shape.rank() == self.rank()`
    pub fn num_chunks(&self, shape: &Shape) -> SmallVec<[usize; 8]> {
        shape
            .current_dims()
            .iter()
            .zip(self.dims.iter())
            .map(|(&s, &c)| s.div_ceil(c))
            .collect()
    }

    /// Total number of chunks for a given dataset shape.
    ///
    /// `∏_{i=0}^{rank-1} ⌈shape[i] / chunk[i]⌉`
    pub fn total_chunks(&self, shape: &Shape) -> usize {
        self.num_chunks(shape).iter().product()
    }
}
