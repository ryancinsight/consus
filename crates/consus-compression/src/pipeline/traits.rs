//! Filter trait and direction enum.
//!
//! ## Contract
//!
//! For any reversible filter `F` and input data `D`:
//!
//! ```text
//! F.apply(Forward, F.apply(Reverse, D)?) == D
//! F.apply(Reverse, F.apply(Forward, D)?) == D
//! ```
//!
//! This bijectivity requirement ensures that every filter in a pipeline
//! can be composed and inverted without information loss.

use alloc::vec::Vec;

use consus_core::Result;

/// Direction of filter application.
///
/// Determines whether a [`Filter`] performs its forward transformation
/// (used during writes) or its reverse transformation (used during reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterDirection {
    /// Apply the filter (e.g., shuffle, pack, compress).
    Forward,
    /// Reverse the filter (e.g., unshuffle, unpack, decompress).
    Reverse,
}

/// A data transformation filter.
///
/// Filters are the building blocks of a [`super::FilterPipeline`]. Each filter
/// transforms a byte slice in one direction (forward for writing, reverse for
/// reading) and returns the transformed bytes as a new allocation.
///
/// ## Invariant
///
/// For all valid inputs `data`:
///
/// ```text
/// apply(Reverse, apply(Forward, data)?) == data
/// ```
///
/// Implementations that violate this invariant corrupt data on round-trip.
///
/// ## Thread Safety
///
/// Filters are `Send + Sync` so that a single pipeline instance can be
/// shared across threads (e.g., compressing multiple chunks in parallel).
pub trait Filter: Send + Sync {
    /// Human-readable filter name (e.g., `"shuffle"`, `"nbit"`).
    fn name(&self) -> &str;

    /// Apply this filter in the given direction.
    ///
    /// # Errors
    ///
    /// Returns [`consus_core::Error::CompressionError`] if the transformation
    /// fails (e.g., input length is not aligned to the element size).
    fn apply(&self, direction: FilterDirection, data: &[u8]) -> Result<Vec<u8>>;
}
