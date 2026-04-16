//! Combined trait aliases for full random-access I/O.
//!
//! ## Design
//!
//! `RandomAccess` is a convenience super-trait combining positioned
//! read, write, length query, and truncation. It is blanket-implemented
//! for any type satisfying all four constituent traits.
//!
//! ## Object Safety
//!
//! `RandomAccess` is object-safe and `dyn`-compatible.

use super::extent::{Length, Truncate};
use super::read::ReadAt;
use super::write::WriteAt;

/// Full random-access I/O: positioned read + write + length + truncate.
///
/// Blanket-implemented for all types satisfying the constituent bounds.
/// This trait adds no additional methods; it exists solely as a
/// convenience bound.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait RandomAccess: ReadAt + WriteAt + Length + Truncate {}

/// Blanket implementation: any type implementing all four constituent
/// traits automatically implements `RandomAccess`.
impl<T: ReadAt + WriteAt + Length + Truncate> RandomAccess for T {}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RandomAccess` is object-safe.
    #[test]
    fn random_access_is_object_safe() {
        fn _assert(_: &dyn RandomAccess) {}
    }
}
