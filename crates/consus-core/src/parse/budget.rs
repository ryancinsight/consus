//! Resource ceilings applied while decoding untrusted input.

#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::core::error::{Error, Result};

/// Resource ceilings applied while decoding an untrusted document.
///
/// A format header is attacker-chosen data, not a promise. A declared record
/// count, node size, or nesting depth therefore selects an allocation or a
/// stack depth only after it clears these ceilings. The three failure modes
/// this closes are all uncatchable in Rust — a capacity-overflow panic, an
/// allocator abort on a failed `alloc_zeroed`, and a stack overflow — so a
/// budget check is the only place the input can still be rejected.
///
/// The ceilings are deliberately generous: they exist to separate "plausible
/// file" from "hostile integer", not to constrain honest scientific data.
/// Bounding against the ceiling alone is the weaker half of the contract;
/// callers that can also bound against the bytes actually available should do
/// so (see `consus_io::read_at_bounded`, which grows only as reads confirm
/// the input).
///
/// # Examples
///
/// ```
/// use consus_core::ParseBudget;
///
/// let budget = ParseBudget::default();
/// // A plausible node size passes through unchanged.
/// assert_eq!(budget.checked_bytes(4096, "node").unwrap(), 4096);
/// // A hostile one is a typed error, not an abort.
/// assert!(budget.checked_bytes(u64::MAX, "node").is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseBudget {
    /// Largest single buffer, in bytes, that one declared length may request.
    pub max_alloc_bytes: usize,
    /// Largest element count that one declared count may request.
    pub max_elements: usize,
    /// Deepest nesting of self-referential structures (compound datatypes,
    /// B-tree levels) that recursive descent may enter.
    pub max_depth: u16,
}

/// Largest buffer one declared length may request: 64 MiB.
///
/// Matches `consus_hdf5::object_header::v2::MAX_CHUNK_BYTES`, the ceiling
/// already enforced on object-header continuation blocks.
const DEFAULT_MAX_ALLOC_BYTES: usize = 64 * 1024 * 1024;

/// Largest element count one declared count may request.
///
/// Sized so that the smallest useful element (one byte) still cannot exceed
/// [`DEFAULT_MAX_ALLOC_BYTES`] by more than a small factor.
const DEFAULT_MAX_ELEMENTS: usize = 16 * 1024 * 1024;

/// Deepest recursive descent permitted.
///
/// HDF5 compound datatypes nest a handful of levels in practice; 64 leaves
/// two orders of magnitude of headroom over real files while keeping the
/// worst-case stack consumption bounded by a constant.
const DEFAULT_MAX_DEPTH: u16 = 64;

impl Default for ParseBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ParseBudget {
    /// The default ceilings, usable in `const` context.
    pub const DEFAULT: Self = Self::new(
        DEFAULT_MAX_ALLOC_BYTES,
        DEFAULT_MAX_ELEMENTS,
        DEFAULT_MAX_DEPTH,
    );

    /// Construct a budget with explicit ceilings.
    #[must_use]
    pub const fn new(max_alloc_bytes: usize, max_elements: usize, max_depth: u16) -> Self {
        Self {
            max_alloc_bytes,
            max_elements,
            max_depth,
        }
    }

    /// Accept a declared byte length, or reject it as a resource-limit error.
    ///
    /// `what` names the field the length came from so the error identifies the
    /// violated invariant rather than reporting an anonymous number.
    ///
    /// # Errors
    ///
    /// [`Error::ResourceLimit`] when `declared` exceeds [`Self::max_alloc_bytes`]
    /// or does not fit in a `usize`.
    pub const fn checked_bytes(&self, declared: u64, what: &'static str) -> Result<usize> {
        if declared > self.max_alloc_bytes as u64 {
            return Err(Error::ResourceLimit {
                what,
                requested: declared,
                limit: self.max_alloc_bytes as u64,
            });
        }
        // The comparison above already proved `declared` fits in a usize.
        Ok(declared as usize)
    }

    /// Accept a declared element count, or reject it as a resource-limit error.
    ///
    /// Both the count itself and its footprint (`count × element_bytes`) are
    /// checked, so a small count of enormous elements is caught alongside an
    /// enormous count of small ones.
    ///
    /// # Errors
    ///
    /// [`Error::ResourceLimit`] when the count exceeds [`Self::max_elements`]
    /// or its footprint exceeds [`Self::max_alloc_bytes`].
    pub const fn checked_elements(
        &self,
        declared: u64,
        element_bytes: usize,
        what: &'static str,
    ) -> Result<usize> {
        if declared > self.max_elements as u64 {
            return Err(Error::ResourceLimit {
                what,
                requested: declared,
                limit: self.max_elements as u64,
            });
        }
        let count = declared as usize;
        match count.checked_mul(element_bytes) {
            Some(footprint) if footprint <= self.max_alloc_bytes => Ok(count),
            _ => Err(Error::ResourceLimit {
                what,
                requested: declared,
                limit: self.element_ceiling(element_bytes) as u64,
            }),
        }
    }

    /// How many elements of `element_bytes` fit inside the byte ceiling.
    ///
    /// A zero element size is treated as one byte: hostile metadata must not
    /// be able to divide by zero, and a zero-sized element cannot exhaust
    /// memory in any case.
    const fn element_ceiling(&self, element_bytes: usize) -> usize {
        match self.max_alloc_bytes.checked_div(element_bytes) {
            Some(ceiling) => ceiling,
            None => self.max_alloc_bytes,
        }
    }

    /// Clamp a declared element count to what may be reserved speculatively.
    ///
    /// Unlike [`Self::checked_elements`] this never rejects: it caps the
    /// up-front reservation while leaving the collection free to grow as
    /// elements are actually decoded. Use it where the count is a hint whose
    /// truth the subsequent parse will establish, and `checked_elements`
    /// where an over-large count is itself a format violation.
    #[must_use]
    pub const fn capacity_hint(&self, declared: u64, element_bytes: usize) -> usize {
        let ceiling = self.element_ceiling(element_bytes);
        if declared < ceiling as u64 {
            declared as usize
        } else {
            ceiling
        }
    }

    /// Enter one level of recursive descent, returning the new depth.
    ///
    /// # Errors
    ///
    /// [`Error::ResourceLimit`] when the new depth would exceed
    /// [`Self::max_depth`]. Rust guarantees no tail-call elimination, so a
    /// self-referential format must be bounded here or it overflows the stack
    /// — an abort no `Result` can express.
    pub const fn descend(&self, depth: u16, what: &'static str) -> Result<u16> {
        // Saturating, not wrapping: at `u16::MAX` the ceiling is already far
        // exceeded, and a wrap would restart the descent from zero — turning
        // the bound into an unbounded loop.
        let next = depth.saturating_add(1);
        if next > self.max_depth {
            return Err(Error::ResourceLimit {
                what,
                requested: next as u64,
                limit: self.max_depth as u64,
            });
        }
        Ok(next)
    }

    /// Reserve a zeroed byte buffer of a declared length, recoverably.
    ///
    /// The length clears [`Self::checked_bytes`] first, then the allocation
    /// itself goes through `try_reserve`, so exhaustion under a legitimate but
    /// large request surfaces as an error instead of the global allocator's
    /// abort.
    ///
    /// # Errors
    ///
    /// [`Error::ResourceLimit`] when the length exceeds the budget or the
    /// allocation cannot be satisfied.
    #[cfg(feature = "alloc")]
    pub fn zeroed(&self, declared: u64, what: &'static str) -> Result<Vec<u8>> {
        let len = self.checked_bytes(declared, what)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(len)
            .map_err(|_| Error::ResourceLimit {
                what,
                requested: declared,
                limit: self.max_alloc_bytes as u64,
            })?;
        buffer.resize(len, 0);
        Ok(buffer)
    }

    /// Reserve a collection of a declared element count, recoverably.
    ///
    /// # Errors
    ///
    /// [`Error::ResourceLimit`] when the count exceeds the budget or the
    /// allocation cannot be satisfied.
    #[cfg(feature = "alloc")]
    pub fn vec_with_capacity<T>(&self, declared: u64, what: &'static str) -> Result<Vec<T>> {
        let count = self.checked_elements(declared, core::mem::size_of::<T>(), what)?;
        let mut collection = Vec::new();
        collection
            .try_reserve_exact(count)
            .map_err(|_| Error::ResourceLimit {
                what,
                requested: declared,
                limit: self.max_elements as u64,
            })?;
        Ok(collection)
    }
    /// Read a stream into a bounded buffer without trusting its declared size.
    ///
    /// The initial capacity is only a hint from the format metadata. Every
    /// subsequent growth is checked against `max_alloc_bytes` and uses
    /// fallible reservation, so a compressed stream cannot turn expansion or
    /// allocator exhaustion into a process abort.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ResourceLimit`] when the stream exceeds the byte
    /// ceiling or allocation cannot be satisfied, [`Error::Io`] when reading
    /// fails, or [`Error::Overflow`] when the accumulated length overflows.
    #[cfg(all(feature = "alloc", feature = "std"))]
    pub fn read_bounded<R: std::io::Read>(
        &self,
        reader: &mut R,
        initial_capacity: usize,
        what: &'static str,
    ) -> Result<Vec<u8>> {
        let initial_capacity = self.checked_bytes(
            u64::try_from(initial_capacity).map_err(|_| Error::Overflow)?,
            what,
        )?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(initial_capacity)
            .map_err(|_| Error::ResourceLimit {
                what,
                requested: initial_capacity as u64,
                limit: self.max_alloc_bytes as u64,
            })?;

        let mut chunk = [0u8; 8192];
        loop {
            let read = reader.read(&mut chunk).map_err(Error::Io)?;
            if read == 0 {
                break;
            }
            let next_len = output.len().checked_add(read).ok_or(Error::Overflow)?;
            if next_len > self.max_alloc_bytes {
                return Err(Error::ResourceLimit {
                    what,
                    requested: next_len as u64,
                    limit: self.max_alloc_bytes as u64,
                });
            }
            output.try_reserve(read).map_err(|_| Error::ResourceLimit {
                what,
                requested: next_len as u64,
                limit: self.max_alloc_bytes as u64,
            })?;
            let Some(chunk) = chunk.get(..read) else {
                return Err(Error::InternalError {
                    message: String::from("reader returned more bytes than its buffer"),
                });
            };
            output.extend_from_slice(chunk);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    // Test code is exempt from the production panic policy: an `unwrap` here
    // asserts, it does not ship.
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;

    #[test]
    fn plausible_byte_length_passes_through() {
        let budget = ParseBudget::default();
        assert_eq!(budget.checked_bytes(4096, "node").unwrap(), 4096);
        assert_eq!(
            budget
                .checked_bytes(DEFAULT_MAX_ALLOC_BYTES as u64, "node")
                .unwrap(),
            DEFAULT_MAX_ALLOC_BYTES
        );
    }

    #[test]
    fn hostile_byte_length_is_a_typed_error() {
        let budget = ParseBudget::default();
        let error = budget.checked_bytes(u64::MAX, "node").unwrap_err();
        assert!(matches!(
            error,
            Error::ResourceLimit {
                what: "node",
                requested: u64::MAX,
                ..
            }
        ));
    }

    /// The element footprint, not just the count, is bounded: 2^20 elements of
    /// 4 KiB each is a modest count and a 4 GiB allocation.
    #[test]
    fn element_footprint_is_bounded_independently_of_count() {
        let budget = ParseBudget::default();
        assert!(budget.checked_elements(1_048_576, 4096, "records").is_err());
        assert_eq!(
            budget.checked_elements(1024, 4096, "records").unwrap(),
            1024
        );
    }

    #[test]
    fn capacity_hint_clamps_instead_of_rejecting() {
        let budget = ParseBudget::default();
        assert_eq!(budget.capacity_hint(10, 24), 10);
        assert_eq!(
            budget.capacity_hint(u64::MAX, 24),
            DEFAULT_MAX_ALLOC_BYTES / 24
        );
        // A zero element size must not divide by zero.
        assert_eq!(budget.capacity_hint(u64::MAX, 0), DEFAULT_MAX_ALLOC_BYTES);
    }

    #[test]
    fn descend_stops_at_the_depth_ceiling() {
        let budget = ParseBudget::new(1024, 1024, 3);
        assert_eq!(budget.descend(0, "nesting").unwrap(), 1);
        assert_eq!(budget.descend(2, "nesting").unwrap(), 3);
        let error = budget.descend(3, "nesting").unwrap_err();
        assert!(matches!(
            error,
            Error::ResourceLimit {
                what: "nesting",
                requested: 4,
                limit: 3
            }
        ));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn zeroed_returns_an_error_rather_than_aborting() {
        let budget = ParseBudget::default();
        assert_eq!(budget.zeroed(8, "buffer").unwrap(), vec![0u8; 8]);
        assert!(budget.zeroed(u64::MAX, "buffer").is_err());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn vec_with_capacity_bounds_by_element_footprint() {
        let budget = ParseBudget::default();
        let reserved: Vec<u64> = budget.vec_with_capacity(16, "records").unwrap();
        assert!(reserved.capacity() >= 16);
        assert!(
            budget
                .vec_with_capacity::<u64>(u64::MAX, "records")
                .is_err()
        );
    }
    #[cfg(feature = "std")]
    #[test]
    fn read_bounded_rejects_output_above_budget() {
        let budget = ParseBudget::new(4, 4, 1);
        let mut reader = std::io::Cursor::new([1u8, 2, 3, 4, 5]);
        let error = budget
            .read_bounded(&mut reader, 0, "decompressed output")
            .unwrap_err();
        assert!(matches!(error, Error::ResourceLimit { .. }));
    }
}
