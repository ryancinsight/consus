//! Contiguous (CSR-style) storage for multiple disjoint byte-range reads.
//!
//! [`RangeBytes`] is the canonical home for this container: `read_ranges`
//! and `par_read_ranges` in the parent module populate it, but the storage
//! and access invariants live here so the range-read functions stay
//! format-agnostic I/O orchestration and this type stays a pure container.

use alloc::vec::Vec;

use consus_core::Result;

use super::IoRange;

/// Owned results of multiple disjoint byte-range reads, stored contiguously.
///
/// Ranges are read into one backing buffer addressed by offset (a CSR-style
/// layout) rather than as one heap allocation per range, so a batch of `n`
/// range reads costs one allocation instead of `n`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RangeBytes {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl RangeBytes {
    pub(super) fn with_capacity(ranges: &[IoRange]) -> Self {
        let total: usize = ranges.iter().map(|range| range.len).sum();
        let mut offsets = Vec::with_capacity(ranges.len() + 1);
        offsets.push(0);
        Self {
            bytes: Vec::with_capacity(total),
            offsets,
        }
    }

    /// Consumes per-range results (in range order) into one contiguous buffer.
    pub(super) fn try_from_chunks(chunks: Vec<Result<Vec<u8>>>) -> Result<Self> {
        let mut offsets = Vec::with_capacity(chunks.len() + 1);
        offsets.push(0);
        let mut bytes = Vec::new();
        for chunk in chunks {
            let chunk = chunk?;
            bytes.extend_from_slice(&chunk);
            offsets.push(bytes.len());
        }
        Ok(Self { bytes, offsets })
    }

    /// Appends `len` zero-initialized bytes and returns a mutable view into
    /// them for the caller to fill (e.g. via a positioned read), recording
    /// the new range's boundary.
    pub(super) fn push_zeroed(&mut self, len: usize) -> &mut [u8] {
        let start = self.bytes.len();
        self.bytes.resize(start + len, 0);
        self.offsets.push(self.bytes.len());
        &mut self.bytes[start..]
    }

    /// Number of ranges stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Returns `true` when no ranges were read.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows the bytes read for range `index`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(index)?;
        let end = *self.offsets.get(index + 1)?;
        Some(&self.bytes[start..end])
    }

    /// Iterates over each range's bytes in range order.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        self.offsets
            .windows(2)
            .map(move |w| &self.bytes[w[0]..w[1]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ranges_yield_empty_range_bytes() {
        let out = RangeBytes::with_capacity(&[]);
        assert_eq!(out.len(), 0);
        assert!(out.is_empty());
        assert_eq!(out.get(0), None);
        assert_eq!(out.iter().collect::<Vec<_>>(), Vec::<&[u8]>::new());
    }

    #[test]
    fn push_zeroed_records_offsets_in_call_order() {
        let mut out = RangeBytes::with_capacity(&[IoRange::new(0, 3), IoRange::new(0, 2)]);
        out.push_zeroed(3).copy_from_slice(&[1, 2, 3]);
        out.push_zeroed(2).copy_from_slice(&[9, 9]);

        assert_eq!(out.len(), 2);
        assert_eq!(out.get(0), Some(&[1u8, 2, 3][..]));
        assert_eq!(out.get(1), Some(&[9u8, 9][..]));
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            vec![&[1u8, 2, 3][..], &[9u8, 9][..]]
        );
    }

    #[test]
    fn try_from_chunks_concatenates_in_input_order() {
        let chunks: Vec<Result<Vec<u8>>> =
            alloc::vec![Ok(alloc::vec![1, 2]), Ok(alloc::vec![]), Ok(alloc::vec![3])];
        let out = RangeBytes::try_from_chunks(chunks).expect("chunks");

        assert_eq!(out.len(), 3);
        assert_eq!(out.get(0), Some(&[1u8, 2][..]));
        assert_eq!(out.get(1), Some(&[][..]));
        assert_eq!(out.get(2), Some(&[3u8][..]));
    }
}
