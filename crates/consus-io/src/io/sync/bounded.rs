//! Bounded-allocation reads for untrusted declared lengths.
//!
//! Sequential ([`read_exact_bounded`]) and positioned ([`read_at_bounded`])
//! forms of one rule: a length read out of a file header never becomes an
//! allocation until reads have confirmed the bytes exist.

use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::io::{Error, ErrorKind, Read, Result};

use consus_core::{Error as ConsusError, ParseBudget, Result as ConsusResult};

use crate::io::traits::read::ReadAt;

const READ_CHUNK_BYTES: usize = 64 * 1024;
const MAX_EAGER_BYTES: usize = 16 * 1024 * 1024;

/// Cap an element count so speculative reservation stays within 16 MiB.
///
/// The collection may still grow as validated elements are appended. A zero
/// element size is treated as one byte so hostile metadata cannot divide by
/// zero or bypass the cap.
#[must_use]
pub const fn bounded_capacity(count: usize, element_bytes: usize) -> usize {
    let element_bytes = if element_bytes == 0 { 1 } else { element_bytes };
    let limit = MAX_EAGER_BYTES / element_bytes;
    if count < limit { count } else { limit }
}

/// Reads exactly `length` bytes without reserving the untrusted length upfront.
///
/// Storage grows by at most 64 KiB before each confirmed read. This prevents a
/// hostile format header from turning a declared length into one speculative
/// allocation while retaining standard `Read` streaming semantics.
///
/// # Errors
///
/// Returns [`ErrorKind::UnexpectedEof`] when the source ends before `length`,
/// the source error for other read failures, or [`ErrorKind::Other`] when the
/// output allocation cannot grow.
#[cfg(feature = "std")]
pub fn read_exact_bounded<R: Read + ?Sized>(reader: &mut R, length: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    while output.len() < length {
        let remaining = length - output.len();
        let chunk = remaining.min(READ_CHUNK_BYTES);
        output.try_reserve(chunk).map_err(Error::other)?;
        let start = output.len();
        output.resize(start + chunk, 0);
        let mut filled = 0;
        while filled < chunk {
            match reader.read(&mut output[start + filled..start + chunk])? {
                0 => {
                    output.truncate(start + filled);
                    return Err(Error::new(
                        ErrorKind::UnexpectedEof,
                        format!(
                            "bounded read expected {length} bytes but received {}",
                            output.len()
                        ),
                    ));
                }
                count => filled += count,
            }
        }
    }
    Ok(output)
}

/// Read `declared` bytes from `pos` without reserving `declared` up front.
///
/// The positioned counterpart to [`read_exact_bounded`], and the bound that
/// [`ParseBudget`] alone cannot supply: the ceiling rejects an absurd length,
/// but a length that is merely far larger than the file still reaches the
/// allocator. Here the buffer grows by at most 64 KiB before each read that
/// must succeed, so a header claiming 60 MiB inside a 4 KiB file costs one
/// chunk and then fails — the allocation is bounded by the input that
/// actually exists, not only by the constant.
///
/// [`ReadAt`] carries no length, which is why the bound has to be expressed
/// as incremental confirmation rather than a single up-front comparison.
///
/// # Errors
///
/// - [`ConsusError::ResourceLimit`] when `declared` exceeds the budget's byte
///   ceiling or an allocation cannot be satisfied.
/// - The source's own error (typically [`ConsusError::BufferTooSmall`]) when
///   the input ends before `declared` bytes have been read.
pub fn read_at_bounded<R: ReadAt + ?Sized>(
    source: &R,
    pos: u64,
    declared: u64,
    budget: &ParseBudget,
    what: &'static str,
) -> ConsusResult<Vec<u8>> {
    let length = budget.checked_bytes(declared, what)?;

    let mut output = Vec::new();
    while output.len() < length {
        let chunk = (length - output.len()).min(READ_CHUNK_BYTES);
        output
            .try_reserve(chunk)
            .map_err(|_| ConsusError::ResourceLimit {
                what,
                requested: declared,
                limit: output.len() as u64,
            })?;
        let start = output.len();
        output.resize(start + chunk, 0);
        let offset = pos.checked_add(start as u64).ok_or(ConsusError::Overflow)?;
        // A short source fails here, after one chunk of growth rather than
        // after reserving the whole declared length.
        source.read_at(offset, &mut output[start..start + chunk])?;
    }
    Ok(output)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::io::sync::cursor::MemCursor;
    use std::io::Cursor;

    #[test]
    fn exact_read_crosses_chunk_boundaries_without_data_loss() {
        let input: Vec<u8> = (0..READ_CHUNK_BYTES + 17)
            .map(|index| (index % 251) as u8)
            .collect();
        let output = read_exact_bounded(&mut Cursor::new(&input), input.len()).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn truncated_read_reports_received_length() {
        let error = read_exact_bounded(&mut Cursor::new([1_u8, 2, 3]), 5).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
        assert!(error.to_string().contains("received 3"));
    }

    #[test]
    fn zero_length_does_not_touch_the_reader() {
        struct FailingReader;
        impl Read for FailingReader {
            fn read(&mut self, _: &mut [u8]) -> Result<usize> {
                panic!("zero-length bounded read must not access its source")
            }
        }
        assert_eq!(
            read_exact_bounded(&mut FailingReader, 0).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn capacity_is_exact_below_budget_and_capped_above_it() {
        assert_eq!(bounded_capacity(10, 4), 10);
        assert_eq!(bounded_capacity(usize::MAX, 4), MAX_EAGER_BYTES / 4);
        assert_eq!(bounded_capacity(usize::MAX, 0), MAX_EAGER_BYTES);
    }

    #[test]
    fn exact_read_accepts_trait_object_readers() {
        let mut cursor = Cursor::new([1_u8, 2, 3]);
        let reader: &mut dyn Read = &mut cursor;
        assert_eq!(read_exact_bounded(reader, 3).unwrap(), [1, 2, 3]);
    }

    #[test]
    fn positioned_read_crosses_chunk_boundaries_without_data_loss() {
        let input: Vec<u8> = (0..READ_CHUNK_BYTES + 17)
            .map(|index| (index % 251) as u8)
            .collect();
        let source = MemCursor::from_bytes(input.clone());
        let output = read_at_bounded(
            &source,
            0,
            input.len() as u64,
            &ParseBudget::default(),
            "data",
        )
        .unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn positioned_read_honours_the_offset() {
        let source = MemCursor::from_bytes(vec![1_u8, 2, 3, 4, 5]);
        let output = read_at_bounded(&source, 2, 3, &ParseBudget::default(), "data").unwrap();
        assert_eq!(output, [3, 4, 5]);
    }

    /// A declared length far beyond the file must cost one chunk, not the
    /// declared length, and must surface as an error.
    #[test]
    fn positioned_read_over_declaring_length_fails_without_reserving_it() {
        let source = MemCursor::from_bytes(vec![0_u8; 4096]);
        let error = read_at_bounded(
            &source,
            0,
            60 * 1024 * 1024,
            &ParseBudget::default(),
            "node",
        )
        .unwrap_err();
        assert!(
            matches!(error, ConsusError::BufferTooSmall { .. }),
            "expected a short-source error, got {error:?}"
        );
    }

    #[test]
    fn positioned_read_rejects_a_length_beyond_the_budget() {
        let source = MemCursor::from_bytes(vec![0_u8; 4096]);
        let error =
            read_at_bounded(&source, 0, u64::MAX, &ParseBudget::default(), "node").unwrap_err();
        assert!(matches!(
            error,
            ConsusError::ResourceLimit { what: "node", .. }
        ));
    }

    #[test]
    fn positioned_zero_length_read_does_not_touch_the_source() {
        let source = MemCursor::from_bytes(Vec::new());
        assert_eq!(
            read_at_bounded(&source, 0, 0, &ParseBudget::default(), "data").unwrap(),
            Vec::<u8>::new()
        );
    }
}
