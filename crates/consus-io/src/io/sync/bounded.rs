//! Bounded-allocation sequential reads for untrusted declared lengths.

use alloc::vec::Vec;
use std::io::{Error, ErrorKind, Read, Result};

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
