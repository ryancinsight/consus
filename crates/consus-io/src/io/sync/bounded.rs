//! Bounded-allocation sequential reads for untrusted declared lengths.

use alloc::vec::Vec;
use std::io::{Error, ErrorKind, Read, Result};

const READ_CHUNK_BYTES: usize = 64 * 1024;

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
pub fn read_exact_bounded<R: Read>(reader: &mut R, length: usize) -> Result<Vec<u8>> {
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
            Vec::new()
        );
    }
}
