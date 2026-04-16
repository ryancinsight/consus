//! Read-only slice adapter for `no_std` positioned I/O.
//!
//! ## Design
//!
//! `SliceReader` wraps a `&[u8]` and implements `ReadAt` + `Length`.
//! This is the most lightweight I/O source: no allocation, no
//! mutability, no feature gates. Suitable for parsing byte slices
//! directly (e.g., memory-mapped file contents).
//!
//! ## Invariants
//!
//! - `read_at(pos, buf)` succeeds iff `pos + buf.len() <= slice.len()`.
//! - `len()` returns `slice.len()` as `u64`.

use consus_core::{Error, Result};

use crate::io::traits::{Length, ReadAt};

/// Read-only positioned I/O over a byte slice.
///
/// Zero-allocation, `no_std`-compatible. Implements `ReadAt` and
/// `Length` but not `WriteAt` or `Truncate` (immutable source).
///
/// # Lifetime
///
/// The reader borrows the slice for lifetime `'a`. The slice must
/// outlive all I/O operations.
///
/// # Examples
///
/// ```
/// use consus_io::io::sync::slice::SliceReader;
/// use consus_io::io::traits::ReadAt;
///
/// let data = [0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A];
/// let reader = SliceReader::new(&data);
/// let mut magic = [0u8; 4];
/// reader.read_at(0, &mut magic).unwrap();
/// assert_eq!(&magic, b"\x89HDF");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SliceReader<'a> {
    /// Borrowed byte slice.
    data: &'a [u8],
}

impl<'a> SliceReader<'a> {
    /// Create a reader over the given byte slice.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Borrow the underlying slice.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Byte length of the underlying slice.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Whether the underlying slice is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> ReadAt for SliceReader<'a> {
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let pos = pos as usize;
        let end = pos.checked_add(buf.len()).ok_or(Error::Overflow)?;
        if end > self.data.len() {
            return Err(Error::BufferTooSmall {
                required: end,
                provided: self.data.len(),
            });
        }
        buf.copy_from_slice(&self.data[pos..end]);
        Ok(())
    }
}

impl<'a> Length for SliceReader<'a> {
    fn len(&self) -> Result<u64> {
        Ok(self.data.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read at the start of the slice.
    #[test]
    fn read_at_start() {
        let data = [10u8, 20, 30, 40, 50];
        let reader = SliceReader::new(&data);
        let mut buf = [0u8; 3];
        reader.read_at(0, &mut buf).expect("read must succeed");
        assert_eq!(buf, [10, 20, 30]);
    }

    /// Read at a non-zero offset.
    #[test]
    fn read_at_offset() {
        let data = [0xAA, 0xBB, 0xCC, 0xDD];
        let reader = SliceReader::new(&data);
        let mut buf = [0u8; 2];
        reader.read_at(2, &mut buf).expect("read must succeed");
        assert_eq!(buf, [0xCC, 0xDD]);
    }

    /// Read exactly all bytes.
    #[test]
    fn read_entire_slice() {
        let data = [1, 2, 3];
        let reader = SliceReader::new(&data);
        let mut buf = [0u8; 3];
        reader.read_at(0, &mut buf).expect("read must succeed");
        assert_eq!(buf, [1, 2, 3]);
    }

    /// Read beyond bounds fails.
    #[test]
    fn read_out_of_bounds() {
        let data = [1, 2, 3];
        let reader = SliceReader::new(&data);
        let mut buf = [0u8; 4];
        let err = reader.read_at(0, &mut buf).unwrap_err();
        match err {
            Error::BufferTooSmall {
                required: 4,
                provided: 3,
            } => {}
            other => panic!("expected BufferTooSmall, got: {other}"),
        }
    }

    /// Read with offset past end fails.
    #[test]
    fn read_past_end() {
        let data = [1, 2];
        let reader = SliceReader::new(&data);
        let mut buf = [0u8; 1];
        let err = reader.read_at(5, &mut buf).unwrap_err();
        match err {
            Error::BufferTooSmall { .. } => {}
            other => panic!("expected BufferTooSmall, got: {other}"),
        }
    }

    /// Zero-length read succeeds on empty slice.
    #[test]
    fn zero_length_read_on_empty() {
        let reader = SliceReader::new(&[]);
        let mut buf = [];
        reader
            .read_at(0, &mut buf)
            .expect("zero-length read must succeed");
    }

    /// Length returns correct value.
    #[test]
    fn length_returns_slice_len() {
        let data = [0u8; 256];
        let reader = SliceReader::new(&data);
        assert_eq!(Length::len(&reader).unwrap(), 256);
    }

    /// Empty slice has length zero.
    #[test]
    fn empty_slice_length() {
        let reader = SliceReader::new(&[]);
        assert_eq!(Length::len(&reader).unwrap(), 0);
        assert!(Length::is_empty(&reader).unwrap());
    }

    /// `SliceReader` is Copy.
    #[test]
    fn slice_reader_is_copy() {
        let data = [1, 2, 3];
        let reader = SliceReader::new(&data);
        let copy = reader;
        let mut buf1 = [0u8; 3];
        let mut buf2 = [0u8; 3];
        reader.read_at(0, &mut buf1).unwrap();
        copy.read_at(0, &mut buf2).unwrap();
        assert_eq!(buf1, buf2);
    }
}
