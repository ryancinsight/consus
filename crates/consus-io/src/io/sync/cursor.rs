//! In-memory I/O buffer implementing positioned and stream I/O.
//!
//! ## Design
//!
//! `MemCursor` wraps a `Vec<u8>` and implements `ReadAt`, `WriteAt`,
//! `Length`, and `Truncate`. Writes beyond the current length
//! auto-extend the buffer with zero bytes.
//!
//! Thread-safety is the caller's responsibility (wrap in `Mutex`
//! or `RwLock` if concurrent access is needed).
//!
//! ## Invariants
//!
//! - `read_at(pos, buf)` succeeds iff `pos + buf.len() <= self.data.len()`.
//! - `write_at(pos, buf)` extends the buffer to `pos + buf.len()` if needed.
//! - `set_len(size)` resizes the buffer, zero-filling on extension.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use consus_core::{Error, Result};

use crate::io::traits::{Length, ReadAt, Truncate, WriteAt};

/// In-memory byte buffer implementing positioned I/O.
///
/// Useful for testing format backends without filesystem access,
/// and for constructing byte sequences programmatically.
///
/// # Examples
///
/// ```
/// use consus_io::io::sync::cursor::MemCursor;
/// use consus_io::io::traits::ReadAt;
///
/// let cursor = MemCursor::from_bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]);
/// let mut buf = [0u8; 2];
/// cursor.read_at(2, &mut buf).unwrap();
/// assert_eq!(buf, [0xBE, 0xEF]);
/// ```
#[derive(Debug, Clone)]
pub struct MemCursor {
    /// Backing storage.
    data: Vec<u8>,
}

impl MemCursor {
    /// Create an empty in-memory buffer.
    #[must_use]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create a buffer with the specified initial capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Create a buffer pre-populated with `data`.
    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Borrow the underlying bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Mutably borrow the underlying bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consume the cursor and return the underlying buffer.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Current byte length of the buffer.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer contains zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for MemCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadAt for MemCursor {
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

impl WriteAt for MemCursor {
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let pos = pos as usize;
        let end = pos.checked_add(buf.len()).ok_or(Error::Overflow)?;
        if end > self.data.len() {
            self.data.resize(end, 0);
        }
        self.data[pos..end].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // No-op for in-memory buffer: all writes are immediately visible.
        Ok(())
    }
}

impl Length for MemCursor {
    fn len(&self) -> Result<u64> {
        Ok(self.data.len() as u64)
    }
}

impl Truncate for MemCursor {
    fn set_len(&mut self, size: u64) -> Result<()> {
        self.data.resize(size as usize, 0);
        Ok(())
    }
}

// MemCursor implements ReadAt + WriteAt + Length + Truncate,
// so it automatically satisfies RandomAccess via blanket impl.

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::io::traits::RandomAccess;

    /// MemCursor satisfies `RandomAccess`.
    #[test]
    fn mem_cursor_is_random_access() {
        fn _assert(_: &dyn RandomAccess) {}
    }

    /// Write then read round-trip at identical offset.
    #[test]
    fn write_then_read_roundtrip() {
        let mut cursor = MemCursor::new();
        let payload = b"consus-io positioned I/O";
        cursor.write_at(0, payload).expect("write must succeed");

        assert_eq!(cursor.byte_len(), payload.len());

        let mut buf = vec![0u8; payload.len()];
        cursor.read_at(0, &mut buf).expect("read must succeed");
        assert_eq!(&buf, payload);
    }

    /// Write beyond current length auto-extends with zero fill.
    #[test]
    fn write_extends_with_zero_fill() {
        let mut cursor = MemCursor::new();
        cursor.write_at(10, b"data").expect("write must succeed");
        assert_eq!(cursor.byte_len(), 14);

        // Bytes [0..10) must be zero.
        let mut prefix = vec![0u8; 10];
        cursor.read_at(0, &mut prefix).expect("read must succeed");
        assert_eq!(prefix, vec![0u8; 10]);

        // Bytes [10..14) must be "data".
        let mut suffix = vec![0u8; 4];
        cursor.read_at(10, &mut suffix).expect("read must succeed");
        assert_eq!(&suffix, b"data");
    }

    /// Read beyond buffer bounds returns `BufferTooSmall`.
    #[test]
    fn read_out_of_bounds() {
        let cursor = MemCursor::from_bytes(vec![1, 2, 3]);
        let mut buf = [0u8; 4];
        let err = cursor.read_at(0, &mut buf).unwrap_err();
        match err {
            Error::BufferTooSmall {
                required: 4,
                provided: 3,
            } => {}
            other => panic!("expected BufferTooSmall, got: {other}"),
        }
    }

    /// Read at offset beyond buffer length returns error.
    #[test]
    fn read_past_end() {
        let cursor = MemCursor::from_bytes(vec![1, 2]);
        let mut buf = [0u8; 1];
        let err = cursor.read_at(3, &mut buf).unwrap_err();
        match err {
            Error::BufferTooSmall { .. } => {}
            other => panic!("expected BufferTooSmall, got: {other}"),
        }
    }

    /// Zero-length read always succeeds.
    #[test]
    fn zero_length_read_succeeds() {
        let cursor = MemCursor::new();
        let mut buf = [];
        cursor
            .read_at(0, &mut buf)
            .expect("zero-length read must succeed");
    }

    /// Zero-length write always succeeds.
    #[test]
    fn zero_length_write_succeeds() {
        let mut cursor = MemCursor::new();
        cursor
            .write_at(0, &[])
            .expect("zero-length write must succeed");
        assert_eq!(cursor.byte_len(), 0);
    }

    /// `set_len` truncates when smaller.
    #[test]
    fn set_len_truncates() {
        let mut cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);
        cursor.set_len(3).expect("set_len must succeed");
        assert_eq!(cursor.byte_len(), 3);
        assert_eq!(cursor.as_bytes(), &[1, 2, 3]);
    }

    /// `set_len` zero-extends when larger.
    #[test]
    fn set_len_extends() {
        let mut cursor = MemCursor::from_bytes(vec![1, 2]);
        cursor.set_len(5).expect("set_len must succeed");
        assert_eq!(cursor.byte_len(), 5);
        assert_eq!(cursor.as_bytes(), &[1, 2, 0, 0, 0]);
    }

    /// `Length::len` returns correct value.
    #[test]
    fn length_trait_returns_byte_count() {
        let cursor = MemCursor::from_bytes(vec![0; 42]);
        assert_eq!(Length::len(&cursor).unwrap(), 42);
    }

    /// `Length::is_empty` returns true for empty cursor.
    #[test]
    fn is_empty_on_empty_cursor() {
        let cursor = MemCursor::new();
        assert!(Length::is_empty(&cursor).unwrap());
    }

    /// `Length::is_empty` returns false for non-empty cursor.
    #[test]
    fn is_empty_on_nonempty_cursor() {
        let cursor = MemCursor::from_bytes(vec![1]);
        assert!(!Length::is_empty(&cursor).unwrap());
    }

    /// Overwriting existing data preserves surrounding bytes.
    #[test]
    fn overwrite_preserves_surrounding() {
        let mut cursor = MemCursor::from_bytes(vec![0xAA; 10]);
        cursor
            .write_at(3, &[0xBB, 0xCC])
            .expect("write must succeed");
        assert_eq!(cursor.byte_len(), 10);
        assert_eq!(cursor.as_bytes()[2], 0xAA);
        assert_eq!(cursor.as_bytes()[3], 0xBB);
        assert_eq!(cursor.as_bytes()[4], 0xCC);
        assert_eq!(cursor.as_bytes()[5], 0xAA);
    }

    /// `from_bytes` + `into_bytes` round-trip.
    #[test]
    fn from_into_roundtrip() {
        let original = vec![10, 20, 30, 40, 50];
        let cursor = MemCursor::from_bytes(original.clone());
        let recovered = cursor.into_bytes();
        assert_eq!(recovered, original);
    }
}
