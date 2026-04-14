//! In-memory I/O source for testing and embedded use.
//!
//! ## Design
//!
//! `MemCursor` wraps a `Vec<u8>` and implements `ReadAt` + `WriteAt`.
//! It auto-extends on writes beyond the current length.
//! Thread-safety is the caller's responsibility (wrap in `Mutex` if needed).

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec;
#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use consus_core::error::{Error, Result};

#[cfg(feature = "alloc")]
use crate::source::{ReadAt, WriteAt};

/// In-memory byte buffer implementing positioned I/O.
///
/// Useful for testing format backends without touching the filesystem.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct MemCursor {
    data: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl MemCursor {
    /// Create an empty in-memory buffer.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create a buffer pre-populated with `data`.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Access the underlying bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the cursor and return the underlying buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Current length of the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(feature = "alloc")]
impl Default for MemCursor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl ReadAt for MemCursor {
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        let pos = pos as usize;
        let end = pos.checked_add(buf.len()).ok_or(Error::BufferTooSmall {
            required: buf.len(),
            provided: 0,
        })?;
        if end > self.data.len() {
            return Err(Error::BufferTooSmall {
                required: end,
                provided: self.data.len(),
            });
        }
        buf.copy_from_slice(&self.data[pos..end]);
        Ok(())
    }

    fn len(&self) -> Result<u64> {
        Ok(self.data.len() as u64)
    }
}

#[cfg(feature = "alloc")]
impl WriteAt for MemCursor {
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        let pos = pos as usize;
        let end = pos.checked_add(buf.len()).ok_or(Error::BufferTooSmall {
            required: buf.len(),
            provided: 0,
        })?;
        if end > self.data.len() {
            self.data.resize(end, 0);
        }
        self.data[pos..end].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(()) // no-op for in-memory
    }

    fn set_len(&mut self, size: u64) -> Result<()> {
        self.data.resize(size as usize, 0);
        Ok(())
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn write_then_read() {
        let mut cursor = MemCursor::new();
        let data = b"hello, consus";
        cursor.write_at(0, data).expect("write must succeed");
        assert_eq!(cursor.len(), data.len());

        let mut buf = vec![0u8; data.len()];
        cursor.read_at(0, &mut buf).expect("read must succeed");
        assert_eq!(&buf, data);
    }

    #[test]
    fn write_extends_buffer() {
        let mut cursor = MemCursor::new();
        cursor
            .write_at(10, b"test")
            .expect("write beyond length must succeed");
        assert_eq!(cursor.len(), 14);
        // Bytes 0..10 should be zero-filled
        let mut prefix = vec![0u8; 10];
        cursor.read_at(0, &mut prefix).expect("read must succeed");
        assert_eq!(prefix, vec![0u8; 10]);
    }

    #[test]
    fn read_out_of_bounds_fails() {
        let cursor = MemCursor::from_bytes(vec![1, 2, 3]);
        let mut buf = [0u8; 4];
        let result = cursor.read_at(0, &mut buf);
        assert!(result.is_err());
    }
}
