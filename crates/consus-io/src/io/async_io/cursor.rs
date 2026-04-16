//! Asynchronous in-memory I/O buffer.
//!
//! ## Design
//!
//! `AsyncMemCursor` wraps a `Vec<u8>` and implements the async I/O
//! traits (`AsyncReadAt`, `AsyncWriteAt`, `AsyncLength`, `AsyncTruncate`).
//! Since all operations are in-memory, the futures complete immediately
//! without actually suspending. This makes `AsyncMemCursor` suitable
//! for testing async I/O paths without requiring a real async runtime
//! for correctness (though tests use `tokio::test` for the executor).
//!
//! ## Invariants
//!
//! - Same semantics as `MemCursor`: reads require sufficient bytes,
//!   writes auto-extend, `set_len` truncates or zero-extends.
//! - All trait methods are `Send + Sync`-compatible.

use alloc::vec::Vec;

use consus_core::{Error, Result};

use crate::io::traits::{AsyncLength, AsyncReadAt, AsyncTruncate, AsyncWriteAt};

/// Asynchronous in-memory byte buffer.
///
/// Implements `AsyncReadAt`, `AsyncWriteAt`, `AsyncLength`, and
/// `AsyncTruncate`. All operations complete immediately (no actual
/// suspension) since the backing store is in-memory.
///
/// # Thread Safety
///
/// `AsyncMemCursor` is `Send + Sync` because `Vec<u8>` is `Send + Sync`.
#[derive(Debug, Clone)]
pub struct AsyncMemCursor {
    /// Backing storage.
    data: Vec<u8>,
}

impl AsyncMemCursor {
    /// Create an empty async in-memory buffer.
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

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for AsyncMemCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncReadAt for AsyncMemCursor {
    async fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
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

impl AsyncWriteAt for AsyncMemCursor {
    async fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
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

    async fn flush(&mut self) -> Result<()> {
        // No-op for in-memory buffer.
        Ok(())
    }
}

impl AsyncLength for AsyncMemCursor {
    async fn len(&self) -> Result<u64> {
        Ok(self.data.len() as u64)
    }
}

impl AsyncTruncate for AsyncMemCursor {
    async fn set_len(&mut self, size: u64) -> Result<()> {
        self.data.resize(size as usize, 0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write then read round-trip.
    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let mut cursor = AsyncMemCursor::new();
        let payload = b"async consus-io test";
        cursor
            .write_at(0, payload)
            .await
            .expect("write must succeed");

        assert_eq!(cursor.byte_len(), payload.len());

        let mut buf = vec![0u8; payload.len()];
        cursor
            .read_at(0, &mut buf)
            .await
            .expect("read must succeed");
        assert_eq!(&buf, payload);
    }

    /// Write beyond length auto-extends with zero fill.
    #[tokio::test]
    async fn write_extends_with_zero_fill() {
        let mut cursor = AsyncMemCursor::new();
        cursor
            .write_at(10, b"data")
            .await
            .expect("write must succeed");
        assert_eq!(cursor.byte_len(), 14);

        let mut prefix = vec![0u8; 10];
        cursor
            .read_at(0, &mut prefix)
            .await
            .expect("read must succeed");
        assert_eq!(prefix, vec![0u8; 10]);

        let mut suffix = vec![0u8; 4];
        cursor
            .read_at(10, &mut suffix)
            .await
            .expect("read must succeed");
        assert_eq!(&suffix, b"data");
    }

    /// Read out of bounds returns error.
    #[tokio::test]
    async fn read_out_of_bounds() {
        let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3]);
        let mut buf = [0u8; 4];
        let err = cursor.read_at(0, &mut buf).await.unwrap_err();
        match err {
            Error::BufferTooSmall {
                required: 4,
                provided: 3,
            } => {}
            other => panic!("expected BufferTooSmall, got: {other}"),
        }
    }

    /// Async length returns correct value.
    #[tokio::test]
    async fn async_length() {
        let cursor = AsyncMemCursor::from_bytes(vec![0; 42]);
        assert_eq!(AsyncLength::len(&cursor).await.unwrap(), 42);
    }

    /// Async is_empty.
    #[tokio::test]
    async fn async_is_empty() {
        let empty = AsyncMemCursor::new();
        assert!(AsyncLength::is_empty(&empty).await.unwrap());

        let non_empty = AsyncMemCursor::from_bytes(vec![1]);
        assert!(!AsyncLength::is_empty(&non_empty).await.unwrap());
    }

    /// Async set_len truncates.
    #[tokio::test]
    async fn async_truncate() {
        let mut cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3, 4, 5]);
        cursor.set_len(3).await.expect("set_len must succeed");
        assert_eq!(cursor.byte_len(), 3);
        assert_eq!(cursor.as_bytes(), &[1, 2, 3]);
    }

    /// Async set_len extends with zeros.
    #[tokio::test]
    async fn async_extend() {
        let mut cursor = AsyncMemCursor::from_bytes(vec![1, 2]);
        cursor.set_len(5).await.expect("set_len must succeed");
        assert_eq!(cursor.byte_len(), 5);
        assert_eq!(cursor.as_bytes(), &[1, 2, 0, 0, 0]);
    }

    /// Zero-length operations succeed.
    #[tokio::test]
    async fn zero_length_ops() {
        let cursor = AsyncMemCursor::new();
        let mut buf = [];
        cursor
            .read_at(0, &mut buf)
            .await
            .expect("zero-length read must succeed");

        let mut cursor = AsyncMemCursor::new();
        cursor
            .write_at(0, &[])
            .await
            .expect("zero-length write must succeed");
        assert_eq!(cursor.byte_len(), 0);
    }

    /// Flush is a no-op that succeeds.
    #[tokio::test]
    async fn flush_succeeds() {
        let mut cursor = AsyncMemCursor::new();
        cursor.flush().await.expect("flush must succeed");
    }

    /// `from_bytes` + `into_bytes` round-trip.
    #[tokio::test]
    async fn from_into_roundtrip() {
        let original = vec![10, 20, 30];
        let cursor = AsyncMemCursor::from_bytes(original.clone());
        let recovered = cursor.into_bytes();
        assert_eq!(recovered, original);
    }
}
