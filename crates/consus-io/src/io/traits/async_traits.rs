//! Asynchronous equivalents of the synchronous I/O traits.
//!
//! ## Feature Gate
//!
//! This module is available only when the `async-io` feature is enabled.
//!
//! ## Design
//!
//! Async traits use return-position `impl Future` (RPITIT, stable since
//! Rust 1.75). This provides zero-cost async dispatch. These traits
//! require `Send + Sync` bounds for compatibility with multi-threaded
//! async runtimes (tokio).
//!
//! ## Trait Hierarchy
//!
//! ```text
//! AsyncReadAt:   Send + Sync  (positioned async read)
//! AsyncWriteAt:  Send + Sync  (positioned async write)
//! AsyncLength:   Send + Sync  (async length query)
//! AsyncTruncate: Send + Sync  (async resize)
//! AsyncSeekable: Send + Sync  (async cursor seeking)
//! AsyncRandomAccess = AsyncReadAt + AsyncWriteAt + AsyncLength + AsyncTruncate
//! ```

use super::seek::SeekFrom;
use consus_core::Result;

/// Asynchronous positioned byte read.
///
/// Async equivalent of [`super::read::ReadAt`]. Intended for object
/// store backends (S3, GCS, Azure Blob) that perform range-request reads.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` for use with multi-threaded
/// async runtimes.
pub trait AsyncReadAt: Send + Sync {
    /// Read exactly `buf.len()` bytes starting at byte offset `pos`.
    ///
    /// # Errors
    ///
    /// - `Error::Io` on underlying I/O failure.
    /// - `Error::BufferTooSmall` if the source has insufficient bytes.
    fn read_at(
        &self,
        pos: u64,
        buf: &mut [u8],
    ) -> impl core::future::Future<Output = Result<()>> + Send;
}

/// Asynchronous positioned byte write.
///
/// Async equivalent of [`super::write::WriteAt`].
pub trait AsyncWriteAt: Send + Sync {
    /// Write all of `buf` starting at byte offset `pos`.
    ///
    /// # Errors
    ///
    /// - `Error::Io` on I/O failure.
    /// - `Error::ReadOnly` if the sink does not support writes.
    fn write_at(
        &mut self,
        pos: u64,
        buf: &[u8],
    ) -> impl core::future::Future<Output = Result<()>> + Send;

    /// Flush any buffered data to the underlying sink.
    fn flush(&mut self) -> impl core::future::Future<Output = Result<()>> + Send;
}

/// Asynchronous byte length query.
///
/// Async equivalent of [`super::extent::Length`].
pub trait AsyncLength: Send + Sync {
    /// Total number of bytes in the source.
    fn len(&self) -> impl core::future::Future<Output = Result<u64>> + Send;

    /// Whether the source contains zero bytes.
    fn is_empty(&self) -> impl core::future::Future<Output = Result<bool>> + Send {
        async { self.len().await.map(|l| l == 0) }
    }
}

/// Asynchronous truncation.
///
/// Async equivalent of [`super::extent::Truncate`].
pub trait AsyncTruncate: Send + Sync {
    /// Set the sink's byte length to `size`.
    fn set_len(&mut self, size: u64) -> impl core::future::Future<Output = Result<()>> + Send;
}

/// Asynchronous cursor-based seeking.
///
/// Async equivalent of [`super::seek::Seekable`].
pub trait AsyncSeekable: Send + Sync {
    /// Move the cursor to a new position.
    ///
    /// Returns the new absolute byte position.
    fn seek(&mut self, pos: SeekFrom) -> impl core::future::Future<Output = Result<u64>> + Send;

    /// Return the current cursor position.
    fn stream_position(&mut self) -> impl core::future::Future<Output = Result<u64>> + Send {
        async { self.seek(SeekFrom::Current(0)).await }
    }
}

/// Asynchronous full random-access I/O.
///
/// Blanket-implemented for all types satisfying the constituent bounds.
pub trait AsyncRandomAccess: AsyncReadAt + AsyncWriteAt + AsyncLength + AsyncTruncate {}

impl<T: AsyncReadAt + AsyncWriteAt + AsyncLength + AsyncTruncate> AsyncRandomAccess for T {}
