//! Async positioned I/O traits.
//!
//! Enabled by the `async-io` feature. Provides async equivalents of
//! `ReadAt` and `WriteAt` for use with object stores and async runtimes.

use consus_core::error::Result;

/// Async positioned read.
///
/// Object store backends (S3, GCS, Azure Blob) implement this trait
/// to provide range-request-based reads.
pub trait AsyncReadAt: Send + Sync {
    /// Read exactly `buf.len()` bytes starting at byte offset `pos`.
    fn read_at(
        &self,
        pos: u64,
        buf: &mut [u8],
    ) -> impl core::future::Future<Output = Result<()>> + Send;

    /// Total length of the underlying source in bytes.
    fn len(&self) -> impl core::future::Future<Output = Result<u64>> + Send;
}

/// Async positioned write.
pub trait AsyncWriteAt: Send + Sync {
    /// Write all of `buf` starting at byte offset `pos`.
    fn write_at(
        &mut self,
        pos: u64,
        buf: &[u8],
    ) -> impl core::future::Future<Output = Result<()>> + Send;

    /// Flush any buffered data.
    fn flush(&mut self) -> impl core::future::Future<Output = Result<()>> + Send;
}
