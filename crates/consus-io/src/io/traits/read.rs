//! Positioned read trait for stateless, concurrent-safe byte access.
//!
//! ## Contract
//!
//! `ReadAt::read_at` reads exactly `buf.len()` bytes from the specified
//! absolute byte offset. The operation is stateless: no internal cursor
//! is maintained, enabling safe concurrent reads without external
//! synchronization (pread(2) / ReadFile-with-offset semantics).
//!
//! ## Object Safety
//!
//! `ReadAt` is object-safe and `dyn`-compatible.

use consus_core::Result;

/// Positioned byte read without cursor state.
///
/// # Thread Safety
///
/// Implementations must be safe for concurrent reads from multiple
/// threads. The read offset is an explicit parameter, not internal
/// mutable state, following `pread(2)` / `ReadFile`-with-offset
/// semantics.
///
/// # Contract
///
/// - `read_at(pos, buf)` reads exactly `buf.len()` bytes starting at
///   byte offset `pos`.
/// - If the source contains fewer than `pos + buf.len()` bytes, the
///   call returns an error.
/// - A zero-length `buf` succeeds without performing I/O.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait ReadAt {
    /// Read exactly `buf.len()` bytes starting at byte offset `pos`.
    ///
    /// # Errors
    ///
    /// - `Error::Io` on underlying I/O failure (requires `std` feature).
    /// - `Error::BufferTooSmall` if the source has fewer bytes available
    ///   than `pos + buf.len()`.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ReadAt` is object-safe.
    #[test]
    fn read_at_is_object_safe() {
        fn _assert(_: &dyn ReadAt) {}
    }
}
