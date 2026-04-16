//! Positioned write trait for stateless byte output.
//!
//! ## Contract
//!
//! `WriteAt::write_at` writes all `buf.len()` bytes at the specified
//! absolute byte offset. Implementations may extend the sink if the
//! write position exceeds the current length.
//!
//! ## Object Safety
//!
//! `WriteAt` is object-safe and `dyn`-compatible.

use consus_core::Result;

/// Positioned byte write at absolute offset.
///
/// # Contract
///
/// - `write_at(pos, buf)` writes all `buf.len()` bytes starting at
///   byte offset `pos`.
/// - Implementations may extend the sink if `pos + buf.len()` exceeds
///   the current length (implementation-defined behavior).
/// - A zero-length `buf` succeeds without performing I/O.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait WriteAt {
    /// Write all of `buf` starting at byte offset `pos`.
    ///
    /// # Errors
    ///
    /// - `Error::Io` on underlying I/O failure.
    /// - `Error::ReadOnly` if the sink does not support writes.
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<()>;

    /// Flush any buffered data to the underlying sink.
    ///
    /// Implementations that do not buffer may implement this as a no-op.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` on I/O failure.
    fn flush(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `WriteAt` is object-safe.
    #[test]
    fn write_at_is_object_safe() {
        fn _assert(_: &dyn WriteAt) {}
    }
}
