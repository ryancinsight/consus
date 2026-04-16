//! Cursor-based seeking for stream-oriented I/O.
//!
//! ## Design
//!
//! Provides `no_std`-compatible seek primitives. Format parsers that
//! need sequential access over a positioned source use `Seekable`
//! via the `StreamReader` adapter in the `sync` module.
//!
//! `SeekFrom` is a `no_std` replacement for `std::io::SeekFrom`.
//!
//! ## Object Safety
//!
//! `Seekable` is object-safe and `dyn`-compatible.

use consus_core::Result;

/// Byte offset reference point for seek operations.
///
/// `no_std`-compatible replacement for `std::io::SeekFrom`.
///
/// ## Invariants
///
/// - `Start(n)`: absolute offset `n` from byte 0. Always non-negative.
/// - `End(n)`: signed offset from the end. `End(0)` = end of source.
///   `End(-k)` = `k` bytes before end.
/// - `Current(n)`: signed offset relative to current cursor position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SeekFrom {
    /// Absolute byte offset from the start of the source.
    Start(u64),
    /// Signed byte offset from the end of the source.
    ///
    /// `End(0)` positions the cursor at the end. `End(-n)` positions
    /// it `n` bytes before the end. Positive values seek past the end.
    End(i64),
    /// Signed byte offset relative to the current cursor position.
    Current(i64),
}

/// Cursor-based byte seeking.
///
/// Implementations maintain an internal cursor position that is
/// advanced by `seek()`. This trait is typically implemented by
/// stream adapters that wrap positioned I/O (`ReadAt` + `Length`).
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait Seekable {
    /// Move the internal cursor to a new position.
    ///
    /// Returns the new absolute byte position after seeking.
    ///
    /// # Errors
    ///
    /// - `Error::Overflow` if the computed position would be negative
    ///   or exceed `u64::MAX`.
    fn seek(&mut self, pos: SeekFrom) -> Result<u64>;

    /// Return the current cursor position without moving it.
    ///
    /// Equivalent to `self.seek(SeekFrom::Current(0))`.
    fn stream_position(&mut self) -> Result<u64> {
        self.seek(SeekFrom::Current(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Seekable` is object-safe.
    #[test]
    fn seekable_is_object_safe() {
        fn _assert(_: &dyn Seekable) {}
    }

    /// `SeekFrom` equality.
    #[test]
    fn seek_from_equality() {
        assert_eq!(SeekFrom::Start(0), SeekFrom::Start(0));
        assert_ne!(SeekFrom::Start(0), SeekFrom::End(0));
        assert_ne!(SeekFrom::Start(0), SeekFrom::Current(0));
        assert_eq!(SeekFrom::End(-10), SeekFrom::End(-10));
        assert_eq!(SeekFrom::Current(5), SeekFrom::Current(5));
    }
}
