//! Byte extent traits: length query and truncation.
//!
//! ## Design
//!
//! `Length` and `Truncate` are separated from `ReadAt`/`WriteAt` (SRP):
//! knowing the size of a source is orthogonal to reading/writing content.
//! Resizing a sink is orthogonal to writing content at a given offset.
//!
//! ## Object Safety
//!
//! Both traits are object-safe and `dyn`-compatible.

use consus_core::Result;

/// Query the byte length of an I/O source.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait Length {
    /// Total number of bytes in the source.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` if the length cannot be determined.
    fn len(&self) -> Result<u64>;

    /// Whether the source contains zero bytes.
    ///
    /// Default implementation delegates to `len()`.
    fn is_empty(&self) -> Result<bool> {
        self.len().map(|l| l == 0)
    }
}

/// Truncate or extend a sink to a specified byte length.
///
/// # Object Safety
///
/// This trait is object-safe.
pub trait Truncate {
    /// Set the sink's byte length to `size`.
    ///
    /// - If `size < current_len`, the sink is truncated.
    /// - If `size > current_len`, the sink is zero-extended.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` on I/O failure.
    fn set_len(&mut self, size: u64) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Length` is object-safe.
    #[test]
    fn length_is_object_safe() {
        fn _assert(_: &dyn Length) {}
    }

    /// `Truncate` is object-safe.
    #[test]
    fn truncate_is_object_safe() {
        fn _assert(_: &dyn Truncate) {}
    }
}
