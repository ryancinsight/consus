//! Zarr storage backend abstraction.
//!
//! ## Design
//!
//! Zarr separates storage from format: chunks and metadata are accessed
//! through a key-value interface. This trait abstracts over:
//! - Local filesystem directories
//! - Zip archives
//! - Object stores (S3, GCS, Azure Blob)
//! - In-memory stores (testing)

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::error::Result;

/// Key-value store interface for Zarr backends.
///
/// Keys are path-like strings (e.g., `"array/.zarray"`, `"array/0.0.0"`).
#[cfg(feature = "alloc")]
pub trait Store {
    /// Read the value for a key.
    fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// Write a value for a key.
    fn set(&mut self, key: &str, value: &[u8]) -> Result<()>;

    /// Delete a key.
    fn delete(&mut self, key: &str) -> Result<()>;

    /// List keys with a given prefix.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Check if a key exists.
    fn contains(&self, key: &str) -> Result<bool>;
}
