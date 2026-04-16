//! Zarr storage backend abstraction.
//!
//! ## Design
//!
//! Zarr separates storage from format: chunks and metadata are accessed
//! through a key-value [`Store`] trait. This module provides concrete
//! implementations for:
//!
//! - **Local filesystem** — directory tree with `.zarray`, `.zgroup`, chunk files
//! - **In-memory** — `Vec<u8>`-backed, no I/O, for testing and ephemeral data
//! - **S3-compatible** — any object store that implements the S3 API
//!
//! ## Key Semantics
//!
//! Keys are path-like strings using `/` as the separator (Zarr convention):
//!
//! | Zarr v2 key | Content |
//! |-------------|---------|
//! | `"my_array/.zarray"` | Array metadata |
//! | `"my_array/.zattrs"` | Array attributes |
//! | `"my_array/.zgroup"` | Group metadata |
//! | `"my_array/c/0.0"` | Chunk at grid coordinate (0, 0) |
//!
//! ## Python zarr Compatibility
//!
//! All stores use the canonical Zarr key scheme, ensuring full
//! interoperability with zarr-python. A Zarr hierarchy produced by
//! consus-zarr and stored in S3 is directly readable by zarr-python
//! when served via an S3-compatible HTTP server.
//!
//! ## Module Hierarchy
//!
//! ```text
//! store/
//! ├── mod.rs       # Store trait and re-exports
//! ├── memory.rs    # InMemoryStore
//! ├── filesystem.rs # FsStore
//! └── s3.rs        # S3Store (feature = "s3")
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod filesystem;
#[cfg(feature = "alloc")]
mod memory;

#[cfg(feature = "s3")]
mod s3;

// Re-export store backends at the crate root.
#[cfg(feature = "alloc")]
pub use filesystem::FsStore;
#[cfg(feature = "alloc")]
pub use memory::InMemoryStore;
#[cfg(feature = "s3")]
pub use s3::S3Store;

// ---------------------------------------------------------------------------
// Store trait
// ---------------------------------------------------------------------------

/// Key-value store interface for Zarr backends.
///
/// This trait abstracts over the storage medium, enabling the same Zarr
/// logic to operate against local files, in-memory buffers, or remote
/// object stores without changes to the format layer.
///
/// ## Contract
///
/// - `get(key)` returns `Ok(data)` iff `set(key, data)` was previously
///   called with the same key.
/// - `delete(key)` removes the key; subsequent `get(key)` returns `NotFound`.
/// - `list(prefix)` returns all keys that start with `prefix`, sorted
///   lexicographically.
/// - `contains(key)` returns `true` iff the key exists.
///
/// ## Thread Safety
///
/// Implementations must be `Send + Sync` if they are to be shared across
/// threads. `InMemoryStore` is single-threaded only; wrap in a `Mutex`
/// or `RwLock` for thread-safe access.
#[cfg(feature = "alloc")]
pub trait Store: Send + Sync {
    /// Read the value for a key.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if the key does not exist.
    fn get(&self, key: &str) -> consus_core::Result<Vec<u8>>;

    /// Write a value for a key, replacing any existing value.
    fn set(&mut self, key: &str, value: &[u8]) -> consus_core::Result<()>;

    /// Delete a key.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` if the key does not exist.
    fn delete(&mut self, key: &str) -> consus_core::Result<()>;

    /// List all keys with the given prefix.
    ///
    /// The returned keys are sorted lexicographically.
    /// If no keys match, returns an empty vector (not an error).
    fn list(&self, prefix: &str) -> consus_core::Result<Vec<alloc::string::String>>;

    /// Check whether a key exists in the store.
    fn contains(&self, key: &str) -> consus_core::Result<bool>;
}

// ---------------------------------------------------------------------------
// Composite store wrappers
// ---------------------------------------------------------------------------

/// A store that is simultaneously readable and writable.
///
/// `Store` requires `&mut self` for write operations, which can be
/// inconvenient when the store is shared via `&Arc<Mutex<S>>` or similar.
/// This blanket impl allows any `Store` to also satisfy `ReadWriteStore`
/// when the caller holds a mutable reference.
#[cfg(feature = "alloc")]

/// Extends `Store` with a `get_mut` method for in-place mutation when
/// the store is already mutably borrowed.
#[cfg(feature = "alloc")]
pub trait ReadWriteStore: Store {
    /// Get a mutable reference to this store.
    fn get_mut(&mut self) -> &mut Self {
        self
    }
}

#[cfg(feature = "alloc")]
impl<T: Store> ReadWriteStore for T {}

// ---------------------------------------------------------------------------
// Store utilities
// ---------------------------------------------------------------------------

/// A store that prefixes all keys with a given string.
///
/// This is useful for namespacing: for example, a single S3 bucket can
/// hold multiple independent Zarr hierarchies by using different prefixes.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct PrefixedStore<S> {
    /// Inner store.
    store: S,
    /// Prefix applied to all keys.
    prefix: alloc::string::String,
}

#[cfg(feature = "alloc")]
impl<S: Store> PrefixedStore<S> {
    /// Wrap a store with a key prefix.
    ///
    /// The prefix is added to every key operation. If the prefix does
    /// not end with `/`, one is appended automatically.
    pub fn new(store: S, prefix: impl Into<alloc::string::String>) -> Self {
        let prefix_str = prefix.into();
        let prefix = if prefix_str.ends_with('/') {
            prefix_str
        } else {
            alloc::format!("{}/", prefix_str)
        };
        Self { store, prefix }
    }

    /// Returns a reference to the inner store.
    pub fn inner(&self) -> &S {
        &self.store
    }

    /// Returns a mutable reference to the inner store.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Consume this prefixed store and return the inner store.
    pub fn into_inner(self) -> S {
        self.store
    }

    /// Full key including the prefix.
    fn full_key(&self, key: &str) -> alloc::string::String {
        alloc::format!("{}{}", self.prefix, key)
    }
}

#[cfg(feature = "alloc")]
impl<S: Store> Store for PrefixedStore<S> {
    fn get(&self, key: &str) -> consus_core::Result<Vec<u8>> {
        self.store.get(&self.full_key(key))
    }

    fn set(&mut self, key: &str, value: &[u8]) -> consus_core::Result<()> {
        self.store.set(&self.full_key(key), value)
    }

    fn delete(&mut self, key: &str) -> consus_core::Result<()> {
        self.store.delete(&self.full_key(key))
    }

    fn list(&self, prefix: &str) -> consus_core::Result<Vec<alloc::string::String>> {
        let full_prefix = self.full_key(prefix);
        let keys = self.store.list(&full_prefix)?;
        let stripped = keys
            .into_iter()
            .map(|key| {
                key.strip_prefix(&full_prefix)
                    .map(alloc::string::String::from)
                    .unwrap_or(key)
            })
            .collect();
        Ok(stripped)
    }

    fn contains(&self, key: &str) -> consus_core::Result<bool> {
        self.store.contains(&self.full_key(key))
    }
}

/// Combine a readable store and a writable store into one struct.
///
/// This is useful when read and write handles are separate (e.g., a
/// read-only filesystem mount and a separate write handle).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SplitStore<R, W> {
    pub read: R,
    pub write: W,
}

#[cfg(feature = "alloc")]
impl<R: Store, W: Store> Store for SplitStore<R, W> {
    fn get(&self, key: &str) -> consus_core::Result<Vec<u8>> {
        self.read.get(key)
    }

    fn set(&mut self, key: &str, value: &[u8]) -> consus_core::Result<()> {
        self.write.set(key, value)
    }

    fn delete(&mut self, key: &str) -> consus_core::Result<()> {
        self.write.delete(key)
    }

    fn list(&self, prefix: &str) -> consus_core::Result<Vec<alloc::string::String>> {
        self.read.list(prefix)
    }

    fn contains(&self, key: &str) -> consus_core::Result<bool> {
        self.read.contains(key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixed_store_adds_prefix() {
        let inner = InMemoryStore::new();
        let mut store = PrefixedStore::new(inner, "my_data.zarr");
        store.set(".zarray", b"{}").unwrap();

        // The prefixed key should be reachable through the namespace-aware API.
        assert!(store.contains(".zarray").unwrap());
        assert!(!store.contains("my_data.zarr/.zarray").unwrap());
    }

    #[test]
    fn prefixed_store_list_strips_prefix() {
        let inner = InMemoryStore::new();
        let mut store = PrefixedStore::new(inner, "root/");
        store.set("arr/.zarray", b"{}").unwrap();
        store.set("arr/c/0.0", b"chunk").unwrap();

        let keys = store.list("arr/").unwrap();
        assert!(keys.is_empty() || keys.iter().all(|k| !k.starts_with("root/")));
    }

    #[test]
    fn prefixed_store_trailing_slash_normalized() {
        let inner = InMemoryStore::new();
        let store = PrefixedStore::new(inner, "no_trailing");
        // Should have normalized to "no_trailing/"
        assert!(store.prefix.ends_with('/'));
    }

    #[test]
    fn prefixed_store_roundtrip() {
        let inner = InMemoryStore::new();
        let mut store = PrefixedStore::new(inner, "ns");
        let data = b"test data for round-trip";
        store.set("key/path", data).unwrap();
        let retrieved = store.get("key/path").unwrap();
        assert_eq!(&retrieved, data);
    }

    #[test]
    fn split_store_read_write_separate() {
        let read_store = InMemoryStore::new();
        let mut write_store = InMemoryStore::new();

        write_store.set("key", b"value").unwrap();

        let split = SplitStore {
            read: read_store,
            write: write_store,
        };

        assert!(split.get("key").is_err());
        assert_eq!(split.write.get("key").unwrap(), b"value");
    }

    #[test]
    fn split_store_clone_is_deep() {
        let read_store = InMemoryStore::new();
        let write_store = InMemoryStore::new();
        let split = SplitStore {
            read: read_store,
            write: write_store,
        };
        let split2 = split.clone();
        drop(split);
        // split2 should still have its own stores
        assert!(split2.read.list("").is_ok());
    }

    #[test]
    fn read_write_store_blanket() {
        fn _assert<T: ReadWriteStore>(_: &T) {}
        let store = InMemoryStore::new();
        _assert(&store);
    }
}
