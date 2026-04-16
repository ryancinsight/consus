//! In-memory store backend for Zarr.
//!
//! Provides a `Vec`-backed `Store` implementation for testing and
//! in-process usage without filesystem or network access.
//!
//! ## Thread Safety
//!
//! `InMemoryStore` is single-threaded. Wrap in a `Mutex` or `RwLock`
//! from `std::sync` for multi-threaded access.
//!
//! ## Key Semantics
//!
//! Keys are full paths within the Zarr hierarchy (e.g., `"my_array/.zarray"`).
//! The store treats keys as opaque byte sequences and does not interpret
//! path separators.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{collections::BTreeMap, string::String, vec::Vec};

#[cfg(feature = "alloc")]
use crate::store::Store;

/// In-memory key-value store backed by a `BTreeMap`.
///
/// Stores each key-value pair in a `BTreeMap<String, Vec<u8>>`, providing
/// ordered iteration over keys (which matters for `list` operations).
///
/// ## Invariants
///
/// - `get` returns `Ok(data)` iff `data` was previously stored with `set`.
/// - `delete` removes the key; subsequent `get` returns `NotFound`.
/// - `list(prefix)` returns all keys starting with `prefix`, sorted.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct InMemoryStore {
    /// Ordered map from key to value.
    entries: BTreeMap<String, Vec<u8>>,
}

#[cfg(feature = "alloc")]
impl InMemoryStore {
    /// Create an empty in-memory store.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Create a store pre-populated with the given key-value pairs.
    pub fn from_entries(entries: impl IntoIterator<Item = (String, Vec<u8>)>) -> Self {
        Self {
            entries: BTreeMap::from_iter(entries),
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all keys in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|k| k.as_str())
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(feature = "alloc")]
impl Store for InMemoryStore {
    fn get(&self, key: &str) -> consus_core::Result<Vec<u8>> {
        self.entries
            .get(key)
            .cloned()
            .ok_or_else(|| consus_core::Error::NotFound {
                path: key.to_string(),
            })
    }

    fn set(&mut self, key: &str, value: &[u8]) -> consus_core::Result<()> {
        self.entries.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    fn delete(&mut self, key: &str) -> consus_core::Result<()> {
        self.entries
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| consus_core::Error::NotFound {
                path: key.to_string(),
            })
    }

    fn list(&self, prefix: &str) -> consus_core::Result<Vec<String>> {
        let prefix_str = prefix.to_string();
        Ok(self
            .entries
            .range(prefix_str..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect())
    }

    fn contains(&self, key: &str) -> consus_core::Result<bool> {
        Ok(self.entries.contains_key(key))
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
    fn set_get_roundtrip() {
        let mut store = InMemoryStore::new();
        store.set("array/.zarray", b"{}").unwrap();
        let data = store.get("array/.zarray").unwrap();
        assert_eq!(&data, b"{}");
    }

    #[test]
    fn delete_then_not_found() {
        let mut store = InMemoryStore::new();
        store.set("key", b"value").unwrap();
        store.delete("key").unwrap();
        let err = store.get("key").unwrap_err();
        assert!(matches!(err, consus_core::Error::NotFound { .. }));
    }

    #[test]
    fn list_prefix() {
        let mut store = InMemoryStore::new();
        store.set("arr/.zarray", b"{}").unwrap();
        store.set("arr/c/0.0", b"chunk").unwrap();
        store.set("arr/c/1.0", b"chunk").unwrap();
        store.set("other/.zgroup", b"{}").unwrap();

        let mut keys = store.list("arr/").unwrap();
        keys.sort();
        assert_eq!(keys, &["arr/.zarray", "arr/c/0.0", "arr/c/1.0"]);
    }

    #[test]
    fn list_empty_prefix() {
        let store = InMemoryStore::new();
        assert!(store.list("anything/").unwrap().is_empty());
    }

    #[test]
    fn contains() {
        let mut store = InMemoryStore::new();
        store.set("key", b"val").unwrap();
        assert!(store.contains("key").unwrap());
        assert!(!store.contains("missing").unwrap());
    }

    #[test]
    fn from_entries() {
        let store =
            InMemoryStore::from_entries([("a".to_string(), vec![1]), ("b".to_string(), vec![2])]);
        assert_eq!(store.get("a").unwrap(), &[1]);
        assert_eq!(store.get("b").unwrap(), &[2]);
    }

    #[test]
    fn clear() {
        let mut store = InMemoryStore::new();
        store.set("k", b"v").unwrap();
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn clone_is_independent() {
        let mut store = InMemoryStore::new();
        store.set("k", b"v1").unwrap();
        let clone = store.clone();
        drop(store);
        // Clone still has the data
        assert_eq!(clone.get("k").unwrap(), b"v1");
    }
}
