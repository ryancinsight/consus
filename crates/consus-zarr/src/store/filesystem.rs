//! Local filesystem store backend for Zarr.
//!
//! Stores Zarr hierarchy as a directory tree on the local filesystem.
//! Each key maps to a file path relative to the store root.
//!
//! ## Directory Structure
//!
//! ```text
//! /path/to/root.zarr/
//! ├── .zarray
//! ├── .zgroup
//! ├── .zattrs
//! ├── c/
//! │   ├── 0.0.0
//! │   └── ...
//! └── sub_group/
//!     ├── .zgroup
//!     └── ...
//! ```
//!
//! ## Thread Safety
//!
//! Concurrent reads are safe. Concurrent writes to the same key require
//! external synchronization (e.g., a filesystem lock or per-key mutex).
//! Concurrent writes to different keys are safe.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::fs;
#[cfg(feature = "std")]
use std::io::{Read, Write};
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::Result;

#[cfg(feature = "alloc")]
use crate::store::Store;

/// A Zarr store backed by a local filesystem directory.
///
/// Each key maps to a file relative to `root`. Keys use '/' as the
/// separator (Zarr convention) and are mapped to OS-native paths.
///
/// ## Invariants
///
/// - `root` must be a directory that exists when the store is created.
/// - All keys are relative paths; no absolute paths or `..` escapes.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct FsStore {
    /// Root directory of the store.
    root: PathBuf,
}

#[cfg(feature = "alloc")]
impl FsStore {
    /// Open an existing directory as a Zarr store.
    ///
    /// Returns an error if `path` does not exist or is not a directory.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = path.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("not a directory: {}", root.display()),
            ));
        }
        Ok(Self { root })
    }

    /// Create a new directory store at the given path.
    ///
    /// Creates the directory and all intermediate parents if they do not exist.
    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Returns the root path of this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Convert a Zarr key to an absolute filesystem path.
    fn key_to_path(&self, key: &str) -> PathBuf {
        // Zarr keys use '/' separators; map to OS-native separator
        let native_key = key.replace('/', std::path::MAIN_SEPARATOR_STR);
        self.root.join(native_key)
    }
}

#[cfg(feature = "alloc")]
impl Store for FsStore {
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.key_to_path(key);
        let mut file = fs::File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                consus_core::Error::NotFound {
                    path: key.to_string(),
                }
            } else {
                consus_core::Error::Io(e)
            }
        })?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(consus_core::Error::Io)?;
        Ok(buf)
    }

    fn set(&mut self, key: &str, value: &[u8]) -> Result<()> {
        let path = self.key_to_path(key);
        // Create parent directories if they do not exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(consus_core::Error::Io)?;
        }
        let mut file = fs::File::create(&path).map_err(consus_core::Error::Io)?;
        file.write_all(value).map_err(consus_core::Error::Io)?;
        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        let path = self.key_to_path(key);
        fs::remove_file(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                consus_core::Error::NotFound {
                    path: key.to_string(),
                }
            } else {
                consus_core::Error::Io(e)
            }
        })?;
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut results = Vec::new();
        let dir_path = if prefix.is_empty() {
            self.root.clone()
        } else {
            self.key_to_path(prefix)
        };

        // If prefix maps to a file (not a directory), return just that key
        if prefix.is_empty() || dir_path.is_dir() {
            Self::collect_files_recursive(&self.root, &dir_path, prefix, &mut results);
            // Deduplicate and sort
            results.sort();
            results.dedup();
        } else if dir_path.is_file() {
            results.push(prefix.to_string());
        }

        Ok(results)
    }

    fn contains(&self, key: &str) -> Result<bool> {
        let path = self.key_to_path(key);
        Ok(path.is_file())
    }
}

#[cfg(feature = "alloc")]
impl FsStore {
    /// Recursively collect all file paths under `dir` that start with `prefix`.
    fn collect_files_recursive(root: &Path, dir: &Path, prefix: &str, out: &mut Vec<String>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Recurse into subdirectory
                Self::collect_files_recursive(root, &path, prefix, out);
            } else if path.is_file() {
                // Compute the Zarr key for this file
                let relative = path.strip_prefix(root).unwrap_or(&path);
                let key: String = relative
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push(key);
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl Clone for FsStore {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn create_store() {
        let tmp = TempDir::new().unwrap();
        let mut store = FsStore::create(tmp.path()).unwrap();
        store.set("array/.zarray", b"{}").unwrap();
        assert!(store.contains("array/.zarray").unwrap());
    }

    #[test]
    fn get_missing() {
        let tmp = TempDir::new().unwrap();
        let store = FsStore::create(tmp.path()).unwrap();
        let err = store.get("missing").unwrap_err();
        assert!(matches!(err, consus_core::Error::NotFound { .. }));
    }

    #[test]
    fn delete_existing() {
        let tmp = TempDir::new().unwrap();
        let mut store = FsStore::create(tmp.path()).unwrap();
        store.set("file.txt", b"data").unwrap();
        store.delete("file.txt").unwrap();
        assert!(!store.contains("file.txt").unwrap());
    }

    #[test]
    fn list_empty() {
        let tmp = TempDir::new().unwrap();
        let store = FsStore::create(tmp.path()).unwrap();
        assert!(store.list("anything/").unwrap().is_empty());
    }

    #[test]
    fn roundtrip_nested() {
        let tmp = TempDir::new().unwrap();
        let mut store = FsStore::create(tmp.path()).unwrap();
        store
            .set("group/sub/.zarray", br#"{"zarr_format":2,"shape":[5],"chunks":[5],"dtype":"<f8","fill_value":0.0,"order":"C","compressor":null,"filters":null}"#)
            .unwrap();
        let data = store.get("group/sub/.zarray").unwrap();
        assert!(data.starts_with(b"{\"zarr_format"));
    }

    #[test]
    fn list_prefix() {
        let tmp = TempDir::new().unwrap();
        let mut store = FsStore::create(tmp.path()).unwrap();
        store.set("arr/.zarray", b"{}").unwrap();
        store.set("arr/c/0.0", b"chunk").unwrap();
        store.set("arr/c/1.0", b"chunk").unwrap();

        let mut keys = store.list("arr/").unwrap();
        keys.sort();
        assert!(keys.iter().any(|k| k == "arr/.zarray"));
    }

    #[test]
    fn clone_is_independent() {
        let tmp = TempDir::new().unwrap();
        let store = FsStore::create(tmp.path()).unwrap();
        drop(store);
        // Files persist after store is dropped
        let store2 = FsStore::open(tmp.path()).unwrap();
        assert!(store2.contains("missing").is_ok());
    }
}
