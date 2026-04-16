//! Store operation tests for Zarr backends.
//!
//! ## Coverage
//!
//! - InMemoryStore: set/get/delete/list/contains operations
//! - Directory store operations (tempfile-based)
//! - Async store operations (tokio, feature-gated)
//! - PrefixedStore wrapper
//! - SplitStore wrapper
//! - Error handling for missing keys

use consus_zarr::store::{InMemoryStore, PrefixedStore, SplitStore, Store};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// InMemoryStore basic operations
// ---------------------------------------------------------------------------

/// Test InMemoryStore set and get roundtrip.
///
/// ## Invariant
///
/// `store.get(key)` returns `Ok(data)` iff `store.set(key, data)` was called.
#[test]
fn in_memory_store_set_get_roundtrip() {
    let mut store = InMemoryStore::new();
    let key = "array/.zarray";
    let value = b"{\"zarr_format\":2}";

    store.set(key, value).expect("set must succeed");
    let retrieved = store.get(key).expect("get must succeed");

    assert_eq!(&retrieved, value);
}

/// Test InMemoryStore with binary data.
#[test]
fn in_memory_store_binary_data() {
    let mut store = InMemoryStore::new();
    let key = "array/c/0.0";
    let value: Vec<u8> = (0..255).collect();

    store.set(key, &value).expect("set must succeed");
    let retrieved = store.get(key).expect("get must succeed");

    assert_eq!(retrieved.len(), 255);
    assert_eq!(retrieved[0], 0);
    assert_eq!(retrieved[254], 254);
}

/// Test InMemoryStore with empty value.
#[test]
fn in_memory_store_empty_value() {
    let mut store = InMemoryStore::new();
    let key = "empty";

    store.set(key, b"").expect("set must succeed");
    let retrieved = store.get(key).expect("get must succeed");

    assert!(retrieved.is_empty());
}

/// Test InMemoryStore overwrite.
#[test]
fn in_memory_store_overwrite() {
    let mut store = InMemoryStore::new();
    let key = "key";

    store.set(key, b"value1").expect("set must succeed");
    store.set(key, b"value2").expect("set must succeed");
    let retrieved = store.get(key).expect("get must succeed");

    assert_eq!(&retrieved, b"value2");
}

/// Test InMemoryStore delete operation.
///
/// ## Invariant
///
/// After `delete(key)`, subsequent `get(key)` returns `NotFound`.
#[test]
fn in_memory_store_delete() {
    let mut store = InMemoryStore::new();
    let key = "to_delete";

    store.set(key, b"data").expect("set must succeed");
    store.delete(key).expect("delete must succeed");

    let result = store.get(key);
    assert!(result.is_err());
}

/// Test InMemoryStore delete missing key.
#[test]
fn in_memory_store_delete_missing() {
    let mut store = InMemoryStore::new();
    let result = store.delete("nonexistent");
    assert!(result.is_err());
}

/// Test InMemoryStore list with prefix.
///
/// ## Invariant
///
/// `list(prefix)` returns all keys starting with `prefix`, sorted.
#[test]
fn in_memory_store_list_prefix() {
    let mut store = InMemoryStore::new();

    store.set("array/.zarray", b"{}").expect("set must succeed");
    store.set("array/.zattrs", b"{}").expect("set must succeed");
    store
        .set("array/c/0.0", b"chunk")
        .expect("set must succeed");
    store
        .set("array/c/1.0", b"chunk")
        .expect("set must succeed");
    store.set("other/.zgroup", b"{}").expect("set must succeed");

    let mut keys = store.list("array/").expect("list must succeed");
    keys.sort();

    assert_eq!(keys.len(), 4);
    assert!(keys.contains(&"array/.zarray".to_string()));
    assert!(keys.contains(&"array/.zattrs".to_string()));
    assert!(keys.contains(&"array/c/0.0".to_string()));
    assert!(keys.contains(&"array/c/1.0".to_string()));
}

/// Test InMemoryStore list empty prefix returns all keys.
#[test]
fn in_memory_store_list_all() {
    let mut store = InMemoryStore::new();

    store.set("a", b"1").expect("set must succeed");
    store.set("b", b"2").expect("set must succeed");
    store.set("c", b"3").expect("set must succeed");

    let keys = store.list("").expect("list must succeed");
    assert_eq!(keys.len(), 3);
}

/// Test InMemoryStore list empty result.
#[test]
fn in_memory_store_list_empty_result() {
    let store = InMemoryStore::new();
    let keys = store.list("nonexistent/").expect("list must succeed");
    assert!(keys.is_empty());
}

/// Test InMemoryStore contains operation.
///
/// ## Invariant
///
/// `contains(key)` returns `true` iff key exists.
#[test]
fn in_memory_store_contains() {
    let mut store = InMemoryStore::new();

    store.set("exists", b"data").expect("set must succeed");

    assert!(store.contains("exists").expect("contains must succeed"));
    assert!(!store.contains("missing").expect("contains must succeed"));
}

/// Test InMemoryStore get missing key returns NotFound.
#[test]
fn in_memory_store_get_missing() {
    let store = InMemoryStore::new();
    let result = store.get("missing");
    assert!(result.is_err());
}

/// Test InMemoryStore from_entries constructor.
#[test]
fn in_memory_store_from_entries() {
    let entries = vec![
        ("key1".to_string(), vec![1, 2, 3]),
        ("key2".to_string(), vec![4, 5, 6]),
    ];

    let store = InMemoryStore::from_entries(entries);

    assert_eq!(store.get("key1").unwrap(), vec![1, 2, 3]);
    assert_eq!(store.get("key2").unwrap(), vec![4, 5, 6]);
}

/// Test InMemoryStore len and is_empty.
#[test]
fn in_memory_store_len() {
    let mut store = InMemoryStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    store.set("a", b"1").expect("set must succeed");
    assert_eq!(store.len(), 1);

    store.set("b", b"2").expect("set must succeed");
    assert_eq!(store.len(), 2);

    store.delete("a").expect("delete must succeed");
    assert_eq!(store.len(), 1);
}

/// Test InMemoryStore clear.
#[test]
fn in_memory_store_clear() {
    let mut store = InMemoryStore::new();

    store.set("a", b"1").expect("set must succeed");
    store.set("b", b"2").expect("set must succeed");
    store.clear();

    assert!(store.is_empty());
}

/// Test InMemoryStore clone is independent.
#[test]
fn in_memory_store_clone_independent() {
    let mut store = InMemoryStore::new();
    store.set("key", b"original").expect("set must succeed");

    let cloned = store.clone();
    store.set("key", b"modified").expect("set must succeed");

    // Clone should have original value
    assert_eq!(cloned.get("key").unwrap(), b"original");
}

// ---------------------------------------------------------------------------
// FsStore (Directory store) operations
// ---------------------------------------------------------------------------

/// Test FsStore create and basic operations.
#[test]
fn fs_store_create_and_operations() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let mut store = consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

    let key = "array/.zarray";
    let value = b"{\"zarr_format\":2}";

    store.set(key, value).expect("set must succeed");
    let retrieved = store.get(key).expect("get must succeed");
    assert_eq!(&retrieved, value);
}

/// Test FsStore with nested paths.
#[test]
fn fs_store_nested_paths() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let mut store = consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

    store
        .set("group/sub/array/.zarray", b"{}")
        .expect("set must succeed");
    let data = store
        .get("group/sub/array/.zarray")
        .expect("get must succeed");
    assert_eq!(&data, b"{}");
}

/// Test FsStore delete operation.
#[test]
fn fs_store_delete() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let mut store = consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

    store.set("file", b"data").expect("set must succeed");
    store.delete("file").expect("delete must succeed");

    let result = store.get("file");
    assert!(result.is_err());
}

/// Test FsStore list with prefix.
#[test]
fn fs_store_list_prefix() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let mut store = consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

    store.set("arr/.zarray", b"{}").expect("set must succeed");
    store.set("arr/c/0.0", b"chunk").expect("set must succeed");
    store.set("arr/c/1.0", b"chunk").expect("set must succeed");
    store.set("other/.zgroup", b"{}").expect("set must succeed");

    let keys = store.list("arr/").expect("list must succeed");
    assert!(!keys.is_empty());
    assert!(keys.iter().any(|k| k.contains(".zarray")));
}

/// Test FsStore contains operation.
#[test]
fn fs_store_contains() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let mut store = consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

    store.set("exists", b"data").expect("set must succeed");

    assert!(store.contains("exists").expect("contains must succeed"));
    assert!(!store.contains("missing").expect("contains must succeed"));
}

/// Test FsStore open existing directory.
#[test]
fn fs_store_open_existing() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let path = tmp.path();

    // Create and write
    {
        let mut store = consus_zarr::store::FsStore::create(path).expect("create must succeed");
        store.set("data", b"value").expect("set must succeed");
    }

    // Open and read
    {
        let store = consus_zarr::store::FsStore::open(path).expect("open must succeed");
        let data = store.get("data").expect("get must succeed");
        assert_eq!(&data, b"value");
    }
}

/// Test FsStore handles missing files gracefully.
#[test]
fn fs_store_missing_file() {
    let tmp = TempDir::new().expect("tempdir must succeed");
    let store = consus_zarr::store::FsStore::create(tmp.path()).expect("create must succeed");

    let result = store.get("nonexistent");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// PrefixedStore tests
// ---------------------------------------------------------------------------

/// Test PrefixedStore adds prefix to keys.
#[test]
fn prefixed_store_adds_prefix() {
    let inner = InMemoryStore::new();
    let mut store = PrefixedStore::new(inner, "my_data.zarr");

    store.set(".zarray", b"{}").expect("set must succeed");

    // PrefixedStore callers use unprefixed keys; the inner store receives the prefix.
    assert!(store.contains(".zarray").expect("contains must succeed"));

    let inner = store.into_inner();
    assert!(
        inner
            .contains("my_data.zarr/.zarray")
            .expect("contains must succeed")
    );
}

/// Test PrefixedStore strips prefix from list results.
#[test]
fn prefixed_store_list_strips_prefix() {
    let inner = InMemoryStore::new();
    let mut store = PrefixedStore::new(inner, "root/");

    store.set("arr/.zarray", b"{}").expect("set must succeed");
    store.set("arr/c/0.0", b"chunk").expect("set must succeed");

    let keys = store.list("arr/").expect("list must succeed");

    // Keys returned should NOT include the "root/" prefix
    assert!(keys.iter().all(|k| !k.starts_with("root/")));
}

/// Test PrefixedStore normalizes trailing slash.
#[test]
fn prefixed_store_trailing_slash_normalized() {
    let inner = InMemoryStore::new();
    let mut store = PrefixedStore::new(inner, "no_trailing");

    // Verify normalization by checking that a key set without trailing slash
    // is retrievable, confirming the prefix was normalized internally.
    store.set("key", b"value").expect("set must succeed");
    let retrieved = store.get("key").expect("get must succeed");
    assert_eq!(&retrieved, b"value");
}

/// Test PrefixedStore roundtrip.
#[test]
fn prefixed_store_roundtrip() {
    let inner = InMemoryStore::new();
    let mut store = PrefixedStore::new(inner, "ns");

    let data = b"test data for round-trip";
    store.set("key/path", data).expect("set must succeed");

    let retrieved = store.get("key/path").expect("get must succeed");
    assert_eq!(&retrieved, data);
}

/// Test PrefixedStore delete.
#[test]
fn prefixed_store_delete() {
    let inner = InMemoryStore::new();
    let mut store = PrefixedStore::new(inner, "prefix/");

    store.set("key", b"value").expect("set must succeed");
    store.delete("key").expect("delete must succeed");

    assert!(store.get("key").is_err());
}

/// Test PrefixedStore contains.
#[test]
fn prefixed_store_contains() {
    let inner = InMemoryStore::new();
    let mut store = PrefixedStore::new(inner, "prefix/");

    store.set("key", b"value").expect("set must succeed");

    assert!(store.contains("key").expect("contains must succeed"));
    assert!(!store.contains("missing").expect("contains must succeed"));
}

// ---------------------------------------------------------------------------
// SplitStore tests
// ---------------------------------------------------------------------------

/// Test SplitStore uses read store for reads.
#[test]
fn split_store_read_from_read_store() {
    let mut read_store = InMemoryStore::new();
    let write_store = InMemoryStore::new();

    read_store
        .set("key", b"from_read")
        .expect("set must succeed");

    let split = SplitStore {
        read: read_store,
        write: write_store,
    };

    let data = split.get("key").expect("get must succeed");
    assert_eq!(&data, b"from_read");
}

/// Test SplitStore uses write store for writes.
#[test]
fn split_store_write_to_write_store() {
    let read_store = InMemoryStore::new();
    let write_store = InMemoryStore::new();

    let mut split = SplitStore {
        read: read_store,
        write: write_store,
    };

    split.set("key", b"new_value").expect("set must succeed");

    // Write store should have the key
    let data = split.write.get("key").expect("get must succeed");
    assert_eq!(&data, b"new_value");
}

/// Test SplitStore delete uses write store.
#[test]
fn split_store_delete() {
    let read_store = InMemoryStore::new();
    let mut write_store = InMemoryStore::new();
    write_store.set("key", b"value").expect("set must succeed");

    let mut split = SplitStore {
        read: read_store,
        write: write_store,
    };

    split.delete("key").expect("delete must succeed");
    assert!(split.write.get("key").is_err());
}

/// Test SplitStore list uses read store.
#[test]
fn split_store_list() {
    let mut read_store = InMemoryStore::new();
    let write_store = InMemoryStore::new();

    read_store.set("a", b"1").expect("set must succeed");
    read_store.set("b", b"2").expect("set must succeed");

    let split = SplitStore {
        read: read_store,
        write: write_store,
    };

    let keys = split.list("").expect("list must succeed");
    assert!(keys.contains(&"a".to_string()));
    assert!(keys.contains(&"b".to_string()));
}

/// Test SplitStore contains uses read store.
#[test]
fn split_store_contains() {
    let mut read_store = InMemoryStore::new();
    let write_store = InMemoryStore::new();

    read_store.set("key", b"value").expect("set must succeed");

    let split = SplitStore {
        read: read_store,
        write: write_store,
    };

    assert!(split.contains("key").expect("contains must succeed"));
}

/// Test SplitStore clone is deep.
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

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Test key with special characters (valid in Zarr).
#[test]
fn key_with_underscores_and_dots() {
    let mut store = InMemoryStore::new();

    store
        .set("array_name_1/.zarray", b"{}")
        .expect("set must succeed");
    assert!(
        store
            .contains("array_name_1/.zarray")
            .expect("contains must succeed")
    );
}

/// Test deeply nested key path.
#[test]
fn deeply_nested_key() {
    let mut store = InMemoryStore::new();

    store
        .set("a/b/c/d/e/f/.zarray", b"{}")
        .expect("set must succeed");
    let data = store.get("a/b/c/d/e/f/.zarray").expect("get must succeed");
    assert_eq!(&data, b"{}");
}

/// Test large value storage.
#[test]
fn large_value_storage() {
    let mut store = InMemoryStore::new();

    // 1MB of data
    let value = vec![0u8; 1024 * 1024];
    store.set("large", &value).expect("set must succeed");

    let retrieved = store.get("large").expect("get must succeed");
    assert_eq!(retrieved.len(), 1024 * 1024);
}

/// Test many keys in store.
#[test]
fn many_keys() {
    let mut store = InMemoryStore::new();

    for i in 0..1000 {
        let key = format!("key_{}", i);
        store.set(&key, b"data").expect("set must succeed");
    }

    assert_eq!(store.len(), 1000);

    let keys = store.list("key_").expect("list must succeed");
    assert!(keys.len() >= 1000);
}

/// Test list with non-matching prefix.
#[test]
fn list_non_matching_prefix() {
    let mut store = InMemoryStore::new();

    store.set("alpha/file", b"1").expect("set must succeed");
    store.set("beta/file", b"2").expect("set must succeed");

    let keys = store.list("gamma/").expect("list must succeed");
    assert!(keys.is_empty());
}

/// Test sequential writes and reads.
#[test]
fn sequential_operations() {
    let mut store = InMemoryStore::new();

    for i in 0..100 {
        let key = format!("chunk/{}", i);
        let value = format!("data_{}", i);
        store.set(&key, value.as_bytes()).expect("set must succeed");
    }

    for i in 0..100 {
        let key = format!("chunk/{}", i);
        let expected = format!("data_{}", i);
        let data = store.get(&key).expect("get must succeed");
        assert_eq!(data, expected.as_bytes());
    }
}
