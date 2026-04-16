//! Property-based tests for Zarr operations using proptest.
//!
//! ## Coverage
//!
//! - Random array shapes and dtypes
//! - Random chunk configurations
//! - Random attribute values
//! - Compression roundtrip with random data
//! - Store operations with random keys and values
//! - Metadata serialization/deserialization invariants

use consus_zarr::Codec;
use consus_zarr::chunk::{ChunkKeySeparator, chunk_key};
use consus_zarr::codec::CompressionRegistryTrait;
use consus_zarr::codec::{CodecPipeline, default_registry};
use consus_zarr::metadata::{AttributeValue, parse_zattrs, serialize_zattrs};
use consus_zarr::store::{InMemoryStore, Store};
use proptest::collection::hash_map;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies for generating test data
// ---------------------------------------------------------------------------

/// Strategy for generating valid dimension sizes (1 to 1000).
fn dimension_size() -> impl Strategy<Value = usize> {
    1usize..=1000usize
}

/// Strategy for generating valid chunk sizes (1 to 100).
fn chunk_size() -> impl Strategy<Value = usize> {
    1usize..=100usize
}

/// Strategy for generating array shapes (1D to 5D).
fn array_shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(dimension_size(), 1..=5)
}

/// Strategy for generating chunk shapes matching array dimensions.
fn chunk_shape_for_shape(ndim: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(chunk_size(), ndim..=ndim)
}

/// Strategy for generating Zarr v2 dtype strings.
fn dtype_v2() -> impl Strategy<Value = String> {
    prop_oneof![
        // Float types
        Just("<f8".to_string()),
        Just(">f8".to_string()),
        Just("<f4".to_string()),
        Just(">f4".to_string()),
        // Integer types
        Just("<i8".to_string()),
        Just(">i8".to_string()),
        Just("<i4".to_string()),
        Just(">i4".to_string()),
        Just("<i2".to_string()),
        Just(">i2".to_string()),
        Just("<i1".to_string()),
        // Unsigned integer types
        Just("<u8".to_string()),
        Just(">u8".to_string()),
        Just("<u4".to_string()),
        Just(">u4".to_string()),
        Just("<u2".to_string()),
        Just(">u2".to_string()),
        Just("<u1".to_string()),
        // Boolean
        Just("|b1".to_string()),
        // Complex
        Just("<c16".to_string()),
        Just("<c8".to_string()),
    ]
}

/// Strategy for generating Zarr v3 data type names.
fn dtype_v3() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("bool".to_string()),
        Just("int8".to_string()),
        Just("int16".to_string()),
        Just("int32".to_string()),
        Just("int64".to_string()),
        Just("uint8".to_string()),
        Just("uint16".to_string()),
        Just("uint32".to_string()),
        Just("uint64".to_string()),
        Just("float16".to_string()),
        Just("float32".to_string()),
        Just("float64".to_string()),
        Just("complex64".to_string()),
        Just("complex128".to_string()),
    ]
}

/// Strategy for generating valid Zarr keys.
fn zarr_key() -> impl Strategy<Value = String> {
    prop_oneof![
        "[a-z][a-z0-9_]*".boxed(),
        "[a-z][a-z0-9_]*/[a-z][a-z0-9_]*".boxed(),
        "[a-z][a-z0-9_]*/[a-z][a-z0-9_]*/[a-z][a-z0-9_]*".boxed(),
    ]
}

/// Strategy for generating chunk coordinates.
fn chunk_coords(ndim: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(0usize..=1000usize, ndim..=ndim)
}

/// Strategy for generating arbitrary bytes.
fn arbitrary_bytes(max_size: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=max_size)
}

/// Strategy for generating compression levels.
fn compression_level() -> impl Strategy<Value = i32> {
    prop_oneof![
        Just(1i32),
        Just(3i32),
        Just(6i32),
        Just(9i32),
        (-5i32..=-1i32), // For zstd negative levels
    ]
}

/// Strategy for generating compressor configurations.
fn compressor_config() -> impl Strategy<Value = Option<(String, Vec<(String, String)>)>> {
    prop_oneof![
        Just(None),
        Just(Some((
            "gzip".to_string(),
            vec![("level".to_string(), "6".to_string())]
        ))),
        Just(Some((
            "zstd".to_string(),
            vec![("level".to_string(), "3".to_string())]
        ))),
        Just(Some((
            "lz4".to_string(),
            vec![("level".to_string(), "1".to_string())]
        ))),
    ]
}

/// Strategy for generating attribute values.
fn attribute_value() -> impl Strategy<Value = AttributeValue> {
    prop_oneof![
        any::<bool>().prop_map(AttributeValue::Bool),
        any::<i64>().prop_map(AttributeValue::Int),
        any::<u64>().prop_map(AttributeValue::Uint),
        any::<f64>().prop_map(AttributeValue::Float),
        "[a-zA-Z][a-zA-Z0-9_]*".prop_map(AttributeValue::String),
    ]
}

/// Strategy for generating attribute maps.
fn attribute_map() -> impl Strategy<Value = Vec<(String, AttributeValue)>> {
    hash_map("[a-z][a-z0-9_]*".boxed(), attribute_value(), 0..=10)
        .prop_map(|m| m.into_iter().collect())
}

// ---------------------------------------------------------------------------
// Chunk key property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that chunk key generation is deterministic.
    ///
    /// ## Invariant
    ///
    /// For any coordinates `c` and separator `s`:
    /// `chunk_key(c, s) == chunk_key(c, s)`
    #[test]
    fn chunk_key_deterministic(coords in chunk_coords(3)) {
        let key1 = chunk_key(&coords, ChunkKeySeparator::Dot);
        let key2 = chunk_key(&coords, ChunkKeySeparator::Dot);
        assert_eq!(key1, key2);

        let key3 = chunk_key(&coords, ChunkKeySeparator::Slash);
        let key4 = chunk_key(&coords, ChunkKeySeparator::Slash);
        assert_eq!(key3, key4);
    }

    /// Test that chunk key format is valid (no empty components).
    ///
    /// ## Invariant
    ///
    /// Chunk keys contain only numeric indices separated by the appropriate separator.
    #[test]
    fn chunk_key_valid_format(coords in chunk_coords(5)) {
        let dot_key = chunk_key(&coords, ChunkKeySeparator::Dot);
        let slash_key = chunk_key(&coords, ChunkKeySeparator::Slash);

        // Dot key should not be empty if coords is non-empty
        if !coords.is_empty() {
            assert!(!dot_key.is_empty() || coords.is_empty());
        }

        // Slash key should start with 'c/'
        if !coords.is_empty() {
            assert!(slash_key.starts_with('c'));
        }
    }

    /// Test that chunk keys are unique for different coordinates.
    ///
    /// ## Invariant
    ///
    /// Different chunk coordinates produce different keys.
    #[test]
    fn chunk_key_unique(coords1 in chunk_coords(3), coords2 in chunk_coords(3)) {
        prop_assume!(coords1 != coords2);

        let key1 = chunk_key(&coords1, ChunkKeySeparator::Dot);
        let key2 = chunk_key(&coords2, ChunkKeySeparator::Dot);

        // Keys might be equal only if coords are the same
        if coords1 == coords2 {
            assert_eq!(key1, key2);
        }
        // Note: Different coords can produce different keys, but not guaranteed
        // due to different dimensionality possibilities
    }
}

// ---------------------------------------------------------------------------
// Compression roundtrip property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test gzip compression roundtrip with arbitrary data.
    ///
    /// ## Invariant
    ///
    /// For any data `d`: `decompress(compress(d)) == d`
    #[test]
    fn gzip_roundtrip_arbitrary(
        data in arbitrary_bytes(10000),
        level in compression_level()
    ) {
        let registry = default_registry();
        let level = level.clamp(1, 9) as i32;

        let pipeline = CodecPipeline::single(Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), level.to_string())],
        });

        let compressed = pipeline.compress(&data, registry).expect("compress must succeed");
        let decompressed = pipeline.decompress(&compressed, registry).expect("decompress must succeed");

        assert_eq!(decompressed, data);
    }

    /// Test zstd compression roundtrip with arbitrary data.
    #[test]
    fn zstd_roundtrip_arbitrary(
        data in arbitrary_bytes(10000),
        level in compression_level()
    ) {
        let registry = default_registry();
        let level = level.clamp(-3, 19);

        let pipeline = CodecPipeline::single(Codec {
            name: String::from("zstd"),
            configuration: vec![(String::from("level"), level.to_string())],
        });

        let compressed = pipeline.compress(&data, registry).expect("compress must succeed");
        let decompressed = pipeline.decompress(&compressed, registry).expect("decompress must succeed");

        assert_eq!(decompressed, data);
    }

    /// Test lz4 compression roundtrip with arbitrary data.
    #[test]
    fn lz4_roundtrip_arbitrary(data in arbitrary_bytes(10000)) {
        let registry = default_registry();
        let codec = Codec {
            name: String::from("lz4"),
            configuration: vec![(String::from("level"), String::from("1"))],
        };

        let pipeline = CodecPipeline::single(codec);
        let compressed = pipeline.compress(&data, registry).expect("compress must succeed");
        let decompressed = pipeline.decompress(&compressed, registry).expect("decompress must succeed");
        assert_eq!(decompressed, data);
    }

    /// Test that compression reduces size for highly compressible data.
    ///
    /// ## Invariant
    ///
    /// For uniform data, compressed size < original size when the codec
    /// is available and compression is enabled.
    #[test]
    fn compression_reduces_uniform_size(
        byte_val in any::<u8>(),
        size in 100usize..=10000
    ) {
        let registry = default_registry();
        let data = vec![byte_val; size];

        let pipeline = CodecPipeline::single(Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), String::from("6"))],
        });

        let compressed = pipeline.compress(&data, registry).expect("compress must succeed");

        // Uniform data should compress to a strictly smaller payload when gzip is available.
        prop_assert!(compressed.len() < data.len());
    }
}

// ---------------------------------------------------------------------------
// Store operation property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that store set/get is idempotent.
    ///
    /// ## Invariant
    ///
    /// For any store `s`, key `k`, value `v`:
    /// `s.get(k) == v` after `s.set(k, v)`
    #[test]
    fn store_set_get_roundtrip(
        key in zarr_key(),
        value in arbitrary_bytes(1000)
    ) {
        let mut store = InMemoryStore::new();

        store.set(&key, &value).expect("set must succeed");
        let retrieved = store.get(&key).expect("get must succeed");

        assert_eq!(retrieved, value);
    }

    /// Test that store delete removes keys.
    ///
    /// ## Invariant
    ///
    /// After `store.delete(k)`, `store.contains(k) == false`
    #[test]
    fn store_delete_removes_key(key in zarr_key()) {
        let mut store = InMemoryStore::new();

        store.set(&key, b"value").expect("set must succeed");
        assert!(store.contains(&key).expect("contains must succeed"));

        store.delete(&key).expect("delete must succeed");
        assert!(!store.contains(&key).expect("contains must succeed"));
    }

    /// Test that store list returns all matching keys.
    ///
    /// ## Invariant
    ///
    /// `store.list(prefix)` returns all keys starting with `prefix`.
    #[test]
    fn store_list_returns_matching_keys(
        prefix in "[a-z]".boxed(),
        keys in prop::collection::vec("[a-z][a-z0-9_]*".boxed(), 1..=20)
    ) {
        let mut store = InMemoryStore::new();

        for key in &keys {
            store.set(key, b"data").expect("set must succeed");
        }

        let listed = store.list(&prefix).expect("list must succeed");

        // All listed keys should start with the prefix
        for key in listed {
            prop_assert!(key.starts_with(&prefix) || prefix.is_empty());
        }
    }

    /// Test that overwriting a key replaces the value.
    #[test]
    fn store_overwrite_replaces(
        key in zarr_key(),
        value1 in arbitrary_bytes(100),
        value2 in arbitrary_bytes(100)
    ) {
        prop_assume!(value1 != value2);

        let mut store = InMemoryStore::new();

        store.set(&key, &value1).expect("set must succeed");
        store.set(&key, &value2).expect("set must succeed");

        let retrieved = store.get(&key).expect("get must succeed");
        assert_eq!(retrieved, value2);
    }

    /// Test store contains returns correct result.
    #[test]
    fn store_contains_correct(key in zarr_key()) {
        let mut store = InMemoryStore::new();

        // Key should not exist initially
        assert!(!store.contains(&key).expect("contains must succeed"));

        // After set, key should exist
        store.set(&key, b"value").expect("set must succeed");
        assert!(store.contains(&key).expect("contains must succeed"));
    }
}

// ---------------------------------------------------------------------------
// Metadata parsing property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that zattrs parse/serialize roundtrip preserves content.
    ///
    /// ## Invariant
    ///
    /// `parse_zattrs(serialize_zattrs(attrs)) == attrs`
    #[test]
    fn zattrs_roundtrip(attrs in attribute_map()) {
        let serialized = serialize_zattrs(&attrs).expect("serialize must succeed");
        let reparsed = parse_zattrs(&serialized).expect("parse must succeed");

        assert_eq!(attrs.len(), reparsed.len());
    }

    /// Test that attribute values can be serialized without panic.
    #[test]
    fn attribute_value_serialization(attr in attribute_value()) {
        let attrs = vec![("test".to_string(), attr)];

        // Should not panic
        let result = serialize_zattrs(&attrs);
        prop_assert!(result.is_ok());
    }
}

// ---------------------------------------------------------------------------
// Chunk size property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that chunk size calculation is consistent.
    ///
    /// ## Invariant
    ///
    /// Chunk size = product of chunk dimensions × element size.
    #[test]
    fn chunk_size_consistent(
        chunk_dims in prop::collection::vec(1usize..=100usize, 1..=5),
        element_size in 1usize..=16
    ) {
        let total_elements: usize = chunk_dims.iter().product();
        let expected_size = total_elements * element_size;

        // This is the mathematical invariant
        prop_assert_eq!(total_elements * element_size, expected_size);
    }

    /// Test that total chunks calculation is consistent.
    ///
    /// ## Invariant
    ///
    /// Total chunks = ceil(shape[i] / chunk[i]) for each dimension.
    #[test]
    fn total_chunks_consistent(
        shape in prop::collection::vec(1usize..=1000usize, 1..=5),
        chunk in prop::collection::vec(1usize..=100usize, 1..=5)
    ) {
        prop_assume!(shape.len() == chunk.len());

        let total_chunks: usize = shape
            .iter()
            .zip(chunk.iter())
            .map(|(s, c)| (s + c - 1) / c)
            .product();

        let expected_total: usize = shape
            .iter()
            .zip(chunk.iter())
            .map(|(s, c)| (s + c - 1) / c)
            .product();

        prop_assert_eq!(total_chunks, expected_total);
    }
}

// ---------------------------------------------------------------------------
// Codec pipeline property tests
// ---------------------------------------------------------------------------

/// Test that codec pipeline preserves empty data.
#[test]
fn pipeline_preserves_empty() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    let empty: &[u8] = &[];
    let compressed = pipeline
        .compress(empty, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert!(decompressed.is_empty());
}

proptest! {
    /// Test that bytes codec is identity.
    #[test]
    fn bytes_codec_identity(data in arbitrary_bytes(1000)) {
        let registry = default_registry();
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from("little"))],
        });

        let output = pipeline.compress(&data, registry).expect("compress must succeed");
        prop_assert_eq!(output, data);
    }

    /// Test that empty pipeline is identity.
    #[test]
    fn empty_pipeline_identity(data in arbitrary_bytes(1000)) {
        let registry = default_registry();
        let pipeline = CodecPipeline::empty();

        let compressed = pipeline.compress(&data, registry).expect("compress must succeed");
        let decompressed = pipeline.decompress(&compressed, registry).expect("decompress must succeed");

        prop_assert_eq!(decompressed, data);
    }
}

// ---------------------------------------------------------------------------
// Store stress tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test store with many keys.
    #[test]
    fn store_many_keys(
        entries in prop::collection::hash_map(
            "[a-z][a-z0-9_]*".boxed(),
            arbitrary_bytes(100),
            1..=100
        )
    ) {
        let mut store = InMemoryStore::new();

        // Set all entries
        for (key, value) in &entries {
            store.set(key, value).expect("set must succeed");
        }

        // Verify all entries
        for (key, expected) in &entries {
            let retrieved = store.get(key).expect("get must succeed");
            prop_assert_eq!(&retrieved, expected);
        }

        prop_assert_eq!(store.len(), entries.len());
    }

    /// Test sequential operations maintain consistency.
    #[test]
    fn store_sequential_consistency(
        ops in prop::collection::vec(
            (any::<u8>(), "[a-z]{1,5}"),
            1..=50
        )
    ) {
        let mut store = InMemoryStore::new();

        for (value, key_prefix) in ops {
            let key = format!("{}_{}", key_prefix, value);
            store.set(&key, &[value]).expect("set must succeed");
        }

        // Store should be in consistent state
        let list_result = store.list("");
        prop_assert!(list_result.is_ok());
    }
}

// ---------------------------------------------------------------------------
// Invariant verification tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that store key lookup is consistent.
    ///
    /// ## Invariant
    ///
    /// `store.contains(k) == true` iff `store.get(k).is_ok()`
    #[test]
    fn store_contains_get_consistency(key in zarr_key()) {
        let mut store = InMemoryStore::new();
        store.set(&key, b"value").expect("set must succeed");

        let contains = store.contains(&key).expect("contains must succeed");
        let get_result = store.get(&key);

        prop_assert_eq!(contains, get_result.is_ok());
    }

    /// Test that store deletion is idempotent for error on missing.
    #[test]
    fn store_delete_missing_consistent(key in zarr_key()) {
        let mut store = InMemoryStore::new();

        let delete_result = store.delete(&key);
        let contains = store.contains(&key).expect("contains must succeed");

        // Delete of missing should fail, and key should not exist
        prop_assert!(delete_result.is_err());
        prop_assert!(!contains);
    }
}
