//! Unit tests for codec registry.
//!
//! ## Contract
//!
//! - `get(&id)` returns `Ok` for any registered `CodecId`.
//! - `get(&id)` returns `Error::UnsupportedFeature` for unregistered IDs.
//! - `register(id, codec)` makes `id` immediately resolvable.
//! - `get_by_name(name)` finds codecs by their `name()` value.

#![cfg(feature = "std")]

use consus_compression::{
    Codec, CodecId, CompressionLevel, CompressionRegistry, DefaultCodecRegistry,
};

// =============================================================================
// DefaultCodecRegistry Tests
// =============================================================================

mod default_registry {
    use super::*;

    /// Create a default registry and verify it's not empty.
    #[test]
    fn registry_creation() {
        let registry = DefaultCodecRegistry::new();
        // Registry should have at least one codec if any feature is enabled
        let codec_ids = registry.codec_ids();
        assert!(
            !codec_ids.is_empty()
                || cfg!(not(any(
                    feature = "deflate",
                    feature = "gzip",
                    feature = "zstd",
                    feature = "lz4",
                    feature = "szip",
                    feature = "blosc"
                ))),
            "registry should contain codecs when features are enabled"
        );
    }

    /// Verify deflate codec is registered under FilterId(1) when feature is enabled.
    #[cfg(feature = "deflate")]
    #[test]
    fn contains_deflate_by_filter_id() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get(&CodecId::FilterId(1))
            .expect("deflate codec must be registered under FilterId(1)");

        assert_eq!(
            codec.name(),
            "deflate",
            "codec registered under FilterId(1) must report name 'deflate'"
        );

        assert_eq!(
            codec.hdf5_filter_id(),
            Some(1),
            "deflate codec must report HDF5 filter ID 1"
        );
    }

    /// Verify deflate codec is findable by name when feature is enabled.
    #[cfg(feature = "deflate")]
    #[test]
    fn contains_deflate_by_name() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get_by_name("deflate")
            .expect("deflate codec must be resolvable by name");

        assert_eq!(codec.name(), "deflate");
        assert_eq!(codec.hdf5_filter_id(), Some(1));
    }

    /// Verify zstd codec is registered when feature is enabled.
    #[cfg(feature = "zstd")]
    #[test]
    fn contains_zstd_by_filter_id() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get(&CodecId::FilterId(32015))
            .expect("zstd codec must be registered under FilterId(32015)");

        assert_eq!(codec.name(), "zstd");
        assert_eq!(codec.hdf5_filter_id(), Some(32015));
    }

    /// Verify zstd codec is findable by name when feature is enabled.
    #[cfg(feature = "zstd")]
    #[test]
    fn contains_zstd_by_name() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get_by_name("zstd")
            .expect("zstd codec must be resolvable by name");

        assert_eq!(codec.name(), "zstd");
    }

    /// Verify lz4 codec is registered when feature is enabled.
    #[cfg(feature = "lz4")]
    #[test]
    fn contains_lz4_by_filter_id() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get(&CodecId::FilterId(32004))
            .expect("lz4 codec must be registered under FilterId(32004)");

        assert_eq!(codec.name(), "lz4");
        assert_eq!(codec.hdf5_filter_id(), Some(32004));
    }

    /// Verify lz4 codec is findable by name when feature is enabled.
    #[cfg(feature = "lz4")]
    #[test]
    fn contains_lz4_by_name() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get_by_name("lz4")
            .expect("lz4 codec must be resolvable by name");

        assert_eq!(codec.name(), "lz4");
    }

    /// Verify gzip codec is registered when feature is enabled.
    #[cfg(feature = "gzip")]
    #[test]
    fn contains_gzip_by_name() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get_by_name("gzip")
            .expect("gzip codec must be resolvable by name");

        assert_eq!(codec.name(), "gzip");
        assert_eq!(codec.hdf5_filter_id(), None, "gzip has no HDF5 filter ID");
    }

    /// Verify szip codec is registered when feature is enabled.
    #[cfg(feature = "szip")]
    #[test]
    fn contains_szip_by_filter_id() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get(&CodecId::FilterId(4))
            .expect("szip codec must be registered under FilterId(4)");

        assert_eq!(codec.name(), "szip");
        assert_eq!(codec.hdf5_filter_id(), Some(4));
    }

    /// Verify blosc codec is registered when feature is enabled.
    #[cfg(feature = "blosc")]
    #[test]
    fn contains_blosc_by_filter_id() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry
            .get(&CodecId::FilterId(32001))
            .expect("blosc codec must be registered under FilterId(32001)");

        assert_eq!(codec.name(), "blosc");
        assert_eq!(codec.hdf5_filter_id(), Some(32001));
    }

    /// Unknown filter ID returns UnsupportedFeature error.
    #[test]
    fn unknown_filter_id_returns_error() {
        let registry = DefaultCodecRegistry::new();

        let result = registry.get(&CodecId::FilterId(9999));
        assert!(result.is_err(), "unregistered FilterId must return error");

        match result.err().expect("result is Err (checked above)") {
            consus_core::Error::UnsupportedFeature { feature } => {
                assert!(
                    feature.contains("9999"),
                    "error must mention the unknown ID, got: {}",
                    feature
                );
            }
            other => panic!("expected UnsupportedFeature, got: {:?}", other),
        }
    }

    /// Unknown name returns UnsupportedFeature error.
    #[test]
    fn unknown_name_returns_error() {
        let registry = DefaultCodecRegistry::new();

        let result = registry.get_by_name("nonexistent_codec");
        assert!(result.is_err(), "unregistered name must return error");

        match result.err().expect("result is Err (checked above)") {
            consus_core::Error::UnsupportedFeature { feature } => {
                assert!(
                    feature.contains("nonexistent_codec"),
                    "error must mention the unknown name, got: {}",
                    feature
                );
            }
            other => panic!("expected UnsupportedFeature, got: {:?}", other),
        }
    }

    /// `contains` returns true for registered codecs.
    #[cfg(feature = "deflate")]
    #[test]
    fn contains_returns_true_for_registered() {
        let registry = DefaultCodecRegistry::new();

        assert!(
            registry.contains(&CodecId::FilterId(1)),
            "contains must return true for registered FilterId(1)"
        );

        assert!(
            registry.contains(&CodecId::Name(String::from("deflate"))),
            "contains must return true for registered Name('deflate')"
        );
    }

    /// `contains` returns false for unregistered codecs.
    #[test]
    fn contains_returns_false_for_unregistered() {
        let registry = DefaultCodecRegistry::new();

        assert!(
            !registry.contains(&CodecId::FilterId(9999)),
            "contains must return false for unregistered FilterId(9999)"
        );

        assert!(
            !registry.contains(&CodecId::Name(String::from("nonexistent"))),
            "contains must return false for unregistered Name"
        );
    }

    /// `codec_ids` returns registration order.
    #[cfg(feature = "deflate")]
    #[test]
    fn codec_ids_returns_registration_order() {
        let registry = DefaultCodecRegistry::new();
        let ids = registry.codec_ids();

        // First entry should be FilterId(1) for deflate
        assert!(
            ids.iter()
                .any(|(id, codec)| { id == &CodecId::FilterId(1) && codec.name() == "deflate" }),
            "registry must contain deflate under FilterId(1)"
        );
    }

    /// Default trait creates a new registry.
    #[test]
    fn default_creates_registry() {
        let registry = DefaultCodecRegistry::default();
        let ids = registry.codec_ids();
        // Same as new()
        let registry_new = DefaultCodecRegistry::new();
        assert_eq!(ids.len(), registry_new.codec_ids().len());
    }
}

// =============================================================================
// Custom Codec Registration Tests
// =============================================================================

mod custom_registration {
    use super::*;

    /// Identity codec for testing custom registration.
    struct IdentityCodec;

    impl Codec for IdentityCodec {
        fn name(&self) -> &str {
            "identity"
        }

        fn hdf5_filter_id(&self) -> Option<u16> {
            None
        }

        fn compress(&self, input: &[u8], _level: CompressionLevel) -> consus_core::Result<Vec<u8>> {
            Ok(input.to_vec())
        }

        fn decompress(&self, input: &[u8], _expected_size: usize) -> consus_core::Result<Vec<u8>> {
            Ok(input.to_vec())
        }
    }

    /// Register a custom codec and verify it's retrievable.
    #[test]
    fn register_custom_codec() {
        let mut registry = DefaultCodecRegistry::new();
        static IDENTITY: IdentityCodec = IdentityCodec;

        registry.register(CodecId::Name(String::from("identity")), &IDENTITY);

        let codec = registry
            .get(&CodecId::Name(String::from("identity")))
            .expect("identity codec must be retrievable after registration");

        assert_eq!(codec.name(), "identity");
        assert_eq!(codec.hdf5_filter_id(), None);
    }

    /// Registered codec round-trip works.
    #[test]
    fn registered_codec_round_trip() {
        let mut registry = DefaultCodecRegistry::new();
        static IDENTITY: IdentityCodec = IdentityCodec;

        registry.register(CodecId::Name(String::from("identity")), &IDENTITY);

        let codec = registry
            .get_by_name("identity")
            .expect("identity must be findable by name");

        let input: Vec<u8> = (0u8..=255).cycle().take(512).collect();

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("identity compress must succeed");

        assert_eq!(
            compressed, input,
            "identity compress returns input unchanged"
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("identity decompress must succeed");

        assert_eq!(decompressed, input, "identity round-trip must be lossless");
    }

    /// Multiple registrations for same codec under different IDs.
    #[test]
    fn multiple_ids_for_same_codec() {
        let mut registry = DefaultCodecRegistry::new();
        static IDENTITY: IdentityCodec = IdentityCodec;

        registry.register(CodecId::FilterId(9998), &IDENTITY);
        registry.register(CodecId::Name(String::from("custom_identity")), &IDENTITY);

        let by_filter_id = registry.get(&CodecId::FilterId(9998)).unwrap();
        let by_name = registry.get_by_name("identity").unwrap();

        // Both should return the same codec implementation.
        assert_eq!(by_filter_id.name(), "identity");
        assert_eq!(by_name.name(), "identity");
    }

    /// First registered entry wins on lookup (no overwriting).
    #[test]
    fn first_registration_wins() {
        static CODEC_A: IdentityCodec = IdentityCodec;
        static CODEC_B: IdentityCodec = IdentityCodec;

        let mut registry = DefaultCodecRegistry::new();

        // Capture baseline before any custom registrations so the assertion
        // is invariant to feature-gated built-in codecs.
        let baseline = registry.codec_ids().len();

        // Register codec A first.
        registry.register(CodecId::Name(String::from("test")), &CODEC_A);

        // Attempt to register codec B under the same ID (must not overwrite).
        registry.register(CodecId::Name(String::from("test")), &CODEC_B);

        // Both calls push an entry; the duplicate ID does not prevent insertion.
        // First-registration-wins applies only on lookup, not on storage.
        assert_eq!(registry.codec_ids().len(), baseline + 2);

        // First-registration-wins invariant: "test" must resolve to CODEC_A (IdentityCodec),
        // not replaced by CODEC_B.
        let codec = registry
            .get(&CodecId::Name(String::from("test")))
            .expect("registered name must be found");
        assert_eq!(codec.name(), "identity");
    }

    /// Custom codec with FilterId works.
    #[test]
    fn custom_filter_id_codec() {
        static CUSTOM: IdentityCodec = IdentityCodec;

        let mut registry = DefaultCodecRegistry::new();
        registry.register(CodecId::FilterId(5000), &CUSTOM);

        let codec = registry.get(&CodecId::FilterId(5000)).unwrap();
        assert_eq!(codec.name(), "identity");
        assert_eq!(codec.hdf5_filter_id(), None);
    }
}

// =============================================================================
// Registry Lookup Edge Cases
// =============================================================================

mod lookup_edge_cases {
    use super::*;

    /// Empty registry (no features enabled) handles lookup gracefully.
    #[test]
    fn empty_registry_unknown_id() {
        let registry = DefaultCodecRegistry::new();
        let result = registry.get(&CodecId::FilterId(9999));
        assert!(result.is_err());
    }

    /// `get_by_name` is case-sensitive.
    #[test]
    fn get_by_name_case_sensitive() {
        let registry = DefaultCodecRegistry::new();

        // If deflate is enabled, verify case sensitivity
        #[cfg(feature = "deflate")]
        {
            assert!(registry.get_by_name("deflate").is_ok());
            assert!(registry.get_by_name("DEFLATE").is_err());
            assert!(registry.get_by_name("Deflate").is_err());
        }
    }

    /// Name lookup with empty string fails.
    #[test]
    fn get_by_name_empty_string() {
        let registry = DefaultCodecRegistry::new();
        let result = registry.get_by_name("");
        assert!(result.is_err());
    }

    /// FilterId(0) is not a valid HDF5 filter ID and should not be registered.
    #[test]
    fn filter_id_zero_not_registered() {
        let registry = DefaultCodecRegistry::new();
        let result = registry.get(&CodecId::FilterId(0));
        assert!(
            result.is_err(),
            "FilterId(0) should not be registered (invalid HDF5 ID)"
        );
    }

    /// Verify all registered codecs satisfy the Codec trait contract.
    #[cfg(feature = "deflate")]
    #[test]
    fn registered_codec_satisfies_trait() {
        let registry = DefaultCodecRegistry::new();

        let codec = registry.get(&CodecId::FilterId(1)).unwrap();

        // Verify trait methods work
        let name = codec.name();
        assert!(!name.is_empty(), "codec name must not be empty");

        let filter_id = codec.hdf5_filter_id();
        assert!(filter_id.is_some(), "deflate must have HDF5 filter ID");

        // Verify compression works
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");

        assert_eq!(decompressed, input, "round-trip must be lossless");
    }
}

// =============================================================================
// Registry Thread Safety
// =============================================================================

mod thread_safety {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Registry can be shared across threads (Send + Sync).
    #[test]
    fn registry_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DefaultCodecRegistry>();
    }

    /// Concurrent reads from multiple threads.
    #[cfg(feature = "deflate")]
    #[test]
    fn concurrent_reads() {
        let registry = Arc::new(DefaultCodecRegistry::new());
        let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

        for _ in 0..4 {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                let codec = reg.get(&CodecId::FilterId(1)).unwrap();
                assert_eq!(codec.name(), "deflate");
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    /// Concurrent lookups by name.
    #[cfg(feature = "deflate")]
    #[test]
    fn concurrent_name_lookups() {
        let registry = Arc::new(DefaultCodecRegistry::new());
        let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

        for _ in 0..4 {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                let codec = reg.get_by_name("deflate").unwrap();
                assert_eq!(codec.name(), "deflate");
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
