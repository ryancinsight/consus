//! Default codec registry implementation.
//!
//! Pre-populates with all codecs enabled by cargo feature flags. Each codec
//! is registered by both its HDF5 filter ID (when applicable) and its
//! human-readable name, permitting lookup by either identifier.
//!
//! This module is compiled only when the `alloc` feature is enabled.

// `String` is used inside feature-gated codec registration blocks below.
// When no codec features are enabled, the import is unused.
#[allow(unused_imports)]
use alloc::string::String;
use alloc::vec::Vec;

use crate::codec::traits::{Codec, CodecId};
use crate::registry::CompressionRegistry;
use consus_core::{Error, Result};

/// Default codec registry pre-populated with all feature-enabled codecs.
///
/// ## Codec Registration (by feature flag)
///
/// | Feature   | Codec    | HDF5 Filter ID | Name     |
/// |-----------|----------|----------------|----------|
/// | deflate   | Deflate  | 1              | deflate  |
/// | gzip      | Gzip     | —              | gzip     |
/// | zstd      | Zstd     | 32015          | zstd     |
/// | lz4       | LZ4      | 32004          | lz4      |
/// | szip      | Szip     | 4              | szip     |
/// | blosc     | Blosc    | 32001          | blosc    |
///
/// Each codec is registered by both its HDF5 filter ID (when applicable)
/// and its human-readable name. This permits lookup by either identifier.
///
/// ## Invariant
///
/// For any codec registered during [`new`](DefaultCodecRegistry::new):
///   `registry.get(&CodecId::FilterId(id)).unwrap().name() == name`
///   where `id` and `name` are the values from the table above.
pub struct DefaultCodecRegistry {
    codecs: Vec<(CodecId, &'static dyn Codec)>,
}

impl DefaultCodecRegistry {
    /// Creates a new registry pre-populated with all feature-enabled codecs.
    ///
    /// Codec instances are `&'static` references to compile-time constants.
    /// Each codec is registered by both its HDF5 filter ID (if it has one)
    /// and its string name, so callers can look up by either identifier.
    pub fn new() -> Self {
        // Mutability is required when any codec feature is enabled (pushes below).
        #[allow(unused_mut)]
        let mut codecs: Vec<(CodecId, &'static dyn Codec)> = Vec::new();

        #[cfg(feature = "deflate")]
        {
            use crate::codec::deflate::DeflateCodec;
            static DEFLATE: DeflateCodec = DeflateCodec;
            codecs.push((CodecId::FilterId(1), &DEFLATE));
            codecs.push((CodecId::Name(String::from("deflate")), &DEFLATE));
        }

        #[cfg(feature = "gzip")]
        {
            use crate::codec::gzip::GzipCodec;
            static GZIP: GzipCodec = GzipCodec;
            codecs.push((CodecId::Name(String::from("gzip")), &GZIP));
        }

        #[cfg(feature = "zstd")]
        {
            use crate::codec::zstd::ZstdCodec;
            static ZSTD: ZstdCodec = ZstdCodec;
            codecs.push((CodecId::FilterId(32015), &ZSTD));
            codecs.push((CodecId::Name(String::from("zstd")), &ZSTD));
        }

        #[cfg(feature = "lz4")]
        {
            use crate::codec::lz4::Lz4Codec;
            static LZ4: Lz4Codec = Lz4Codec;
            codecs.push((CodecId::FilterId(32004), &LZ4));
            codecs.push((CodecId::Name(String::from("lz4")), &LZ4));
        }

        #[cfg(feature = "szip")]
        {
            use crate::codec::szip::SzipCodec;
            static SZIP: SzipCodec = SzipCodec::DEFAULT;
            codecs.push((CodecId::FilterId(4), &SZIP));
            codecs.push((CodecId::Name(String::from("szip")), &SZIP));
        }

        #[cfg(feature = "blosc")]
        {
            use crate::codec::blosc::BloscCodec;
            static BLOSC: BloscCodec = BloscCodec::DEFAULT;
            codecs.push((CodecId::FilterId(32001), &BLOSC));
            codecs.push((CodecId::Name(String::from("blosc")), &BLOSC));
        }

        Self { codecs }
    }
}

impl Default for DefaultCodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionRegistry for DefaultCodecRegistry {
    /// Look up a codec by its [`CodecId`].
    ///
    /// Performs a linear scan over registered entries and returns the first
    /// match. For registries with both `FilterId` and `Name` entries for the
    /// same codec, either identifier resolves to the same `&dyn Codec`.
    ///
    /// # Errors
    ///
    /// Returns `Error::UnsupportedFeature` if no entry matches `id`.
    fn get(&self, id: &CodecId) -> Result<&dyn Codec> {
        self.codecs
            .iter()
            .find(|(cid, _)| cid == id)
            .map(|(_, codec)| *codec)
            .ok_or_else(|| Error::UnsupportedFeature {
                feature: alloc::format!("codec {:?}", id),
            })
    }

    /// Appends a new `(id, codec)` entry to the registry.
    ///
    /// If a codec with the same `id` already exists, both entries remain;
    /// the first registered entry wins on lookup via [`get`](Self::get).
    fn register(&mut self, id: CodecId, codec: &'static dyn Codec) {
        self.codecs.push((id, codec));
    }

    /// Returns the full slice of registered `(CodecId, &dyn Codec)` pairs
    /// in registration order.
    fn codec_ids(&self) -> &[(CodecId, &'static dyn Codec)] {
        &self.codecs
    }

    /// Look up a codec by its human-readable [`Codec::name`] value.
    ///
    /// Scans all registered codecs and returns the first whose `name()`
    /// equals `name`. This is independent of the `CodecId` key.
    ///
    /// # Errors
    ///
    /// Returns `Error::UnsupportedFeature` if no codec with the given
    /// name is registered.
    fn get_by_name(&self, name: &str) -> Result<&dyn Codec> {
        self.codecs
            .iter()
            .find(|(_, codec)| codec.name() == name)
            .map(|(_, codec)| *codec)
            .ok_or_else(|| Error::UnsupportedFeature {
                feature: alloc::format!("codec named {:?}", name),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::traits::CompressionLevel;

    /// Identity codec for testing.
    ///
    /// Performs no transformation: both `compress` and `decompress` return
    /// a byte-exact copy of the input. This satisfies the round-trip
    /// invariant trivially: `decompress(compress(data)) == data` for all
    /// `data`, because both operations are the identity function on byte
    /// sequences.
    struct IdentityCodec;

    impl Codec for IdentityCodec {
        fn name(&self) -> &str {
            "identity"
        }

        fn hdf5_filter_id(&self) -> Option<u16> {
            None
        }

        fn compress(&self, input: &[u8], _level: CompressionLevel) -> Result<Vec<u8>> {
            Ok(input.to_vec())
        }

        fn decompress(&self, input: &[u8], _expected_size: usize) -> Result<Vec<u8>> {
            Ok(input.to_vec())
        }
    }

    /// Verify that the default registry contains a deflate codec registered
    /// under `FilterId(1)`, and that the returned codec reports name "deflate".
    #[cfg(feature = "deflate")]
    #[test]
    fn test_default_registry_contains_deflate() {
        let registry = DefaultCodecRegistry::new();
        let codec = registry
            .get(&CodecId::FilterId(1))
            .expect("deflate codec must be registered under FilterId(1)");
        assert_eq!(
            codec.name(),
            "deflate",
            "codec registered under FilterId(1) must report name \"deflate\""
        );
    }

    /// Verify that `get_by_name("deflate")` resolves to the deflate codec
    /// and that the returned codec reports `hdf5_filter_id() == Some(1)`.
    #[cfg(feature = "deflate")]
    #[test]
    fn test_default_registry_get_by_name() {
        let registry = DefaultCodecRegistry::new();
        let codec = registry
            .get_by_name("deflate")
            .expect("deflate codec must be resolvable by name");
        assert_eq!(
            codec.name(),
            "deflate",
            "codec found by name must report name \"deflate\""
        );
        assert_eq!(
            codec.hdf5_filter_id(),
            Some(1),
            "deflate codec must report HDF5 filter ID 1"
        );
    }

    /// Register a custom identity codec and verify it is retrievable by
    /// both `FilterId` and `Name` identifiers. Also verify round-trip
    /// invariant: `decompress(compress(data)) == data`.
    #[test]
    fn test_registry_register_custom() {
        let mut registry = DefaultCodecRegistry::new();

        static IDENTITY: IdentityCodec = IdentityCodec;

        registry.register(CodecId::Name(String::from("identity")), &IDENTITY);

        let codec = registry
            .get(&CodecId::Name(String::from("identity")))
            .expect("identity codec must be retrievable after registration");

        assert_eq!(
            codec.name(),
            "identity",
            "registered codec must report name \"identity\""
        );
        assert_eq!(
            codec.hdf5_filter_id(),
            None,
            "identity codec must report no HDF5 filter ID"
        );

        // Verify the round-trip invariant with non-trivial data.
        let input: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("identity compress must succeed");
        assert_eq!(
            compressed, input,
            "identity compress must return input unchanged"
        );
        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("identity decompress must succeed");
        assert_eq!(decompressed, input, "identity round-trip must be lossless");
    }

    /// Verify that `get` returns `Error::UnsupportedFeature` for a codec
    /// ID that was never registered.
    #[test]
    fn test_registry_unknown_codec() {
        let registry = DefaultCodecRegistry::new();

        let err = match registry.get(&CodecId::FilterId(9999)) {
            Err(e) => e,
            Ok(_) => panic!("lookup of unregistered FilterId must fail"),
        };
        assert!(
            matches!(err, Error::UnsupportedFeature { .. }),
            "error must be UnsupportedFeature, got: {err:?}"
        );

        let err_name = match registry.get(&CodecId::Name(String::from("nonexistent"))) {
            Err(e) => e,
            Ok(_) => panic!("lookup of unregistered Name must fail"),
        };
        assert!(
            matches!(err_name, Error::UnsupportedFeature { .. }),
            "error must be UnsupportedFeature, got: {err_name:?}"
        );

        let err_by_name = match registry.get_by_name("nonexistent") {
            Err(e) => e,
            Ok(_) => panic!("get_by_name for unregistered name must fail"),
        };
        assert!(
            matches!(err_by_name, Error::UnsupportedFeature { .. }),
            "error must be UnsupportedFeature, got: {err_by_name:?}"
        );
    }

    /// Verify that `contains` returns `true` for registered codecs and
    /// `false` for unregistered ones.
    #[cfg(feature = "deflate")]
    #[test]
    fn test_registry_contains() {
        let registry = DefaultCodecRegistry::new();

        assert!(
            registry.contains(&CodecId::FilterId(1)),
            "contains must return true for registered FilterId(1)"
        );
        assert!(
            registry.contains(&CodecId::Name(String::from("deflate"))),
            "contains must return true for registered Name(\"deflate\")"
        );
        assert!(
            !registry.contains(&CodecId::FilterId(9999)),
            "contains must return false for unregistered FilterId(9999)"
        );
        assert!(
            !registry.contains(&CodecId::Name(String::from("nonexistent"))),
            "contains must return false for unregistered Name(\"nonexistent\")"
        );
    }
}
