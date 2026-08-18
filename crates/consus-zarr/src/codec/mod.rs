//! Codec pipeline execution for Zarr chunks.
//!
//! ## Zarr v3 Codec Chain
//!
//! Zarr v3 stores a codec chain in `zarr.json`. Each codec in the chain
//! is applied in order when writing and reversed when reading. The chain
//! always starts with a bytes-level codec (for endianness) and may include
//! compression/decompression codecs.
//!
//! ## Codec Order
//!
//! For reading (decompress): chain is applied in reverse order.
//! For writing (compress): chain is applied in forward order.
//!
//! ## Supported Codecs
//!
//! | Name | Direction | Description |
//! |------|-----------|-------------|
//! | "bytes" | Both | Raw byte transport; identity when the stored byte order matches the host, `UnsupportedFeature` otherwise |
//! | "crc32" | — | Refused in both directions: Zarr v3 specifies crc32c, and only CRC-32/IEEE is implemented here |
//! | "gzip" | Both | Gzip compression |
//! | "zstd" | Both | Zstandard compression |
//! | "lz4" | Both | LZ4 block compression |
//! | "blosc" | Both | Blosc meta-compressor |

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use alloc::string::ToString;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use consus_core::Result;

#[cfg(feature = "alloc")]
use crate::metadata::Codec;

// ---------------------------------------------------------------------------
// Compression level
// ---------------------------------------------------------------------------

/// Compression level hint passed to codecs.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy)]
pub struct CompressionLevel(pub i32);

#[cfg(feature = "alloc")]
impl Default for CompressionLevel {
    fn default() -> Self {
        Self(6)
    }
}

// ---------------------------------------------------------------------------
// Codec pipeline
// ---------------------------------------------------------------------------

/// A codec pipeline that applies a chain of codecs in sequence.
///
/// The pipeline maintains a registry reference for looking up codec
/// implementations by name. Codecs are applied in forward order for
/// compression and in reverse order for decompression.
///
/// ## Invariant
///
/// For any pipeline `p` and input data `d`:
/// `p.decompress(p.compress(d)?)? == d`
///
/// This is guaranteed when each registered codec satisfies the round-trip
/// invariant individually.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct CodecPipeline {
    /// Ordered list of codec configurations.
    codecs: Vec<Codec>,
}

#[cfg(feature = "alloc")]
impl CodecPipeline {
    /// Create a new pipeline from an ordered list of codec configurations.
    ///
    /// The codecs are applied in the order given for compression.
    pub fn new(codecs: Vec<Codec>) -> Self {
        Self { codecs }
    }

    /// Create a pipeline from a single codec.
    pub fn single(codec: Codec) -> Self {
        Self {
            codecs: vec![codec],
        }
    }

    /// Create an empty (identity) pipeline.
    pub fn empty() -> Self {
        Self { codecs: Vec::new() }
    }

    /// Returns the number of codecs in this pipeline.
    pub fn len(&self) -> usize {
        self.codecs.len()
    }

    /// Returns true if this pipeline has no codecs.
    pub fn is_empty(&self) -> bool {
        self.codecs.is_empty()
    }

    /// Returns a slice of the codecs in this pipeline.
    pub fn codecs(&self) -> &[Codec] {
        &self.codecs
    }

    /// Compress data through the full codec chain.
    ///
    /// Codecs are applied in forward order: the first codec receives
    /// the raw chunk bytes, its output is passed to the second, and so on.
    ///
    /// Returns an error if any codec in the chain fails.
    pub fn compress(
        &self,
        data: &[u8],
        registry: &dyn CompressionRegistryTrait,
    ) -> Result<Vec<u8>> {
        let mut current = data.to_vec();
        for codec in &self.codecs {
            let encoded = self.apply_compress(codec, &current, registry)?;
            current = encoded;
        }
        Ok(current)
    }

    /// Decompress data through the full codec chain in reverse order.
    ///
    /// Codecs are applied in reverse: the last codec in the chain receives
    /// the compressed bytes first, its output is passed to the second-to-last,
    /// and so on until the raw chunk bytes are produced.
    ///
    /// Returns an error if any codec in the chain fails.
    pub fn decompress(
        &self,
        data: &[u8],
        registry: &dyn CompressionRegistryTrait,
    ) -> Result<Vec<u8>> {
        let mut current = data.to_vec();
        for codec in self.codecs.iter().rev() {
            let decoded = self.apply_decompress(codec, &current, registry)?;
            current = decoded;
        }
        Ok(current)
    }

    /// Apply a single codec in the compress direction.
    fn apply_compress(
        &self,
        codec: &Codec,
        data: &[u8],
        registry: &dyn CompressionRegistryTrait,
    ) -> Result<Vec<u8>> {
        match codec.name.as_str() {
            "bytes" => bytes_codec_passthrough(codec, data),

            "crc32" => Err(unsupported_crc32()),

            // Compression codecs
            "gzip" | "zlib" => {
                let level = codec.gzip_level().unwrap_or(6) as i32;
                registry
                    .get_by_name(&codec.name)
                    .and_then(|c| c.compress(data, CompressionLevel(level)))
            }
            "zstd" => {
                let level = codec.zstd_level().unwrap_or(3);
                registry
                    .get_by_name(&codec.name)
                    .and_then(|c| c.compress(data, CompressionLevel(level)))
            }
            "lz4" => {
                let level = codec.lz4_level().unwrap_or(0);
                registry
                    .get_by_name(&codec.name)
                    .and_then(|c| c.compress(data, CompressionLevel(level)))
            }
            // blosc, deflate, etc. — look up by name
            name => {
                let level = codec
                    .configuration
                    .iter()
                    .find(|(k, _)| k == "level")
                    .and_then(|(_, v)| v.parse::<i32>().ok())
                    .unwrap_or(6);
                registry
                    .get_by_name(name)
                    .and_then(|c| c.compress(data, CompressionLevel(level)))
            }
        }
    }

    /// Apply a single codec in the decompress direction.
    fn apply_decompress(
        &self,
        codec: &Codec,
        data: &[u8],
        registry: &dyn CompressionRegistryTrait,
    ) -> Result<Vec<u8>> {
        match codec.name.as_str() {
            "bytes" => bytes_codec_passthrough(codec, data),

            "crc32" => Err(unsupported_crc32()),

            // All compression codecs use the same decompress interface
            name => registry
                .get_by_name(name)
                .and_then(|c| c.decompress(data, 0)),
        }
    }
}

/// The `bytes` codec, which is an identity transform exactly when the stored
/// byte order already matches this host's.
///
/// Both directions share this: byte-order conversion is its own inverse, so
/// the arm that is legal on write is the arm that is legal on read.
///
/// A differing byte order is rejected rather than swapped, because a swap
/// needs the element width and this signature does not carry it — the
/// pipeline is `(codec, bytes)` with no dtype. Returning the bytes unchanged
/// would decode every multi-byte element reversed with no error raised, which
/// is the defect this replaces; an honest `UnsupportedFeature` is the most
/// this layer can say until the element width is threaded through.
#[cfg(feature = "alloc")]
fn bytes_codec_passthrough(codec: &Codec, data: &[u8]) -> Result<Vec<u8>> {
    // Absent configuration means "no reordering requested". `native` is not a
    // Zarr v3 value — the spec admits only `little` and `big` — but this crate
    // writes it in its own pipeline constructors, and it means the host order
    // by definition, so it is accepted here. Replacing it with a spec-legal
    // value changes serialized metadata, so it is filed rather than folded in.
    let host = if cfg!(target_endian = "big") {
        "big"
    } else {
        "little"
    };

    match codec.bytes_endian() {
        None | Some("native") => Ok(data.to_vec()),
        Some(endian) if endian == host => Ok(data.to_vec()),
        Some(endian) => Err(consus_core::Error::UnsupportedFeature {
            feature: alloc::format!(
                "zarr `bytes` codec with endian={endian} on a {host}-endian host: \
                 byte-order conversion needs the element width, which the codec \
                 pipeline does not receive"
            ),
        }),
    }
}

/// Reject the `crc32` codec in both directions.
///
/// Two independent reasons, either sufficient:
///
/// 1. **Wrong algorithm for the format.** Zarr v3's checksum codec is
///    `crc32c` (Castagnoli, reflected `0x82F63B78`). The only checksum this
///    workspace implements is `consus_compression::Crc32`, which is CRC-32/IEEE
///    (reflected `0xEDB88320`). Wiring that in would append checksums no
///    conformant reader accepts, turning a silent read defect into a silent
///    *write* defect.
/// 2. **No length plumbing.** The codec appends four bytes on write and must
///    strip and verify them on read. Nothing in this pipeline tracks that the
///    payload length changed.
///
/// Until both are addressed, refusing is the only honest behaviour: the
/// previous pass-through neither appended the checksum on write (so a
/// conformant reader strips four bytes of real payload) nor stripped it on
/// read (so a conformant writer's output comes back with four trailing
/// garbage bytes and a length that no longer matches the chunk shape).
#[cfg(feature = "alloc")]
fn unsupported_crc32() -> consus_core::Error {
    consus_core::Error::UnsupportedFeature {
        feature: String::from(
            "zarr `crc32` checksum codec: Zarr v3 specifies crc32c (Castagnoli) \
             and this workspace implements only CRC-32/IEEE; the codec is \
             refused rather than applied with the wrong polynomial",
        ),
    }
}

// ---------------------------------------------------------------------------
// Trait for codec registry access
// ---------------------------------------------------------------------------

/// Trait for looking up codec implementations by name.
///
/// This abstracts over `consus_compression::CompressionRegistry` so the
/// pipeline does not need to depend on a concrete type.
#[cfg(feature = "alloc")]
pub trait CompressionRegistryTrait: Send + Sync {
    /// Look up a codec by name.
    fn get_by_name(&self, name: &str) -> Result<Box<dyn CodecTrait + '_>>;
    /// Look up a codec by HDF5 filter ID.
    #[allow(unused)]
    fn get_by_filter_id(&self, _id: u16) -> Result<Box<dyn CodecTrait>> {
        Err(consus_core::Error::UnsupportedFeature {
            feature: "filter_id_lookup".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Adapter from consus-compression CompressionRegistry
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<T: consus_compression::CompressionRegistry> CompressionRegistryTrait for T {
    fn get_by_name(&self, name: &str) -> Result<Box<dyn CodecTrait + '_>> {
        let codec = consus_compression::CompressionRegistry::get_by_name(self, name)?;
        Ok(Box::new(CodecAdapterWrapper(codec)))
    }
}

/// Wrapper type that owns a dynamically dispatched codec reference.
pub struct CodecAdapterWrapper<'a>(&'a dyn consus_compression::Codec);

impl CodecTrait for CodecAdapterWrapper<'_> {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn hdf5_filter_id(&self) -> Option<u16> {
        self.0.hdf5_filter_id()
    }
    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        self.0
            .compress(input, consus_compression::CompressionLevel(level.0))
            .map_err(|e| consus_core::Error::CompressionError {
                message: alloc::format!("{:?}", e),
            })
    }
    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        self.0
            .decompress(input, expected_size)
            .map_err(|e| consus_core::Error::CompressionError {
                message: alloc::format!("{:?}", e),
            })
    }
}

/// Unified codec trait used within the codec pipeline.
///
/// This is a thin wrapper around `consus_compression::Codec` that
/// normalizes the compression level type.
#[cfg(feature = "alloc")]
pub trait CodecTrait: Send + Sync {
    /// Human-readable name of this codec.
    fn name(&self) -> &str;

    /// HDF5 filter ID, if applicable.
    fn hdf5_filter_id(&self) -> Option<u16>;

    /// Compress input data.
    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>>;

    /// Decompress input data.
    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>>;
}

// ---------------------------------------------------------------------------
// Default registry (lazy static)
// ---------------------------------------------------------------------------

/// Returns the default compression registry.
///
/// The registry is initialized lazily on first access with all codecs
/// enabled by cargo features (deflate, gzip, zstd, lz4, blosc, szip).
#[cfg(feature = "std")]
pub fn default_registry() -> &'static impl CompressionRegistryTrait {
    static REGISTRY: std::sync::OnceLock<consus_compression::DefaultCodecRegistry> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(consus_compression::DefaultCodecRegistry::new)
}

/// Look up a codec by name from the default registry.
/// This is a convenience wrapper for cases where a full registry
/// reference is not available.
#[cfg(feature = "std")]
pub fn get_codec_by_name(name: &str) -> Result<Box<dyn CodecTrait + 'static>> {
    default_registry().get_by_name(name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_codec_pipeline_roundtrip() {
        let registry = default_registry();
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), String::from("1"))],
        });

        let input = b"The quick brown fox jumps over the lazy dog";
        let compressed = pipeline
            .compress(input, registry)
            .expect("compress must succeed");
        let decompressed = pipeline
            .decompress(&compressed, registry)
            .expect("decompress must succeed");
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn gzip_level_extraction() {
        let codec = Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), String::from("9"))],
        };
        assert_eq!(codec.gzip_level(), Some(9));
    }

    #[test]
    fn zstd_level_extraction() {
        let codec = Codec {
            name: String::from("zstd"),
            configuration: vec![(String::from("level"), String::from("-3"))],
        };
        assert_eq!(codec.zstd_level(), Some(-3));
    }

    #[test]
    fn compression_level_default() {
        let level = CompressionLevel::default();
        assert_eq!(level.0, 6);
    }

    #[test]
    fn pipeline_len() {
        assert!(CodecPipeline::empty().is_empty());
        assert_eq!(CodecPipeline::empty().len(), 0);

        let pipeline = CodecPipeline::single(Codec {
            name: String::from("bytes"),
            configuration: vec![],
        });
        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn bytes_codec_is_identity() {
        let registry = default_registry();
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from("native"))],
        });

        let input = b"raw bytes";
        let result = pipeline.compress(input, registry).unwrap();
        assert_eq!(&result, input);

        let decompressed = pipeline.decompress(&result, registry).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn codec_is_identity_for_bytes_native() {
        let codec = Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from("native"))],
        };
        assert!(codec.is_identity());
    }

    #[test]
    fn zstd_roundtrip() {
        let registry = default_registry();
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("zstd"),
            configuration: vec![(String::from("level"), String::from("1"))],
        });

        let input = b"Zstandard is a real-time compression algorithm";
        let compressed = pipeline
            .compress(input, registry)
            .expect("compress must succeed");
        let decompressed = pipeline
            .decompress(&compressed, registry)
            .expect("decompress must succeed");
        assert_eq!(&decompressed, input);
    }

    /// The host's own byte order is a legal no-op in both directions.
    #[test]
    fn bytes_codec_accepts_host_endianness() {
        let registry = default_registry();
        let host = if cfg!(target_endian = "big") {
            "big"
        } else {
            "little"
        };
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from(host))],
        });

        let input = b"raw bytes";
        assert_eq!(&pipeline.compress(input, registry).unwrap(), input);
        assert_eq!(&pipeline.decompress(input, registry).unwrap(), input);
    }

    /// The opposite byte order must fail, not pass the bytes through.
    ///
    /// Before this was fixed the call returned `Ok` with every multi-byte
    /// element reversed, so the assertion that matters is that it is an
    /// error at all — a value-returning path here is silent corruption.
    #[test]
    fn bytes_codec_rejects_foreign_endianness() {
        let registry = default_registry();
        let foreign = if cfg!(target_endian = "big") {
            "little"
        } else {
            "big"
        };
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from(foreign))],
        });

        let input = &[0x01u8, 0x02, 0x03, 0x04];
        assert!(matches!(
            pipeline.decompress(input, registry),
            Err(consus_core::Error::UnsupportedFeature { .. })
        ));
        assert!(matches!(
            pipeline.compress(input, registry),
            Err(consus_core::Error::UnsupportedFeature { .. })
        ));
    }

    /// crc32 is refused in both directions rather than silently skipped.
    ///
    /// The write direction matters as much as the read: pass-through there
    /// emits a chunk with no checksum, which a conformant reader "verifies"
    /// by stripping four bytes of real payload.
    #[test]
    fn crc32_codec_is_refused_in_both_directions() {
        let registry = default_registry();
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("crc32"),
            configuration: vec![],
        });

        let input = b"chunk payload";
        assert!(matches!(
            pipeline.compress(input, registry),
            Err(consus_core::Error::UnsupportedFeature { .. })
        ));
        assert!(matches!(
            pipeline.decompress(input, registry),
            Err(consus_core::Error::UnsupportedFeature { .. })
        ));
    }
}
