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
//! | "bytes" | Both | Raw byte transport; handles endianness |
//! | "crc32" | Read | Checksum filter; validates integrity |
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
// Endianness helpers
// ---------------------------------------------------------------------------

/// Returns `true` when `configured` matches the host byte order.
///
/// Zarr v3 legal values are `"little"` and `"big"`. The value `"native"` is
/// **not** a legal Zarr v3 endian specification; we treat it as matching the
/// host to preserve existing round-trip behaviour for internally-written stores
/// while avoiding silent corruption for cross-platform interop.
#[cfg(feature = "alloc")]
fn is_native_endian(configured: &str) -> bool {
    #[cfg(target_endian = "little")]
    let host_is_little = true;
    #[cfg(not(target_endian = "little"))]
    let host_is_little = false;

    match configured {
        "little" => host_is_little,
        "big" => !host_is_little,
        _ => true, // "native" or unknown: no swap
    }
}

/// Byte-swap every `element_size`-byte word in `data` in place.
///
/// If `data.len()` is not a multiple of `element_size` the trailing partial
/// element is left unchanged (no panic, no silent discard).
#[cfg(feature = "alloc")]
fn byte_swap_elements(data: &[u8], element_size: usize) -> Vec<u8> {
    let mut out = data.to_vec();
    let n_elements = out.len() / element_size;
    for i in 0..n_elements {
        let start = i * element_size;
        let slice = &mut out[start..start + element_size];
        slice.reverse();
    }
    out
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
    /// Element size in bytes, used for endianness conversion in the `bytes` codec.
    /// `1` disables byte-swapping (single-byte or variable-length types).
    element_size: usize,
}

#[cfg(feature = "alloc")]
impl CodecPipeline {
    /// Create a new pipeline from an ordered list of codec configurations.
    ///
    /// The codecs are applied in the order given for compression.
    pub fn new(codecs: Vec<Codec>) -> Self {
        Self {
            codecs,
            element_size: 1,
        }
    }

    /// Set the element size (bytes per scalar element) for byte-order conversion.
    ///
    /// This is used when the `"bytes"` codec specifies an endianness different
    /// from the host. Call with the result of `dtype_to_element_size(&meta.dtype)`.
    /// A value of `1` (the default) disables byte-swapping.
    pub fn with_element_size(mut self, element_size: usize) -> Self {
        self.element_size = element_size.max(1);
        self
    }

    /// Create a pipeline from a single codec.
    pub fn single(codec: Codec) -> Self {
        Self {
            codecs: vec![codec],
            element_size: 1,
        }
    }

    /// Create an empty (identity) pipeline.
    pub fn empty() -> Self {
        Self {
            codecs: Vec::new(),
            element_size: 1,
        }
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
            // Bytes codec: applies endian byte-swapping when the configured
            // endianness differs from the host byte order.
            "bytes" => {
                let configured = codec.bytes_endian().unwrap_or("little");
                if self.element_size > 1 && !is_native_endian(configured) {
                    Ok(byte_swap_elements(data, self.element_size))
                } else {
                    Ok(data.to_vec())
                }
            }

            // CRC32 checksum: not yet implemented as a full codec.
            // Returning pass-through is incorrect (silent corruption), so we
            // reject it at pipeline-application time instead.
            "crc32" => Err(consus_core::Error::UnsupportedFeature {
                feature: "crc32 codec (not implemented; use a compression codec without crc32)"
                    .to_string(),
            }),

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
            // Bytes codec: reverse the byte-swap applied on compress (symmetric).
            "bytes" => {
                let configured = codec.bytes_endian().unwrap_or("little");
                if self.element_size > 1 && !is_native_endian(configured) {
                    Ok(byte_swap_elements(data, self.element_size))
                } else {
                    Ok(data.to_vec())
                }
            }

            // CRC32 checksum: not yet implemented.
            // Pass-through in either direction is incorrect (strips 4 bytes of
            // real payload on read or silently accepts corrupted data), so we
            // reject it.
            "crc32" => Err(consus_core::Error::UnsupportedFeature {
                feature: "crc32 codec (not implemented; use a compression codec without crc32)"
                    .to_string(),
            }),

            // All compression codecs use the same decompress interface
            name => registry
                .get_by_name(name)
                .and_then(|c| c.decompress(data, 0)),
        }
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

    // ── Endianness (ATLAS-CONSUS-ZARR-BYTES-ENDIAN-206) ──────────────────────

    /// Native-endian `bytes` codec must not alter the data.
    #[test]
    fn bytes_codec_native_endian_is_identity() {
        let registry = default_registry();
        let codec = Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from("little"))],
        };
        let pipeline = CodecPipeline::single(codec).with_element_size(4);
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        #[cfg(target_endian = "little")]
        {
            let out = pipeline.compress(&input, registry).expect("compress");
            assert_eq!(out, input, "native endian: data must be unchanged");
            let back = pipeline.decompress(&input, registry).expect("decompress");
            assert_eq!(back, input, "native endian: data must be unchanged");
        }
    }

    /// Non-native-endian `bytes` codec must byte-swap each element.
    #[test]
    fn bytes_codec_non_native_endian_swaps_elements() {
        let registry = default_registry();
        #[cfg(target_endian = "little")]
        let endian = "big";
        #[cfg(target_endian = "big")]
        let endian = "little";
        let codec = Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from(endian))],
        };
        let pipeline = CodecPipeline::single(codec).with_element_size(4);
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let swapped: Vec<u8> = vec![0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05];
        let compressed = pipeline.compress(&input, registry).expect("compress");
        assert_eq!(
            compressed, swapped,
            "non-native endian: 4-byte elements must be swapped"
        );
        let back = pipeline
            .decompress(&compressed, registry)
            .expect("decompress");
        assert_eq!(back, input, "round-trip must recover original");
    }

    /// Single-byte types must not be swapped regardless of endian config.
    #[test]
    fn bytes_codec_element_size_1_is_identity() {
        let registry = default_registry();
        let codec = Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from("big"))],
        };
        let pipeline = CodecPipeline::single(codec).with_element_size(1);
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
        assert_eq!(
            pipeline.compress(&input, registry).expect("compress"),
            input
        );
    }

    /// Big-endian fixture: 1.0f32 must decode correctly on a little-endian host.
    #[test]
    fn bytes_codec_big_endian_f32_oracle() {
        let registry = default_registry();
        #[cfg(target_endian = "little")]
        {
            let codec = Codec {
                name: String::from("bytes"),
                configuration: vec![(String::from("endian"), String::from("big"))],
            };
            let pipeline = CodecPipeline::single(codec).with_element_size(4);
            // 1.0f32 in big-endian: 3F 80 00 00
            let on_disk: Vec<u8> = vec![0x3F, 0x80, 0x00, 0x00];
            let decoded = pipeline.decompress(&on_disk, registry).expect("decompress");
            // After swap: 00 00 80 3F = 1.0f32 on little-endian host
            assert_eq!(decoded, vec![0x00, 0x00, 0x80, 0x3F]);
            let val = f32::from_le_bytes([decoded[0], decoded[1], decoded[2], decoded[3]]);
            assert_eq!(val, 1.0f32);
        }
    }

    // ── CRC32 rejection (ATLAS-CONSUS-ZARR-CRC32-207) ────────────────────────

    /// `crc32` compress must be rejected with an error, not silently pass through.
    #[test]
    fn crc32_codec_compress_is_rejected() {
        let registry = default_registry();
        let codec = Codec {
            name: String::from("crc32"),
            configuration: vec![],
        };
        assert!(
            CodecPipeline::single(codec)
                .compress(b"payload", registry)
                .is_err(),
            "crc32 compress must return an error"
        );
    }

    /// `crc32` decompress must be rejected with an error, not silently pass through.
    #[test]
    fn crc32_codec_decompress_is_rejected() {
        let registry = default_registry();
        let codec = Codec {
            name: String::from("crc32"),
            configuration: vec![],
        };
        assert!(
            CodecPipeline::single(codec)
                .decompress(b"payload", registry)
                .is_err(),
            "crc32 decompress must return an error"
        );
    }
}
