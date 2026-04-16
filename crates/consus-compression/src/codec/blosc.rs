//! Blosc meta-compressor container format codec.
//!
//! ## HDF5 Mapping
//!
//! HDF5 filter ID 32001. Blosc is a meta-compressor that wraps an inner
//! codec (blosclz, lz4, zlib, zstd) with optional byte-shuffle preprocessing
//! inside a self-describing 16-byte header.
//!
//! ## Container Format
//!
//! ### Header (16 bytes)
//!
//! | Offset  | Size    | Description                                        |
//! |---------|---------|----------------------------------------------------|
//! | 0       | 1 byte  | Format version (2)                                 |
//! | 1       | 1 byte  | Compressor code (0=blosclz, 1=lz4, 5=zlib, 6=zstd)|
//! | 2       | 1 byte  | Flags (bit 0: byte-shuffle, bit 1: memcpyed, bit 2: bit-shuffle) |
//! | 3       | 1 byte  | Type size (element width in bytes)                  |
//! | 4..8    | 4 bytes | `nbytes`: uncompressed size (LE u32)               |
//! | 8..12   | 4 bytes | `blocksize` (LE u32)                               |
//! | 12..16  | 4 bytes | `cbytes`: total compressed size including header (LE u32) |
//!
//! ### Memcpy Mode (flag bit 1 set)
//!
//! Data immediately follows the 16-byte header without block framing.
//! The data may be shuffled (flag bit 0) prior to storage.
//!
//! ### Block Mode (flag bit 1 clear)
//!
//! After the header: `nblocks` LE u32 offsets (each relative to buffer
//! start), followed by compressed blocks. `nblocks = ceil(nbytes / blocksize)`.
//! Each block is compressed with the sub-compressor identified by byte 1.
//!
//! ## Shuffle Algorithm
//!
//! Byte-shuffle reorders elements of `typesize` bytes so that all byte-0
//! values are contiguous, then all byte-1 values, etc. For `N` elements of
//! `T` bytes each:
//!
//! ```text
//! shuffle:   output[j*N + i] = input[i*T + j]   for j in 0..T, i in 0..N
//! unshuffle: output[i*T + j] = input[j*N + i]   (inverse)
//! ```
//!
//! Trailing bytes that do not form a complete element are copied verbatim.
//!
//! ## Invariant
//!
//! For all byte sequences `data` and valid `BloscConfig`:
//!
//! ```text
//! decompress(compress(data)?) == data
//! ```
//!
//! ## Implementation Scope
//!
//! - Compression: memcpy mode with optional byte-shuffle.
//! - Decompression: memcpy mode fully supported. Block mode returns
//!   `Error::UnsupportedFeature` because sub-compressor implementations
//!   are provided by their respective codec modules and not embedded here.

use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};

use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Blosc container header size in bytes.
const HEADER_SIZE: usize = 16;

/// Blosc format version emitted by this implementation.
const BLOSC_VERSION: u8 = 2;

/// Flag bit: byte-shuffle is active.
const FLAG_BYTE_SHUFFLE: u8 = 0x01;

/// Flag bit: data is stored uncompressed (memcpy mode).
const FLAG_MEMCPYED: u8 = 0x02;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the [`BloscCodec`].
///
/// ## Fields
///
/// - `typesize`: element width in bytes used by the shuffle transform.
///   Must be ≥ 1. Default: 1 (no effective shuffle).
/// - `do_shuffle`: enable byte-shuffle preprocessing. Default: `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BloscConfig {
    /// Element width in bytes for shuffle. Must be ≥ 1.
    pub typesize: u8,
    /// Enable byte-shuffle before storage.
    pub do_shuffle: bool,
}

impl Default for BloscConfig {
    /// Default: `typesize = 1`, `do_shuffle = false`.
    fn default() -> Self {
        Self {
            typesize: 1,
            do_shuffle: false,
        }
    }
}

/// Blosc meta-compressor container codec (HDF5 filter ID 32001).
///
/// This implementation uses memcpy mode (uncompressed storage inside
/// the blosc container) with optional byte-shuffle. It can decompress
/// any memcpy-mode blosc buffer and correctly parses block-mode headers,
/// returning [`Error::UnsupportedFeature`] when a sub-compressor is
/// required.
///
/// ## Construction
///
/// ```text
/// BloscCodec::default()           // typesize=1, no shuffle
/// BloscCodec::new(BloscConfig { typesize: 4, do_shuffle: true })
/// ```
#[derive(Debug, Clone)]
pub struct BloscCodec {
    config: BloscConfig,
}

impl BloscCodec {
    /// Compile-time default instance: `typesize = 1`, `do_shuffle = false`.
    ///
    /// Used by the codec registry for `static` initialization.
    pub const DEFAULT: Self = Self {
        config: BloscConfig {
            typesize: 1,
            do_shuffle: false,
        },
    };

    /// Create a `BloscCodec` with the given configuration.
    pub fn new(config: BloscConfig) -> Self {
        Self { config }
    }
}

impl Default for BloscCodec {
    /// Default: `typesize = 1`, `do_shuffle = false`.
    fn default() -> Self {
        Self::new(BloscConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Codec trait implementation
// ---------------------------------------------------------------------------

impl Codec for BloscCodec {
    fn name(&self) -> &str {
        "blosc"
    }

    fn hdf5_filter_id(&self) -> Option<u16> {
        Some(32001)
    }

    fn compress(&self, input: &[u8], _level: CompressionLevel) -> Result<Vec<u8>> {
        let nbytes = input.len();

        // Apply shuffle if requested and typesize > 1
        let payload = if self.config.do_shuffle && self.config.typesize > 1 {
            shuffle(input, self.config.typesize as usize)
        } else {
            input.to_vec()
        };

        // Build flags
        let mut flags: u8 = FLAG_MEMCPYED;
        if self.config.do_shuffle && self.config.typesize > 1 {
            flags |= FLAG_BYTE_SHUFFLE;
        }

        // Total compressed size = header + payload
        let cbytes = HEADER_SIZE + payload.len();

        // Allocate output
        let mut output = Vec::with_capacity(cbytes);

        // Write 16-byte header
        let mut header = [0u8; HEADER_SIZE];
        header[0] = BLOSC_VERSION;
        header[1] = 0; // compressor code (irrelevant for memcpy mode)
        header[2] = flags;
        header[3] = self.config.typesize;
        LittleEndian::write_u32(&mut header[4..8], nbytes as u32);
        LittleEndian::write_u32(&mut header[8..12], nbytes as u32); // blocksize = nbytes
        LittleEndian::write_u32(&mut header[12..16], cbytes as u32);

        output.extend_from_slice(&header);
        output.extend_from_slice(&payload);

        Ok(output)
    }

    fn decompress(&self, input: &[u8], _expected_size: usize) -> Result<Vec<u8>> {
        if input.len() < HEADER_SIZE {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "blosc: input too short for header ({} < {HEADER_SIZE})",
                    input.len()
                ),
            });
        }

        // Parse header
        let version = input[0];
        let compressor_code = input[1];
        let flags = input[2];
        let typesize = input[3] as usize;
        let nbytes = LittleEndian::read_u32(&input[4..8]) as usize;
        let blocksize = LittleEndian::read_u32(&input[8..12]) as usize;
        let cbytes = LittleEndian::read_u32(&input[12..16]) as usize;

        // Validate header
        if version == 0 {
            return Err(Error::InvalidFormat {
                message: "blosc: invalid version byte 0".to_string(),
            });
        }

        if cbytes != input.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "blosc: header cbytes ({cbytes}) does not match input length ({})",
                    input.len()
                ),
            });
        }

        let memcpyed = (flags & FLAG_MEMCPYED) != 0;
        let shuffled = (flags & FLAG_BYTE_SHUFFLE) != 0;

        if memcpyed {
            // Memcpy mode: raw data follows the header
            let data_start = HEADER_SIZE;
            let data_end = data_start + nbytes;
            if data_end > input.len() {
                return Err(Error::InvalidFormat {
                    message: alloc::format!(
                        "blosc: memcpy payload extends past input \
                         (need {data_end}, have {})",
                        input.len()
                    ),
                });
            }

            let raw = &input[data_start..data_end];

            // Unshuffle if needed
            let output = if shuffled && typesize > 1 {
                unshuffle(raw, typesize)
            } else {
                raw.to_vec()
            };

            Ok(output)
        } else {
            // Block mode: sub-compressor required
            if blocksize == 0 {
                // Zero blocksize with non-memcpy mode: the data is empty
                if nbytes == 0 {
                    return Ok(Vec::new());
                }
                return Err(Error::InvalidFormat {
                    message: alloc::format!(
                        "blosc: blocksize is 0 but nbytes is {nbytes} in block mode"
                    ),
                });
            }

            let nblocks = nbytes.div_ceil(blocksize);
            let offsets_size = nblocks * 4;
            let offsets_end = HEADER_SIZE + offsets_size;

            if offsets_end > input.len() {
                return Err(Error::InvalidFormat {
                    message: alloc::format!(
                        "blosc: block offsets table extends past input \
                         (need {offsets_end}, have {})",
                        input.len()
                    ),
                });
            }

            // Parse block offsets for validation
            let mut offsets = Vec::with_capacity(nblocks);
            for i in 0..nblocks {
                let off_start = HEADER_SIZE + i * 4;
                let offset = LittleEndian::read_u32(&input[off_start..off_start + 4]) as usize;
                offsets.push(offset);
            }

            // Validate offsets are within bounds
            for (i, &offset) in offsets.iter().enumerate() {
                if offset >= input.len() {
                    return Err(Error::InvalidFormat {
                        message: alloc::format!(
                            "blosc: block {i} offset {offset} exceeds input length {}",
                            input.len()
                        ),
                    });
                }
            }

            let compressor_name = match compressor_code {
                0 => "blosclz",
                1 => "lz4",
                5 => "zlib",
                6 => "zstd",
                other => {
                    return Err(Error::UnsupportedFeature {
                        feature: alloc::format!("blosc: unknown compressor code {other}"),
                    });
                }
            };

            Err(Error::UnsupportedFeature {
                feature: alloc::format!(
                    "blosc: decompression of {compressor_name}-compressed blocks \
                     ({nblocks} blocks, blocksize={blocksize}) requires the \
                     {compressor_name} sub-compressor"
                ),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-shuffle transform
// ---------------------------------------------------------------------------

/// Byte-shuffle: group byte-position `j` of all elements contiguously.
///
/// For `N` elements of `typesize` bytes:
///   `output[j*N + i] = input[i*typesize + j]` for `j` in `0..typesize`, `i` in `0..N`
///
/// Trailing bytes (if `data.len() % typesize != 0`) are copied verbatim.
///
/// ## Theorem
///
/// `unshuffle(shuffle(d, T), T) == d` for all `d` and `T >= 1`.
/// Proof: the index mapping is a permutation on `[0, N*T)` composed with
/// identity on the remainder segment.
fn shuffle(data: &[u8], typesize: usize) -> Vec<u8> {
    if typesize <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let n_elements = data.len() / typesize;
    let shuffled_len = n_elements * typesize;
    let mut output = alloc::vec![0; data.len()];

    for i in 0..n_elements {
        for j in 0..typesize {
            output[j * n_elements + i] = data[i * typesize + j];
        }
    }

    // Copy trailing bytes that do not form a complete element
    output[shuffled_len..].copy_from_slice(&data[shuffled_len..]);

    output
}

/// Inverse byte-shuffle: reconstruct element-interleaved layout.
///
/// `output[i*typesize + j] = input[j*N + i]` for `j` in `0..typesize`, `i` in `0..N`
///
/// Inverse of [`shuffle`].
fn unshuffle(data: &[u8], typesize: usize) -> Vec<u8> {
    if typesize <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let n_elements = data.len() / typesize;
    let shuffled_len = n_elements * typesize;
    let mut output = alloc::vec![0; data.len()];

    for i in 0..n_elements {
        for j in 0..typesize {
            output[i * typesize + j] = data[j * n_elements + i];
        }
    }

    // Copy trailing bytes verbatim
    output[shuffled_len..].copy_from_slice(&data[shuffled_len..]);

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Shuffle unit tests --------------------------------------------------

    /// Shuffle/unshuffle round-trip for typesize=4.
    ///
    /// Input: two 4-byte elements [A0,A1,A2,A3, B0,B1,B2,B3]
    /// Shuffled: [A0,B0, A1,B1, A2,B2, A3,B3]
    #[test]
    fn shuffle_unshuffle_typesize_4() {
        let input: Vec<u8> = alloc::vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
        let shuffled = shuffle(&input, 4);
        assert_eq!(
            shuffled,
            alloc::vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3],
            "shuffle must group same-position bytes"
        );

        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(unshuffled, input, "unshuffle must invert shuffle");
    }

    /// Shuffle with typesize=1 is identity (no reordering).
    #[test]
    fn shuffle_typesize_1_is_identity() {
        let input: Vec<u8> = (0..32).collect();
        let shuffled = shuffle(&input, 1);
        assert_eq!(shuffled, input);
    }

    /// Shuffle with trailing bytes: 9 bytes, typesize=4.
    /// 2 full elements (8 bytes) + 1 trailing byte.
    #[test]
    fn shuffle_with_trailing_bytes() {
        let input: Vec<u8> = alloc::vec![1, 2, 3, 4, 5, 6, 7, 8, 0xFF];
        let shuffled = shuffle(&input, 4);
        // 2 elements: [1,2,3,4] and [5,6,7,8]
        // shuffled part: [1,5, 2,6, 3,7, 4,8]
        // trailing: [0xFF]
        assert_eq!(
            shuffled,
            alloc::vec![1, 5, 2, 6, 3, 7, 4, 8, 0xFF],
            "trailing byte must be preserved"
        );
        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(unshuffled, input);
    }

    /// Shuffle on empty input returns empty.
    #[test]
    fn shuffle_empty() {
        assert!(shuffle(&[], 4).is_empty());
        assert!(unshuffle(&[], 4).is_empty());
    }

    /// Shuffle round-trip with larger data and typesize=8 (f64-sized elements).
    #[test]
    fn shuffle_round_trip_typesize_8() {
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        assert_eq!(
            input.len() % 8,
            0,
            "input length must be multiple of typesize"
        );
        let shuffled = shuffle(&input, 8);
        assert_eq!(shuffled.len(), input.len());
        // Shuffled data must differ from input for non-trivial typesize
        assert_ne!(shuffled, input, "shuffle must reorder bytes");
        let unshuffled = unshuffle(&shuffled, 8);
        assert_eq!(unshuffled, input, "unshuffle must recover original");
    }

    // -- Header verification -------------------------------------------------

    /// Verify all 16 header bytes after compression without shuffle.
    #[test]
    fn header_bytes_no_shuffle() {
        let codec = BloscCodec::new(BloscConfig {
            typesize: 1,
            do_shuffle: false,
        });
        let input: Vec<u8> = (0u8..=99).collect();
        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        assert!(compressed.len() >= HEADER_SIZE);

        // Byte 0: version
        assert_eq!(
            compressed[0], BLOSC_VERSION,
            "version must be {BLOSC_VERSION}"
        );
        // Byte 1: compressor code (0 for memcpy mode)
        assert_eq!(compressed[1], 0, "compressor code must be 0");
        // Byte 2: flags (memcpy only, no shuffle because typesize=1)
        assert_eq!(
            compressed[2], FLAG_MEMCPYED,
            "flags must be memcpy-only (0x02)"
        );
        // Byte 3: typesize
        assert_eq!(compressed[3], 1, "typesize must be 1");
        // Bytes 4-7: nbytes
        assert_eq!(
            LittleEndian::read_u32(&compressed[4..8]),
            100,
            "nbytes must equal input length"
        );
        // Bytes 8-11: blocksize
        assert_eq!(
            LittleEndian::read_u32(&compressed[8..12]),
            100,
            "blocksize must equal nbytes in memcpy mode"
        );
        // Bytes 12-15: cbytes
        assert_eq!(
            LittleEndian::read_u32(&compressed[12..16]),
            (HEADER_SIZE + 100) as u32,
            "cbytes must equal header + payload"
        );
        // Total size
        assert_eq!(compressed.len(), HEADER_SIZE + 100);
    }

    /// Verify header flags when shuffle is active.
    #[test]
    fn header_bytes_with_shuffle() {
        let codec = BloscCodec::new(BloscConfig {
            typesize: 4,
            do_shuffle: true,
        });
        let input: Vec<u8> = alloc::vec![1, 2, 3, 4, 5, 6, 7, 8];
        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        assert_eq!(compressed[0], BLOSC_VERSION);
        assert_eq!(compressed[1], 0); // compressor code
        assert_eq!(
            compressed[2],
            FLAG_MEMCPYED | FLAG_BYTE_SHUFFLE,
            "flags must have both memcpy and shuffle bits"
        );
        assert_eq!(compressed[3], 4, "typesize must be 4");
        assert_eq!(LittleEndian::read_u32(&compressed[4..8]), 8);
        assert_eq!(LittleEndian::read_u32(&compressed[8..12]), 8);
        assert_eq!(
            LittleEndian::read_u32(&compressed[12..16]),
            (HEADER_SIZE + 8) as u32
        );
    }

    // -- Memcpy round-trip ---------------------------------------------------

    /// Round-trip in memcpy mode without shuffle.
    #[test]
    fn round_trip_memcpy_no_shuffle() {
        let codec = BloscCodec::default();
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        assert_eq!(input.len(), 1024);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");
        assert_eq!(
            compressed.len(),
            HEADER_SIZE + 1024,
            "memcpy mode: output = header + input"
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input, "memcpy round-trip must be lossless");
    }

    /// Round-trip in memcpy mode with byte-shuffle (typesize=4).
    #[test]
    fn round_trip_memcpy_with_shuffle() {
        let codec = BloscCodec::new(BloscConfig {
            typesize: 4,
            do_shuffle: true,
        });
        // 256 elements of 4 bytes each = 1024 bytes
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        assert_eq!(input.len(), 1024);
        assert_eq!(input.len() % 4, 0);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");
        assert_eq!(compressed.len(), HEADER_SIZE + 1024);

        // Verify the stored data is shuffled (not identical to input)
        let stored_data = &compressed[HEADER_SIZE..];
        assert_ne!(
            stored_data,
            &input[..],
            "shuffled payload must differ from input"
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input, "shuffle round-trip must be lossless");
    }

    /// Round-trip with shuffle on data whose length is not a multiple of typesize.
    #[test]
    fn round_trip_shuffle_trailing_bytes() {
        let codec = BloscCodec::new(BloscConfig {
            typesize: 4,
            do_shuffle: true,
        });
        // 103 bytes: 25 full elements (100 bytes) + 3 trailing
        let input: Vec<u8> = (0u8..103).collect();
        assert_eq!(input.len(), 103);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), 103);
        assert_eq!(
            decompressed, input,
            "trailing-byte round-trip must be lossless"
        );
    }

    // -- Empty input ---------------------------------------------------------

    /// Round-trip with empty input.
    #[test]
    fn round_trip_empty() {
        let codec = BloscCodec::default();
        let input: &[u8] = &[];

        let compressed = codec
            .compress(input, CompressionLevel::default())
            .expect("compress empty must succeed");
        assert_eq!(compressed.len(), HEADER_SIZE);
        assert_eq!(
            LittleEndian::read_u32(&compressed[4..8]),
            0,
            "nbytes must be 0"
        );
        assert_eq!(
            LittleEndian::read_u32(&compressed[12..16]),
            HEADER_SIZE as u32,
            "cbytes must equal header size"
        );

        let decompressed = codec
            .decompress(&compressed, 0)
            .expect("decompress empty must succeed");
        assert!(decompressed.is_empty());
    }

    // -- Error cases ---------------------------------------------------------

    /// Decompression rejects input shorter than the header.
    #[test]
    fn reject_truncated_header() {
        let codec = BloscCodec::default();
        let short = alloc::vec![0u8; 10];
        let err = codec.decompress(&short, 0).unwrap_err();
        let msg = alloc::format!("{err}");
        assert!(
            msg.contains("too short"),
            "error must mention truncation: {msg}"
        );
    }

    /// Decompression rejects mismatched cbytes.
    #[test]
    fn reject_cbytes_mismatch() {
        let codec = BloscCodec::default();
        let input = alloc::vec![42u8; 8];
        let mut compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        // Corrupt cbytes to a wrong value
        LittleEndian::write_u32(&mut compressed[12..16], 999);

        let err = codec.decompress(&compressed, 8).unwrap_err();
        let msg = alloc::format!("{err}");
        assert!(
            msg.contains("cbytes"),
            "error must mention cbytes mismatch: {msg}"
        );
    }

    /// Codec metadata accessors return expected values.
    #[test]
    fn codec_metadata() {
        let codec = BloscCodec::default();
        assert_eq!(codec.name(), "blosc");
        assert_eq!(codec.hdf5_filter_id(), Some(32001));
    }

    /// Round-trip with typesize=8 and shuffle on a larger dataset.
    ///
    /// 2048 bytes of non-trivial patterned data (simulating f64 array).
    #[test]
    fn round_trip_large_typesize_8_shuffle() {
        let codec = BloscCodec::new(BloscConfig {
            typesize: 8,
            do_shuffle: true,
        });
        // 256 elements * 8 bytes = 2048 bytes
        let mut input = Vec::with_capacity(2048);
        for i in 0u64..256 {
            input.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(input.len(), 2048);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");
        assert_eq!(compressed.len(), HEADER_SIZE + 2048);

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), 2048);
        assert_eq!(
            decompressed, input,
            "large shuffle round-trip must be lossless"
        );
    }

    /// Verify that shuffle=true with typesize=1 does not modify the data,
    /// because single-byte shuffle is a no-op.
    #[test]
    fn shuffle_typesize_1_is_noop() {
        let codec = BloscCodec::new(BloscConfig {
            typesize: 1,
            do_shuffle: true,
        });
        let input: Vec<u8> = (0u8..64).collect();

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        // Flags should NOT have shuffle bit because typesize=1 skips it
        assert_eq!(
            compressed[2], FLAG_MEMCPYED,
            "typesize=1 must suppress shuffle flag"
        );

        // Payload should be identical to input (no shuffle effect)
        assert_eq!(&compressed[HEADER_SIZE..], &input[..]);

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed, input);
    }
}
