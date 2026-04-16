//! Szip codec implementing Rice entropy coding.
//!
//! ## HDF5 Mapping
//!
//! HDF5 filter ID 4. This codec implements the Rice entropy coding algorithm
//! used by the HDF5 szip filter, with configurable coding mode and block size.
//!
//! ## Algorithm
//!
//! 1. **Preprocessing** (Nearest-Neighbor mode only): compute deltas between
//!    consecutive sample values via wrapping subtraction. This concentrates
//!    energy near zero for slowly varying signals.
//! 2. **Block partitioning**: divide the sample stream into blocks of
//!    `pixels_per_block` values. The final block may be shorter.
//! 3. **Per-block Rice parameter selection**: compute `k = floor(log2(mean))`
//!    where `mean` is the arithmetic mean of sample values in the block
//!    (integer division), clamped to `[0, bits_per_sample - 1]`. When the
//!    mean is zero, `k = 0`.
//! 4. **Rice encoding**: for each sample value `v`:
//!    - Quotient `q = v >> k`
//!    - Remainder `r = v & ((1 << k) - 1)`
//!    - Write `q` in unary (q zero-bits followed by one 1-bit)
//!    - Write `r` in `k` bits, MSB-first
//! 5. **Bit packing**: bits are packed into bytes MSB-first. Each block's
//!    bit stream is independently padded to a byte boundary with zero bits.
//!
//! ## Wire Format
//!
//! | Offset | Size    | Description                         |
//! |--------|---------|-------------------------------------|
//! | 0      | 1 byte  | Coding method (0 = EC, 1 = NN)     |
//! | 1      | 1 byte  | `pixels_per_block`                  |
//! | 2      | 1 byte  | `bits_per_sample` (8 for u8 data)   |
//! | 3      | 4 bytes | Number of samples (LE u32)          |
//! | 7      | varies  | Per-block: 1 byte `k`, then Rice-coded bits padded to byte boundary |
//!
//! ## Invariant
//!
//! For all byte sequences `data` and valid `SzipCodec` configurations:
//!
//! ```text
//! decompress(compress(data)?) == data
//! ```
//!
//! ## Constraints
//!
//! - `pixels_per_block` must be even and in `[2, 32]`.
//! - `bits_per_sample` is derived from the input representation; this
//!   implementation treats all input bytes as 8-bit samples.

use alloc::vec::Vec;

use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Coding mode selection for the szip Rice entropy coder.
///
/// - `EntropyCoding`: encode sample values directly (no preprocessing).
/// - `NearestNeighbor`: apply delta preprocessing before coding. Effective
///   when consecutive samples are correlated (slowly varying signals).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SzipCoding {
    /// Entropy Coding — sample values are Rice-coded without preprocessing.
    EntropyCoding,
    /// Nearest Neighbor — deltas (wrapping differences) between consecutive
    /// samples are computed before Rice coding.
    NearestNeighbor,
}

/// Szip codec implementing Rice entropy coding (HDF5 filter ID 4).
///
/// ## Construction
///
/// Use [`SzipCodec::new`] for validated construction, or [`SzipCodec::default`]
/// for default parameters (`pixels_per_block = 32`, `EntropyCoding`).
///
/// ## Parameters
///
/// - `pixels_per_block`: number of samples per coding block. Must be even
///   and in `[2, 32]`. Larger blocks amortize per-block overhead but reduce
///   adaptivity. Default: 32.
/// - `coding`: [`SzipCoding`] variant selecting preprocessing mode.
#[derive(Debug, Clone)]
pub struct SzipCodec {
    pixels_per_block: u8,
    coding: SzipCoding,
}

impl SzipCodec {
    /// Compile-time default instance: `pixels_per_block = 32`, `EntropyCoding`.
    ///
    /// Used by the codec registry for `static` initialization.
    pub const DEFAULT: Self = Self {
        pixels_per_block: 32,
        coding: SzipCoding::EntropyCoding,
    };

    /// Create a new `SzipCodec` with validated parameters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFormat`] if `pixels_per_block` is odd or
    /// outside the range `[2, 32]`.
    pub fn new(pixels_per_block: u8, coding: SzipCoding) -> Result<Self> {
        if !(2..=32).contains(&pixels_per_block) || pixels_per_block % 2 != 0 {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "szip: pixels_per_block must be even and in [2, 32], got {pixels_per_block}"
                ),
            });
        }
        Ok(Self {
            pixels_per_block,
            coding,
        })
    }

    /// Returns the configured `pixels_per_block`.
    pub fn pixels_per_block(&self) -> u8 {
        self.pixels_per_block
    }

    /// Returns the configured coding mode.
    pub fn coding(&self) -> SzipCoding {
        self.coding
    }
}

impl Default for SzipCodec {
    /// Default: `pixels_per_block = 32`, `EntropyCoding`.
    fn default() -> Self {
        Self {
            pixels_per_block: 32,
            coding: SzipCoding::EntropyCoding,
        }
    }
}

// ---------------------------------------------------------------------------
// Codec trait implementation
// ---------------------------------------------------------------------------

/// Header size: 1 (coding) + 1 (ppb) + 1 (bps) + 4 (n_samples LE u32) = 7.
const HEADER_SIZE: usize = 7;

/// Bits per sample for u8 data.
const BITS_PER_SAMPLE: u8 = 8;

impl Codec for SzipCodec {
    fn name(&self) -> &str {
        "szip"
    }

    fn hdf5_filter_id(&self) -> Option<u16> {
        Some(4)
    }

    fn compress(&self, input: &[u8], _level: CompressionLevel) -> Result<Vec<u8>> {
        let n_samples = input.len();

        // Preprocessing
        let processed = match self.coding {
            SzipCoding::EntropyCoding => input.to_vec(),
            SzipCoding::NearestNeighbor => delta_encode(input),
        };

        // Allocate output with header
        let mut output = Vec::with_capacity(HEADER_SIZE + n_samples);

        // Write header
        let coding_byte: u8 = match self.coding {
            SzipCoding::EntropyCoding => 0,
            SzipCoding::NearestNeighbor => 1,
        };
        output.push(coding_byte);
        output.push(self.pixels_per_block);
        output.push(BITS_PER_SAMPLE);
        output.extend_from_slice(&(n_samples as u32).to_le_bytes());

        // Encode blocks
        let ppb = self.pixels_per_block as usize;
        let mut block_start = 0;
        while block_start < n_samples {
            let block_end = (block_start + ppb).min(n_samples);
            let block = &processed[block_start..block_end];

            let k = rice_parameter(block, BITS_PER_SAMPLE);
            output.push(k);

            let mut writer = BitWriter::new();
            for &v in block {
                let v32 = u32::from(v);
                let q = v32 >> k;
                writer.write_unary(q);
                if k > 0 {
                    let r = v32 & ((1u32 << k) - 1);
                    writer.write_bits(r, k);
                }
            }
            output.extend_from_slice(&writer.finish());

            block_start = block_end;
        }

        Ok(output)
    }

    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        if input.len() < HEADER_SIZE {
            return Err(Error::CompressionError {
                message: alloc::format!(
                    "szip: compressed data too short for header ({} < {HEADER_SIZE})",
                    input.len()
                ),
            });
        }

        // Parse header
        let coding_byte = input[0];
        let ppb = input[1] as usize;
        let bps = input[2];
        let n_samples = u32::from_le_bytes([input[3], input[4], input[5], input[6]]) as usize;

        let coding = match coding_byte {
            0 => SzipCoding::EntropyCoding,
            1 => SzipCoding::NearestNeighbor,
            other => {
                return Err(Error::CompressionError {
                    message: alloc::format!("szip: invalid coding method byte {other}"),
                });
            }
        };

        if !(2..=32).contains(&ppb) || ppb % 2 != 0 {
            return Err(Error::CompressionError {
                message: alloc::format!(
                    "szip: invalid pixels_per_block {ppb} in header (must be even, 2..=32)"
                ),
            });
        }

        if bps != BITS_PER_SAMPLE {
            return Err(Error::UnsupportedFeature {
                feature: alloc::format!(
                    "szip: bits_per_sample {bps} not supported (only 8-bit samples implemented)"
                ),
            });
        }

        // Decode blocks
        let mut decoded = Vec::with_capacity(n_samples);
        let mut pos: usize = HEADER_SIZE;
        let mut remaining = n_samples;

        while remaining > 0 {
            let block_size = remaining.min(ppb);

            if pos >= input.len() {
                return Err(Error::CompressionError {
                    message: alloc::format!(
                        "szip: unexpected end of data at offset {pos} reading k parameter \
                         ({remaining} samples remaining)"
                    ),
                });
            }
            let k = input[pos];
            pos += 1;

            if k >= BITS_PER_SAMPLE {
                return Err(Error::CompressionError {
                    message: alloc::format!(
                        "szip: Rice parameter k={k} exceeds bits_per_sample-1={}",
                        BITS_PER_SAMPLE - 1
                    ),
                });
            }

            let bit_data = if pos < input.len() {
                &input[pos..]
            } else if block_size == 0 {
                // Edge case: no samples to decode and no bit data needed.
                &[][..]
            } else {
                return Err(Error::CompressionError {
                    message: alloc::format!(
                        "szip: unexpected end of data at offset {pos} reading Rice-coded block"
                    ),
                });
            };

            let mut reader = BitReader::new(bit_data);

            for _ in 0..block_size {
                let q = reader.read_unary()?;
                let r = if k > 0 { reader.read_bits(k)? } else { 0 };
                let v = (q << k) | r;
                if v > u32::from(u8::MAX) {
                    return Err(Error::CompressionError {
                        message: alloc::format!(
                            "szip: decoded value {v} exceeds u8 range (q={q}, k={k}, r={r})"
                        ),
                    });
                }
                decoded.push(v as u8);
            }

            pos += reader.bytes_consumed();
            remaining -= block_size;
        }

        // Undo preprocessing
        let output = match coding {
            SzipCoding::EntropyCoding => decoded,
            SzipCoding::NearestNeighbor => delta_decode(&decoded),
        };

        if output.len() != expected_size {
            return Err(Error::CompressionError {
                message: alloc::format!(
                    "szip: decoded {} bytes but expected_size is {expected_size}",
                    output.len()
                ),
            });
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Delta preprocessing (Nearest-Neighbor mode)
// ---------------------------------------------------------------------------

/// Compute wrapping deltas: `out[0] = data[0]`, `out[i] = data[i] - data[i-1]`.
///
/// ## Theorem
///
/// `delta_decode(delta_encode(d)) == d` for all `d: &[u8]`, proved by
/// induction on the prefix length with wrapping arithmetic identity
/// `(a.wrapping_sub(b)).wrapping_add(b) == a` for all `a, b: u8`.
fn delta_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        out.push(data[i].wrapping_sub(data[i - 1]));
    }
    out
}

/// Reconstruct original values from wrapping deltas.
///
/// Inverse of [`delta_encode`].
fn delta_decode(deltas: &[u8]) -> Vec<u8> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(deltas.len());
    out.push(deltas[0]);
    for i in 1..deltas.len() {
        out.push(out[i - 1].wrapping_add(deltas[i]));
    }
    out
}

// ---------------------------------------------------------------------------
// Rice parameter selection
// ---------------------------------------------------------------------------

/// Compute the optimal Rice parameter `k` for a block of u8 samples.
///
/// `k = floor(log2(mean))` where `mean = sum(block) / len(block)` (integer
/// division), clamped to `[0, bits_per_sample - 1]`. Returns 0 when the
/// block is empty or the mean is zero.
///
/// ## Derivation
///
/// For a geometric distribution with parameter `p`, the optimal Golomb
/// parameter satisfies `m ≈ -1 / ln(1 - p)`. For Rice codes (`m = 2^k`),
/// `k ≈ log2(mean)` minimizes expected code length when the source mean
/// equals `mean`. Integer floor and clamping ensure `k` is a valid bit
/// count for the sample width.
fn rice_parameter(block: &[u8], bits_per_sample: u8) -> u8 {
    if block.is_empty() {
        return 0;
    }
    let sum: u64 = block.iter().map(|&v| u64::from(v)).sum();
    let mean = sum / block.len() as u64;
    if mean == 0 {
        return 0;
    }
    // floor(log2(mean)) for positive integer mean:
    // For u64, leading_zeros gives the number of leading zero bits.
    // 63 - leading_zeros = position of the highest set bit = floor(log2(mean)).
    let k = (63 - mean.leading_zeros()) as u8;
    k.min(bits_per_sample - 1)
}

// ---------------------------------------------------------------------------
// Bit-level I/O — MSB-first packing
// ---------------------------------------------------------------------------

/// Bit writer that packs bits into bytes MSB-first.
///
/// On [`finish`](BitWriter::finish), any partially-filled trailing byte is
/// padded with zero bits in the LSB positions.
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    /// Number of bits written into `current` (0..=7).
    bits_used: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            bits_used: 0,
        }
    }

    /// Write a single bit (MSB-first packing).
    #[inline]
    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current |= 1 << (7 - self.bits_used);
        }
        self.bits_used += 1;
        if self.bits_used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.bits_used = 0;
        }
    }

    /// Write `count` bits from `value`, MSB-first.
    ///
    /// Only the lowest `count` bits of `value` are written.
    /// `count` must be in `[0, 32]`.
    fn write_bits(&mut self, value: u32, count: u8) {
        for i in (0..count).rev() {
            self.write_bit(((value >> i) & 1) != 0);
        }
    }

    /// Write a unary code: `q` zero-bits followed by one 1-bit.
    fn write_unary(&mut self, q: u32) {
        for _ in 0..q {
            self.write_bit(false);
        }
        self.write_bit(true);
    }

    /// Flush the writer, returning the packed byte vector.
    ///
    /// A partially-filled final byte is padded with zero bits.
    fn finish(mut self) -> Vec<u8> {
        if self.bits_used > 0 {
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

/// Bit reader that unpacks bits from bytes MSB-first.
struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    /// Bit offset within `bytes[byte_pos]` (0..=7, 0 = MSB).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit (MSB-first unpacking).
    #[inline]
    fn read_bit(&mut self) -> Result<bool> {
        if self.byte_pos >= self.bytes.len() {
            return Err(Error::CompressionError {
                message: "szip: unexpected end of bit stream".to_string(),
            });
        }
        let bit = (self.bytes[self.byte_pos] >> (7 - self.bit_pos)) & 1 != 0;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        Ok(bit)
    }

    /// Read `count` bits and return them as a `u32` (MSB-first).
    fn read_bits(&mut self, count: u8) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// Read a unary code: count zero-bits until a 1-bit is encountered.
    ///
    /// Returns the number of leading zeros (the decoded quotient).
    fn read_unary(&mut self) -> Result<u32> {
        let mut count = 0u32;
        loop {
            if self.read_bit()? {
                return Ok(count);
            }
            count += 1;
        }
    }

    /// Number of full bytes consumed from the input slice.
    ///
    /// If the reader is partway through a byte, that byte is counted as
    /// consumed (it contains padding bits from the writer).
    fn bytes_consumed(&self) -> usize {
        if self.bit_pos > 0 {
            self.byte_pos + 1
        } else {
            self.byte_pos
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper --------------------------------------------------------------

    /// Deterministic xorshift32 pseudo-random byte generator.
    ///
    /// Produces a repeatable sequence parameterized by a fixed seed.
    /// Period: 2^32 - 1. Distribution: uniform over u32, truncated to u8.
    fn xorshift_data(len: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(len);
        let mut state: u32 = 0xDEAD_BEEF;
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            data.push(state as u8);
        }
        data
    }

    // -- Constructor validation ----------------------------------------------

    #[test]
    fn reject_odd_pixels_per_block() {
        let err = SzipCodec::new(7, SzipCoding::EntropyCoding).unwrap_err();
        let msg = alloc::format!("{err}");
        assert!(
            msg.contains("pixels_per_block"),
            "error message must mention pixels_per_block: {msg}"
        );
    }

    #[test]
    fn reject_pixels_per_block_out_of_range() {
        assert!(SzipCodec::new(0, SzipCoding::EntropyCoding).is_err());
        assert!(SzipCodec::new(1, SzipCoding::EntropyCoding).is_err());
        assert!(SzipCodec::new(34, SzipCoding::EntropyCoding).is_err());
        assert!(SzipCodec::new(64, SzipCoding::EntropyCoding).is_err());
    }

    #[test]
    fn accept_valid_pixels_per_block() {
        for ppb in [2, 4, 8, 16, 32] {
            assert!(
                SzipCodec::new(ppb, SzipCoding::EntropyCoding).is_ok(),
                "pixels_per_block={ppb} must be accepted"
            );
        }
    }

    // -- Delta encode/decode unit tests --------------------------------------

    #[test]
    fn delta_round_trip_identity() {
        let data: Vec<u8> = (0u8..=255).collect();
        let encoded = delta_encode(&data);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn delta_encode_values() {
        // [10, 12, 11, 15] → [10, 2, 255, 4]
        // 12 - 10 = 2
        // 11 - 12 = 255 (wrapping)
        // 15 - 11 = 4
        let data: Vec<u8> = alloc::vec![10, 12, 11, 15];
        let encoded = delta_encode(&data);
        assert_eq!(encoded, alloc::vec![10, 2, 255, 4]);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn delta_empty() {
        assert!(delta_encode(&[]).is_empty());
        assert!(delta_decode(&[]).is_empty());
    }

    // -- Rice parameter unit tests -------------------------------------------

    #[test]
    fn rice_parameter_all_zeros() {
        let block = [0u8; 8];
        assert_eq!(rice_parameter(&block, 8), 0);
    }

    #[test]
    fn rice_parameter_mean_one() {
        // mean = 8/8 = 1, floor(log2(1)) = 0
        let block = [1u8; 8];
        assert_eq!(rice_parameter(&block, 8), 0);
    }

    #[test]
    fn rice_parameter_mean_two() {
        // mean = 16/8 = 2, floor(log2(2)) = 1
        let block = [2u8; 8];
        assert_eq!(rice_parameter(&block, 8), 1);
    }

    #[test]
    fn rice_parameter_mean_255() {
        // mean = 255, floor(log2(255)) = 7
        let block = [255u8; 8];
        assert_eq!(rice_parameter(&block, 8), 7);
    }

    #[test]
    fn rice_parameter_clamped() {
        // bits_per_sample = 4 → max k = 3
        // mean = 255, floor(log2(255)) = 7, clamped to 3
        let block = [255u8; 4];
        assert_eq!(rice_parameter(&block, 4), 3);
    }

    // -- BitWriter / BitReader round-trip ------------------------------------

    #[test]
    fn bit_io_round_trip() {
        let mut w = BitWriter::new();
        // Write unary 3 (0001), then 5 bits of value 19 (10011)
        w.write_unary(3);
        w.write_bits(19, 5);
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        let q = r.read_unary().unwrap();
        assert_eq!(q, 3);
        let v = r.read_bits(5).unwrap();
        assert_eq!(v, 19);
    }

    // -- Codec round-trip: Entropy Coding mode -------------------------------

    /// Round-trip with Entropy Coding on a repeating pattern (1024 bytes).
    #[test]
    fn round_trip_ec_patterned() {
        let codec = SzipCodec::new(16, SzipCoding::EntropyCoding).unwrap();
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        assert_eq!(input.len(), 1024);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input, "EC round-trip must be lossless");
    }

    /// Round-trip with EC on pseudo-random data (512 bytes).
    ///
    /// Random data may not compress (Rice coding can expand), but the
    /// round-trip must remain lossless.
    #[test]
    fn round_trip_ec_random() {
        let codec = SzipCodec::new(16, SzipCoding::EntropyCoding).unwrap();
        let input = xorshift_data(512);
        assert_eq!(input.len(), 512);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), input.len());
        assert_eq!(
            decompressed, input,
            "EC round-trip must be lossless for random data"
        );
    }

    // -- Codec round-trip: Nearest Neighbor mode -----------------------------

    /// Round-trip with NN on slowly varying data (1024 bytes).
    ///
    /// Data: `0, 1, 2, ..., 255, 0, 1, ...` — all deltas are 1 (wrapping),
    /// which compresses well under NN + Rice coding.
    #[test]
    fn round_trip_nn_slowly_varying() {
        let codec = SzipCodec::new(8, SzipCoding::NearestNeighbor).unwrap();
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        assert_eq!(input.len(), 1024);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        // NN + Rice on slowly varying data: deltas are mostly 1, k=0,
        // each sample is 2 bits (unary "01"). Expect compression gain.
        assert!(
            compressed.len() < input.len(),
            "NN mode on slowly varying data must compress: {} vs {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input, "NN round-trip must be lossless");
    }

    /// Round-trip with NN on pseudo-random data (512 bytes).
    ///
    /// Random data produces uniformly distributed deltas; compression gain
    /// is not expected, but lossless round-trip must hold.
    #[test]
    fn round_trip_nn_random() {
        let codec = SzipCodec::new(8, SzipCoding::NearestNeighbor).unwrap();
        let input = xorshift_data(512);
        assert_eq!(input.len(), 512);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), input.len());
        assert_eq!(
            decompressed, input,
            "NN round-trip must be lossless for random data"
        );
    }

    // -- Edge cases ----------------------------------------------------------

    /// Round-trip with empty input.
    #[test]
    fn round_trip_empty() {
        let codec = SzipCodec::default();
        let input: &[u8] = &[];

        let compressed = codec
            .compress(input, CompressionLevel::default())
            .expect("compress empty must succeed");
        // Header only: 7 bytes, n_samples = 0
        assert_eq!(compressed.len(), HEADER_SIZE);
        assert_eq!(compressed[0], 0); // EC
        assert_eq!(compressed[1], 32); // ppb
        assert_eq!(compressed[2], 8); // bps
        assert_eq!(
            u32::from_le_bytes([compressed[3], compressed[4], compressed[5], compressed[6]]),
            0
        );

        let decompressed = codec
            .decompress(&compressed, 0)
            .expect("decompress empty must succeed");
        assert!(decompressed.is_empty());
    }

    /// Round-trip with all-zero input.
    ///
    /// k = 0 for all blocks; each sample encodes as a single 1-bit (unary of 0).
    /// Expect significant compression (1 bit per sample vs 8 bits).
    #[test]
    fn round_trip_all_zeros() {
        let codec = SzipCodec::new(8, SzipCoding::EntropyCoding).unwrap();
        let input = alloc::vec![0u8; 256];

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");
        // 256 samples, 1 bit each = 32 bytes of bit data + 32 blocks * 1 byte k + 7 header
        // = 32 + 32 + 7 = 71 bytes, much less than 256
        assert!(
            compressed.len() < input.len(),
            "all-zero data must compress: {} vs {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed, input);
    }

    /// Round-trip with a single sample.
    #[test]
    fn round_trip_single_byte() {
        let codec = SzipCodec::new(2, SzipCoding::EntropyCoding).unwrap();
        let input: &[u8] = &[42];

        let compressed = codec
            .compress(input, CompressionLevel::default())
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0], 42);
    }

    /// Codec metadata accessors return expected values.
    #[test]
    fn codec_metadata() {
        let codec = SzipCodec::default();
        assert_eq!(codec.name(), "szip");
        assert_eq!(codec.hdf5_filter_id(), Some(4));
        assert_eq!(codec.pixels_per_block(), 32);
        assert_eq!(codec.coding(), SzipCoding::EntropyCoding);
    }

    /// Header fields are written correctly.
    #[test]
    fn header_encoding() {
        let codec = SzipCodec::new(16, SzipCoding::NearestNeighbor).unwrap();
        let input = alloc::vec![0u8; 100];

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        assert!(compressed.len() >= HEADER_SIZE);
        assert_eq!(compressed[0], 1, "coding byte must be 1 (NN)");
        assert_eq!(compressed[1], 16, "pixels_per_block must be 16");
        assert_eq!(compressed[2], 8, "bits_per_sample must be 8");
        let n = u32::from_le_bytes([compressed[3], compressed[4], compressed[5], compressed[6]]);
        assert_eq!(n, 100, "n_samples must be 100");
    }

    /// Non-block-aligned sample count: 100 samples with ppb=16 produces
    /// 6 full blocks (96 samples) + 1 partial block (4 samples).
    #[test]
    fn round_trip_non_aligned_block_count() {
        let codec = SzipCodec::new(16, SzipCoding::NearestNeighbor).unwrap();
        // 100 samples: not a multiple of 16
        let input: Vec<u8> = (0u8..100).collect();
        assert_eq!(input.len(), 100);

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input);
    }
}
