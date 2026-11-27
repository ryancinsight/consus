//! Shared, bounded `flate2` decompression scaffolding.
//!
//! Deflate/zlib, gzip, and raw DEFLATE all decode through `flate2` reader
//! types that must never be drained with an unbounded `read_to_end`: a hostile
//! stream would otherwise grow the output `Vec` until the allocator aborts an
//! uncatchable process. This module implements the bounded read once and
//! selects the wire framing with a zero-sized [`FlateFraming`](crate::codec::flate::FlateFraming) marker type.
//!
//! The output of any stream is at most [`MAX_DEFLATE_EXPANSION`](crate::codec::flate::MAX_DEFLATE_EXPANSION) times its
//! length, so that product bounds the decode and is the most the codec ever
//! reserves up front. A caller that needs a smaller output limit passes it as
//! the budget's byte ceiling.

use std::io::Read;

use consus_core::{Error, ParseBudget, Result};

use super::traits::CompressionLevel;

/// Selects the framing of a `flate2` byte stream.
///
/// Implementations are zero-sized marker types; the associated [`Self::Decoder`]
/// is the concrete `flate2::read` type produced for that framing.
pub trait FlateFraming {
    /// Codec label used in decompression error messages.
    const LABEL: &'static str;

    /// Reader that decodes the compressed input under this framing.
    type Decoder<'a>: Read;

    /// Wrap `input` in this framing's decoder.
    fn decoder(input: &[u8]) -> Self::Decoder<'_>;

    /// Output ceiling, in bytes, that a caller's budget and declaration allow.
    ///
    /// One rule serves every framing: the larger of the declared
    /// `expected_size` and the budget's byte ceiling. [`flate2_decompress`]
    /// takes the smaller of this and the size the compressed bytes can
    /// produce, so it caps a caller's own limit and never exceeds what the
    /// stream can carry.
    #[must_use]
    fn ceiling(expected_size: usize, budget: &ParseBudget) -> usize {
        expected_size.max(budget.max_alloc_bytes)
    }
}

/// zlib framing (RFC 1950), used by the HDF5 deflate filter (ID 1).
pub struct ZlibFraming;

impl FlateFraming for ZlibFraming {
    const LABEL: &'static str = "deflate (zlib)";

    type Decoder<'a> = flate2::read::ZlibDecoder<&'a [u8]>;

    fn decoder(input: &[u8]) -> Self::Decoder<'_> {
        flate2::read::ZlibDecoder::new(input)
    }
}

/// gzip framing (RFC 1952), used by the Zarr gzip codec and Parquet's GZIP codec.
///
/// RFC 1952 section 2.2 defines a gzip file as a series of members, so the
/// decoder returns the concatenated data of every member; a decoder that stops
/// after the first member silently truncates the output.
pub struct GzipFraming;

impl FlateFraming for GzipFraming {
    const LABEL: &'static str = "gzip";

    type Decoder<'a> = flate2::read::MultiGzDecoder<&'a [u8]>;

    fn decoder(input: &[u8]) -> Self::Decoder<'_> {
        flate2::read::MultiGzDecoder::new(input)
    }
}

/// Raw DEFLATE framing (RFC 1951), used by ZIP method 8.
pub struct RawDeflateFraming;

impl FlateFraming for RawDeflateFraming {
    const LABEL: &'static str = "deflate";

    type Decoder<'a> = flate2::read::DeflateDecoder<&'a [u8]>;

    fn decoder(input: &[u8]) -> Self::Decoder<'_> {
        flate2::read::DeflateDecoder::new(input)
    }
}

/// Largest number of output bytes one input byte of DEFLATE can produce.
///
/// Derivation (RFC 1951 sections 3.2.5 and 3.2.7): a match copies at most 258
/// bytes (length code 285, no extra bits). Every used symbol costs at least one
/// bit, and a block with a single distance code still spends one bit on it, so
/// a match costs at least two bits: eight bits yield at most `4 * 258 = 1032`
/// bytes. A literal yields one byte for at least one bit, a stored block `LEN`
/// bytes for `LEN + 4` bytes, and headers, end-of-block codes and zlib or gzip
/// framing yield nothing, so the bound holds for a whole stream and for
/// concatenated gzip members. `all_zero_input_stays_within_the_expansion_bound`
/// measures the densest encoder output against it and
/// `densest_stream_decodes_in_full_under_a_zero_declaration` decodes a
/// hand-built block at 1031.81 to one.
pub const MAX_DEFLATE_EXPANSION: usize = 1032;

/// Budget that leaves [`flate2_decompress`] bounded by the producible size alone.
///
/// The codecs pass it: they have no output size to enforce beyond what the
/// stream can produce, and the shared 64 MiB budget would refuse honest data.
pub(crate) const PRODUCIBLE_ONLY: ParseBudget = ParseBudget::new(
    usize::MAX,
    ParseBudget::DEFAULT.max_elements,
    ParseBudget::DEFAULT.max_depth,
);

/// Most output bytes any DEFLATE stream of `input_len` bytes can produce.
///
/// Saturates, so a length whose product overflows `usize` is bounded by
/// `usize::MAX` instead of wrapping to a small ceiling.
fn producible(input_len: usize) -> usize {
    input_len.saturating_mul(MAX_DEFLATE_EXPANSION)
}

/// Decompress `input` under framing `F`.
///
/// The declared `expected_size` is a capacity hint, never a refusal: the initial
/// reservation is the smaller of it and the output ceiling, so a hostile
/// declaration reserves at most 1032 bytes per byte of input. The output grows
/// only as decoded data arrives, through the budget's fallible reservation, and
/// ends at the ceiling:
/// `min(input.len() * MAX_DEFLATE_EXPANSION, F::ceiling(expected_size, budget))`.
/// The first term is what no stream can exceed; the second is the caller's own
/// limit, which is what a budget bounds:
///
/// - the codecs pass a budget with no byte cap, so only the first term applies
///   and any honest stream decodes whatever it declares;
/// - `consus-zip` passes the member's declared size, so a stream that expands
///   past its directory entry is refused;
/// - the Parquet reader passes the shared 64 MiB budget, which its own check
///   already applies to the page's declaration.
///
/// # Errors
///
/// Returns [`Error::CompressionError`] when the stream is malformed and
/// [`Error::ResourceLimit`] when the decompressed output exceeds the
/// ceiling or the reservation cannot be satisfied.
pub fn flate2_decompress<F: FlateFraming>(
    input: &[u8],
    expected_size: usize,
    budget: &ParseBudget,
) -> Result<Vec<u8>> {
    let ceiling = producible(input.len()).min(F::ceiling(expected_size, budget));
    let mut decoder = F::decoder(input);
    let bounded = ParseBudget::new(ceiling, budget.max_elements, budget.max_depth);
    bounded
        .read_bounded(
            &mut decoder,
            expected_size.min(ceiling),
            "decompressed output",
        )
        .map_err(|error| match error {
            Error::Io(error) => Error::CompressionError {
                message: alloc::format!("{} decompress failed: {error}", F::LABEL),
            },
            error => error,
        })
}

/// Compress `input` into one zlib stream (RFC 1950).
///
/// `level` is clamped to `[0, 9]`. The stream is `CMF FLG <DEFLATE> ADLER32`
/// with the Adler-32 of `input` stored most-significant byte first
/// (RFC 1950 section 2.2), which is what `zlib.compress` in CPython and the
/// HDF5 deflate filter write.
///
/// # Errors
///
/// Returns [`Error::CompressionError`] when the encoder fails.
pub fn zlib_compress(input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
    let clamped = level.0.clamp(0, 9).unsigned_abs();
    let mut encoder = flate2::read::ZlibEncoder::new(input, flate2::Compression::new(clamped));
    let mut output = Vec::new();
    encoder
        .read_to_end(&mut output)
        .map_err(|e| Error::CompressionError {
            message: alloc::format!("{} compress failed: {e}", ZlibFraming::LABEL),
        })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A declared size one byte above the default budget gives the same
    /// ceiling under zlib, gzip and raw DEFLATE framing.
    ///
    /// Derivation: the rule is `max(declared, budget)`, so a declaration of
    /// `budget + 1` yields `budget + 1` for every framing; the capacity hint
    /// does not enter it.
    #[test]
    fn ceiling_above_default_budget_is_one_rule() {
        let budget = ParseBudget::default();
        let declared = budget.max_alloc_bytes + 1;

        let zlib = ZlibFraming::ceiling(declared, &budget);
        let gzip = GzipFraming::ceiling(declared, &budget);
        let raw = RawDeflateFraming::ceiling(declared, &budget);

        assert_eq!(zlib, declared);
        assert_eq!(gzip, declared);
        assert_eq!(raw, declared);
    }

    /// A declaration at or below the budget keeps the budget ceiling.
    #[test]
    fn ceiling_at_or_below_default_budget_is_the_budget() {
        let budget = ParseBudget::default();
        for declared in [0, 1, budget.max_alloc_bytes] {
            assert_eq!(
                ZlibFraming::ceiling(declared, &budget),
                budget.max_alloc_bytes
            );
            assert_eq!(
                GzipFraming::ceiling(declared, &budget),
                budget.max_alloc_bytes
            );
            assert_eq!(
                RawDeflateFraming::ceiling(declared, &budget),
                budget.max_alloc_bytes
            );
        }
    }

    /// The expansion bound is eight bits of two-bit matches, 258 bytes each.
    #[test]
    fn expansion_bound_is_four_maximal_matches_per_byte() {
        assert_eq!(MAX_DEFLATE_EXPANSION, 4 * 258);
    }

    /// The producible size is the exact product below overflow and saturates
    /// above it, so a 32-bit length of 4,161,791 bytes or more never wraps.
    #[test]
    fn producible_size_saturates_instead_of_wrapping() {
        assert_eq!(producible(0), 0);
        assert_eq!(producible(1), 1032);
        let last_exact = usize::MAX / MAX_DEFLATE_EXPANSION;
        assert_eq!(producible(last_exact), last_exact * MAX_DEFLATE_EXPANSION);
        assert_eq!(producible(last_exact.saturating_add(1)), usize::MAX);
        assert_eq!(producible(usize::MAX), usize::MAX);
    }
}
