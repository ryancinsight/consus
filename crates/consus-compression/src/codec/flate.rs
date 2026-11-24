//! Shared, bounded `flate2` decompression scaffolding.
//!
//! Deflate/zlib, gzip, and raw DEFLATE all decode through `flate2` reader
//! types that must never be drained with an unbounded `read_to_end`: a hostile
//! stream would otherwise grow the output `Vec` until the allocator aborts an
//! uncatchable process. This module implements the bounded read once and
//! selects the wire framing with a zero-sized [`FlateFraming`](crate::codec::flate::FlateFraming) marker type.
//!
//! Every framing keeps at least the shared [`ParseBudget`](consus_core::ParseBudget) byte ceiling and
//! widens it to the caller's declared size, so a stream that expands past both is rejected with
//! [`Error::ResourceLimit`](consus_core::Error::ResourceLimit) instead of exhausting memory.

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

    /// Output ceiling, in bytes, for one decompression.
    ///
    /// One rule serves every framing: the larger of the declared
    /// `expected_size` and the shared [`ParseBudget`] byte ceiling. A caller
    /// that declares a larger legitimate size therefore decodes it, whichever
    /// framing carries the stream, and a stream that declares less is still
    /// held to the budget.
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

/// Decompress `input` under framing `F`, bounding the output exactly once.
///
/// The declared `expected_size` is the initial reservation and, when it
/// exceeds the budget, the ceiling ([`FlateFraming::ceiling`]), so a caller
/// that takes the declaration from untrusted input bounds it first. A stream
/// that expands past the ceiling is rejected, and every growth goes through
/// the budget's fallible reservation.
///
/// # Errors
///
/// Returns [`Error::CompressionError`] when the stream is malformed and
/// [`Error::ResourceLimit`] when the decompressed output would exceed the
/// ceiling.
pub fn flate2_decompress<F: FlateFraming>(
    input: &[u8],
    expected_size: usize,
    budget: &ParseBudget,
) -> Result<Vec<u8>> {
    let mut decoder = F::decoder(input);
    let bounded = ParseBudget::new(
        F::ceiling(expected_size, budget),
        budget.max_elements,
        budget.max_depth,
    );
    bounded
        .read_bounded(&mut decoder, expected_size, "decompressed output")
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
}
