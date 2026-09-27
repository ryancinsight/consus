//! Shared, bounded `flate2` decompression scaffolding.
//!
//! Deflate/zlib, gzip, and raw DEFLATE all decode through `flate2` reader
//! types that must never be drained with an unbounded `read_to_end`: a hostile
//! stream would otherwise grow the output `Vec` until the allocator aborts an
//! uncatchable process. This module implements the bounded read once and
//! selects the wire framing with a zero-sized [`FlateFraming`](crate::codec::flate::FlateFraming) marker type.
//!
//! Every framing keeps at least the shared [`ParseBudget`](consus_core::ParseBudget) byte ceiling, so a
//! stream that expands past it is rejected with [`Error::ResourceLimit`](consus_core::Error::ResourceLimit)
//! instead of exhausting memory.

use std::io::Read;

use consus_core::{Error, ParseBudget, Result};

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
    /// Every framing keeps at least the shared [`ParseBudget`] ceiling; a
    /// framing whose callers can declare a larger legitimate size widens the
    /// ceiling to that declaration so honest data is never rejected.
    fn ceiling(expected_size: usize, budget: &ParseBudget) -> usize;
}

/// zlib framing (RFC 1950), used by the HDF5 deflate filter (ID 1).
pub struct ZlibFraming;

impl FlateFraming for ZlibFraming {
    const LABEL: &'static str = "deflate (zlib)";

    type Decoder<'a> = flate2::read::ZlibDecoder<&'a [u8]>;

    fn decoder(input: &[u8]) -> Self::Decoder<'_> {
        flate2::read::ZlibDecoder::new(input)
    }

    fn ceiling(_expected_size: usize, budget: &ParseBudget) -> usize {
        budget.max_alloc_bytes
    }
}

/// gzip framing (RFC 1952), used by Zarr's gzip codec.
pub struct GzipFraming;

impl FlateFraming for GzipFraming {
    const LABEL: &'static str = "gzip";

    type Decoder<'a> = flate2::read::GzDecoder<&'a [u8]>;

    fn decoder(input: &[u8]) -> Self::Decoder<'_> {
        flate2::read::GzDecoder::new(input)
    }

    fn ceiling(expected_size: usize, budget: &ParseBudget) -> usize {
        expected_size.max(budget.max_alloc_bytes)
    }
}

/// Raw DEFLATE framing (RFC 1951), used by Parquet's GZIP/ZLIB codecs.
pub struct RawDeflateFraming;

impl FlateFraming for RawDeflateFraming {
    const LABEL: &'static str = "deflate";

    type Decoder<'a> = flate2::read::DeflateDecoder<&'a [u8]>;

    fn decoder(input: &[u8]) -> Self::Decoder<'_> {
        flate2::read::DeflateDecoder::new(input)
    }

    fn ceiling(expected_size: usize, budget: &ParseBudget) -> usize {
        expected_size.max(budget.max_alloc_bytes)
    }
}

/// Decompress `input` under framing `F`, bounding the output exactly once.
///
/// The declared `expected_size` is a capacity hint only: the stream is capped
/// at [`FlateFraming::ceiling`] and every growth goes through the budget's
/// fallible reservation, so a hostile stream is rejected rather than allowed
/// to exhaust memory.
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
