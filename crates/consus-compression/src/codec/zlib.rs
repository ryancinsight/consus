//! Zlib codec (RFC 1950) for the Zarr v2 `zlib` compressor id.
//!
//! ## Format
//!
//! A zlib stream is `CMF FLG [DICTID] <DEFLATE data> ADLER32`
//! (RFC 1950 section 2.2, <https://www.rfc-editor.org/rfc/rfc1950>):
//!
//! - `CMF` low nibble `CM = 8` selects DEFLATE; `CM = 15` is reserved. The
//!   high nibble `CINFO` is the base-2 logarithm of the window size minus 8,
//!   so the common 32 KiB window gives `CMF = 0x78`.
//! - `FLG` carries `FCHECK`, chosen so that `CMF * 256 + FLG` is a multiple of
//!   31 (for example `0x789c`, `0x7801`).
//! - `ADLER32` is the Adler-32 of the uncompressed data, stored
//!   most-significant byte first.
//!
//! numcodecs `Zlib` (`src/numcodecs/zlib.py` at zarr-developers/numcodecs
//! commit `4afa6287f8305a824942c9d76e00249cba5e55d8`, `codec_id = 'zlib'`)
//! encodes with CPython `zlib.compress(buf, level)` and decodes with
//! `zlib.decompress`, so the chunks zarr-python writes under the `zlib`
//! compressor id are exactly this framing. This codec is that mapping; it
//! uses the same encoder and decoder as `DeflateCodec` (the HDF5 deflate
//! filter, available under the `deflate` feature), which writes the same
//! framing.
//!
//! ## Gzip fallback on decode
//!
//! Earlier versions registered the name `zlib` against the gzip codec, so
//! `CodecId::Name("zlib")` resolved to a gzip encoder and decoder. A caller
//! that stored chunks through that entry holds RFC 1952 streams under the
//! name `zlib`. [`ZlibCodec::decompress`] keeps them readable: input that
//! begins with the gzip magic `1f 8b` is decoded as gzip. The test is
//! unambiguous because `0x1f` as a `CMF` byte has `CM = 15`, which RFC 1950
//! reserves, and `0x1f8b` is not a multiple of 31 either, so no valid zlib
//! stream starts with it. The fallback is decode-only: `compress` always
//! writes RFC 1950, and a gzip stream is accepted under this name only
//! through this check.

use alloc::vec::Vec;

use super::flate::{GzipFraming, PRODUCIBLE_ONLY, ZlibFraming, flate2_decompress, zlib_compress};
use super::traits::{Codec, CompressionLevel};
use consus_core::Result;

/// First two bytes of every gzip member (RFC 1952 section 2.3.1, `ID1 ID2`).
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Codec registered under the name `zlib`, writing and reading RFC 1950.
///
/// ## Contract
///
/// - `compress` produces one zlib stream whose first byte is `0x78` for every
///   level in `[0, 9]` (32 KiB window); `decompress` inverts it exactly.
/// - `decompress` validates the header check bits and the Adler-32 trailer and
///   returns [`Error::CompressionError`](consus_core::Error::CompressionError)
///   on any stream or integrity failure.
/// - `decompress` also accepts a gzip stream (leading `1f 8b`) written by
///   earlier versions of this crate under the same name (see the module
///   documentation).
#[derive(Debug, Default)]
pub struct ZlibCodec;

impl Codec for ZlibCodec {
    fn name(&self) -> &str {
        "zlib"
    }

    /// Returns `None`; HDF5 filter 1 is the `deflate` codec.
    fn hdf5_filter_id(&self) -> Option<u16> {
        None
    }

    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        zlib_compress(input, level)
    }

    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        if input.starts_with(&GZIP_MAGIC) {
            flate2_decompress::<GzipFraming>(input, expected_size, &PRODUCIBLE_ONLY)
        } else {
            flate2_decompress::<ZlibFraming>(input, expected_size, &PRODUCIBLE_ONLY)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::gzip::GzipCodec;
    use consus_core::Error;

    /// `b"consus zlib framing " * 4`: 80 bytes.
    fn payload() -> Vec<u8> {
        b"consus zlib framing ".repeat(4)
    }

    /// `zlib.compress(payload, 6)` from CPython 3.13.12 with zlib 1.3.1.
    /// numcodecs `Zlib.encode` returns exactly this call's result.
    const CPYTHON_ZLIB: [u8; 31] = [
        0x78, 0x9c, 0x4b, 0xce, 0xcf, 0x2b, 0x2e, 0x2d, 0x56, 0xa8, 0xca, 0xc9, 0x4c, 0x52, 0x48,
        0x2b, 0x4a, 0xcc, 0xcd, 0xcc, 0x4b, 0x57, 0x48, 0xa6, 0x40, 0x0c, 0x00, 0xd5, 0xc8, 0x1e,
        0x41,
    ];

    /// `gzip.compress(payload, 6, mtime=0)` from CPython 3.13.12.
    const CPYTHON_GZIP: [u8; 43] = [
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x4b, 0xce, 0xcf, 0x2b, 0x2e,
        0x2d, 0x56, 0xa8, 0xca, 0xc9, 0x4c, 0x52, 0x48, 0x2b, 0x4a, 0xcc, 0xcd, 0xcc, 0x4b, 0x57,
        0x48, 0xa6, 0x40, 0x0c, 0x00, 0x00, 0x91, 0x29, 0x4b, 0x50, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn decodes_cpython_zlib_stream() {
        let decoded = ZlibCodec
            .decompress(&CPYTHON_ZLIB, payload().len())
            .expect("RFC 1950 stream must decode");
        assert_eq!(decoded, payload());
    }

    #[test]
    fn decodes_legacy_gzip_stream_under_zlib_name() {
        let decoded = ZlibCodec
            .decompress(&CPYTHON_GZIP, payload().len())
            .expect("gzip-framed legacy chunk must decode");
        assert_eq!(decoded, payload());

        let written_by_previous_mapping = GzipCodec
            .compress(&payload(), CompressionLevel(6))
            .expect("gzip compress must succeed");
        assert_eq!(&written_by_previous_mapping[..2], &GZIP_MAGIC);
        let decoded = ZlibCodec
            .decompress(&written_by_previous_mapping, payload().len())
            .expect("gzip chunk written by the previous mapping must decode");
        assert_eq!(decoded, payload());
    }

    /// RFC 1950 section 2.2: `CM = 8`, `CMF * 256 + FLG` a multiple of 31, and
    /// the trailer is the big-endian Adler-32 of the input.
    #[test]
    fn compress_writes_rfc1950_framing_at_every_level() {
        for level in 0..=9 {
            let encoded = ZlibCodec
                .compress(&payload(), CompressionLevel(level))
                .expect("compress must succeed");
            assert_eq!(encoded[0], 0x78, "level {level}: CMF is CM=8, CINFO=7");
            let header = u16::from_be_bytes([encoded[0], encoded[1]]);
            assert_eq!(header % 31, 0, "level {level}: FCHECK");
            let trailer = &encoded[encoded.len() - 4..];
            assert_eq!(
                u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]),
                adler32(&payload()),
                "level {level}: ADLER32 trailer"
            );
            assert_eq!(
                ZlibCodec
                    .decompress(&encoded, payload().len())
                    .expect("round trip must decode"),
                payload()
            );
        }
    }

    /// Level 6 output is the CPython bytes: the encoder is a zlib-format
    /// deflate, and the header for the default level is `78 9c`.
    #[test]
    fn level_six_header_matches_cpython() {
        let encoded = ZlibCodec
            .compress(&payload(), CompressionLevel(6))
            .expect("compress must succeed");
        assert_eq!(&encoded[..2], &CPYTHON_ZLIB[..2]);
    }

    #[test]
    fn malformed_streams_are_typed_errors() {
        let mut bad_check = CPYTHON_ZLIB;
        bad_check[1] = 0x00;
        let mut bad_trailer = CPYTHON_ZLIB;
        *bad_trailer
            .last_mut()
            .expect("invariant: non-empty literal") ^= 0xff;
        let truncated = &CPYTHON_ZLIB[..CPYTHON_ZLIB.len() - 6];
        for input in [&bad_check[..], &bad_trailer[..], truncated, &[0x78]] {
            let error = ZlibCodec
                .decompress(input, payload().len())
                .expect_err("malformed stream must be rejected");
            assert!(
                matches!(error, Error::CompressionError { .. }),
                "got {error:?}"
            );
        }
    }

    /// Adler-32 per RFC 1950 section 2.2: `s1` is the byte sum starting at 1,
    /// `s2` the sum of every `s1`, both modulo 65521, result `s2 * 65536 + s1`.
    fn adler32(data: &[u8]) -> u32 {
        const MODULUS: u32 = 65_521;
        let (mut s1, mut s2) = (1u32, 0u32);
        for &byte in data {
            s1 = (s1 + u32::from(byte)) % MODULUS;
            s2 = (s2 + s1) % MODULUS;
        }
        (s2 << 16) | s1
    }
}
