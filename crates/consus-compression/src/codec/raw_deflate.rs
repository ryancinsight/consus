//! Raw DEFLATE codec (RFC 1951) with no zlib or gzip framing.
//!
//! ZIP stores method-8 members as bare DEFLATE streams: no header, no
//! trailer, and the integrity check lives in the archive's CRC-32 field.
//! Decompression goes through the shared bounded [`flate2_decompress`]
//! path, so a hostile stream cannot grow its output past the ceiling.

use alloc::vec::Vec;

use flate2::read::DeflateEncoder;
use std::io::Read;

use super::flate::{RawDeflateFraming, flate2_decompress};
use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, ParseBudget, Result};

/// DEFLATE codec without framing (RFC 1951).
///
/// Used by ZIP compression method 8. It has no HDF5 filter ID: HDF5's
/// deflate filter is zlib-framed ([`DeflateCodec`](super::deflate::DeflateCodec)).
#[derive(Debug, Default)]
pub struct RawDeflateCodec;

impl Codec for RawDeflateCodec {
    fn name(&self) -> &str {
        "raw-deflate"
    }

    fn hdf5_filter_id(&self) -> Option<u16> {
        None
    }

    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let clamped = level.0.clamp(0, 9).cast_unsigned();
        let mut encoder = DeflateEncoder::new(input, flate2::Compression::new(clamped));
        let mut output = Vec::new();
        encoder
            .read_to_end(&mut output)
            .map_err(|e| Error::CompressionError {
                message: alloc::format!("raw deflate compress failed: {e}"),
            })?;
        Ok(output)
    }

    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        flate2_decompress::<RawDeflateFraming>(input, expected_size, &ParseBudget::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_restores_input_bytes() {
        let codec = RawDeflateCodec;
        let input: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let compressed = codec.compress(&input, CompressionLevel(6)).unwrap();
        assert!(compressed.len() < input.len());
        assert_eq!(codec.decompress(&compressed, input.len()).unwrap(), input);
    }

    #[test]
    fn output_has_no_zlib_header() {
        // A zlib stream opens with CMF 0x78, whose bit 0 is clear. A raw
        // stream opens with a block header whose bit 0 is BFINAL, and an input
        // this short fits one final block, so bit 0 is set.
        let compressed = RawDeflateCodec
            .compress(b"consus raw deflate", CompressionLevel(6))
            .unwrap();
        assert_eq!(compressed[0] & 1, 1);
        let decoded = RawDeflateCodec.decompress(&compressed, 18).unwrap();
        assert_eq!(decoded, b"consus raw deflate");
    }

    #[test]
    fn truncated_stream_is_a_compression_error() {
        let compressed = RawDeflateCodec
            .compress(&[7u8; 1024], CompressionLevel(9))
            .unwrap();
        let err = RawDeflateCodec
            .decompress(&compressed[..compressed.len() / 2], 1024)
            .unwrap_err();
        assert!(matches!(err, Error::CompressionError { .. }), "{err:?}");
    }
}
