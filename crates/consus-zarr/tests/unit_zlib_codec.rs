//! `zlib` codec behavior of the Zarr codec pipeline.
//!
//! The Zarr v2 `zlib` compressor is numcodecs `Zlib`, whose `encode` is
//! CPython `zlib.compress(buf, level)` and whose `decode` is
//! `zlib.decompress`: RFC 1950 streams (`CMF FLG <DEFLATE> ADLER32`,
//! <https://www.rfc-editor.org/rfc/rfc1950> section 2.2; numcodecs source
//! `src/numcodecs/zlib.py`). The reference bytes below are produced by that
//! call, so decoding them is a cross-implementation check, not a round trip.

use consus_core::Error;
use consus_zarr::Codec;
use consus_zarr::codec::{CodecPipeline, default_registry};

/// `b"consus zlib framing " * 4`, 80 bytes.
fn payload() -> Vec<u8> {
    b"consus zlib framing ".repeat(4)
}

/// `zlib.compress(payload, 6)`, CPython 3.13.12 with zlib 1.3.1.
const CPYTHON_ZLIB: [u8; 31] = [
    0x78, 0x9c, 0x4b, 0xce, 0xcf, 0x2b, 0x2e, 0x2d, 0x56, 0xa8, 0xca, 0xc9, 0x4c, 0x52, 0x48, 0x2b,
    0x4a, 0xcc, 0xcd, 0xcc, 0x4b, 0x57, 0x48, 0xa6, 0x40, 0x0c, 0x00, 0xd5, 0xc8, 0x1e, 0x41,
];

/// `gzip.compress(payload, 6, mtime=0)`, CPython 3.13.12; the framing earlier
/// versions of this crate wrote under the `zlib` name.
const CPYTHON_GZIP: [u8; 43] = [
    0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x4b, 0xce, 0xcf, 0x2b, 0x2e, 0x2d,
    0x56, 0xa8, 0xca, 0xc9, 0x4c, 0x52, 0x48, 0x2b, 0x4a, 0xcc, 0xcd, 0xcc, 0x4b, 0x57, 0x48, 0xa6,
    0x40, 0x0c, 0x00, 0x00, 0x91, 0x29, 0x4b, 0x50, 0x00, 0x00, 0x00,
];

fn zlib_pipeline() -> CodecPipeline {
    CodecPipeline::single(Codec {
        name: String::from("zlib"),
        configuration: Vec::new(),
    })
}

#[test]
fn decodes_numcodecs_zlib_stream() {
    let decoded = zlib_pipeline()
        .decompress(&CPYTHON_ZLIB, default_registry())
        .expect("RFC 1950 stream must decode under \"zlib\"");
    assert_eq!(decoded, payload());
}

#[test]
fn legacy_gzip_chunk_still_decodes_under_zlib() {
    let decoded = zlib_pipeline()
        .decompress(&CPYTHON_GZIP, default_registry())
        .expect("gzip-framed legacy chunk must decode under \"zlib\"");
    assert_eq!(decoded, payload());

    let gzip_written_by_consus = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: Vec::new(),
    })
    .compress(&payload(), default_registry())
    .expect("gzip compress must succeed");
    let decoded = zlib_pipeline()
        .decompress(&gzip_written_by_consus, default_registry())
        .expect("consus gzip chunk must decode under \"zlib\"");
    assert_eq!(decoded, payload());
}

#[test]
fn zlib_encode_writes_rfc1950_framing_and_round_trips() {
    let encoded = zlib_pipeline()
        .compress(&payload(), default_registry())
        .expect("zlib compress must succeed");
    assert_eq!(
        encoded[0], 0x78,
        "CMF: CM=8 deflate, CINFO=7 (32 KiB window)"
    );
    let header = u16::from_be_bytes([encoded[0], encoded[1]]);
    assert_eq!(header % 31, 0, "FCHECK makes CMF*256+FLG a multiple of 31");
    assert_ne!(&encoded[..2], &[0x1f, 0x8b], "must not be gzip-framed");
    let decoded = zlib_pipeline()
        .decompress(&encoded, default_registry())
        .expect("round trip must decode");
    assert_eq!(decoded, payload());
}

#[test]
fn malformed_zlib_stream_is_a_compression_error() {
    let mut bad_check = CPYTHON_ZLIB;
    bad_check[1] = 0x00;
    let mut bad_trailer = CPYTHON_ZLIB;
    *bad_trailer
        .last_mut()
        .expect("invariant: reference literal is non-empty") ^= 0xff;
    let truncated = &CPYTHON_ZLIB[..CPYTHON_ZLIB.len() - 6];
    for input in [&bad_check[..], &bad_trailer[..], truncated] {
        let error = zlib_pipeline()
            .decompress(input, default_registry())
            .expect_err("malformed stream must be rejected");
        assert!(
            matches!(error, Error::CompressionError { .. }),
            "got {error:?}"
        );
    }
}
