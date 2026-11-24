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
use consus_zarr::chunk::{ChunkError, Selection, read_array, write_array};
use consus_zarr::codec::{CodecPipeline, CompressionRegistryTrait, default_registry};
use consus_zarr::metadata::{ArrayMetadata, ArrayMetadataV2, ZarrJson};
use consus_zarr::store::{InMemoryStore, Store};

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

/// zlib's `Z_DEFAULT_COMPRESSION` resolves to level 6 (`zlib.h`), and numcodecs
/// `Zlib` and `GZip` default to the same call.
const ZLIB_DEFAULT_LEVEL: i32 = 6;

/// `b"configured compression level "` repeated 1024 times, 29,696 bytes.
fn level_input() -> Vec<u8> {
    b"configured compression level ".repeat(1024)
}

fn v2_metadata(name: &str, level: Option<i32>) -> ArrayMetadata {
    let len = level_input().len();
    let configuration = level
        .map(|n| format!(", \"level\": {n}"))
        .unwrap_or_default();
    let json = format!(
        r#"{{"zarr_format":2,"shape":[{len}],"chunks":[{len}],
        "dtype":"|u1","fill_value":0,"order":"C",
        "compressor":{{"id":"{name}"{configuration}}},"filters":null}}"#
    );
    ArrayMetadataV2::parse(&json)
        .expect("valid v2 metadata")
        .to_canonical()
}

fn v3_metadata(name: &str, level: &str) -> ArrayMetadata {
    let len = level_input().len();
    let json = format!(
        r#"{{"zarr_format":3,"node_type":"array","shape":[{len}],"data_type":"uint8",
        "chunk_grid":{{"name":"regular","configuration":{{"chunk_shape":[{len}]}}}},
        "chunk_key_encoding":{{"name":"default","configuration":{{"separator":"/"}}}},
        "codecs":[{{"name":"bytes","configuration":{{"endian":"little"}}}},
                  {{"name":"{name}","configuration":{{"level":{level}}}}}],
        "fill_value":0}}"#
    );
    ZarrJson::parse(&json)
        .expect("valid v3 metadata")
        .to_array_canonical()
        .expect("array node")
}

fn write_level_input(metadata: &ArrayMetadata) -> (InMemoryStore, Result<(), ChunkError>) {
    let mut store = InMemoryStore::new();
    let result = write_array(&mut store, "array", metadata, &level_input());
    (store, result)
}

/// The single chunk object a write left under the array key, whatever the
/// metadata version's chunk key encoding.
fn stored_chunk(store: &InMemoryStore) -> Vec<u8> {
    let keys = store.list("array/").expect("list array keys");
    assert_eq!(keys.len(), 1, "one chunk object: {keys:?}");
    store.get(&keys[0]).expect("written chunk")
}

/// Writes `level_input` through the Zarr array path, checks the stored chunk
/// against the direct encoder at `expected_level`, and decodes it.
fn write_compressed_chunk(name: &str, level: Option<i32>, expected_level: i32) -> Vec<u8> {
    let metadata = v2_metadata(name, level);
    let (store, result) = write_level_input(&metadata);
    result.expect("write full Zarr array");
    let encoded = stored_chunk(&store);
    let expected = default_registry()
        .get_by_name(name)
        .expect("codec")
        .compress(
            &level_input(),
            consus_zarr::codec::CompressionLevel(expected_level),
        )
        .expect("direct codec compression");
    assert_eq!(
        encoded, expected,
        "{name} level {level:?} must reach the encoder as {expected_level}"
    );
    assert_eq!(
        CodecPipeline::single(metadata.codecs[0].clone())
            .decompress(&encoded, default_registry())
            .expect("read written chunk"),
        level_input()
    );
    encoded
}

/// Smallest stream a conformant encoder emits at level 0 for `level_input`:
/// one stored block per 65,535 input bytes, each with a 5-byte header
/// (RFC 1951 section 3.2.4), plus the container framing: zlib is a 2-byte
/// header and 4-byte Adler-32 (RFC 1950 section 2.2), gzip a 10-byte header
/// and 8-byte CRC-32 and length (RFC 1952 section 2.3).
fn stored_floor(framing: usize) -> usize {
    const STORED_BLOCK_MAX: usize = 65_535;
    const STORED_BLOCK_HEADER: usize = 5;
    let input = level_input().len();
    input + input.div_ceil(STORED_BLOCK_MAX) * STORED_BLOCK_HEADER + framing
}

const ZLIB_FRAMING: usize = 2 + 4;
const GZIP_FRAMING: usize = 10 + 8;

#[test]
fn zarr_writes_honor_zlib_levels_and_default() {
    let stored = write_compressed_chunk("zlib", Some(0), 0);
    let compressed = write_compressed_chunk("zlib", Some(9), 9);
    assert!(stored.len() >= stored_floor(ZLIB_FRAMING), "level 0 stores");
    assert!(
        compressed.len() < level_input().len(),
        "level 9 shrinks repetitive input"
    );
    let default = write_compressed_chunk("zlib", None, ZLIB_DEFAULT_LEVEL);
    assert_eq!(
        default,
        write_compressed_chunk("zlib", Some(ZLIB_DEFAULT_LEVEL), ZLIB_DEFAULT_LEVEL)
    );
    assert_eq!(
        default,
        write_compressed_chunk("zlib", Some(-1), ZLIB_DEFAULT_LEVEL),
        "-1 is Z_DEFAULT_COMPRESSION"
    );
}

#[test]
fn zarr_writes_preserve_gzip_levels_and_default() {
    let stored = write_compressed_chunk("gzip", Some(0), 0);
    let compressed = write_compressed_chunk("gzip", Some(9), 9);
    assert!(stored.len() >= stored_floor(GZIP_FRAMING), "level 0 stores");
    assert!(
        compressed.len() < level_input().len(),
        "level 9 shrinks repetitive input"
    );
    let default = write_compressed_chunk("gzip", None, ZLIB_DEFAULT_LEVEL);
    assert_eq!(
        default,
        write_compressed_chunk("gzip", Some(ZLIB_DEFAULT_LEVEL), ZLIB_DEFAULT_LEVEL)
    );
    assert_eq!(
        default,
        write_compressed_chunk("gzip", Some(-1), ZLIB_DEFAULT_LEVEL),
        "-1 is Z_DEFAULT_COMPRESSION"
    );
}

fn assert_rejected_level(metadata: &ArrayMetadata, name: &str, declared: &str) {
    let (store, result) = write_level_input(metadata);
    let error = result.expect_err("out-of-range level must be rejected");
    assert!(
        matches!(
            &error,
            ChunkError::InvalidCodecConfiguration(message)
                if message.contains(&format!("{declared:?}"))
                    && message.contains(&format!("`{name}`"))
        ),
        "{name} level {declared}: {error:?}"
    );
    assert_eq!(
        store.list("array/").expect("list array keys"),
        Vec::<String>::new(),
        "a rejected write stores no chunk"
    );
}

#[test]
fn v2_out_of_range_levels_are_rejected_not_clamped() {
    for name in ["gzip", "zlib"] {
        for level in [10, i32::MAX, -2, i32::MIN] {
            assert_rejected_level(&v2_metadata(name, Some(level)), name, &level.to_string());
        }
    }
}

#[test]
fn v3_out_of_range_levels_are_rejected_not_clamped() {
    for name in ["gzip", "zlib"] {
        for level in ["10", "-2", "1.5", "\"six\""] {
            let declared = level.trim_matches('"');
            assert_rejected_level(&v3_metadata(name, level), name, declared);
        }
    }
}

#[test]
fn v3_levels_in_range_reach_the_encoder() {
    for (name, framing) in [("gzip", GZIP_FRAMING), ("zlib", ZLIB_FRAMING)] {
        let written = |level: &str| {
            let (store, result) = write_level_input(&v3_metadata(name, level));
            result.expect("in-range level");
            stored_chunk(&store)
        };
        assert!(written("0").len() >= stored_floor(framing));
        assert!(written("9").len() < level_input().len());
    }
}

#[test]
fn decoding_does_not_consult_the_declared_level() {
    let written = write_compressed_chunk("zlib", Some(9), 9);
    for declared in [10, -2, i32::MAX] {
        let metadata = v2_metadata("zlib", Some(declared));
        let mut store = InMemoryStore::new();
        store.set("array/0", &written).expect("store chunk");
        let read = read_array(&store, "array", &Selection::full(1), &metadata)
            .expect("a stored level never blocks a read");
        assert_eq!(read, level_input(), "declared level {declared}");
    }
}
