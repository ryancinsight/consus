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

/// Level written when a `gzip` or `zlib` codec declares none: this crate's
/// policy (CHANGELOG, Unreleased). It is also what zlib's `-1` resolves to
/// (`Z_DEFAULT_COMPRESSION`, "currently equivalent to level 6", `zlib.h`).
const ZLIB_DEFAULT_LEVEL: i32 = 6;

/// 29,696 bytes: `b"configured compression level "` repeated 512 times,
/// then the same count of `(7 i + 13 (i >> 5)) mod 251`. The structured tail
/// makes the levels the zstd test writes (-8, 1, 3, 4) produce different
/// bytes, which a repeated phrase alone does not.
fn level_input() -> Vec<u8> {
    let phrase = b"configured compression level ".repeat(512);
    let tail = (0..phrase.len())
        .map(|i| u8::try_from((7 * i + 13 * (i >> 5)) % 251).expect("invariant: residue < 251"));
    phrase.iter().copied().chain(tail).collect()
}

/// Where a codec declares its level: Zarr v2 inline beside `id`, Zarr v2
/// inside the compressor's `configuration` object, Zarr v2 inline on a
/// `filters` entry (the compressor is then `null`), Zarr v3 `configuration`,
/// or Zarr v2 inline beside a numeric HDF5 filter id (`zstd` only, so not in
/// `PLACEMENTS`, which `gzip` and `zlib` share).
#[derive(Clone, Copy, Debug)]
enum LevelAt {
    Inline,
    Nested,
    Filter,
    V3,
    NumericId,
}

const PLACEMENTS: [LevelAt; 4] = [
    LevelAt::Inline,
    LevelAt::Nested,
    LevelAt::Filter,
    LevelAt::V3,
];

/// A one-chunk `uint8` Zarr v3 array whose codec chain is the JSON `codecs` list body.
fn v3_metadata(codecs: &str) -> ArrayMetadata {
    let len = level_input().len();
    let json = format!(
        r#"{{"zarr_format":3,"node_type":"array","shape":[{len}],"data_type":"uint8",
        "chunk_grid":{{"name":"regular","configuration":{{"chunk_shape":[{len}]}}}},
        "chunk_key_encoding":{{"name":"default","configuration":{{"separator":"/"}}}},
        "codecs":[{codecs}],"fill_value":0}}"#
    );
    ZarrJson::parse(&json)
        .expect("valid v3 metadata")
        .to_array_canonical()
        .expect("array node")
}

/// A one-chunk `uint8` Zarr v2 array with the given `compressor` and `filters`
/// JSON.
fn v2_metadata(compressor: &str, filters: &str) -> ArrayMetadata {
    let len = level_input().len();
    let json = format!(
        r#"{{"zarr_format":2,"shape":[{len}],"chunks":[{len}],
        "dtype":"|u1","fill_value":0,"order":"C",
        "compressor":{compressor},"filters":{filters}}}"#
    );
    ArrayMetadataV2::parse(&json)
        .expect("valid v2 metadata")
        .to_canonical()
}

/// A one-chunk `uint8` array whose `name` codec declares `level` (JSON text)
/// at `at`, or declares none.
fn metadata(name: &str, level: Option<&str>, at: LevelAt) -> ArrayMetadata {
    let declaration = level
        .map(|n| match at {
            LevelAt::Inline | LevelAt::Filter | LevelAt::NumericId => {
                format!(r#", "level": {n}"#)
            }
            LevelAt::Nested | LevelAt::V3 => {
                format!(r#", "configuration": {{"level": {n}}}"#)
            }
        })
        .unwrap_or_default();
    if let LevelAt::V3 = at {
        let codec = format!(r#"{{"name":"{name}"{declaration}}}"#);
        return v3_metadata(&format!(
            r#"{{"name":"bytes","configuration":{{"endian":"little"}}}},{codec}"#
        ));
    }
    let id = match at {
        LevelAt::NumericId => {
            assert_eq!(name, "zstd", "32015 is the HDF5 filter id of zstd");
            String::from("32015")
        }
        _ => format!(r#""{name}""#),
    };
    let codec = format!(r#"{{"id":{id}{declaration}}}"#);
    match at {
        LevelAt::Filter => v2_metadata("null", &format!("[{codec}]")),
        _ => v2_metadata(&codec, "null"),
    }
}

fn write_level_input(metadata: &ArrayMetadata) -> (InMemoryStore, Result<(), ChunkError>) {
    let mut store = InMemoryStore::new();
    let result = write_array(&mut store, "array", metadata, &level_input());
    (store, result)
}

/// The single chunk object a write left under the array key.
fn stored_chunk(store: &InMemoryStore) -> Vec<u8> {
    let keys = store.list("array/").expect("list array keys");
    assert_eq!(keys.len(), 1, "one chunk object: {keys:?}");
    store.get(&keys[0]).expect("written chunk")
}

/// Writes `level_input` through the Zarr array path, checks the stored chunk
/// against the direct encoder at `expected_level`, and decodes it.
fn write_compressed_chunk(
    name: &str,
    level: Option<&str>,
    at: LevelAt,
    expected_level: i32,
) -> Vec<u8> {
    let metadata = metadata(name, level, at);
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
        "{name} level {level:?} ({at:?}) must reach the encoder as {expected_level}"
    );
    assert_eq!(
        CodecPipeline::new(metadata.codecs.clone())
            .decompress(&encoded, default_registry())
            .expect("read written chunk"),
        level_input()
    );
    encoded
}

/// Smallest level-0 stream for `level_input`: one stored block per 65,535
/// input bytes with a 5-byte header (RFC 1951 section 3.2.4), plus framing:
/// zlib 2-byte header and 4-byte Adler-32 (RFC 1950 section 2.2), gzip
/// 10-byte header and 8-byte CRC-32 and length (RFC 1952 section 2.3).
fn stored_floor(framing: usize) -> usize {
    const STORED_BLOCK_MAX: usize = 65_535;
    const STORED_BLOCK_HEADER: usize = 5;
    let input = level_input().len();
    input + input.div_ceil(STORED_BLOCK_MAX) * STORED_BLOCK_HEADER + framing
}

/// Level a `zstd` codec writes at when it declares none, and the range its
/// encoder clamps to (consus-compression `ZstdCodec::compress`: -8..=4).
const ZSTD_DEFAULT_LEVEL: i32 = 3;
const ZSTD_MIN_LEVEL: i32 = -8;
const ZSTD_MAX_LEVEL: i32 = 4;

const ZLIB_FRAMING: usize = 2 + 4;
const GZIP_FRAMING: usize = 10 + 8;

#[test]
fn zarr_writes_honor_levels_and_default() {
    for (name, framing) in [("zlib", ZLIB_FRAMING), ("gzip", GZIP_FRAMING)] {
        for at in PLACEMENTS {
            let write = |level, expected| write_compressed_chunk(name, level, at, expected);
            let stored = write(Some("0"), 0);
            let compressed = write(Some("9"), 9);
            let default = write(None, ZLIB_DEFAULT_LEVEL);
            assert!(
                stored.len() >= stored_floor(framing),
                "{name} {at:?} stores"
            );
            assert!(
                compressed.len() < level_input().len(),
                "{name} {at:?} shrinks"
            );
            assert_ne!(write(Some("1"), 1), default, "{name} {at:?} level 1");
            assert_eq!(default, write(Some("6"), ZLIB_DEFAULT_LEVEL));
            assert_eq!(
                default,
                write(Some("-1"), ZLIB_DEFAULT_LEVEL),
                "-1 is Z_DEFAULT_COMPRESSION"
            );
        }
    }
}

fn assert_rejected_level(metadata: &ArrayMetadata, name: &str, declared: &str) {
    let (store, result) = write_level_input(metadata);
    assert_eq!(
        result.expect_err("out-of-range level must be rejected"),
        ChunkError::InvalidCodecConfiguration(format!(
            "zarr `{name}` codec: level {declared:?} is not an integer in 0..=9 or the zlib default -1"
        ))
    );
    assert_eq!(
        store.list("array/").expect("list array keys"),
        Vec::<String>::new(),
        "a rejected write stores no chunk"
    );
}

#[test]
fn zstd_writes_honor_a_nested_or_inline_level_after_its_clamp() {
    for at in PLACEMENTS.into_iter().chain([LevelAt::NumericId]) {
        let write = |level, expected| write_compressed_chunk("zstd", level, at, expected);
        let default = write(None, ZSTD_DEFAULT_LEVEL);
        let level_one = write(Some("1"), 1);
        assert_ne!(level_one, default, "zstd level 1 ({at:?}) differs from 3");
        assert_eq!(level_one, write(Some("0"), 1), "0 selects level 1");
        let ceiling = write(Some("4"), ZSTD_MAX_LEVEL);
        assert_ne!(ceiling, default, "zstd level 4 ({at:?}) differs from 3");
        assert_eq!(ceiling, write(Some("9"), ZSTD_MAX_LEVEL), "9 clamps to 4");
        let floor = write(Some("-8"), ZSTD_MIN_LEVEL);
        assert_ne!(floor, default, "zstd level -8 ({at:?}) differs from 3");
        assert_eq!(floor, write(Some("-9"), ZSTD_MIN_LEVEL), "-9 clamps to -8");
    }
}

#[test]
fn level_outside_range_is_rejected() {
    for name in ["gzip", "zlib"] {
        for at in PLACEMENTS {
            for level in [10, i32::MAX, -2, i32::MIN] {
                let text = level.to_string();
                assert_rejected_level(&metadata(name, Some(&text), at), name, &text);
            }
        }
        for at in [LevelAt::Nested, LevelAt::Filter, LevelAt::V3] {
            for (json, declared) in [("1.5", "1.5"), ("\"six\"", "six"), ("null", "")] {
                assert_rejected_level(&metadata(name, Some(json), at), name, declared);
            }
        }
    }
}

#[test]
fn inline_null_level_on_a_string_id_compressor_is_absent() {
    let metadata = v2_metadata(r#"{"id":"zlib","level":null}"#, "null");
    let (store, result) = write_level_input(&metadata);
    result.expect("the form `to_json` writes for no level");
    let default = write_compressed_chunk("zlib", None, LevelAt::Inline, ZLIB_DEFAULT_LEVEL);
    assert_eq!(stored_chunk(&store), default);
}

#[test]
fn v2_conflicting_level_declarations_are_rejected() {
    let codecs = [
        ("\"gzip\"", "gzip"),
        ("\"zlib\"", "zlib"),
        ("\"zstd\"", "zstd"),
        ("32015", "zstd"),
        ("\"lz4\"", "lz4"),
        ("\"blosc\"", "blosc"),
    ];
    for (id, name) in codecs {
        let codec = format!(r#"{{"id":{id},"level":1,"configuration":{{"level":9}}}}"#);
        let as_filter = format!("[{codec}]");
        for (compressor, filters) in [(codec.as_str(), "null"), ("null", as_filter.as_str())] {
            let json = format!("{compressor} {filters}");
            let metadata = v2_metadata(compressor, filters);
            let (store, result) = write_level_input(&metadata);
            assert_eq!(
                result.expect_err("conflicting levels must be rejected"),
                ChunkError::InvalidCodecConfiguration(format!(
                    "zarr `{name}` codec: level is declared as both \"9\" and \"1\""
                )),
                "{json}"
            );
            assert!(store.list("array/").expect("list array keys").is_empty());
        }
    }
}

#[test]
fn decoding_does_not_consult_the_declared_level() {
    let written = write_compressed_chunk("zlib", Some("9"), LevelAt::Inline, 9);
    for at in [LevelAt::Inline, LevelAt::Nested, LevelAt::Filter] {
        for declared in ["10", "-2", "2147483647"] {
            let metadata = metadata("zlib", Some(declared), at);
            let mut store = InMemoryStore::new();
            store.set("array/0", &written).expect("store chunk");
            let read = read_array(&store, "array", &Selection::full(1), &metadata)
                .expect("a stored level never blocks a read");
            assert_eq!(read, level_input(), "declared level {declared} ({at:?})");
        }
    }
}

#[test]
fn sharded_write_rejects_inner_codec_level_outside_range() {
    let inner = r#"[{"name":"bytes","configuration":{"endian":"little"}},
                    {"name":"gzip","configuration":{"level":10}}]"#;
    let sharding = format!(
        r#"{{"name":"sharding_indexed","configuration":{{"chunk_shape":[2],"codecs":{inner},"index_codecs":[]}}}}"#
    );
    assert_rejected_level(&v3_metadata(&sharding), "gzip", "10");
}
