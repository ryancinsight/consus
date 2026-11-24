//! Stored-chunk checks for the Zarr v2 filters and primary compressor.
use consus_zarr::chunk::{Selection, read_array, write_array};
use consus_zarr::codec::{CompressionLevel, CompressionRegistryTrait, default_registry};
use consus_zarr::metadata::{ArrayMetadata, ArrayMetadataV2};
use consus_zarr::store::{InMemoryStore, Store};

const INPUT: [u8; 16] = *b"aAbBcCdDeEfFgGhH";
const SHUFFLED: [u8; 16] = *b"abcdefghABCDEFGH";

fn metadata(len: usize, filters: &str, compressor: &str) -> ArrayMetadata {
    let json = format!(
        r#"{{"zarr_format":2,"shape":[{len}],"chunks":[{len}],"dtype":"|u1","order":"C",
        "fill_value":0,"filters":{filters},"compressor":{compressor}}}"#
    );
    ArrayMetadataV2::parse(&json)
        .expect("v2 metadata")
        .to_canonical()
}

fn compress(name: &str, level: i32, input: &[u8]) -> Vec<u8> {
    default_registry()
        .get_by_name(name)
        .expect("registered codec")
        .compress(input, CompressionLevel(level))
        .expect("direct compression")
}

#[test]
fn writes_filters_in_list_order_then_compresses() {
    for (filters, filtered) in [
        (r#"[{"id":"shuffle","elementsize":2}]"#, SHUFFLED.to_vec()),
        (
            r#"[{"id":"shuffle","elementsize":2},{"id":"zlib","level":1}]"#,
            compress("zlib", 1, &SHUFFLED),
        ),
    ] {
        let meta = metadata(INPUT.len(), filters, r#"{"id":"zstd","level":3}"#);
        let mut store = InMemoryStore::new();
        write_array(&mut store, "", &meta, &INPUT).expect("write filtered array");
        assert_eq!(
            store.get("0").expect("stored chunk"),
            compress("zstd", 3, &filtered)
        );
        assert_eq!(
            read_array(&store, "", &Selection::full(1), &meta).expect("read filtered array"),
            INPUT
        );
    }
}

#[test]
fn reads_numcodecs_shuffle_zstd_chunk() {
    // numcodecs 0.16.5: Zstd(level=3).encode(Shuffle(2).encode(b"aAbBcCdDeEfFgGhH" * 32))
    let chunk = [
        0x28, 0xb5, 0x2f, 0xfd, 0x60, 0x00, 0x01, 0xd5, 0x00, 0x00, 0x80, 0x61, 0x62, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x02, 0x00, 0x75,
        0x40, 0xf1, 0xfa, 0xa8, 0xc2, 0x05,
    ];
    let expected = INPUT.repeat(32);
    let meta = metadata(
        expected.len(),
        r#"[{"id":"shuffle","elementsize":2}]"#,
        r#"{"id":"zstd","level":3}"#,
    );
    let mut store = InMemoryStore::new();
    store.set("0", &chunk).expect("seed external chunk");
    assert_eq!(
        read_array(&store, "", &Selection::full(1), &meta).expect("read numcodecs chunk"),
        expected
    );
}

#[test]
fn shuffle_without_a_compressor_honors_its_own_width() {
    for (filters, expected) in [
        (r#"[{"id":"shuffle"}]"#, *b"acegACEGbdfhBDFH"),
        (r#"[{"id":"shuffle","elementsize":1}]"#, INPUT),
        (r#"[{"id":"shuffle","elementsize":0}]"#, INPUT),
    ] {
        let meta = metadata(INPUT.len(), filters, "null");
        let mut store = InMemoryStore::new();
        write_array(&mut store, "", &meta, &INPUT).expect("write filter-only array");
        assert_eq!(store.get("0").expect("stored chunk"), expected);
    }
}

#[test]
fn invalid_shuffle_metadata_or_misaligned_input_fails_before_storage() {
    for width in ["null", "1.5", "3"] {
        let filters = format!(r#"[{{"id":"shuffle","elementsize":{width}}}]"#);
        let meta = metadata(INPUT.len(), &filters, "null");
        let mut store = InMemoryStore::new();
        let error = write_array(&mut store, "", &meta, &INPUT).expect_err("invalid shuffle");
        assert!(matches!(
            error,
            consus_zarr::chunk::ChunkError::InvalidCodecConfiguration(_)
        ));
        assert!(store.list("").expect("list store").is_empty());
    }
}
