//! Chunk key encoding, coordinate, stride, and copy helper tests.

use super::super::*;
use crate::metadata::ChunkKeyEncoding;

#[test]
fn chunk_key_v2() {
    let encoding = ChunkKeyEncoding {
        name: "v2".to_string(),
        separator: '.',
    };
    let key = chunk_key_for_array("arr", &[0u64, 0u64, 0u64], &encoding);
    assert_eq!(key, "arr/0.0.0");

    let key = chunk_key_for_array("myarray", &[1u64, 2u64, 3u64], &encoding);
    assert_eq!(key, "myarray/1.2.3");
}

#[test]
fn chunk_key_default() {
    let encoding = ChunkKeyEncoding::default();
    let key = chunk_key_for_array("arr", &[0u64, 0u64, 0u64], &encoding);
    assert_eq!(key, "arr/c/0/0/0");
}

#[test]
fn chunk_key_v2_root_relative() {
    let encoding = ChunkKeyEncoding {
        name: "v2".to_string(),
        separator: '.',
    };
    let key = chunk_key_for_array(".", &[1u64, 2u64], &encoding);
    assert_eq!(key, "1.2");

    let key = chunk_key_for_array("", &[3u64, 4u64], &encoding);
    assert_eq!(key, "3.4");
}

#[test]
fn chunk_key_default_root_relative() {
    let encoding = ChunkKeyEncoding::default();
    let key = chunk_key_for_array(".", &[1u64, 2u64], &encoding);
    assert_eq!(key, "c/1/2");

    let key = chunk_key_for_array("", &[3u64, 4u64], &encoding);
    assert_eq!(key, "c/3/4");
}
