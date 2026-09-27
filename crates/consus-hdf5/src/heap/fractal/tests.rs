//! Unit tests for the fractal heap modules.

use super::*;
use crate::address::ParseContext;
use consus_core::Error;

/// Round-trip: build a minimal fractal heap header image, parse it, and
/// verify every field.
#[test]
fn parse_header_round_trip() {
    let ctx = ParseContext::new(8, 8);
    let s = ctx.length_bytes();
    let o = ctx.offset_bytes();

    // Pre-compute expected total size (before optional filter info and
    // checksum): 14 fixed + 10*s + 3*o + 2 + 2*s + 2 + 2 + o + 2
    // With s=8, o=8: 14 + 80 + 24 + 2 + 16 + 2 + 2 + 8 + 2 = 150
    let header_size = 14 + 10 * s + 3 * o + 2 + 2 * s + 2 + 2 + o + 2;
    let total = header_size + 4; // + checksum
    let mut data = vec![0u8; total.max(256)];

    // Signature + version
    data[0..4].copy_from_slice(b"FRHP");
    data[4] = 0; // version

    // Fixed fields
    data[5..7].copy_from_slice(&42u16.to_le_bytes()); // heap_id_length
    data[7..9].copy_from_slice(&0u16.to_le_bytes()); // io_filter_size
    data[9] = 0x03; // flags
    data[10..14].copy_from_slice(&1024u32.to_le_bytes()); // max_managed_object_size

    let mut pos = 14usize;

    // next_huge_id (skip)
    data[pos..pos + s].copy_from_slice(&99u64.to_le_bytes()[..s]);
    pos += s;

    // huge_object_btree_address
    data[pos..pos + o].copy_from_slice(&0x1000u64.to_le_bytes()[..o]);
    pos += o;

    // free_managed_space
    data[pos..pos + s].copy_from_slice(&50u64.to_le_bytes()[..s]);
    pos += s;

    // free_space_manager_address
    data[pos..pos + o].copy_from_slice(&0x2000u64.to_le_bytes()[..o]);
    pos += o;

    // managed_space
    data[pos..pos + s].copy_from_slice(&4096u64.to_le_bytes()[..s]);
    pos += s;

    // allocated_managed_space
    data[pos..pos + s].copy_from_slice(&2048u64.to_le_bytes()[..s]);
    pos += s;

    // iter_offset (skip)
    pos += s;

    // managed_object_count
    data[pos..pos + s].copy_from_slice(&10u64.to_le_bytes()[..s]);
    pos += s;

    // huge_object_size
    data[pos..pos + s].copy_from_slice(&500u64.to_le_bytes()[..s]);
    pos += s;

    // huge_object_count
    data[pos..pos + s].copy_from_slice(&2u64.to_le_bytes()[..s]);
    pos += s;

    // tiny_object_size
    data[pos..pos + s].copy_from_slice(&30u64.to_le_bytes()[..s]);
    pos += s;

    // tiny_object_count
    data[pos..pos + s].copy_from_slice(&5u64.to_le_bytes()[..s]);
    pos += s;

    // table_width
    data[pos..pos + 2].copy_from_slice(&4u16.to_le_bytes());
    pos += 2;

    // starting_block_size
    data[pos..pos + s].copy_from_slice(&512u64.to_le_bytes()[..s]);
    pos += s;

    // max_direct_block_size
    data[pos..pos + s].copy_from_slice(&65536u64.to_le_bytes()[..s]);
    pos += s;

    // max_heap_size_bits
    data[pos..pos + 2].copy_from_slice(&16u16.to_le_bytes());
    pos += 2;

    // starting_rows
    data[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
    pos += 2;

    // root_block_address
    data[pos..pos + o].copy_from_slice(&0x3000u64.to_le_bytes()[..o]);
    pos += o;

    // root_indirect_rows
    data[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
    // pos += 2;

    let reader = consus_io::SliceReader::new(&data);
    let hdr = FractalHeapHeader::parse(&reader, 0, &ctx).unwrap();

    assert_eq!(hdr.heap_id_length, 42);
    assert_eq!(hdr.io_filter_size, 0);
    assert_eq!(hdr.flags, 0x03);
    assert_eq!(hdr.max_managed_object_size, 1024);
    assert_eq!(hdr.huge_object_btree_address, 0x1000);
    assert_eq!(hdr.free_managed_space, 50);
    assert_eq!(hdr.free_space_manager_address, 0x2000);
    assert_eq!(hdr.managed_space, 4096);
    assert_eq!(hdr.allocated_managed_space, 2048);
    assert_eq!(hdr.managed_object_count, 10);
    assert_eq!(hdr.huge_object_size, 500);
    assert_eq!(hdr.huge_object_count, 2);
    assert_eq!(hdr.tiny_object_size, 30);
    assert_eq!(hdr.tiny_object_count, 5);
    assert_eq!(hdr.table_width, 4);
    assert_eq!(hdr.starting_block_size, 512);
    assert_eq!(hdr.max_direct_block_size, 65536);
    assert_eq!(hdr.max_heap_size_bits, 16);
    assert_eq!(hdr.root_block_address, 0x3000);
    assert_eq!(hdr.root_indirect_rows, 0);
}

#[test]
fn parse_header_bad_signature() {
    let ctx = ParseContext::new(8, 8);
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(b"XXXX");
    let reader = consus_io::SliceReader::new(&data);
    let err = FractalHeapHeader::parse(&reader, 0, &ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidFormat { .. }),
        "expected InvalidFormat, got: {err:?}"
    );
}

#[test]
fn decode_managed_heap_id() {
    let header = FractalHeapHeader {
        heap_id_length: 7,
        io_filter_size: 0,
        flags: 0,
        max_managed_object_size: 1024,
        huge_object_btree_address: 0,
        free_managed_space: 0,
        free_space_manager_address: 0,
        managed_space: 4096,
        allocated_managed_space: 2048,
        managed_object_count: 1,
        huge_object_size: 0,
        huge_object_count: 0,
        tiny_object_size: 0,
        tiny_object_count: 0,
        table_width: 4,
        starting_rows: 1,
        starting_block_size: 512,
        max_direct_block_size: 65536,
        max_heap_size_bits: 16,
        root_block_address: 0,
        root_indirect_rows: 0,
    };

    // Type = 0 (managed), version = 0 → byte 0 = 0x00
    // max_heap_size_bits = 16 → offset_bytes = 2
    // id_bytes = [0x00, offset_lo, offset_hi, len_lo, len_mid, len_hi, len_top]
    //            type+ver   offset=0x0100         length=0x00000080
    let id_bytes: [u8; 7] = [0x00, 0x00, 0x01, 0x80, 0x00, 0x00, 0x00];
    let id = decode_heap_id(&id_bytes, &header).unwrap();
    match id {
        FractalHeapId::Managed { offset, length } => {
            assert_eq!(offset, 0x0100);
            assert_eq!(length, 0x80);
        }
        other => panic!("expected Managed, got: {other:?}"),
    }
}

#[test]
fn decode_tiny_heap_id() {
    let header = FractalHeapHeader {
        heap_id_length: 5,
        io_filter_size: 0,
        flags: 0,
        max_managed_object_size: 64,
        huge_object_btree_address: 0,
        free_managed_space: 0,
        free_space_manager_address: 0,
        managed_space: 0,
        allocated_managed_space: 0,
        managed_object_count: 0,
        huge_object_size: 0,
        huge_object_count: 0,
        tiny_object_size: 0,
        tiny_object_count: 0,
        table_width: 4,
        starting_rows: 1,
        starting_block_size: 256,
        max_direct_block_size: 4096,
        max_heap_size_bits: 8,
        root_block_address: 0,
        root_indirect_rows: 0,
    };

    // Type = 1 (tiny) → bits 6-7 = 01 → byte 0 = 0b0100_0000 = 0x40
    let id_bytes: [u8; 5] = [0x40, 0xAA, 0xBB, 0xCC, 0xDD];
    let id = decode_heap_id(&id_bytes, &header).unwrap();
    match id {
        FractalHeapId::Tiny { data } => {
            assert_eq!(data, vec![0xAA, 0xBB, 0xCC, 0xDD]);
        }
        other => panic!("expected Tiny, got: {other:?}"),
    }
}

#[test]
fn decode_huge_heap_id() {
    let header = FractalHeapHeader {
        heap_id_length: 9,
        io_filter_size: 0,
        flags: 0,
        max_managed_object_size: 128,
        huge_object_btree_address: 0x5000,
        free_managed_space: 0,
        free_space_manager_address: 0,
        managed_space: 0,
        allocated_managed_space: 0,
        managed_object_count: 0,
        huge_object_size: 0,
        huge_object_count: 0,
        tiny_object_size: 0,
        tiny_object_count: 0,
        table_width: 4,
        starting_rows: 1,
        starting_block_size: 256,
        max_direct_block_size: 4096,
        max_heap_size_bits: 16,
        root_block_address: 0,
        root_indirect_rows: 0,
    };

    // Type = 2 (huge) → bits 6-7 = 10 → byte 0 = 0b1000_0000 = 0x80
    let mut id_bytes = [0u8; 9];
    id_bytes[0] = 0x80;
    id_bytes[1..9].copy_from_slice(&0x0000_DEAD_BEEF_0000u64.to_le_bytes());
    let id = decode_heap_id(&id_bytes, &header).unwrap();
    match id {
        FractalHeapId::Huge { btree_key } => {
            assert_eq!(btree_key, 0x0000_DEAD_BEEF_0000);
        }
        other => panic!("expected Huge, got: {other:?}"),
    }
}

#[test]
fn read_managed_object_direct_root() {
    let ctx = ParseContext::new(8, 8);
    let header = FractalHeapHeader {
        heap_id_length: 7,
        io_filter_size: 0,
        flags: 0,
        max_managed_object_size: 1024,
        huge_object_btree_address: 0,
        free_managed_space: 0,
        free_space_manager_address: 0,
        managed_space: 256,
        allocated_managed_space: 256,
        managed_object_count: 1,
        huge_object_size: 0,
        huge_object_count: 0,
        tiny_object_size: 0,
        tiny_object_count: 0,
        table_width: 4,
        starting_rows: 1,
        starting_block_size: 256,
        max_direct_block_size: 65536,
        max_heap_size_bits: 16,
        root_block_address: 0, // root block at file offset 0
        root_indirect_rows: 0, // direct block
    };

    // Build a direct block image at offset 0.
    // Overhead: 5 (sig+ver) + 8 (heap addr) + 2 (block offset, ceil(16/8))
    // = 15 bytes of header before data.
    let overhead = 5 + ctx.offset_bytes() + 2; // 15
    let block_size = overhead + 256; // data area = 256 bytes
    let mut image = vec![0u8; block_size];

    image[0..4].copy_from_slice(b"FHDB");
    image[4] = 0; // version
    // heap header address (8 bytes, irrelevant for this test)
    // block offset (2 bytes, 0)

    // Write a known payload. Under correct HDF5 semantics managed_off
    // encodes the absolute byte offset from block start, so the object at
    // data-area byte +10 lives at block byte (overhead + 10).
    let payload = b"HELLO";
    let data_start = overhead + 10;
    image[data_start..data_start + 5].copy_from_slice(payload);

    let reader = consus_io::SliceReader::new(&image);
    // managed_off = block_heap_off + block_byte_pos = 0 + data_start
    let result = read_managed_object(&reader, &header, data_start as u64, 5, &ctx).unwrap();
    assert_eq!(result, b"HELLO");
}

#[test]
fn read_managed_object_rejects_indirect_root() {
    let ctx = ParseContext::new(8, 8);
    let header = FractalHeapHeader {
        heap_id_length: 7,
        io_filter_size: 0,
        flags: 0,
        max_managed_object_size: 1024,
        huge_object_btree_address: 0,
        free_managed_space: 0,
        free_space_manager_address: 0,
        managed_space: 0,
        allocated_managed_space: 0,
        managed_object_count: 0,
        huge_object_size: 0,
        huge_object_count: 0,
        tiny_object_size: 0,
        tiny_object_count: 0,
        table_width: 4,
        starting_rows: 1,
        starting_block_size: 256,
        max_direct_block_size: 65536,
        max_heap_size_bits: 16,
        root_block_address: 0,
        root_indirect_rows: 2, // indirect
    };

    // Buffer of zeros: FHIB signature check will fail with InvalidFormat.
    let data = vec![0u8; 4096];
    let reader = consus_io::SliceReader::new(&data);
    let err = read_managed_object(&reader, &header, 0, 10, &ctx).unwrap_err();
    assert!(
        matches!(err, Error::InvalidFormat { .. }),
        "expected InvalidFormat, got: {err:?}"
    );
}

#[test]
fn read_uint_le_various_widths() {
    let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    assert_eq!(read_uint_le(&data, 0), 0);
    assert_eq!(read_uint_le(&data, 1), 0x01);
    assert_eq!(read_uint_le(&data, 2), 0x0201);
    assert_eq!(read_uint_le(&data, 3), 0x0003_0201);
    assert_eq!(read_uint_le(&data, 4), 0x0403_0201);
    assert_eq!(read_uint_le(&data, 8), 0x0807_0605_0403_0201);
}
