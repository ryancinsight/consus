use alloc::vec::Vec;

use consus_core::Error;
use consus_io::MemCursor;

use super::*;
use crate::address::ParseContext;

fn ctx8() -> ParseContext {
    ParseContext::new(8, 8)
}

fn ctx4() -> ParseContext {
    ParseContext::new(4, 4)
}

/// Build a minimal global heap collection with one object.
fn build_collection_one_object(ctx: &ParseContext, obj_data: &[u8]) -> Vec<u8> {
    let l = ctx.length_bytes();
    let obj_header_size = 8 + l;
    let padded_data_size = (obj_data.len() + 7) & !7;
    // End marker object header (index=0)
    let end_marker_size = obj_header_size;

    let body_size = obj_header_size + padded_data_size + end_marker_size;
    let header_size = 8 + l;
    let collection_size = header_size + body_size;

    let mut buf = vec![0u8; collection_size];

    // Header
    buf[0..4].copy_from_slice(&GLOBAL_HEAP_SIGNATURE);
    buf[4] = 1; // version
    // reserved [5..8]
    match l {
        4 => {
            let bytes = u32::try_from(collection_size)
                .expect("invariant: collection size fits u32")
                .to_le_bytes();
            buf[8..12].copy_from_slice(&bytes);
        }
        8 => {
            let bytes = u64::try_from(collection_size)
                .expect("invariant: collection size fits u64")
                .to_le_bytes();
            buf[8..16].copy_from_slice(&bytes);
        }
        _ => {}
    }

    // Object 1
    let mut pos = header_size;
    buf[pos] = 1; // index = 1
    buf[pos + 1] = 0;
    buf[pos + 2] = 1; // ref count = 1
    buf[pos + 3] = 0;
    // reserved [pos+4..pos+8]
    match l {
        4 => {
            let bytes = u32::try_from(obj_data.len())
                .expect("invariant: object size fits u32")
                .to_le_bytes();
            buf[pos + 8..pos + 12].copy_from_slice(&bytes);
        }
        8 => {
            let bytes = u64::try_from(obj_data.len())
                .expect("invariant: object size fits u64")
                .to_le_bytes();
            buf[pos + 8..pos + 16].copy_from_slice(&bytes);
        }
        _ => {}
    }
    pos += obj_header_size;
    buf[pos..pos + obj_data.len()].copy_from_slice(obj_data);
    // Remaining bytes are zero (end marker has index=0)

    buf
}

/// Parse a collection with one object using 8-byte lengths.
#[test]
fn parse_one_object_8byte() {
    let ctx = ctx8();
    let obj_data = b"hello global heap";
    let buf = build_collection_one_object(&ctx, obj_data);
    let cursor = MemCursor::from_bytes(buf);

    let coll = GlobalHeapCollection::parse(&cursor, 0, &ctx).unwrap();
    assert_eq!(coll.objects.len(), 1);
    assert_eq!(coll.objects[0].index, 1);
    assert_eq!(coll.objects[0].reference_count, 1);
    assert_eq!(coll.objects[0].data, obj_data);
}

/// Parse a collection with one object using 4-byte lengths.
#[test]
fn parse_one_object_4byte() {
    let ctx = ctx4();
    let obj_data = b"four byte offsets";
    let buf = build_collection_one_object(&ctx, obj_data);
    let cursor = MemCursor::from_bytes(buf);

    let coll = GlobalHeapCollection::parse(&cursor, 0, &ctx).unwrap();
    assert_eq!(coll.objects.len(), 1);
    assert_eq!(coll.objects[0].data, obj_data.as_slice());
}

/// Verify `get` and `get_data` lookups.
#[test]
fn lookup_by_index() {
    let ctx = ctx8();
    let obj_data = b"lookup test";
    let buf = build_collection_one_object(&ctx, obj_data);
    let cursor = MemCursor::from_bytes(buf);

    let coll = GlobalHeapCollection::parse(&cursor, 0, &ctx).unwrap();
    assert!(coll.get(1).is_some());
    assert_eq!(coll.get_data(1).unwrap(), obj_data.as_slice());
    assert!(coll.get(2).is_none());
    assert!(coll.get_data(0).is_none());
}

/// Reject invalid signature.
#[test]
fn reject_bad_signature() {
    let ctx = ctx8();
    let mut buf = vec![0u8; 64];
    buf[0..4].copy_from_slice(b"XXXX");
    buf[4] = 1;
    let cursor = MemCursor::from_bytes(buf);

    let err = GlobalHeapCollection::parse(&cursor, 0, &ctx).unwrap_err();
    match err {
        Error::InvalidFormat { message } => {
            assert!(message.contains("signature"));
        }
        other => panic!("expected InvalidFormat, got: {other:?}"),
    }
}

/// Reject unsupported version.
#[test]
fn reject_bad_version() {
    let ctx = ctx8();
    let mut buf = vec![0u8; 64];
    buf[0..4].copy_from_slice(&GLOBAL_HEAP_SIGNATURE);
    buf[4] = 2; // unsupported
    let bytes = 64u64.to_le_bytes();
    buf[8..16].copy_from_slice(&bytes);
    let cursor = MemCursor::from_bytes(buf);

    let err = GlobalHeapCollection::parse(&cursor, 0, &ctx).unwrap_err();
    match err {
        Error::InvalidFormat { message } => {
            assert!(message.contains("version"));
        }
        other => panic!("expected InvalidFormat, got: {other:?}"),
    }
}

/// Empty collection (header only, no objects before end marker).
#[test]
fn parse_empty_collection() {
    let ctx = ctx8();
    let l = ctx.length_bytes();
    let header_size = 8 + l;
    // End marker object: index(2)=0 + rest doesn't matter, just needs space
    let obj_header_size = 8 + l;
    let collection_size = header_size + obj_header_size;
    let mut buf = vec![0u8; collection_size];
    buf[0..4].copy_from_slice(&GLOBAL_HEAP_SIGNATURE);
    buf[4] = 1;
    let bytes = u64::try_from(collection_size)
        .expect("invariant: collection size fits u64")
        .to_le_bytes();
    buf[8..16].copy_from_slice(&bytes);
    // body is all zeros → index 0 at first object → immediate stop

    let cursor = MemCursor::from_bytes(buf);
    let coll = GlobalHeapCollection::parse(&cursor, 0, &ctx).unwrap();
    assert!(coll.objects.is_empty());
}

/// Object data with odd size is correctly padded on disk.
#[test]
fn parse_padded_object_data() {
    let ctx = ctx8();
    // 5-byte object → padded to 8 bytes on disk
    let obj_data = b"12345";
    let buf = build_collection_one_object(&ctx, obj_data);
    let cursor = MemCursor::from_bytes(buf);

    let coll = GlobalHeapCollection::parse(&cursor, 0, &ctx).unwrap();
    assert_eq!(coll.objects.len(), 1);
    assert_eq!(coll.objects[0].data.len(), 5);
    assert_eq!(coll.objects[0].data, b"12345");
}

/// Parse at a non-zero file offset.
#[test]
fn parse_at_offset() {
    let ctx = ctx8();
    let obj_data = b"offset test";
    let collection_buf = build_collection_one_object(&ctx, obj_data);
    let offset = 128u64;
    let offset_usize = usize::try_from(offset).expect("invariant: offset fits usize");
    let mut file_buf = vec![0u8; offset_usize + collection_buf.len()];
    file_buf[offset_usize..].copy_from_slice(&collection_buf);

    let cursor = MemCursor::from_bytes(file_buf);
    let coll = GlobalHeapCollection::parse(&cursor, offset, &ctx).unwrap();
    assert_eq!(coll.objects.len(), 1);
    assert_eq!(coll.objects[0].data, obj_data.as_slice());
}

// -----------------------------------------------------------------------
// resolve_vl_references tests
// -----------------------------------------------------------------------

/// One valid VL reference (8-byte offsets) resolves to the expected payload.
#[test]
fn resolve_vl_single_ref_8byte() {
    let ctx = ctx8();
    let payload = b"hello";
    let collection_buf = build_collection_one_object(&ctx, payload);

    // ref_size = 4 + 8 + 4 = 16
    let ref_size = 4 + ctx.offset_bytes() + 4;
    let mut raw_data = vec![0u8; ref_size];
    // sequence_length = 5 (number of elements)
    raw_data[0..4].copy_from_slice(
        &u32::try_from(payload.len())
            .expect("invariant: payload length fits u32")
            .to_le_bytes(),
    );
    // heap_collection_address = 0 (collection is at file offset 0)
    raw_data[4..12].copy_from_slice(&0u64.to_le_bytes());
    // object_index = 1 (1-based)
    raw_data[12..16].copy_from_slice(&1u32.to_le_bytes());

    let cursor = MemCursor::from_bytes(collection_buf);
    let result = resolve_vl_references(&cursor, &raw_data, &ctx).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], payload.as_slice());
}

/// One valid VL reference (4-byte offsets) resolves to the expected payload.
#[test]
fn resolve_vl_single_ref_4byte() {
    let ctx = ctx4();
    let payload = b"four";
    let collection_buf = build_collection_one_object(&ctx, payload);

    // ref_size = 4 + 4 + 4 = 12
    let ref_size = 4 + ctx.offset_bytes() + 4;
    assert_eq!(ref_size, 12);
    let mut raw_data = vec![0u8; ref_size];
    raw_data[0..4].copy_from_slice(
        &u32::try_from(payload.len())
            .expect("invariant: payload length fits u32")
            .to_le_bytes(),
    );
    raw_data[4..8].copy_from_slice(&0u32.to_le_bytes()); // 4-byte addr = 0
    raw_data[8..12].copy_from_slice(&1u32.to_le_bytes());

    let cursor = MemCursor::from_bytes(collection_buf);
    let result = resolve_vl_references(&cursor, &raw_data, &ctx).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], payload.as_slice());
}

/// sequence_length == 0 must yield an empty inner vec without reading the heap.
#[test]
fn resolve_vl_zero_sequence_length_yields_empty() {
    let ctx = ctx8();
    let ref_size = 4 + ctx.offset_bytes() + 4;
    let mut raw_data = vec![0u8; ref_size];
    raw_data[0..4].copy_from_slice(&0u32.to_le_bytes()); // sequence_length = 0
    raw_data[4..12].copy_from_slice(&0u64.to_le_bytes()); // addr = 0
    raw_data[12..16].copy_from_slice(&1u32.to_le_bytes()); // object_index = 1

    // Empty source: parse must not be attempted; if it were, it would I/O error.
    let cursor = MemCursor::from_bytes(vec![]);
    let result = resolve_vl_references(&cursor, &raw_data, &ctx).unwrap();

    assert_eq!(result.len(), 1);
    assert!(result[0].is_empty());
}

/// UNDEFINED_ADDRESS (8-byte) with non-zero sequence_length yields empty.
#[test]
fn resolve_vl_undefined_address_yields_empty() {
    let ctx = ctx8();
    let ref_size = 4 + ctx.offset_bytes() + 4;
    let mut raw_data = vec![0u8; ref_size];
    raw_data[0..4].copy_from_slice(&1u32.to_le_bytes()); // sequence_length = 1
    raw_data[4..12].copy_from_slice(&u64::MAX.to_le_bytes()); // UNDEFINED_ADDRESS
    raw_data[12..16].copy_from_slice(&1u32.to_le_bytes());

    let cursor = MemCursor::from_bytes(vec![]);
    let result = resolve_vl_references(&cursor, &raw_data, &ctx).unwrap();

    assert_eq!(result.len(), 1);
    assert!(result[0].is_empty());
}

/// 4-byte offset UNDEFINED_ADDRESS (0xFFFF_FFFF) yields empty inner vec.
#[test]
fn resolve_vl_undefined_address_4byte_yields_empty() {
    let ctx = ctx4();
    let ref_size = 4 + ctx.offset_bytes() + 4;
    let mut raw_data = vec![0u8; ref_size];
    raw_data[0..4].copy_from_slice(&1u32.to_le_bytes()); // sequence_length = 1
    raw_data[4..8].copy_from_slice(&u32::MAX.to_le_bytes()); // 4-byte undef addr
    raw_data[8..12].copy_from_slice(&1u32.to_le_bytes());

    let cursor = MemCursor::from_bytes(vec![]);
    let result = resolve_vl_references(&cursor, &raw_data, &ctx).unwrap();

    assert_eq!(result.len(), 1);
    assert!(result[0].is_empty());
}

/// raw_data length not a multiple of ref_size returns InvalidFormat.
#[test]
fn resolve_vl_bad_length_returns_error() {
    let ctx = ctx8(); // ref_size = 16
    let raw_data = vec![0u8; 7]; // 7 % 16 != 0
    let cursor = MemCursor::from_bytes(vec![]);
    let err = resolve_vl_references(&cursor, &raw_data, &ctx).unwrap_err();
    match err {
        Error::InvalidFormat { message } => {
            assert!(message.contains("multiple"), "message was: {message}");
        }
        other => panic!("expected InvalidFormat, got: {other:?}"),
    }
}

/// Two refs in one call: first valid, second has zero sequence_length.
#[test]
fn resolve_vl_multiple_refs_mixed() {
    let ctx = ctx8();
    let payload = b"world";
    let collection_buf = build_collection_one_object(&ctx, payload);

    // ref_size = 16; two refs = 32 bytes total
    let ref_size = 4 + ctx.offset_bytes() + 4;
    let mut raw_data = vec![0u8; ref_size * 2];

    // Ref 0: valid - points to collection at offset 0, object 1
    raw_data[0..4].copy_from_slice(
        &u32::try_from(payload.len())
            .expect("invariant: payload length fits u32")
            .to_le_bytes(),
    );
    raw_data[4..12].copy_from_slice(&0u64.to_le_bytes());
    raw_data[12..16].copy_from_slice(&1u32.to_le_bytes());

    // Ref 1: zero sequence_length - must not touch the heap
    raw_data[16..20].copy_from_slice(&0u32.to_le_bytes());
    raw_data[20..28].copy_from_slice(&0u64.to_le_bytes());
    raw_data[28..32].copy_from_slice(&1u32.to_le_bytes());

    let cursor = MemCursor::from_bytes(collection_buf);
    let result = resolve_vl_references(&cursor, &raw_data, &ctx).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(result[0], payload.as_slice());
    assert!(result[1].is_empty());
}

/// Empty raw_data returns an empty outer vec (zero VL elements).
#[test]
fn resolve_vl_empty_raw_data() {
    let ctx = ctx8();
    let cursor = MemCursor::from_bytes(vec![]);
    let result = resolve_vl_references(&cursor, &[], &ctx).unwrap();
    assert!(result.is_empty());
}

use core::sync::atomic::{AtomicUsize, Ordering};
use std::collections::BTreeMap;
use std::sync::Mutex;

fn build_reference(ctx: &ParseContext, address: u64, object_index: u32) -> Vec<u8> {
    let reference_size = 4 + ctx.offset_bytes() + 4;
    let mut reference = vec![0_u8; reference_size];
    reference[0..4].copy_from_slice(&1_u32.to_le_bytes());
    match ctx.offset_bytes() {
        4 => reference[4..8].copy_from_slice(
            &u32::try_from(address)
                .expect("invariant: address fits u32")
                .to_le_bytes(),
        ),
        8 => reference[4..12].copy_from_slice(&address.to_le_bytes()),
        _ => panic!("unsupported test offset size"),
    }
    let object_index_offset = 4 + ctx.offset_bytes();
    reference[object_index_offset..object_index_offset + 4]
        .copy_from_slice(&object_index.to_le_bytes());
    reference
}

struct CountingReader {
    inner: MemCursor,
    reads: AtomicUsize,
    collection_reads: Mutex<BTreeMap<u64, usize>>,
}

impl CountingReader {
    fn new(bytes: Vec<u8>, collection_addresses: &[u64]) -> Self {
        let collection_reads = collection_addresses
            .iter()
            .map(|address| (*address, 0))
            .collect();
        Self {
            inner: MemCursor::from_bytes(bytes),
            reads: AtomicUsize::new(0),
            collection_reads: Mutex::new(collection_reads),
        }
    }

    fn read_count(&self) -> usize {
        self.reads.load(Ordering::Relaxed)
    }

    fn collection_read_count(&self, address: u64) -> usize {
        self.collection_reads
            .lock()
            .expect("invariant: collection read counter is not poisoned")
            .get(&address)
            .copied()
            .unwrap_or(0)
    }
}

impl consus_io::ReadAt for CountingReader {
    fn read_at(&self, position: u64, buffer: &mut [u8]) -> consus_core::Result<()> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if let Some(count) = self
            .collection_reads
            .lock()
            .expect("invariant: collection read counter is not poisoned")
            .get_mut(&position)
        {
            *count += 1;
        }
        self.inner.read_at(position, buffer)
    }
}

#[test]
fn repeated_references_read_one_collection_once() {
    let ctx = ctx8();
    let payload = b"repeated";
    let collection = build_collection_one_object(&ctx, payload);
    let reference = build_reference(&ctx, 0, 1);
    let raw_data = reference.repeat(32);
    let reader = CountingReader::new(collection, &[0]);

    let values = resolve_vl_references(&reader, &raw_data, &ctx).expect("references must resolve");

    assert_eq!(values.len(), 32);
    assert!(values.iter().all(|value| value.as_slice() == payload));
    assert_eq!(reader.collection_read_count(0), 1);
    assert_eq!(reader.read_count(), 2);
}

#[test]
fn distinct_collections_are_each_read_once() {
    let ctx = ctx8();
    let first_payload = b"first";
    let second_payload = b"second";
    let first = build_collection_one_object(&ctx, first_payload);
    let second_address = u64::try_from(first.len()).expect("invariant: address fits u64");
    let second = build_collection_one_object(&ctx, second_payload);
    let mut file = first;
    file.extend_from_slice(&second);

    let mut raw_data = build_reference(&ctx, 0, 1);
    raw_data.extend_from_slice(&build_reference(&ctx, second_address, 1));
    let reader = CountingReader::new(file, &[0, second_address]);

    let values = resolve_vl_references(&reader, &raw_data, &ctx).expect("references must resolve");

    assert_eq!(
        values,
        vec![first_payload.to_vec(), second_payload.to_vec()]
    );
    assert_eq!(reader.collection_read_count(0), 1);
    assert_eq!(reader.collection_read_count(second_address), 1);
    assert_eq!(reader.read_count(), 4);
}

#[test]
fn first_failing_element_reports_its_error() {
    let ctx = ctx8();
    let first = build_collection_one_object(&ctx, b"first");
    let second_address = u64::try_from(first.len()).expect("invariant: address fits u64");
    let second = build_collection_one_object(&ctx, b"second");
    let mut file = first;
    file.extend_from_slice(&second);

    // The first element references the higher-addressed collection and the
    // second element references the lower-addressed one. The reported error
    // follows element order, not collection address order.
    let mut raw_data = build_reference(&ctx, second_address, 7);
    raw_data.extend_from_slice(&build_reference(&ctx, 0, 9));

    let error = resolve_vl_references(&MemCursor::from_bytes(file), &raw_data, &ctx)
        .expect_err("missing heap objects must fail");

    match error {
        Error::InvalidFormat { message } => {
            assert!(
                message.contains("VL object index 7"),
                "message was: {message}"
            );
            assert!(message.contains("element 0"), "message was: {message}");
        }
        other => panic!("expected invalid format error, got: {other:?}"),
    }
}

#[test]
fn object_index_above_u16_is_rejected() {
    let ctx = ctx8();
    let reference = build_reference(&ctx, 0, u32::MAX);
    let error = resolve_vl_references(&MemCursor::from_bytes(Vec::new()), &reference, &ctx)
        .expect_err("oversized object index must fail");

    match error {
        Error::InvalidFormat { message } => assert!(message.contains("u16::MAX")),
        other => panic!("expected InvalidFormat, got: {other:?}"),
    }
}

#[test]
fn mixed_object_errors_follow_element_order() {
    let ctx = ctx8();
    let payload = build_collection_one_object(&ctx, b"present");
    let missing = build_reference(&ctx, 0, 7);
    let oversized = build_reference(&ctx, 0, u32::MAX);

    let mut raw_data = missing.clone();
    raw_data.extend_from_slice(&oversized);
    let reader = CountingReader::new(payload.clone(), &[0]);
    let error = resolve_vl_references(&reader, &raw_data, &ctx)
        .expect_err("the earlier missing heap object must fail");
    match error {
        Error::InvalidFormat { message } => {
            assert!(
                message.contains("VL object index 7 not found") && message.contains("element 0"),
                "message was: {message}"
            );
        }
        other => panic!("expected invalid format error, got: {other:?}"),
    }
    assert_eq!(reader.collection_read_count(0), 1);

    let mut raw_data = oversized;
    raw_data.extend_from_slice(&missing);
    let reader = CountingReader::new(payload, &[0]);
    let error = resolve_vl_references(&reader, &raw_data, &ctx)
        .expect_err("the earlier oversized object index must fail");
    match error {
        Error::InvalidFormat { message } => {
            assert!(
                message.contains("exceeds u16::MAX") && message.contains("element 0"),
                "message was: {message}"
            );
        }
        other => panic!("expected invalid format error, got: {other:?}"),
    }
    assert_eq!(reader.read_count(), 0);
}
