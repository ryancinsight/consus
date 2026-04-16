//! Global heap: stores collections of variable-length objects.
//!
//! ## Specification (HDF5 File Format Specification, Section IV.B)
//!
//! A global heap collection stores a set of variable-length objects
//! (VL strings, VL sequences, etc.) indexed by a 1-based integer ID.
//! Object 0 is the free-space marker and terminates iteration.
//!
//! ### Global Heap Collection Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | 4    | Signature "GCOL" |
//! | 4      | 1    | Version (1) |
//! | 5      | 3    | Reserved |
//! | 8      | L    | Collection size (length_size bytes) |
//! | 8+L    | var  | Heap objects |
//!
//! ### Global Heap Object Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | 2    | Heap object index (0 = free space / end marker) |
//! | 2      | 2    | Reference count |
//! | 4      | 4    | Reserved |
//! | 8      | L    | Object size (length_size bytes) |
//! | 8+L    | size | Object data (padded to 8-byte boundary) |
//!
//! ## Invariants
//!
//! - Object index 0 marks the end of the used object list.
//! - Object data is padded to an 8-byte boundary on disk; the
//!   `object_size` field gives the unpadded logical size.
//! - Reference counts track how many links point to this object;
//!   a count of 0 means the object slot can be reclaimed.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;

/// Global heap collection signature.
pub const GLOBAL_HEAP_SIGNATURE: [u8; 4] = *b"GCOL";

/// A single object within a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObject {
    /// Object index (1-based; 0 is the free-space marker).
    pub index: u16,
    /// Reference count.
    pub reference_count: u32,
    /// Object data.
    #[cfg(feature = "alloc")]
    pub data: Vec<u8>,
}

/// A parsed global heap collection.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GlobalHeapCollection {
    /// Total collection size in bytes (including header).
    pub collection_size: u64,
    /// Objects in the collection (index 0 / free-space marker excluded).
    pub objects: Vec<GlobalHeapObject>,
}

#[cfg(feature = "alloc")]
impl GlobalHeapCollection {
    /// Parse a global heap collection from the given file address.
    ///
    /// Reads the collection header, validates the signature and version,
    /// then iterates over heap objects until index 0 (free-space marker)
    /// is encountered or the collection size is exhausted.
    ///
    /// ## Arguments
    ///
    /// - `source`: positioned I/O source.
    /// - `address`: file offset of the collection header.
    /// - `ctx`: parsing context (offset/length sizes from superblock).
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "GCOL".
    /// - [`Error::InvalidFormat`] if the version is not 1.
    /// - [`Error::InvalidFormat`] if the collection is truncated.
    pub fn parse<R: ReadAt>(source: &R, address: u64, ctx: &ParseContext) -> Result<Self> {
        let l = ctx.length_bytes();
        // Header: signature(4) + version(1) + reserved(3) + collection_size(L)
        let header_size = 8 + l;
        let mut header_buf = vec![0u8; header_size];
        source.read_at(address, &mut header_buf)?;

        // Validate signature
        if header_buf[0..4] != GLOBAL_HEAP_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: String::from("invalid global heap collection signature"),
            });
        }

        // Validate version
        let version = header_buf[4];
        if version != 1 {
            return Err(Error::InvalidFormat {
                message: alloc::format!("unsupported global heap version: {version}, expected 1"),
            });
        }

        let collection_size = ctx.read_length(&header_buf[8..]);

        // Read the entire collection body
        let body_size = collection_size as usize - header_size;
        if body_size == 0 {
            return Ok(Self {
                collection_size,
                objects: Vec::new(),
            });
        }

        let mut body = vec![0u8; body_size];
        source.read_at(address + header_size as u64, &mut body)?;

        // Parse objects
        let mut objects = Vec::new();
        let mut cursor = 0usize;

        // Object header: index(2) + ref_count(2) + reserved(4) + size(L) = 8 + L bytes
        let obj_header_size = 8 + l;

        while cursor + obj_header_size <= body.len() {
            let index = u16::from_le_bytes([body[cursor], body[cursor + 1]]);

            // Index 0 marks end of objects (free-space marker)
            if index == 0 {
                break;
            }

            let reference_count = u16::from_le_bytes([body[cursor + 2], body[cursor + 3]]) as u32;
            // bytes [cursor+4..cursor+8] are reserved

            let object_size = ctx.read_length(&body[cursor + 8..]) as usize;
            cursor += obj_header_size;

            // Read object data
            if cursor + object_size > body.len() {
                return Err(Error::InvalidFormat {
                    message: alloc::format!(
                        "global heap object {} truncated: need {} bytes at offset {}, have {}",
                        index,
                        object_size,
                        cursor,
                        body.len()
                    ),
                });
            }

            let data = Vec::from(&body[cursor..cursor + object_size]);
            objects.push(GlobalHeapObject {
                index,
                reference_count,
                data,
            });

            // Advance past object data, padded to 8-byte boundary
            let padded_size = (object_size + 7) & !7;
            cursor += padded_size;
        }

        Ok(Self {
            collection_size,
            objects,
        })
    }

    /// Look up an object by its 1-based index.
    ///
    /// Returns `None` if no object with the given index exists.
    pub fn get(&self, index: u16) -> Option<&GlobalHeapObject> {
        self.objects.iter().find(|o| o.index == index)
    }

    /// Look up an object's data by its 1-based index.
    ///
    /// Returns `None` if no object with the given index exists.
    pub fn get_data(&self, index: u16) -> Option<&[u8]> {
        self.get(index).map(|o| o.data.as_slice())
    }
}

// ---------------------------------------------------------------------------
// VL Reference Resolution
// ---------------------------------------------------------------------------

/// Resolve variable-length data references from raw dataset bytes.
///
/// Each VL reference occupies `4 + offset_size + 4` bytes:
/// - 4-byte sequence length (u32 LE, number of elements in the VL object)
/// - `offset_size` bytes: file address of the GCOL collection
/// - 4-byte object index (u32 LE, 1-based)
///
/// ## Algorithm
///
/// 1. Compute `ref_size = 4 + ctx.offset_bytes() + 4`.
/// 2. Validate `raw_data.len() % ref_size == 0`.
/// 3. For each `ref_size`-byte chunk in `raw_data`:
///    a. Read `sequence_length` (u32 LE, bytes `0..4`).
///    b. Read `heap_collection_address` via `ctx.read_offset(&chunk[4..])`.
///    c. Read `object_index` (u32 LE, bytes `4+offset_bytes..4+offset_bytes+4`).
///    d. If `heap_collection_address` equals the offset-size-specific undefined
///       sentinel or `sequence_length == 0`, push an empty `Vec<u8>`.
///    e. Otherwise call [`GlobalHeapCollection::parse`], then `get_data`, and
///       clone the returned slice.
///       Returns [`Error::InvalidFormat`] if the object index is absent.
/// 4. Return `Vec<Vec<u8>>` - one entry per VL element.
///
/// ## Undefined Address Sentinel
///
/// The sentinel is offset-size-dependent: all bits set within the
/// `offset_bytes`-wide address field.
///
/// | offset_bytes | sentinel value |
/// |---|---|
/// | 4 | `0xFFFF_FFFF` |
/// | 8 | `0xFFFF_FFFF_FFFF_FFFF` |
///
/// This matches the HDF5 specification and the `UNDEFINED_ADDRESS` constant
/// for 8-byte offset files.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] if `raw_data.len()` is not a multiple of `ref_size`.
/// - [`Error::InvalidFormat`] if `object_index` does not fit in `u16`.
/// - [`Error::InvalidFormat`] if the heap object is absent from the collection.
/// - Propagates I/O and format errors from [`GlobalHeapCollection::parse`].
#[cfg(feature = "alloc")]
pub fn resolve_vl_references<R: ReadAt>(
    source: &R,
    raw_data: &[u8],
    ctx: &ParseContext,
) -> Result<Vec<Vec<u8>>> {
    let offset_bytes = ctx.offset_bytes();
    let ref_size = 4 + offset_bytes + 4;

    if raw_data.len() % ref_size != 0 {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "VL raw data length {} is not a multiple of ref_size {} (offset_size={})",
                raw_data.len(),
                ref_size,
                offset_bytes,
            ),
        });
    }

    // Offset-size-specific undefined address: all `offset_bytes` bits set.
    // For 4-byte offsets: 0xFFFF_FFFF; for 8-byte offsets: u64::MAX.
    // The guard `offset_bytes == 8` prevents the `1u64 << 64` overflow.
    let undef_addr: u64 = if offset_bytes == 8 {
        u64::MAX
    } else {
        // offset_bytes in {2, 4} => shift in {16, 32} => no overflow.
        (1u64 << (offset_bytes * 8)).wrapping_sub(1)
    };

    let n = raw_data.len() / ref_size;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let base = i * ref_size;
        let chunk = &raw_data[base..base + ref_size];

        let sequence_length =
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let heap_collection_address = ctx.read_offset(&chunk[4..]);
        let idx_base = 4 + offset_bytes;
        let object_index = u32::from_le_bytes([
            chunk[idx_base],
            chunk[idx_base + 1],
            chunk[idx_base + 2],
            chunk[idx_base + 3],
        ]);

        if heap_collection_address == undef_addr || sequence_length == 0 {
            result.push(Vec::new());
            continue;
        }

        // object_index is stored as u32 in the VL reference but the GCOL
        // indexes objects with a u16 field; a value > u16::MAX indicates
        // file corruption.
        let obj_idx = u16::try_from(object_index).map_err(|_| Error::InvalidFormat {
            message: alloc::format!(
                "VL object index {} exceeds u16::MAX at element {} (heap addr {})",
                object_index,
                i,
                heap_collection_address,
            ),
        })?;

        let coll = GlobalHeapCollection::parse(source, heap_collection_address, ctx)?;

        let data = coll.get_data(obj_idx).ok_or_else(|| Error::InvalidFormat {
            message: alloc::format!(
                "VL object index {} not found in heap collection at address {} (element {})",
                obj_idx,
                heap_collection_address,
                i,
            ),
        })?;

        result.push(Vec::from(data));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        use super::*;
        use consus_io::MemCursor;

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
                    let bytes = (collection_size as u32).to_le_bytes();
                    buf[8..12].copy_from_slice(&bytes);
                }
                8 => {
                    let bytes = (collection_size as u64).to_le_bytes();
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
                    let bytes = (obj_data.len() as u32).to_le_bytes();
                    buf[pos + 8..pos + 12].copy_from_slice(&bytes);
                }
                8 => {
                    let bytes = (obj_data.len() as u64).to_le_bytes();
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
            let bytes = (collection_size as u64).to_le_bytes();
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
            let mut file_buf = vec![0u8; offset as usize + collection_buf.len()];
            file_buf[offset as usize..].copy_from_slice(&collection_buf);

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
            raw_data[0..4].copy_from_slice(&(payload.len() as u32).to_le_bytes());
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
            raw_data[0..4].copy_from_slice(&(payload.len() as u32).to_le_bytes());
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
            raw_data[0..4].copy_from_slice(&(payload.len() as u32).to_le_bytes());
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
    }
}
