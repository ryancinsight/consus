//! Huge-object record localization and traversal.

use super::{BTreeV2Header, BTreeV2InternalNode, BTreeV2LeafNode, BTreeV2Record};
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::format;
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;

// ---------------------------------------------------------------------------
// B-tree v2 key search (for fractal heap HUGE object lookup)
// ---------------------------------------------------------------------------

/// Location of a HUGE object in the file: direct address and length.

///
/// Extracted from a type-48 B-tree v2 record in the fractal heap's
/// HUGE object B-tree. The record layout is:
/// | Offset | Size | Field                  |
/// |--------|------|------------------------|
/// | 0      | 8    | Heap ID / object key   |
/// | 8      | O    | File address           |
/// | 8+O    | S    | Object length          |
///
/// where O = offset_size and S = length_size from the parse context.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct HugeObjectLocation {
    /// File address of the HUGE object data.
    pub address: u64,
    /// Size of the object data in bytes.
    pub length: u64,
    /// B-tree key (heap ID) for this object.
    pub object_key: u64,
}

/// Find a HUGE object record in the fractal heap's v2 B-tree by key.
///
/// This is used when the fractal heap header flag bit 0 is CLEAR (0),
/// meaning the `btree_key` in `FractalHeapId::Huge` is a B-tree key
/// that must be searched in the HUGE object B-tree.
///
/// ## Arguments
///
/// - `source`: I/O source (ReadAt).
/// - `btree_address`: Address of the v2 B-tree that indexes HUGE objects.
/// - `btree_header`: The already-parsed B-tree header.
/// - `key`: The B-tree key to search for (from the heap ID's `btree_key`).
/// - `ctx`: Parse context (for variable-width address/length fields).
///
/// ## Returns
///
/// - `Ok(Some(HugeObjectLocation))` if the record is found.
/// - `Ok(None)` if no record with the given key exists.
/// - `Err(...)` on parse or I/O errors.
#[cfg(feature = "alloc")]
pub fn find_huge_object_record<R: ReadAt>(
    source: &R,
    _btree_address: u64,
    btree_header: &BTreeV2Header,
    key: u64,
    ctx: &ParseContext,
) -> Result<Option<HugeObjectLocation>> {
    if btree_header.root_address == crate::constants::UNDEFINED_ADDRESS {
        return Ok(None);
    }
    find_huge_object_recursive(source, btree_header, btree_header.root_address, key, 0, ctx)
}

/// Recursive helper for [`find_huge_object_record`].
#[cfg(feature = "alloc")]
fn find_huge_object_recursive<R: ReadAt>(
    source: &R,
    header: &BTreeV2Header,
    node_address: u64,
    key: u64,
    descent: u16,
    ctx: &ParseContext,
) -> Result<Option<HugeObjectLocation>> {
    // `header.depth` is loop-invariant here, so the internal-node arm below
    // recurses until the stack runs out unless the descent is bounded: a
    // crafted node whose child pointer targets itself never terminates.
    let descent = ctx
        .budget
        .descend(descent, "b-tree v2 huge-object descent")?;

    if header.depth == 0 {
        // Leaf node: search for the key
        let leaf =
            BTreeV2LeafNode::parse(source, node_address, header, header.root_num_records, ctx)?;
        for record in &leaf.records {
            if let Some(loc) = try_parse_huge_object_record(record, header.record_size, ctx)? {
                // The key is the first 8 bytes of the record data (little-endian u64)
                if loc.object_key == key {
                    return Ok(Some(loc));
                }
            }
        }
        Ok(None)
    } else {
        // Internal node: find the child that might contain the key
        let internal =
            BTreeV2InternalNode::parse(source, node_address, header, header.root_num_records, ctx)?;
        // For type-48 (HUGE_OBJECT) records, the B-tree uses binary search on the key.
        // We find the first child whose record key is > our search key.
        let mut child_idx = 0;
        for (i, rec) in internal.records.iter().enumerate() {
            if let Some(first_key) = extract_record_first_key(rec, header.record_size)
                && first_key > key
            {
                break;
            }
            child_idx = i + 1;
        }
        // A truncated node yields no children at all; indexing `len() - 1`
        // would underflow rather than report the malformed node.
        let Some(last) = internal.child_addresses.len().checked_sub(1) else {
            return Err(Error::InvalidFormat {
                message: format!(
                    "b-tree v2 internal node at offset {node_address:#x} has no child pointers"
                ),
            });
        };
        let child_addr = internal.child_addresses[child_idx.min(last)];
        find_huge_object_recursive(source, header, child_addr, key, descent, ctx)
    }
}

/// Attempt to parse a HUGE object record from a B-tree v2 record.
///
/// Returns `Ok(Some(HugeObjectLocation))` on success, `Ok(None)` if
/// the record is too short to contain a valid type-48 record.
#[cfg(feature = "alloc")]
fn try_parse_huge_object_record(
    record: &BTreeV2Record,
    record_size: u16,
    ctx: &ParseContext,
) -> Result<Option<HugeObjectLocation>> {
    // Minimum: 8 (key) + O (address) + S (length)
    let min_len = 8 + ctx.offset_bytes() + ctx.length_bytes();
    if (record_size as usize) < min_len || record.data.len() < min_len {
        return Ok(None);
    }
    let offset_bytes = ctx.offset_bytes();
    let _length_bytes = ctx.length_bytes();
    let object_key = u64::from_le_bytes([
        record.data[0],
        record.data[1],
        record.data[2],
        record.data[3],
        record.data[4],
        record.data[5],
        record.data[6],
        record.data[7],
    ]);
    let address = ctx.read_offset(&record.data[8..]);
    let length = ctx.read_length(&record.data[8 + offset_bytes..]);
    Ok(Some(HugeObjectLocation {
        address,
        length,
        object_key,
    }))
}

/// Extract the first 8 bytes of a B-tree record as a u64 key.
///
/// Used for navigating internal nodes in the HUGE object B-tree.
#[cfg(feature = "alloc")]
fn extract_record_first_key(record: &BTreeV2Record, _record_size: u16) -> Option<u64> {
    if record.data.len() < 8 {
        return None;
    }
    Some(u64::from_le_bytes([
        record.data[0],
        record.data[1],
        record.data[2],
        record.data[3],
        record.data[4],
        record.data[5],
        record.data[6],
        record.data[7],
    ]))
}
