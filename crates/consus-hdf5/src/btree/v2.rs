//! B-tree version 2 types, structures, and parsing.
//!
//! ## Specification (HDF5 File Format Specification, Section III.A.2)
//!
//! Used in newer files (superblock v2+). B-tree v2 provides a more
//! compact and efficient indexing structure than v1, with explicit
//! record types, configurable node sizes, and depth tracking.
//!
//! ### B-tree v2 Header Layout (signature "BTHD")
//!
//! | Offset | Size | Field                              |
//! |--------|------|------------------------------------|
//! | 0      | 4    | Signature ("BTHD")                 |
//! | 4      | 1    | Version (0)                        |
//! | 5      | 1    | Type (record type identifier)      |
//! | 6      | 4    | Node size in bytes                 |
//! | 10     | 2    | Record size in bytes               |
//! | 12     | 2    | Depth                              |
//! | 14     | 1    | Split percent                      |
//! | 15     | 1    | Merge percent                      |
//! | 16     | S    | Root node address (offset_size)    |
//! | 16+S   | 2    | Number of records in root node     |
//! | 18+S   | S    | Total number of records in tree    |
//! | 18+2S  | 4    | Checksum                           |
//!
//! ### B-tree v2 Record Types
//!
//! | Type | Description                                     |
//! |------|-------------------------------------------------|
//! | 1    | Shared object header message (v1)               |
//! | 2    | Shared object header message (v2)               |
//! | 3    | Unsorted, non-filtered, non-paged              |
//! | 4    | Unsorted, filtered, non-paged                  |
//! | 5    | Link name (for indexed group)                   |
//! | 6    | Creation order (for indexed group)              |
//! | 7    | Shared header message, sorted by ref count      |
//! | 8    | Attribute name (for indexed attributes)         |
//! | 9    | Attribute creation order                        |
//! | 10   | Chunked data, non-filtered, non-paged (v4 layout) |
//! | 11   | Chunked data, filtered, non-paged (v4 layout)  |
//!
//! ### B-tree v2 Internal Node Layout (signature "BTIN")
//!
//! | Offset | Size | Field                              |
//! |--------|------|------------------------------------|
//! | 0      | 4    | Signature ("BTIN")                 |
//! | 4      | 1    | Version (0)                        |
//! | 5      | 1    | Type                               |
//! | 6      | var  | Records (record_count × record_size) |
//! | var    | var  | Child pointers and record counts   |
//! | var    | 4    | Checksum                           |
//!
//! ### B-tree v2 Leaf Node Layout (signature "BTLF")
//!
//! | Offset | Size | Field                              |
//! |--------|------|------------------------------------|
//! | 0      | 4    | Signature ("BTLF")                 |
//! | 4      | 1    | Version (0)                        |
//! | 5      | 1    | Type                               |
//! | 6      | var  | Records (record_count × record_size) |
//! | var    | 4    | Checksum                           |

#[cfg(feature = "alloc")]
use alloc::{format, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::Error;
use consus_core::Result;

#[cfg(feature = "alloc")]
use consus_io::ReadAt;

#[cfg(feature = "alloc")]
use crate::address::ParseContext;

/// B-tree v2 header signature.
pub const BTREE_V2_SIGNATURE: [u8; 4] = *b"BTHD";

/// B-tree v2 internal node signature.
pub const BTREE_V2_INTERNAL_SIGNATURE: [u8; 4] = *b"BTIN";

/// B-tree v2 leaf node signature.
pub const BTREE_V2_LEAF_SIGNATURE: [u8; 4] = *b"BTLF";

/// B-tree v2 record type constants.
pub mod record_type {
    /// Testing only (type 0).
    pub const TESTING: u8 = 0;
    /// Shared object header message index (version 1).
    pub const SHARED_MSG_V1: u8 = 1;
    /// Shared object header message index (version 2).
    pub const SHARED_MSG_V2: u8 = 2;
    /// Unsorted, non-filtered, non-paged chunked data (v3 layout B-tree v1 replacement).
    pub const CHUNK_NON_FILTERED: u8 = 3;
    /// Unsorted, filtered, non-paged chunked data.
    pub const CHUNK_FILTERED: u8 = 4;
    /// Link name index for dense groups.
    pub const LINK_NAME: u8 = 5;
    /// Creation order index for dense groups.
    pub const LINK_CREATION_ORDER: u8 = 6;
    /// Shared header message sorted by reference count.
    pub const SHARED_MSG_BY_REFCOUNT: u8 = 7;
    /// Attribute name index for dense attribute storage.
    pub const ATTRIBUTE_NAME: u8 = 8;
    /// Attribute creation order index.
    pub const ATTRIBUTE_CREATION_ORDER: u8 = 9;
    /// Non-filtered chunked data, non-paged (v4 layout).
    pub const CHUNK_V4_NON_FILTERED: u8 = 10;
    /// Filtered chunked data, non-paged (v4 layout).
    pub const CHUNK_V4_FILTERED: u8 = 11;
    /// Fractal heap huge object index (record type 48).
    /// Used by dense group/attribute storage to locate huge objects
    /// stored outside the managed fractal heap space.
    pub const HUGE_OBJECT: u8 = 48;
}

/// Parsed B-tree v2 header.
///
/// Contains the structural parameters of the B-tree required to
/// navigate internal and leaf nodes.
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    /// Record type identifier (see [`record_type`] constants).
    pub record_type: u8,
    /// Node size in bytes (both internal and leaf nodes).
    pub node_size: u32,
    /// Size of one record in bytes.
    pub record_size: u16,
    /// Tree depth (0 = root is a leaf node).
    pub depth: u16,
    /// Split percent (threshold for splitting a node).
    pub split_percent: u8,
    /// Merge percent (threshold for merging nodes).
    pub merge_percent: u8,
    /// Address of the root node.
    ///
    /// `u64::MAX` (undefined address) indicates an empty tree.
    pub root_address: u64,
    /// Number of records in the root node.
    pub root_num_records: u16,
    /// Total number of records in the entire tree.
    pub total_records: u64,
}

#[cfg(feature = "alloc")]
impl BTreeV2Header {
    /// Parse a B-tree v2 header from the given file address.
    ///
    /// Reads and validates the "BTHD" signature, version, and all
    /// structural fields. The checksum is read but not yet verified.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "BTHD".
    /// - [`Error::InvalidFormat`] if the version is not 0.
    /// - [`Error::InvalidFormat`] if the data is truncated.
    pub fn parse<R: ReadAt>(source: &R, address: u64, ctx: &ParseContext) -> Result<Self> {
        let s = ctx.offset_bytes();
        // Minimum size: signature(4) + version(1) + type(1) + node_size(4) +
        //   record_size(2) + depth(2) + split%(1) + merge%(1) +
        //   root_addr(S) + root_nrec(2) + total_records(S) + checksum(4)
        let min_size = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + s + 2 + s + 4;

        let mut buf = vec![0u8; min_size];
        source.read_at(address, &mut buf)?;

        // Validate signature
        if buf[0..4] != BTREE_V2_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 header signature 'BTHD' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        // Validate version
        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!("unsupported B-tree v2 header version: {version}, expected 0"),
            });
        }

        let record_type = buf[5];
        let node_size = u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]);
        let record_size = u16::from_le_bytes([buf[10], buf[11]]);
        let depth = u16::from_le_bytes([buf[12], buf[13]]);
        let split_percent = buf[14];
        let merge_percent = buf[15];

        let mut pos = 16;
        let root_address = ctx.read_offset(&buf[pos..]);
        pos += s;
        let root_num_records = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let total_records = ctx.read_length(&buf[pos..]);
        // pos += s; // next would be checksum, which we skip for now

        Ok(Self {
            record_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_address,
            root_num_records,
            total_records,
        })
    }
}

/// A raw record extracted from a B-tree v2 leaf or internal node.
///
/// The interpretation of the record bytes depends on the B-tree's
/// `record_type`. Callers must decode the bytes according to the
/// applicable record schema.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct BTreeV2Record {
    /// Raw record bytes (length = `BTreeV2Header::record_size`).
    pub data: Vec<u8>,
}

/// Parsed B-tree v2 leaf node.
///
/// Contains the raw records stored in the leaf. The record format
/// depends on the B-tree's record type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct BTreeV2LeafNode {
    /// Record type (must match the tree header's record type).
    pub record_type: u8,
    /// Records in this leaf.
    pub records: Vec<BTreeV2Record>,
}

#[cfg(feature = "alloc")]
impl BTreeV2LeafNode {
    /// Parse a B-tree v2 leaf node at the given address.
    ///
    /// Reads the "BTLF" signature, version, type, and `num_records`
    /// records of `record_size` bytes each.
    ///
    /// ## Arguments
    ///
    /// - `source`: I/O source.
    /// - `address`: file address of the leaf node.
    /// - `header`: the B-tree v2 header (provides record size and node size).
    /// - `num_records`: number of records in this leaf node.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "BTLF".
    /// - [`Error::InvalidFormat`] if the version is not 0.
    pub fn parse<R: ReadAt>(
        source: &R,
        address: u64,
        header: &BTreeV2Header,
        num_records: u16,
    ) -> Result<Self> {
        // Read the full node (node_size bytes)
        let node_size = header.node_size as usize;
        let mut buf = vec![0u8; node_size];
        source.read_at(address, &mut buf)?;

        // Validate signature
        if buf[0..4] != BTREE_V2_LEAF_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 leaf signature 'BTLF' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!("unsupported B-tree v2 leaf version: {version}, expected 0"),
            });
        }

        let record_type = buf[5];
        let rec_size = header.record_size as usize;

        let mut records = Vec::with_capacity(num_records as usize);
        let mut pos = 6; // after signature(4) + version(1) + type(1)

        for _ in 0..num_records {
            if pos + rec_size > buf.len().saturating_sub(4) {
                // Don't read into checksum territory
                break;
            }
            let data = Vec::from(&buf[pos..pos + rec_size]);
            records.push(BTreeV2Record { data });
            pos += rec_size;
        }

        Ok(Self {
            record_type,
            records,
        })
    }
}

/// Parsed B-tree v2 internal node.
///
/// Contains records and child node pointers for tree traversal.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct BTreeV2InternalNode {
    /// Record type (must match the tree header's record type).
    pub record_type: u8,
    /// Records in this internal node.
    pub records: Vec<BTreeV2Record>,
    /// Addresses of child nodes (length = num_records + 1).
    pub child_addresses: Vec<u64>,
    /// Number of records in each child node.
    pub child_num_records: Vec<u16>,
}

#[cfg(feature = "alloc")]
impl BTreeV2InternalNode {
    /// Parse a B-tree v2 internal node at the given address.
    ///
    /// Reads the "BTIN" signature, records, and child pointer table.
    ///
    /// ## Arguments
    ///
    /// - `source`: I/O source.
    /// - `address`: file address of the internal node.
    /// - `header`: B-tree v2 header (provides record size, node size).
    /// - `num_records`: number of records in this node.
    /// - `ctx`: parsing context for variable-width addresses.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "BTIN".
    pub fn parse<R: ReadAt>(
        source: &R,
        address: u64,
        header: &BTreeV2Header,
        num_records: u16,
        ctx: &ParseContext,
    ) -> Result<Self> {
        let node_size = header.node_size as usize;
        let mut buf = vec![0u8; node_size];
        source.read_at(address, &mut buf)?;

        // Validate signature
        if buf[0..4] != BTREE_V2_INTERNAL_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 internal signature 'BTIN' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!(
                    "unsupported B-tree v2 internal node version: {version}, expected 0"
                ),
            });
        }

        let record_type = buf[5];
        let rec_size = header.record_size as usize;
        let s = ctx.offset_bytes();
        let n_rec = num_records as usize;
        let n_children = n_rec + 1;

        // Parse records
        let mut records = Vec::with_capacity(n_rec);
        let mut pos = 6;
        for _ in 0..n_rec {
            if pos + rec_size > buf.len() {
                break;
            }
            let data = Vec::from(&buf[pos..pos + rec_size]);
            records.push(BTreeV2Record { data });
            pos += rec_size;
        }

        // Parse child pointers: each child has address(S) + num_records(variable).
        // The num_records field width depends on the maximum possible records
        // that can fit in a node. For simplicity, we use 2 bytes (u16) which
        // is sufficient for the vast majority of HDF5 files.
        //
        // Per the spec, the width is ceil(log2(max_records_per_node + 1)) / 8,
        // but 2 bytes handles nodes up to 65535 records.
        let num_records_width = compute_num_records_width(header);

        let mut child_addresses = Vec::with_capacity(n_children);
        let mut child_num_records = Vec::with_capacity(n_children);

        for _ in 0..n_children {
            if pos + s > buf.len() {
                break;
            }
            let addr = ctx.read_offset(&buf[pos..]);
            pos += s;

            let nrec = read_variable_width_uint(&buf[pos..], num_records_width);
            pos += num_records_width;

            child_addresses.push(addr);
            child_num_records.push(nrec as u16);
        }

        // If depth > 1, there may also be total_records fields for each child,
        // but we skip those for the current implementation level.

        Ok(Self {
            record_type,
            records,
            child_addresses,
            child_num_records,
        })
    }
}

/// Compute the byte width of the num_records field in internal node
/// child pointer entries.
///
/// The width is `ceil(log2(max_leaf_records + 1)) / 8`, clamped to
/// at least 1 byte.
///
/// For most practical HDF5 files (node_size ≤ 64 KiB, record_size ≥ 1),
/// this is 1 or 2 bytes.
fn compute_num_records_width(header: &BTreeV2Header) -> usize {
    // Maximum records in a leaf node:
    // node_size - signature(4) - version(1) - type(1) - checksum(4) = usable
    // max_leaf_records = usable / record_size
    let usable = (header.node_size as usize).saturating_sub(10);
    let max_leaf_records = if header.record_size > 0 {
        usable / header.record_size as usize
    } else {
        0
    };

    // Width in bytes = ceil(ceil(log2(max_leaf_records + 1)) / 8)
    if max_leaf_records == 0 {
        return 1;
    }
    let bits_needed = u32::BITS - (max_leaf_records as u32).leading_zeros();
    let bytes_needed = (bits_needed as usize + 7) / 8;
    bytes_needed.max(1)
}

/// Read a variable-width unsigned integer (1–8 bytes, little-endian).
fn read_variable_width_uint(buf: &[u8], width: usize) -> u64 {
    let mut value = 0u64;
    for i in 0..width.min(8).min(buf.len()) {
        value |= (buf[i] as u64) << (i * 8);
    }
    value
}

/// Collect all records from a B-tree v2 by recursive traversal.
///
/// Traverses the tree from the root, visiting all internal and leaf
/// nodes, and returns the raw records in tree order (left to right).
///
/// ## Arguments
///
/// - `source`: I/O source.
/// - `header`: the parsed B-tree v2 header.
/// - `ctx`: parsing context for variable-width addresses.
///
/// ## Returns
///
/// All records in the tree, in order. Returns an empty vector for
/// an empty tree (root_address == `u64::MAX`).
///
/// ## Errors
///
/// Propagates I/O and format errors from node parsing.
#[cfg(feature = "alloc")]
pub fn collect_all_records<R: ReadAt>(
    source: &R,
    header: &BTreeV2Header,
    ctx: &ParseContext,
) -> Result<Vec<BTreeV2Record>> {
    if header.root_address == crate::constants::UNDEFINED_ADDRESS {
        return Ok(Vec::new());
    }

    if header.total_records == 0 {
        return Ok(Vec::new());
    }

    let mut records = Vec::with_capacity(header.total_records as usize);
    collect_records_recursive(
        source,
        header,
        header.root_address,
        header.root_num_records,
        header.depth,
        ctx,
        &mut records,
    )?;
    Ok(records)
}

/// Recursive helper for [`collect_all_records`].
#[cfg(feature = "alloc")]
fn collect_records_recursive<R: ReadAt>(
    source: &R,
    header: &BTreeV2Header,
    node_address: u64,
    num_records: u16,
    depth: u16,
    ctx: &ParseContext,
    records: &mut Vec<BTreeV2Record>,
) -> Result<()> {
    if depth == 0 {
        // Leaf node
        let leaf = BTreeV2LeafNode::parse(source, node_address, header, num_records)?;
        records.extend(leaf.records);
    } else {
        // Internal node
        let internal = BTreeV2InternalNode::parse(source, node_address, header, num_records, ctx)?;

        // Interleave: child[0], record[0], child[1], record[1], ..., child[N]
        let n_rec = internal.records.len();
        let n_children = internal.child_addresses.len();

        for i in 0..n_children {
            // Visit child[i]
            if i < n_children {
                let child_addr = internal.child_addresses[i];
                let child_nrec = internal.child_num_records[i];
                collect_records_recursive(
                    source,
                    header,
                    child_addr,
                    child_nrec,
                    depth - 1,
                    ctx,
                    records,
                )?;
            }

            // Emit record[i] (interleaved between children)
            if i < n_rec {
                records.push(internal.records[i].clone());
            }
        }
    }
    Ok(())
}

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
    find_huge_object_recursive(source, btree_header, btree_header.root_address, key, ctx)
}

/// Recursive helper for [`find_huge_object_record`].
#[cfg(feature = "alloc")]
fn find_huge_object_recursive<R: ReadAt>(
    source: &R,
    header: &BTreeV2Header,
    node_address: u64,
    key: u64,
    ctx: &ParseContext,
) -> Result<Option<HugeObjectLocation>> {
    if header.depth == 0 {
        // Leaf node: search for the key
        let leaf = BTreeV2LeafNode::parse(source, node_address, header, header.root_num_records)?;
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
            if let Some(first_key) = extract_record_first_key(rec, header.record_size) {
                if first_key > key {
                    break;
                }
            }
            child_idx = i + 1;
        }
        if child_idx >= internal.child_addresses.len() {
            child_idx = internal.child_addresses.len() - 1;
        }
        let child_addr = internal.child_addresses[child_idx];
        let _child_nrec = internal.child_num_records[child_idx];
        find_huge_object_recursive(source, header, child_addr, key, ctx)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_constants() {
        assert_eq!(&BTREE_V2_SIGNATURE, b"BTHD");
        assert_eq!(&BTREE_V2_INTERNAL_SIGNATURE, b"BTIN");
        assert_eq!(&BTREE_V2_LEAF_SIGNATURE, b"BTLF");
    }

    #[test]
    fn record_type_constants_distinct() {
        let types = [
            record_type::TESTING,
            record_type::SHARED_MSG_V1,
            record_type::SHARED_MSG_V2,
            record_type::CHUNK_NON_FILTERED,
            record_type::CHUNK_FILTERED,
            record_type::LINK_NAME,
            record_type::LINK_CREATION_ORDER,
            record_type::SHARED_MSG_BY_REFCOUNT,
            record_type::ATTRIBUTE_NAME,
            record_type::ATTRIBUTE_CREATION_ORDER,
            record_type::CHUNK_V4_NON_FILTERED,
            record_type::CHUNK_V4_FILTERED,
        ];
        for (i, &a) in types.iter().enumerate() {
            for (j, &b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "record types at indices {i} and {j} must differ");
                }
            }
        }
    }

    #[test]
    fn read_variable_width_uint_cases() {
        // 1 byte
        assert_eq!(read_variable_width_uint(&[0x42], 1), 0x42);
        // 2 bytes LE
        assert_eq!(read_variable_width_uint(&[0x34, 0x12], 2), 0x1234);
        // 4 bytes LE
        assert_eq!(
            read_variable_width_uint(&[0x78, 0x56, 0x34, 0x12], 4),
            0x12345678
        );
        // 8 bytes LE
        assert_eq!(
            read_variable_width_uint(&[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80], 8),
            0x8000000000000001
        );
    }

    #[test]
    fn read_variable_width_uint_empty() {
        assert_eq!(read_variable_width_uint(&[], 0), 0);
    }

    #[test]
    fn compute_num_records_width_typical() {
        // node_size=4096, record_size=8 → usable=4086, max_leaf=510
        // bits_needed = 10, bytes = 2
        let h = BTreeV2Header {
            record_type: 5,
            node_size: 4096,
            record_size: 8,
            depth: 0,
            split_percent: 98,
            merge_percent: 40,
            root_address: 0,
            root_num_records: 0,
            total_records: 0,
        };
        assert_eq!(compute_num_records_width(&h), 2);
    }

    #[test]
    fn compute_num_records_width_small_node() {
        // node_size=64, record_size=16 → usable=54, max_leaf=3
        // bits_needed = 2, bytes = 1
        let h = BTreeV2Header {
            record_type: 5,
            node_size: 64,
            record_size: 16,
            depth: 0,
            split_percent: 98,
            merge_percent: 40,
            root_address: 0,
            root_num_records: 0,
            total_records: 0,
        };
        assert_eq!(compute_num_records_width(&h), 1);
    }

    #[test]
    fn compute_num_records_width_zero_record_size() {
        let h = BTreeV2Header {
            record_type: 0,
            node_size: 4096,
            record_size: 0,
            depth: 0,
            split_percent: 0,
            merge_percent: 0,
            root_address: 0,
            root_num_records: 0,
            total_records: 0,
        };
        assert_eq!(compute_num_records_width(&h), 1);
    }

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        use super::*;
        use consus_io::MemCursor;

        /// Build a minimal v2 B-tree header in memory and parse it.
        fn build_minimal_bthd(offset_size: u8) -> Vec<u8> {
            let s = offset_size as usize;
            let total = 16 + s + 2 + s + 4; // fixed fields + root_addr + root_nrec + total + cksum
            let mut buf = vec![0u8; total];

            // Signature
            buf[0..4].copy_from_slice(b"BTHD");
            // Version 0
            buf[4] = 0;
            // Record type = 5 (link name)
            buf[5] = record_type::LINK_NAME;
            // Node size = 4096
            buf[6] = 0x00;
            buf[7] = 0x10;
            buf[8] = 0x00;
            buf[9] = 0x00;
            // Record size = 16
            buf[10] = 0x10;
            buf[11] = 0x00;
            // Depth = 0
            buf[12] = 0x00;
            buf[13] = 0x00;
            // Split percent = 98
            buf[14] = 98;
            // Merge percent = 40
            buf[15] = 40;

            let mut pos = 16;
            // Root address = 0x2000
            match s {
                4 => {
                    buf[pos..pos + 4].copy_from_slice(&0x2000u32.to_le_bytes());
                }
                8 => {
                    buf[pos..pos + 8].copy_from_slice(&0x2000u64.to_le_bytes());
                }
                _ => {}
            }
            pos += s;
            // Root num records = 5
            buf[pos] = 5;
            buf[pos + 1] = 0;
            pos += 2;
            // Total records = 42
            match s {
                4 => {
                    buf[pos..pos + 4].copy_from_slice(&42u32.to_le_bytes());
                }
                8 => {
                    buf[pos..pos + 8].copy_from_slice(&42u64.to_le_bytes());
                }
                _ => {}
            }
            // Checksum is at the end; left as zeros.
            buf
        }

        #[test]
        fn parse_header_8byte_offsets() {
            let ctx = ParseContext::new(8, 8);
            let data = build_minimal_bthd(8);
            let cursor = MemCursor::from_bytes(data);
            let header = BTreeV2Header::parse(&cursor, 0, &ctx).unwrap();

            assert_eq!(header.record_type, record_type::LINK_NAME);
            assert_eq!(header.node_size, 4096);
            assert_eq!(header.record_size, 16);
            assert_eq!(header.depth, 0);
            assert_eq!(header.split_percent, 98);
            assert_eq!(header.merge_percent, 40);
            assert_eq!(header.root_address, 0x2000);
            assert_eq!(header.root_num_records, 5);
            assert_eq!(header.total_records, 42);
        }

        #[test]
        fn parse_header_4byte_offsets() {
            let ctx = ParseContext::new(4, 4);
            let data = build_minimal_bthd(4);
            let cursor = MemCursor::from_bytes(data);
            let header = BTreeV2Header::parse(&cursor, 0, &ctx).unwrap();

            assert_eq!(header.root_address, 0x2000);
            assert_eq!(header.total_records, 42);
        }

        #[test]
        fn reject_bad_signature() {
            let ctx = ParseContext::new(8, 8);
            let mut data = build_minimal_bthd(8);
            data[0] = b'X'; // corrupt signature
            let cursor = MemCursor::from_bytes(data);
            let err = BTreeV2Header::parse(&cursor, 0, &ctx).unwrap_err();
            match err {
                Error::InvalidFormat { message } => {
                    assert!(message.contains("BTHD"));
                }
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        #[test]
        fn reject_bad_version() {
            let ctx = ParseContext::new(8, 8);
            let mut data = build_minimal_bthd(8);
            data[4] = 1; // unsupported version
            let cursor = MemCursor::from_bytes(data);
            let err = BTreeV2Header::parse(&cursor, 0, &ctx).unwrap_err();
            match err {
                Error::InvalidFormat { message } => {
                    assert!(message.contains("version"));
                }
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        /// Build a minimal leaf node and verify parsing.
        #[test]
        fn parse_leaf_node() {
            let header = BTreeV2Header {
                record_type: record_type::LINK_NAME,
                node_size: 64,
                record_size: 4,
                depth: 0,
                split_percent: 98,
                merge_percent: 40,
                root_address: 0,
                root_num_records: 3,
                total_records: 3,
            };

            let mut buf = vec![0u8; 64];
            // Signature
            buf[0..4].copy_from_slice(b"BTLF");
            buf[4] = 0; // version
            buf[5] = record_type::LINK_NAME; // type

            // 3 records of 4 bytes each
            buf[6..10].copy_from_slice(&[0x01, 0x02, 0x03, 0x04]); // record 0
            buf[10..14].copy_from_slice(&[0x11, 0x12, 0x13, 0x14]); // record 1
            buf[14..18].copy_from_slice(&[0x21, 0x22, 0x23, 0x24]); // record 2

            // Checksum at end (bytes 60..64), left as zeros.

            let cursor = MemCursor::from_bytes(buf);
            let leaf = BTreeV2LeafNode::parse(&cursor, 0, &header, 3).unwrap();

            assert_eq!(leaf.record_type, record_type::LINK_NAME);
            assert_eq!(leaf.records.len(), 3);
            assert_eq!(leaf.records[0].data, [0x01, 0x02, 0x03, 0x04]);
            assert_eq!(leaf.records[1].data, [0x11, 0x12, 0x13, 0x14]);
            assert_eq!(leaf.records[2].data, [0x21, 0x22, 0x23, 0x24]);
        }

        #[test]
        fn reject_bad_leaf_signature() {
            let header = BTreeV2Header {
                record_type: 0,
                node_size: 32,
                record_size: 4,
                depth: 0,
                split_percent: 0,
                merge_percent: 0,
                root_address: 0,
                root_num_records: 0,
                total_records: 0,
            };

            let mut buf = vec![0u8; 32];
            buf[0..4].copy_from_slice(b"XXXX"); // bad signature

            let cursor = MemCursor::from_bytes(buf);
            let err = BTreeV2LeafNode::parse(&cursor, 0, &header, 0).unwrap_err();
            match err {
                Error::InvalidFormat { message } => {
                    assert!(message.contains("BTLF"));
                }
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        /// Collect all records from a single-leaf tree.
        #[test]
        fn collect_all_records_single_leaf() {
            let ctx = ParseContext::new(8, 8);

            // Build header pointing at address 0x100
            let header = BTreeV2Header {
                record_type: record_type::LINK_NAME,
                node_size: 64,
                record_size: 4,
                depth: 0, // leaf is root
                split_percent: 98,
                merge_percent: 40,
                root_address: 0x100,
                root_num_records: 2,
                total_records: 2,
            };

            // Build leaf node at address 0x100
            let mut file_data = vec![0u8; 0x200];
            let leaf_offset = 0x100;
            file_data[leaf_offset..leaf_offset + 4].copy_from_slice(b"BTLF");
            file_data[leaf_offset + 4] = 0; // version
            file_data[leaf_offset + 5] = record_type::LINK_NAME;
            // Record 0
            file_data[leaf_offset + 6..leaf_offset + 10].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
            // Record 1
            file_data[leaf_offset + 10..leaf_offset + 14]
                .copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);

            let cursor = MemCursor::from_bytes(file_data);
            let records = collect_all_records(&cursor, &header, &ctx).unwrap();

            assert_eq!(records.len(), 2);
            assert_eq!(records[0].data, [0xAA, 0xBB, 0xCC, 0xDD]);
            assert_eq!(records[1].data, [0x11, 0x22, 0x33, 0x44]);
        }

        /// Empty tree (undefined root address) returns no records.
        #[test]
        fn collect_all_records_empty_tree() {
            let ctx = ParseContext::new(8, 8);
            let header = BTreeV2Header {
                record_type: 0,
                node_size: 64,
                record_size: 4,
                depth: 0,
                split_percent: 0,
                merge_percent: 0,
                root_address: u64::MAX, // undefined
                root_num_records: 0,
                total_records: 0,
            };

            let cursor = MemCursor::from_bytes(vec![0u8; 64]);
            let records = collect_all_records(&cursor, &header, &ctx).unwrap();
            assert!(records.is_empty());
        }
    }
}
