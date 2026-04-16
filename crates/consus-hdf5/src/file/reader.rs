//! HDF5 file reader: high-level read operations composing all parsers.
//!
//! This module provides functions that orchestrate the low-level parsers
//! (object headers, B-trees, heaps, datatypes, dataspaces, layouts) into
//! coherent read operations: opening files, navigating groups, reading
//! dataset metadata, and extracting raw data.
//!
//! ## Architecture (Dependency Inversion Principle)
//!
//! All I/O is performed through `consus_io::ReadAt`. Compression is
//! accessed through `consus_compression::CompressionRegistry`. No
//! concrete I/O or codec types are referenced directly.
//!
//! ## Read Path
//!
//! ```text
//! Hdf5File::open
//!   └─► Superblock::read_from
//!         └─► read_object_header (root group)
//!               ├─► classify_object → NodeType
//!               ├─► list_group_v1 (symbol table + B-tree v1 + local heap)
//!               ├─► list_group_v2 (link messages or link info)
//!               └─► read_dataset_metadata (datatype + dataspace + layout + filters)
//! ```

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::{Error, NodeType, Result};

#[cfg(feature = "alloc")]
use consus_io::ReadAt;

#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use crate::constants::UNDEFINED_ADDRESS;
#[cfg(feature = "alloc")]
use crate::object_header::message_types;
#[cfg(feature = "alloc")]
use crate::object_header::{HeaderMessage, OHDR_SIGNATURE, ObjectHeader};

// ---------------------------------------------------------------------------
// Object header dispatch
// ---------------------------------------------------------------------------

/// Read and parse the object header at `address`.
///
/// Dispatches to the v1 or v2 parser based on the first 4 bytes:
/// - If the bytes match the `OHDR` signature → version 2.
/// - Otherwise → version 1 (first byte is the version number `1`).
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] if the header is structurally invalid.
/// - I/O errors propagated from the source.
#[cfg(feature = "alloc")]
pub fn read_object_header<R: ReadAt>(
    source: &R,
    address: u64,
    ctx: &ParseContext,
) -> Result<ObjectHeader> {
    let mut peek = [0u8; 4];
    source.read_at(address, &mut peek)?;

    if peek == OHDR_SIGNATURE {
        crate::object_header::v2::parse(source, address, ctx)
    } else {
        crate::object_header::v1::parse(source, address, ctx)
    }
}

// ---------------------------------------------------------------------------
// Message search helpers
// ---------------------------------------------------------------------------

/// Return the first header message matching `msg_type`, or `None`.
#[cfg(feature = "alloc")]
pub fn find_message(header: &ObjectHeader, msg_type: u16) -> Option<&HeaderMessage> {
    header.messages.iter().find(|m| m.message_type == msg_type)
}

/// Return all header messages matching `msg_type`.
#[cfg(feature = "alloc")]
pub fn find_messages(header: &ObjectHeader, msg_type: u16) -> Vec<&HeaderMessage> {
    header
        .messages
        .iter()
        .filter(|m| m.message_type == msg_type)
        .collect()
}

// ---------------------------------------------------------------------------
// Object classification
// ---------------------------------------------------------------------------

/// Classify an object by inspecting its header messages.
///
/// ## Classification Rules
///
/// | Condition | Result |
/// |-----------|--------|
/// | Dataspace + Datatype + Layout present | `Dataset` |
/// | Datatype present without Dataspace | `NamedDatatype` |
/// | Otherwise | `Group` |
#[cfg(feature = "alloc")]
pub fn classify_object(header: &ObjectHeader) -> NodeType {
    let has_dataspace = header
        .messages
        .iter()
        .any(|m| m.message_type == message_types::DATASPACE);
    let has_datatype = header
        .messages
        .iter()
        .any(|m| m.message_type == message_types::DATATYPE);
    let has_layout = header
        .messages
        .iter()
        .any(|m| m.message_type == message_types::DATA_LAYOUT);

    if has_dataspace && has_datatype && has_layout {
        NodeType::Dataset
    } else if has_datatype && !has_dataspace {
        NodeType::NamedDatatype
    } else {
        NodeType::Group
    }
}

// ---------------------------------------------------------------------------
// Dataset metadata extraction
// ---------------------------------------------------------------------------

/// Extract resolved dataset metadata from a parsed object header.
///
/// Parses the datatype (0x0003), dataspace (0x0001), layout (0x0008),
/// and optional filter pipeline (0x000B) messages.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] if a required message is missing or malformed.
#[cfg(feature = "alloc")]
pub fn read_dataset_metadata(
    header: &ObjectHeader,
    ctx: &ParseContext,
) -> Result<crate::dataset::Hdf5Dataset> {
    // --- Datatype ---
    let dt_msg =
        find_message(header, message_types::DATATYPE).ok_or_else(|| Error::InvalidFormat {
            message: String::from("dataset object header missing datatype message"),
        })?;
    let datatype = crate::datatype::compound::parse_datatype(&dt_msg.data)?;

    // --- Dataspace ---
    let ds_msg =
        find_message(header, message_types::DATASPACE).ok_or_else(|| Error::InvalidFormat {
            message: String::from("dataset object header missing dataspace message"),
        })?;
    let shape = crate::dataspace::parse_dataspace(&ds_msg.data, ctx.offset_size)?;

    // --- Layout ---
    let layout_msg =
        find_message(header, message_types::DATA_LAYOUT).ok_or_else(|| Error::InvalidFormat {
            message: String::from("dataset object header missing layout message"),
        })?;
    let layout_info = crate::dataset::layout::DataLayout::parse(&layout_msg.data, ctx)?;

    // --- Filter pipeline (optional) ---
    let filters: Vec<u16> =
        if let Some(fp_msg) = find_message(header, message_types::FILTER_PIPELINE) {
            let pipeline = crate::filter::Hdf5FilterPipeline::parse(&fp_msg.data)?;
            pipeline.filters.iter().map(|f| f.filter_id).collect()
        } else {
            Vec::new()
        };

    // --- Chunk shape ---
    let chunk_shape = layout_info.chunk_dims.as_ref().and_then(|dims| {
        let dims_usize: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        consus_core::ChunkShape::new(&dims_usize)
    });

    Ok(crate::dataset::Hdf5Dataset {
        path: String::new(),
        object_header_address: 0,
        datatype,
        shape,
        layout: layout_info.layout,
        chunk_shape,
        data_address: layout_info.data_address,
        filters,
    })
}

// ---------------------------------------------------------------------------
// Attribute extraction
// ---------------------------------------------------------------------------

/// Extract all attributes from an object header.
///
/// Iterates over all attribute messages (0x000C) and parses each one.
/// Also enumerates dense attributes via the Attribute Info message (0x0015)
/// when present: uses the fractal heap + v2 B-tree (record type 8) as the
/// authoritative attribute store for that object.
/// Returns attributes in the order they appear in the header, compact first.
#[cfg(feature = "alloc")]
pub fn read_attributes<R: ReadAt>(
    source: &R,
    header: &ObjectHeader,
    ctx: &ParseContext,
) -> Result<Vec<crate::attribute::Hdf5Attribute>> {
    let mut attrs = Vec::new();

    // Compact attributes: direct attribute messages (0x000C).
    let attr_msgs = find_messages(header, message_types::ATTRIBUTE);
    for msg in attr_msgs {
        let attr = crate::attribute::Hdf5Attribute::parse(&msg.data, ctx)?;
        attrs.push(attr);
    }

    // Dense attributes: Attribute Info message (0x0015) -> fractal heap + B-tree v2.
    if let Some(ai_msg) = find_message(header, message_types::ATTRIBUTE_INFO) {
        if let Ok(attr_info) = crate::attribute::info::AttributeInfo::parse(&ai_msg.data, ctx) {
            if attr_info.fractal_heap_address != UNDEFINED_ADDRESS {
                let dense = collect_dense_attributes(source, &attr_info, ctx)?;
                attrs.extend(dense);
            }
        }
    }

    Ok(attrs)
}

// ---------------------------------------------------------------------------
// Group child enumeration — version 1 (symbol table)
// ---------------------------------------------------------------------------

/// List children of a v1 group via symbol table + B-tree v1 + local heap.
///
/// Returns `(name, object_header_address)` pairs for each child link.
///
/// ## Algorithm
///
/// 1. Locate the symbol table message (0x0011) in the object header.
/// 2. Parse the local heap to obtain the name string pool.
/// 3. Walk the B-tree v1, collecting leaf-node children (symbol table
///    nodes, "SNOD").
/// 4. Resolve each entry's name from the local heap.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] if the symbol table, local heap, or
///   B-tree structures are malformed.
#[cfg(feature = "alloc")]
pub fn list_group_v1<R: ReadAt>(
    source: &R,
    header: &ObjectHeader,
    ctx: &ParseContext,
) -> Result<Vec<(String, u64)>> {
    use crate::group::symbol_table::SymbolTableMessage;

    let st_msg =
        find_message(header, message_types::SYMBOL_TABLE).ok_or_else(|| Error::InvalidFormat {
            message: String::from("v1 group missing symbol table message"),
        })?;

    let sym_table = SymbolTableMessage::parse(&st_msg.data, ctx)?;

    // Parse local heap header and read data segment.
    let heap = crate::heap::local::LocalHeap::parse(source, sym_table.local_heap_address, ctx)?;

    let heap_data_size = heap.data_segment_size as usize;
    let mut heap_data = vec![0u8; heap_data_size];
    source.read_at(heap.data_address, &mut heap_data)?;

    // Walk B-tree v1 leaves collecting symbol table entries.
    let mut children = Vec::new();
    collect_btree_v1_leaves(
        source,
        sym_table.btree_address,
        ctx,
        &heap_data,
        &mut children,
    )?;

    Ok(children)
}

/// Recursively collect `(name, object_header_address)` from a B-tree v1.
///
/// Leaf nodes (level 0) point to symbol table nodes ("SNOD"); internal
/// nodes recurse into children.
#[cfg(feature = "alloc")]
fn collect_btree_v1_leaves<R: ReadAt>(
    source: &R,
    btree_address: u64,
    ctx: &ParseContext,
    heap_data: &[u8],
    children: &mut Vec<(String, u64)>,
) -> Result<()> {
    use crate::btree::v1::BTreeV1Header;
    use crate::group::symbol_table::SymbolTableNode;

    let header = BTreeV1Header::parse(source, btree_address, ctx)?;

    let s = ctx.offset_bytes();
    // B-tree v1 fixed header: signature(4) + type(1) + level(1) + entries(2) + left(S) + right(S)
    let header_size = 8 + 2 * s;

    // For group B-trees (type 0), each key is one length-size field.
    let key_size = ctx.length_bytes();
    // Each pair is: key + child pointer.
    let pair_size = key_size + s;
    // Total data: N pairs + 1 trailing key.
    let data_size = header.entries_used as usize * pair_size + key_size;

    let mut data = vec![0u8; data_size];
    source.read_at(btree_address + header_size as u64, &mut data)?;

    if header.level == 0 {
        // Leaf: children point to symbol table nodes.
        for i in 0..header.entries_used as usize {
            let child_off = key_size + i * pair_size;
            let child_addr = ctx.read_offset(&data[child_off..]);

            let snod = SymbolTableNode::parse(source, child_addr, ctx)?;

            for entry in &snod.entries {
                let name_off = entry.name_offset as usize;
                if name_off < heap_data.len() {
                    let name_end = heap_data[name_off..]
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(heap_data.len() - name_off);
                    if let Ok(name) =
                        core::str::from_utf8(&heap_data[name_off..name_off + name_end])
                    {
                        if !name.is_empty() {
                            children.push((String::from(name), entry.object_header_address));
                        }
                    }
                }
            }
        }
    } else {
        // Internal node: recurse into each child.
        for i in 0..header.entries_used as usize {
            let child_off = key_size + i * pair_size;
            let child_addr = ctx.read_offset(&data[child_off..]);
            collect_btree_v1_leaves(source, child_addr, ctx, heap_data, children)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Group child enumeration — version 2 (link messages / link info)
// ---------------------------------------------------------------------------

/// List children of a v2 group via link messages (0x0006) or link info.
///
/// Returns `(name, object_header_address, link_type)` triples.
///
/// v2 groups store links in one of two ways:
/// - **Compact**: individual link messages in the object header.
/// - **Dense**: link info message pointing to a fractal heap + B-tree v2
///   (not yet fully supported; returns only compact links).
#[cfg(feature = "alloc")]
pub fn list_group_v2<R: ReadAt>(
    source: &R,
    header: &ObjectHeader,
    ctx: &ParseContext,
) -> Result<Vec<(String, u64, consus_core::LinkType, Option<String>)>> {
    let mut children = Vec::new();

    // Compact storage: direct link messages (0x0006).
    let link_msgs = find_messages(header, message_types::LINK);
    if !link_msgs.is_empty() {
        for msg in link_msgs {
            let link = crate::link::Hdf5Link::parse(&msg.data, ctx)?;
            let addr = link.hard_link_address.unwrap_or(UNDEFINED_ADDRESS);
            children.push((link.name, addr, link.link_type, link.soft_link_target));
        }
        return Ok(children);
    }

    // Dense storage: Link Info message (0x0002) -> fractal heap + B-tree v2.
    if let Some(li_msg) = find_message(header, message_types::LINK_INFO) {
        if let Ok(link_info) = crate::link::LinkInfo::parse(&li_msg.data, ctx) {
            if link_info.fractal_heap_address != UNDEFINED_ADDRESS {
                let dense = collect_dense_links(source, &link_info, ctx)?;
                children.extend(dense);
            }
        }
    }

    Ok(children)
}

// ---------------------------------------------------------------------------
// Dense link enumeration (v2 groups with fractal heap + B-tree v2)
// ---------------------------------------------------------------------------

/// Enumerate links from dense storage: fractal heap + B-tree v2 (record type 5).
///
/// ### B-tree v2 type-5 record layout
///
/// | Offset | Size           | Field          |
/// |--------|----------------|----------------|
/// | 0      | 4              | Name hash      |
/// | 4      | heap_id_length | Heap ID        |
///
/// Each managed or tiny object is a raw link message payload.
/// Huge objects return [].
#[cfg(feature = "alloc")]
fn collect_dense_links<R: ReadAt>(
    source: &R,
    link_info: &crate::link::LinkInfo,
    ctx: &ParseContext,
) -> Result<Vec<(String, u64, consus_core::LinkType, Option<String>)>> {
    use crate::btree::v2::{BTreeV2Header, collect_all_records};
    use crate::heap::fractal::{
        FractalHeapHeader, FractalHeapId, decode_heap_id, read_huge_object, read_managed_object,
    };

    let heap_header = FractalHeapHeader::parse(source, link_info.fractal_heap_address, ctx)?;
    let btree_header = BTreeV2Header::parse(source, link_info.name_btree_address, ctx)?;
    let records = collect_all_records(source, &btree_header, ctx)?;

    let heap_id_len = heap_header.heap_id_length as usize;
    let mut links = Vec::with_capacity(records.len());

    for record in &records {
        // Type-5 record: hash(4) + heap_id(heap_id_len).
        if record.data.len() < 4 + heap_id_len {
            continue;
        }
        let heap_id_bytes = &record.data[4..4 + heap_id_len];
        let heap_id = decode_heap_id(heap_id_bytes, &heap_header)?;

        let raw_bytes = match heap_id {
            FractalHeapId::Managed { offset, length } => {
                read_managed_object(source, &heap_header, offset, length, ctx)?
            }
            FractalHeapId::Tiny { data } => data,
            FractalHeapId::Huge { btree_key } => {
                read_huge_object(source, &heap_header, btree_key, ctx)?
            }
        };

        let link = crate::link::Hdf5Link::parse(&raw_bytes, ctx)?;
        let addr = link.hard_link_address.unwrap_or(UNDEFINED_ADDRESS);
        links.push((link.name, addr, link.link_type, link.soft_link_target));
    }

    Ok(links)
}

// ---------------------------------------------------------------------------
// Dense attribute enumeration (objects with fractal heap + B-tree v2)
// ---------------------------------------------------------------------------

/// Enumerate attributes from dense storage: fractal heap + B-tree v2 (record type 8).
///
/// ### B-tree v2 type-8 record layout
///
/// | Offset | Size           | Field          |
/// |--------|----------------|----------------|
/// | 0      | 4              | Name hash      |
/// | 4      | heap_id_length | Heap ID        |
///
/// Each heap object is a raw attribute message payload.
#[cfg(feature = "alloc")]
fn collect_dense_attributes<R: ReadAt>(
    source: &R,
    attr_info: &crate::attribute::info::AttributeInfo,
    ctx: &ParseContext,
) -> Result<Vec<crate::attribute::Hdf5Attribute>> {
    use crate::btree::v2::{BTreeV2Header, collect_all_records};
    use crate::heap::fractal::{
        FractalHeapHeader, FractalHeapId, decode_heap_id, read_huge_object, read_managed_object,
    };

    // Heap ID within a type-8 B-tree record starts after the 4-byte name hash.
    const HEAP_ID_OFFSET: usize = 4;

    let heap_header = FractalHeapHeader::parse(source, attr_info.fractal_heap_address, ctx)?;
    let btree_header = BTreeV2Header::parse(source, attr_info.name_btree_address, ctx)?;
    let records = collect_all_records(source, &btree_header, ctx)?;

    let heap_id_len = heap_header.heap_id_length as usize;
    let mut attrs = Vec::with_capacity(records.len());

    for record in &records {
        // Type-8 record: hash(4) + heap_id(heap_id_len).
        if record.data.len() < HEAP_ID_OFFSET + heap_id_len {
            continue;
        }
        let heap_id_bytes = &record.data[HEAP_ID_OFFSET..HEAP_ID_OFFSET + heap_id_len];
        let heap_id = decode_heap_id(heap_id_bytes, &heap_header)?;

        let raw_bytes = match heap_id {
            FractalHeapId::Managed { offset, length } => {
                read_managed_object(source, &heap_header, offset, length, ctx)?
            }
            FractalHeapId::Tiny { data } => data,
            FractalHeapId::Huge { btree_key } => {
                read_huge_object(source, &heap_header, btree_key, ctx)?
            }
        };

        let attr = crate::attribute::Hdf5Attribute::parse(&raw_bytes, ctx)?;
        attrs.push(attr);
    }

    Ok(attrs)
}

// ---------------------------------------------------------------------------
// Contiguous data read
// ---------------------------------------------------------------------------

/// Read raw bytes from a contiguously-stored dataset.
///
/// Reads `buf.len()` bytes starting at `data_address + byte_offset`.
///
/// ## Errors
///
/// - I/O errors propagated from the source.
#[cfg(feature = "alloc")]
pub fn read_contiguous_raw<R: ReadAt>(
    source: &R,
    data_address: u64,
    byte_offset: u64,
    buf: &mut [u8],
) -> Result<()> {
    source.read_at(data_address + byte_offset, buf)
}

// ---------------------------------------------------------------------------
// Fill value extraction
// ---------------------------------------------------------------------------

/// Extract the fill value from an object header, if defined.
///
/// Parses the fill value message (0x0005). Returns `None` if no fill
/// value message is present or the fill value is marked as undefined.
///
/// ### Fill Value Message Layout (Version 2/3)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | Version (1, 2, or 3) |
/// | 1 | 1 | Space allocation time |
/// | 2 | 1 | Fill value write time |
/// | 3 | 1 | Fill value defined (0=undefined, 1=default, 2=user-defined) |
/// | 4 | 4 | Fill value size (only if defined == 2) |
/// | 8 | N | Fill value data |
#[cfg(feature = "alloc")]
pub fn read_fill_value(header: &ObjectHeader) -> Option<Vec<u8>> {
    use byteorder::{ByteOrder, LittleEndian};

    let msg = find_message(header, message_types::FILL_VALUE)?;
    let data = &msg.data;

    if data.is_empty() {
        return None;
    }

    let version = data[0];

    match version {
        1 | 2 => {
            // Versions 1 and 2: allocation time (1) + write time (1) +
            // defined flag (1) + optional size (4) + data.
            if data.len() < 4 {
                return None;
            }
            let defined = data[3];
            if defined != 2 || data.len() < 8 {
                return None;
            }
            let size = LittleEndian::read_u32(&data[4..8]) as usize;
            if size == 0 || data.len() < 8 + size {
                return None;
            }
            Some(Vec::from(&data[8..8 + size]))
        }
        3 => {
            // Version 3 encoding.
            if data.len() < 4 {
                return None;
            }
            let flags = data[1];
            let _fill_defined = flags & 0x20 != 0;
            if data.len() < 5 {
                return None;
            }
            let size = LittleEndian::read_u32(&data[4..8]) as usize;
            if size == 0 || data.len() < 8 + size {
                return None;
            }
            Some(Vec::from(&data[8..8 + size]))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use alloc::vec;

    /// Classify an object header with dataspace + datatype + layout → Dataset.
    #[cfg(feature = "alloc")]
    #[test]
    fn classify_dataset() {
        let header = ObjectHeader {
            version: 2,
            messages: vec![
                HeaderMessage {
                    message_type: message_types::DATASPACE,
                    data_size: 0,
                    flags: 0,
                    data: vec![],
                },
                HeaderMessage {
                    message_type: message_types::DATATYPE,
                    data_size: 0,
                    flags: 0,
                    data: vec![],
                },
                HeaderMessage {
                    message_type: message_types::DATA_LAYOUT,
                    data_size: 0,
                    flags: 0,
                    data: vec![],
                },
            ],
        };
        assert_eq!(classify_object(&header), NodeType::Dataset);
    }

    /// Classify an object header with only datatype → NamedDatatype.
    #[cfg(feature = "alloc")]
    #[test]
    fn classify_named_datatype() {
        let header = ObjectHeader {
            version: 2,
            messages: vec![HeaderMessage {
                message_type: message_types::DATATYPE,
                data_size: 0,
                flags: 0,
                data: vec![],
            }],
        };
        assert_eq!(classify_object(&header), NodeType::NamedDatatype);
    }

    /// Classify an empty object header → Group.
    #[cfg(feature = "alloc")]
    #[test]
    fn classify_group() {
        let header = ObjectHeader {
            version: 2,
            messages: vec![],
        };
        assert_eq!(classify_object(&header), NodeType::Group);
    }

    /// Classify a v1 group with a symbol table message → Group.
    #[cfg(feature = "alloc")]
    #[test]
    fn classify_v1_group_with_symbol_table() {
        let header = ObjectHeader {
            version: 1,
            messages: vec![HeaderMessage {
                message_type: message_types::SYMBOL_TABLE,
                data_size: 0,
                flags: 0,
                data: vec![],
            }],
        };
        assert_eq!(classify_object(&header), NodeType::Group);
    }

    /// `find_message` returns the first matching message.
    #[cfg(feature = "alloc")]
    #[test]
    fn find_message_returns_first_match() {
        let header = ObjectHeader {
            version: 2,
            messages: vec![
                HeaderMessage {
                    message_type: 0x0001,
                    data_size: 10,
                    flags: 0,
                    data: vec![1],
                },
                HeaderMessage {
                    message_type: 0x0001,
                    data_size: 20,
                    flags: 0,
                    data: vec![2],
                },
            ],
        };
        let msg = find_message(&header, 0x0001).unwrap();
        assert_eq!(msg.data_size, 10);
        assert_eq!(msg.data, vec![1]);
    }

    /// `find_messages` returns all matching messages.
    #[cfg(feature = "alloc")]
    #[test]
    fn find_messages_returns_all() {
        let header = ObjectHeader {
            version: 2,
            messages: vec![
                HeaderMessage {
                    message_type: 0x000C,
                    data_size: 10,
                    flags: 0,
                    data: vec![1],
                },
                HeaderMessage {
                    message_type: 0x0003,
                    data_size: 5,
                    flags: 0,
                    data: vec![],
                },
                HeaderMessage {
                    message_type: 0x000C,
                    data_size: 20,
                    flags: 0,
                    data: vec![2],
                },
            ],
        };
        let msgs = find_messages(&header, 0x000C);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].data_size, 10);
        assert_eq!(msgs[1].data_size, 20);
    }

    /// `find_message` returns `None` when no message matches.
    #[cfg(feature = "alloc")]
    #[test]
    fn find_message_none_when_absent() {
        let header = ObjectHeader {
            version: 1,
            messages: vec![],
        };
        assert!(find_message(&header, 0x0001).is_none());
    }

    /// `read_fill_value` returns `None` for empty messages.
    #[cfg(feature = "alloc")]
    #[test]
    fn fill_value_none_for_empty() {
        let header = ObjectHeader {
            version: 2,
            messages: vec![],
        };
        assert!(read_fill_value(&header).is_none());
    }

    /// `read_fill_value` extracts a v2 user-defined fill value.
    #[cfg(feature = "alloc")]
    #[test]
    fn fill_value_v2_user_defined() {
        use byteorder::{ByteOrder, LittleEndian};

        // Build a version 2 fill value message with 4-byte fill = [0xDE, 0xAD, 0xBE, 0xEF].
        let mut fv_data = vec![0u8; 12];
        fv_data[0] = 2; // version
        fv_data[1] = 1; // alloc time: late
        fv_data[2] = 0; // fill write time: on allocation
        fv_data[3] = 2; // defined: user-defined
        LittleEndian::write_u32(&mut fv_data[4..8], 4); // size = 4
        fv_data[8] = 0xDE;
        fv_data[9] = 0xAD;
        fv_data[10] = 0xBE;
        fv_data[11] = 0xEF;

        let header = ObjectHeader {
            version: 2,
            messages: vec![HeaderMessage {
                message_type: message_types::FILL_VALUE,
                data_size: fv_data.len() as u16,
                flags: 0,
                data: fv_data,
            }],
        };

        let fv = read_fill_value(&header).expect("fill value must be present");
        assert_eq!(fv, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }
    /// Build a minimal valid FractalHeapHeader at offset 0 in .
    ///
    /// Sets signature, version, and heap_id_length. All other fields
    /// default to zero, which causes collect_all_records to return
    /// immediately (total_records == 0 after B-tree header is built with
    /// build_bthd_empty_in_buf). The buffer is pre-allocated by the caller.
    #[cfg(feature = "alloc")]
    fn build_frhp_in_buf(buf: &mut [u8], heap_id_len: u16, _base: usize) {
        use byteorder::{ByteOrder, LittleEndian};
        // Signature: "FRHP"
        buf[0..4].copy_from_slice(b"FRHP");
        // Version 0
        buf[4] = 0;
        // heap_id_length
        LittleEndian::write_u16(&mut buf[5..7], heap_id_len);
        // io_filter_size = 0, flags = 0, max_managed_object_size = 0
        // Variable-width fields (8-byte lengths, 8-byte offsets):
        //   next_huge_id (14..22): 0
        //   huge_object_btree_address (22..30): UNDEFINED
        for b in &mut buf[22..30] {
            *b = 0xFF;
        }
        //   free_space_manager_address (30..38): UNDEFINED
        for b in &mut buf[30..38] {
            *b = 0xFF;
        }
        //   managed_space, allocated_managed_space, iter_offset,
        //   managed_object_count, huge/tiny sizes and counts: all 0
        // table_width (102..104): 4 (non-zero to avoid division issues)
        LittleEndian::write_u16(&mut buf[102..104], 4);
        // starting_block_size (104..112): 512
        LittleEndian::write_u64(&mut buf[104..112], 512);
        // max_direct_block_size (112..120): 65536
        LittleEndian::write_u64(&mut buf[112..120], 65536);
        // max_heap_size_bits (120..122): 32
        LittleEndian::write_u16(&mut buf[120..122], 32);
        // starting_rows (122..124): 0
        // root_block_address (124..132): UNDEFINED
        for b in &mut buf[124..132] {
            *b = 0xFF;
        }
        // root_indirect_rows (132..134): 0
    }

    /// Build a minimal empty BTreeV2Header at  in .
    ///
    /// Sets signature, version, record_type, and leaves total_records == 0
    /// and root_address == UNDEFINED_ADDRESS so that collect_all_records
    /// returns immediately with an empty Vec.
    #[cfg(feature = "alloc")]
    fn build_bthd_empty_in_buf(buf: &mut [u8], offset: usize, rt: u8) {
        use byteorder::{ByteOrder, LittleEndian};
        // Signature: "BTHD"
        buf[offset..offset + 4].copy_from_slice(b"BTHD");
        // Version 0
        buf[offset + 4] = 0;
        // Record type
        buf[offset + 5] = rt;
        // node_size = 512
        LittleEndian::write_u32(&mut buf[offset + 6..offset + 10], 512);
        // record_size = 11 (typical for type-5 with heap_id_len=7: 4+7)
        LittleEndian::write_u16(&mut buf[offset + 10..offset + 12], 11);
        // depth = 0, split_percent = 0, merge_percent = 0
        // root_address = UNDEFINED_ADDRESS (causes early return)
        for b in &mut buf[offset + 16..offset + 24] {
            *b = 0xFF;
        }
        // root_num_records = 0, total_records = 0, checksum = 0
    }

    /// collect_dense_links returns empty Vec for an empty B-tree.
    ///
    /// Synthetic source layout (8-byte offsets/lengths):
    /// - offset 0:   FractalHeapHeader (256-byte read window, heap_id_len=7)
    /// - offset 256: BTreeV2Header with total_records=0 (UNDEFINED root)
    #[cfg(feature = "alloc")]
    #[test]
    fn collect_dense_links_empty_btree() {
        use consus_io::MemCursor;

        let mut buf = vec![0u8; 512];
        build_frhp_in_buf(&mut buf, 7, 0); // heap at 0, root_block irrelevant (no records)
        build_bthd_empty_in_buf(&mut buf, 256, 5); // btree (type 5 = LINK_NAME) at 256

        let ctx = ParseContext::new(8, 8);
        let link_info = crate::link::LinkInfo {
            flags: 0,
            max_creation_index: None,
            fractal_heap_address: 0,
            name_btree_address: 256,
            creation_order_btree_address: None,
        };

        let cursor = MemCursor::from_bytes(buf);
        let result = collect_dense_links(&cursor, &link_info, &ctx)
            .expect("collect_dense_links must succeed");
        assert!(result.is_empty(), "empty btree must yield no links");
    }

    /// collect_dense_attributes with an empty B-tree returns an empty Vec.
    ///
    /// Same layout as collect_dense_links_empty_btree but uses record type 8.
    #[cfg(feature = "alloc")]
    #[test]
    fn collect_dense_attributes_empty_btree() {
        use crate::attribute::info::AttributeInfo;
        use consus_io::MemCursor;

        let mut buf = vec![0u8; 512];
        build_frhp_in_buf(&mut buf, 7, 0);
        build_bthd_empty_in_buf(&mut buf, 256, 8); // type 8 = ATTRIBUTE_NAME

        let ctx = ParseContext::new(8, 8);
        let attr_info = AttributeInfo {
            flags: 0,
            max_creation_order: None,
            fractal_heap_address: 0,
            name_btree_address: 256,
            creation_order_btree_address: None,
        };

        let cursor = MemCursor::from_bytes(buf);
        let result = collect_dense_attributes(&cursor, &attr_info, &ctx)
            .expect("collect_dense_attributes must succeed");
        assert!(result.is_empty(), "empty btree must yield no attributes");
    }
}
