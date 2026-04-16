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
/// Returns attributes in the order they appear in the header.
#[cfg(feature = "alloc")]
pub fn read_attributes(
    header: &ObjectHeader,
    ctx: &ParseContext,
) -> Result<Vec<crate::attribute::Hdf5Attribute>> {
    let attr_msgs = find_messages(header, message_types::ATTRIBUTE);
    let mut attrs = Vec::with_capacity(attr_msgs.len());
    for msg in attr_msgs {
        let attr = crate::attribute::Hdf5Attribute::parse(&msg.data, ctx)?;
        attrs.push(attr);
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
) -> Result<Vec<(String, u64, consus_core::LinkType)>> {
    let _ = source; // used in future dense-storage path

    let mut children = Vec::new();

    // Try compact storage first: direct link messages.
    let link_msgs = find_messages(header, message_types::LINK);
    if !link_msgs.is_empty() {
        for msg in link_msgs {
            let link = crate::link::Hdf5Link::parse(&msg.data, ctx)?;
            let addr = link
                .hard_link_address
                .unwrap_or(crate::constants::UNDEFINED_ADDRESS);
            children.push((link.name, addr, link.link_type));
        }
        return Ok(children);
    }

    // Dense storage: link info message → fractal heap + B-tree v2.
    // Parse the link info to obtain addresses for future traversal.
    if let Some(li_msg) = find_message(header, message_types::LINK_INFO) {
        let _link_info = crate::attribute::info::AttributeInfo::parse(&li_msg.data, ctx);
        // Full dense-link enumeration requires fractal heap traversal
        // which is tracked as a follow-up work item.
    }

    Ok(children)
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
}
