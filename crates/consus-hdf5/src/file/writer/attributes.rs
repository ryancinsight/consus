//! v2 object-header assembly, attribute message encoding, and v1 group node emission.

#[cfg(feature = "alloc")]
use super::{
    ChildDatasetSpec, ChildGroupSpec, WriteState, chunk_size_encoding, encode_dataspace,
    encode_datatype, encode_group_info, encode_hard_link, encode_layout, encode_link_info,
    write_chunk_size, write_contiguous_data, write_v2_message,
};
#[cfg(feature = "alloc")]
use crate::object_header::message_types;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_compression::Checksum;
#[cfg(feature = "alloc")]
use consus_core::{Datatype, Result, Shape};
#[cfg(feature = "alloc")]
use consus_io::WriteAt;

// ---------------------------------------------------------------------------
// Generalised V2 object header writer
// ---------------------------------------------------------------------------

/// Write a v2 object header containing the given messages.
///
/// Each entry in `messages` is `(message_type, data)`. The header is
/// allocated at an 8-byte-aligned address and appended at the current EOF.
///
/// Returns the file address of the written header.
///
/// ## Buffer sizing
///
/// Uses at least 4 bytes for chunk data to accommodate a minimal NIL
/// message when `messages` is empty, preventing a CRC-field buffer overflow.
#[cfg(feature = "alloc")]
pub fn write_object_header_v2<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    messages: &[(u16, &[u8])],
) -> Result<u64> {
    let chunk_data_size: usize = messages.iter().map(|(_, d)| 4 + d.len()).sum();
    // Minimum 4 bytes: a NIL message header (type:1 + size:2 + flags:1).
    let effective_size = chunk_data_size.max(4);
    let (csf_width, csf_flags) = chunk_size_encoding(effective_size);
    let total = 4 + 1 + 1 + csf_width + effective_size + 4;

    let addr = state.allocate_aligned(total as u64);
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(b"OHDR");
    buf[4] = 2; // version
    buf[5] = csf_flags;
    let mut pos = 6;
    write_chunk_size(&mut buf[pos..], csf_width, effective_size);
    pos += csf_width;

    if messages.is_empty() {
        // Write one NIL message (type=0, size=0, flags=0) to fill space.
        pos += write_v2_message(&mut buf, pos, 0x0000, 0, &[]);
    } else {
        for (msg_type, data) in messages {
            pos += write_v2_message(&mut buf, pos, *msg_type, 0, data);
        }
    }

    // Jenkins lookup3 checksum per HDF5 spec §IV.A.1.2 (Object Header v2)
    let checksum = consus_compression::Lookup3::compute(&buf[..pos]);
    buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

    sink.write_at(addr, &buf)?;
    Ok(addr)
}

// ---------------------------------------------------------------------------
// Attribute message encoding
// ---------------------------------------------------------------------------

/// Encode an attribute as a version-3 attribute message (type 0x000C).
///
/// ### Version 3 Layout
///
/// Per HDF5 spec §IV.A.2.m: the Name Size field includes the null terminator.
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Flags (0) |
/// | 2 | 2 | Name size (byte count INCLUDING null terminator) |
/// | 4 | 2 | Datatype size |
/// | 6 | 2 | Dataspace size |
/// | 8 | 1 | Encoding (0=ASCII, 1=UTF-8) |
/// | 9 | N+1 | Name bytes (null-terminated) |
/// | var | var | Datatype |
/// | var | var | Dataspace |
/// | var | var | Raw data |
#[cfg(feature = "alloc")]
pub fn encode_attribute(
    name: &str,
    dt: &Datatype,
    shape: &Shape,
    raw_data: &[u8],
) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let dt_bytes = encode_datatype(dt)?;
    let ds_bytes = encode_dataspace(shape)?;

    // Detect UTF-8 name (non-ASCII bytes present).
    let encoding: u8 = if name_bytes.iter().any(|&b| b > 0x7F) {
        1
    } else {
        0
    };

    // Per HDF5 spec §IV.A.2.m v3: Name Size includes the null terminator.
    let name_size = name_bytes.len() + 1;
    let total = 9 + name_size + dt_bytes.len() + ds_bytes.len() + raw_data.len();

    let mut buf = vec![0u8; total];
    buf[0] = 3; // version
    buf[1] = 0; // flags
    LittleEndian::write_u16(&mut buf[2..4], name_size as u16);
    LittleEndian::write_u16(&mut buf[4..6], dt_bytes.len() as u16);
    LittleEndian::write_u16(&mut buf[6..8], ds_bytes.len() as u16);
    buf[8] = encoding;

    let mut pos = 9;
    buf[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
    // buf[pos + name_bytes.len()] = 0; // null terminator (already 0 from vec![0u8; total])
    pos += name_size; // advance past name + null
    buf[pos..pos + dt_bytes.len()].copy_from_slice(&dt_bytes);
    pos += dt_bytes.len();
    buf[pos..pos + ds_bytes.len()].copy_from_slice(&ds_bytes);
    pos += ds_bytes.len();
    buf[pos..pos + raw_data.len()].copy_from_slice(raw_data);

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Recursive group node writer
// ---------------------------------------------------------------------------

/// Write a group node recursively: datasets first, then sub-groups, then the
/// group object header.
///
/// Returns the byte offset of the written group object header.
/// The caller is responsible for recording the returned address in its own
/// link table (root group or parent group).
///
/// ### Write order (depth-first, leaf-first)
///
/// 1. Each dataset in `datasets`: contiguous data block → object header.
/// 2. Each sub-group in `sub_groups`: recurse into `write_group_node`.
/// 3. Group object header containing LINK messages for all children and
///    ATTRIBUTE messages for `group_attributes`.
#[cfg(feature = "alloc")]
pub(crate) fn write_group_node(
    sink: &mut consus_io::MemCursor,
    state: &mut WriteState,
    group_attributes: &[(&str, &Datatype, &Shape, &[u8])],
    datasets: &[ChildDatasetSpec<'_>],
    sub_groups: &[ChildGroupSpec<'_>],
) -> Result<u64> {
    let ctx = state.ctx;
    let mut child_links: Vec<(String, u64)> = Vec::with_capacity(datasets.len() + sub_groups.len());

    // Step 1: write each child dataset (data block + object header).
    for child in datasets {
        let data_addr = write_contiguous_data(sink, state, child.raw_data)?;

        let dt_bytes = encode_datatype(child.datatype)?;
        let ds_bytes = encode_dataspace(child.shape)?;
        let layout_bytes = encode_layout(data_addr, &child.dcpl, &ctx)?;

        let mut child_msgs: Vec<(u16, Vec<u8>)> = vec![
            (message_types::DATATYPE, dt_bytes),
            (message_types::DATASPACE, ds_bytes),
            (message_types::DATA_LAYOUT, layout_bytes),
        ];

        for (attr_name, attr_dt, attr_shape, attr_data) in child.attributes {
            child_msgs.push((
                message_types::ATTRIBUTE,
                encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?,
            ));
        }

        let msg_refs: Vec<(u16, &[u8])> =
            child_msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let child_addr = write_object_header_v2(sink, state, &msg_refs)?;
        child_links.push((String::from(child.name), child_addr));
    }

    // Step 2: recursively write each sub-group and record its address.
    for sub in sub_groups {
        let addr = write_group_node(sink, state, sub.attributes, sub.datasets, sub.sub_groups)?;
        child_links.push((String::from(sub.name), addr));
    }

    // Step 3: build group object header messages (LINK_INFO + GROUP_INFO + links + group attributes).
    let mut group_msgs: Vec<(u16, Vec<u8>)> = vec![
        (
            message_types::LINK_INFO,
            encode_link_info(ctx.offset_bytes()),
        ),
        (message_types::GROUP_INFO, encode_group_info()),
    ];

    for (child_name, child_addr) in &child_links {
        group_msgs.push((
            message_types::LINK,
            encode_hard_link(child_name, *child_addr, &ctx)?,
        ));
    }

    for (attr_name, attr_dt, attr_shape, attr_data) in group_attributes {
        group_msgs.push((
            message_types::ATTRIBUTE,
            encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?,
        ));
    }

    // Step 4: write the group object header and return its address.
    let msg_refs: Vec<(u16, &[u8])> = group_msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
    write_object_header_v2(sink, state, &msg_refs)
}
