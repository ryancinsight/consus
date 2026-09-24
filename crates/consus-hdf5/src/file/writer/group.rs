//! Local heaps, v1 symbol-table/index structures, and group object header emission.

#[cfg(feature = "alloc")]
use super::{
    WriteState, chunk_size_encoding, write_chunk_size, write_object_header_v2, write_offset,
    write_v2_message,
};
#[cfg(feature = "alloc")]
use crate::constants::UNDEFINED_ADDRESS;
#[cfg(feature = "alloc")]
use crate::object_header::message_types;
#[cfg(feature = "alloc")]
use crate::property_list::GroupCreationProps;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_compression::Checksum;
#[cfg(feature = "alloc")]
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::WriteAt;

#[cfg(feature = "alloc")]
fn encode_local_heap_data(names: &[&str]) -> Result<(Vec<u8>, Vec<u64>)> {
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(names.len());

    for name in names {
        let offset = data.len() as u64;
        offsets.push(offset);
        data.extend_from_slice(name.as_bytes());
        data.push(0);
    }

    Ok((data, offsets))
}

/// Write a valid HDF5 local heap containing null-terminated link names.
///
/// Returns the heap header address and the byte offsets of each name within
/// the emitted data segment. The data segment is written immediately after the
/// header and its recorded size matches the serialized name pool exactly.
#[cfg(feature = "alloc")]
pub fn write_local_heap<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    names: &[&str],
) -> Result<(u64, Vec<u64>)> {
    let (data, offsets) = encode_local_heap_data(names)?;
    let s = state.ctx.offset_bytes();
    let l = state.ctx.length_bytes();
    let header_size = 4 + 1 + 3 + l + l + s;
    let heap_addr = state.allocate_aligned(header_size as u64);
    let data_addr = heap_addr + header_size as u64;
    state.eof = data_addr + data.len() as u64;

    let mut header = vec![0u8; header_size];
    header[0..4].copy_from_slice(b"HEAP");
    header[4] = 0;
    match l {
        2 => LittleEndian::write_u16(&mut header[8..10], data.len() as u16),
        4 => LittleEndian::write_u32(&mut header[8..12], data.len() as u32),
        8 => LittleEndian::write_u64(&mut header[8..16], data.len() as u64),
        _ => {}
    }
    match l {
        2 => LittleEndian::write_u16(&mut header[8 + l..8 + 2 * l], u16::MAX),
        4 => LittleEndian::write_u32(&mut header[8 + l..8 + 2 * l], u32::MAX),
        8 => LittleEndian::write_u64(&mut header[8 + l..8 + 2 * l], u64::MAX),
        _ => {}
    }
    write_offset(&mut header[8 + 2 * l..], s, data_addr);

    sink.write_at(heap_addr, &header)?;
    if !data.is_empty() {
        sink.write_at(data_addr, &data)?;
    }

    Ok((heap_addr, offsets))
}

#[cfg(feature = "alloc")]
fn write_v1_symbol_table_node<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    name_offsets: &[u64],
    object_addresses: &[u64],
) -> Result<u64> {
    let s = state.ctx.offset_bytes();
    let entry_size = 2 * s + 24;
    let total = 8 + name_offsets.len() * entry_size;
    let addr = state.allocate_aligned(total as u64);
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(b"SNOD");
    buf[4] = 1;
    LittleEndian::write_u16(&mut buf[6..8], name_offsets.len() as u16);

    for (idx, (&name_offset, &object_address)) in
        name_offsets.iter().zip(object_addresses).enumerate()
    {
        let base = 8 + idx * entry_size;
        write_offset(&mut buf[base..base + s], s, name_offset);
        write_offset(&mut buf[base + s..base + 2 * s], s, object_address);
        LittleEndian::write_u32(&mut buf[base + 2 * s..base + 2 * s + 4], 1);
        LittleEndian::write_u32(&mut buf[base + 2 * s + 4..base + 2 * s + 8], 0);
        for byte in &mut buf[base + 2 * s + 8..base + 2 * s + 24] {
            *byte = 0;
        }
    }

    sink.write_at(addr, &buf)?;
    Ok(addr)
}

#[cfg(feature = "alloc")]
fn write_v1_group_index<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    snod_address: u64,
) -> Result<u64> {
    let s = state.ctx.offset_bytes();
    let key_size = state.ctx.length_bytes();
    // B-tree v1 leaf node layout (HDF5 spec §III.B.1, group type 0):
    //   header:  signature(4) + type(1) + level(1) + entries_used(2) + left(S) + right(S)
    //            = 8 + 2*S bytes
    //   data:    key[0](key_size) + child[0](S) + key[1](key_size)
    //            = 2*key_size + S bytes   (N=1 entry → N+1 keys, N child pointers)
    // Reader reads data starting at btree_address + header_size; child_off = key_size.
    let header_size = 8 + 2 * s;
    let data_size = 2 * key_size + s; // key[0] + child[0] + trailing key[1]
    let total = header_size + data_size;
    let addr = state.allocate_aligned(total as u64);
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(b"TREE");
    buf[4] = 0; // node_type = 0 (Group)
    buf[5] = 0; // node_level = 0 (leaf)
    LittleEndian::write_u16(&mut buf[6..8], 1); // entries_used = 1
    write_offset(&mut buf[8..], s, UNDEFINED_ADDRESS); // left sibling (no left)
    write_offset(&mut buf[8 + s..], s, UNDEFINED_ADDRESS); // right sibling (no right)
    // Data: key[0] (zeros) + child[0] = snod_address + key[1] (zeros)
    // child[0] is at data[key_size], i.e. buf[header_size + key_size].
    write_offset(&mut buf[header_size + key_size..], s, snod_address);

    sink.write_at(addr, &buf)?;
    Ok(addr)
}

/// Write a v1 group object header with a local heap, symbol table node,
/// and B-tree v1 root so existing v1 readers can enumerate its names.
#[cfg(feature = "alloc")]
pub fn write_v1_group_header<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    names: &[&str],
    object_addresses: &[u64],
) -> Result<u64> {
    if names.len() != object_addresses.len() {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("v1 group writer requires name/object address count match"),
        });
    }

    let (heap_addr, offsets) = write_local_heap(sink, state, names)?;
    let snod_addr = write_v1_symbol_table_node(sink, state, &offsets, object_addresses)?;
    let btree_addr = write_v1_group_index(sink, state, snod_addr)?;

    let s = state.ctx.offset_bytes();
    let mut st_bytes = vec![0u8; 2 * s];
    write_offset(&mut st_bytes[0..], s, btree_addr);
    write_offset(&mut st_bytes[s..], s, heap_addr);

    let st_msgs: Vec<(u16, &[u8])> = vec![(message_types::SYMBOL_TABLE, st_bytes.as_slice())];
    write_object_header_v2(sink, state, &st_msgs)
}

// ---------------------------------------------------------------------------
/// Encode a Link Info message (type 0x0002) for compact link storage.
///
/// Layout (no creation-order tracking, both addresses UNDEF):
/// | version(1) | flags(1) | heap_addr(s) | btree_addr(s) |  total = 2 + 2*s bytes
///
/// Both addresses are set to 0xFF…FF (UNDEF) indicating compact storage
/// (links are stored directly as LINK messages in the object header).
#[cfg(feature = "alloc")]
pub(crate) fn encode_link_info(offset_bytes: usize) -> Vec<u8> {
    let mut buf = vec![0xFFu8; 2 + offset_bytes * 2];
    buf[0] = 0; // version = 0
    buf[1] = 0; // flags = 0 (no creation order tracking)
    buf
}

/// Encode a Group Info message (type 0x000A) with default thresholds.
///
/// Layout: | version(1) | flags(1) |  total = 2 bytes
#[cfg(feature = "alloc")]
pub(crate) fn encode_group_info() -> Vec<u8> {
    vec![0u8; 2] // version=0, flags=0 (default compact/dense thresholds)
}

///
/// The header is allocated with padding space (NIL messages) to allow
/// future in-place addition of link messages.
///
/// Returns the file address of the written object header.
#[cfg(feature = "alloc")]
pub fn write_group_header<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    _props: &GroupCreationProps,
) -> Result<u64> {
    // Reserve 256 bytes for the group header (generous for future links)
    let reserved = 256usize;

    // Compute chunk data size: one NIL message fills the space.
    // OHDR signature(4) + version(1) + flags(1) + chunk_size_field(?) + data + checksum(4)
    // We solve for data_size such that total = reserved.
    let (csf_width, csf_flags) = chunk_size_encoding(reserved);
    let overhead = 4 + 1 + 1 + csf_width + 4; // signature + version + flags + csf + checksum
    let chunk_data_size = reserved - overhead;

    let addr = state.allocate_aligned(reserved as u64);
    let mut buf = vec![0u8; reserved];

    // OHDR signature
    buf[0..4].copy_from_slice(b"OHDR");
    buf[4] = 2; // version 2
    buf[5] = csf_flags; // flags: chunk size encoding only
    let mut pos = 6;
    write_chunk_size(&mut buf[pos..], csf_width, chunk_data_size);
    pos += csf_width;

    // Fill chunk with a single NIL message spanning all available space.
    let nil_data_len = chunk_data_size.saturating_sub(4); // 4-byte message header
    let written = write_v2_message(&mut buf, pos, 0x0000, 0, &vec![0u8; nil_data_len]);
    pos += written;

    // Jenkins lookup3 checksum per HDF5 spec §IV.A.1.2
    let checksum = consus_compression::Lookup3::compute(&buf[..pos]);
    buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

    sink.write_at(addr, &buf)?;
    Ok(addr)
}
