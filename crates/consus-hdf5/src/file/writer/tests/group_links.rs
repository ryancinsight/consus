//! Group object-header, local heap, v1 index, link, and attribute message tests.

use super::super::*;
use crate::address::ParseContext;
use crate::file::Hdf5File;
use crate::object_header::message_types;
use crate::property_list::{FileCreationProps, GroupCreationProps};
use byteorder::{ByteOrder as _, LittleEndian};
use consus_core::Error;
use core::num::NonZeroUsize;

#[cfg(feature = "alloc")]
#[test]
fn write_group_header_has_ohdr_signature() {
    let mut cursor = consus_io::MemCursor::new();
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    state.eof = 48; // after superblock
    let gcpl = GroupCreationProps::default();
    let addr = write_group_header(&mut cursor, &mut state, &gcpl).unwrap();
    let data = cursor.as_bytes();
    assert_eq!(&data[addr as usize..addr as usize + 4], b"OHDR");
    assert_eq!(data[addr as usize + 4], 2); // version 2
}

#[cfg(feature = "alloc")]
#[test]
fn encode_hard_link_roundtrip() {
    let ctx = ParseContext::new(8, 8);
    let bytes = encode_hard_link("temperature", 0x200, &ctx).unwrap();
    assert_eq!(bytes[0], 1); // version
    // Name length = 11 (fits in 1 byte)
    assert_eq!(bytes[2], 11);
    // Name bytes
    assert_eq!(&bytes[3..14], b"temperature");
    // Address at offset 14
    assert_eq!(LittleEndian::read_u64(&bytes[14..22]), 0x200);
}

#[cfg(feature = "alloc")]
#[test]
fn write_local_heap_roundtrip() {
    let mut cursor = consus_io::MemCursor::new();
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    let names_vec = vec!["alpha", "beta"];
    let (heap_addr, offsets) = write_local_heap(&mut cursor, &mut state, &names_vec)
        .expect("write_local_heap must succeed");
    assert_eq!(offsets, vec![0, 6]);

    let heap = crate::heap::local::LocalHeap::parse(&cursor, heap_addr, &state.ctx).unwrap();
    assert_eq!(heap.data_segment_size, 11);
    assert_eq!(heap.free_list_offset, u64::MAX);
    assert_eq!(heap.data_address, heap_addr + 32);
    assert_eq!(heap.read_name(&cursor, offsets[0]).unwrap(), "alpha");
    assert_eq!(heap.read_name(&cursor, offsets[1]).unwrap(), "beta");
}

#[cfg(feature = "alloc")]
#[test]
fn write_v1_group_roundtrip_via_reader() {
    let mut cursor = consus_io::MemCursor::new();
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    // Reserve superblock space (48 bytes for 8-byte offsets) so that
    // write_superblock (which always writes at offset 0) does not
    // overwrite child headers allocated before it is called.
    // Formula mirrors Hdf5FileBuilder::new: 12 + 4*offset_bytes + 4.
    let sb_size = 12u64 + 4 * state.ctx.offset_bytes() as u64 + 4;
    state.eof = sb_size;
    let empty_msgs: Vec<(u16, &[u8])> = Vec::new();
    let child_a = write_object_header_v2(&mut cursor, &mut state, &empty_msgs).unwrap();
    let child_b = write_object_header_v2(&mut cursor, &mut state, &empty_msgs).unwrap();
    let names_vec = vec!["alpha", "beta"];
    let addresses_vec = vec![child_a, child_b];
    let group_addr =
        write_v1_group_header(&mut cursor, &mut state, &names_vec, &addresses_vec).unwrap();
    let root_link = encode_hard_link("group", group_addr, &state.ctx).unwrap();
    let root_links: Vec<(u16, &[u8])> = vec![(message_types::LINK, root_link.as_slice())];
    let root_group = write_object_header_v2(&mut cursor, &mut state, &root_links).unwrap();
    write_superblock(&mut cursor, &mut state, root_group).unwrap();
    update_superblock_eof(&mut cursor, &state).unwrap();

    let file = Hdf5File::open(cursor).unwrap();
    let children = file.list_group_at(group_addr).unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(children[0].0, "alpha");
    assert_eq!(children[0].1, child_a);
    assert_eq!(children[1].0, "beta");
    assert_eq!(children[1].1, child_b);
}

#[cfg(feature = "alloc")]
#[test]
fn write_local_heap_rejects_missing_null_terminator_on_parse() {
    let mut cursor = consus_io::MemCursor::new();
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    let names_vec = vec!["alpha"];
    let (heap_addr, _) = write_local_heap(&mut cursor, &mut state, &names_vec)
        .expect("write_local_heap must succeed");
    let mut bytes = cursor.as_bytes().to_vec();
    // Local heap header size = 4 (sig) + 1 (ver) + 3 (pad) + l (seg_size)
    // + l (free_list) + s (data_addr) = 4+1+3+8+8+8 = 32 bytes.
    // The data segment immediately follows the header.
    let data_addr = heap_addr + 32;
    bytes[data_addr as usize + 5] = b'X';
    let cursor = consus_io::MemCursor::from_bytes(bytes);
    let heap = crate::heap::local::LocalHeap::parse(&cursor, heap_addr, &state.ctx).unwrap();
    let err = heap.read_name(&cursor, 0).unwrap_err();
    assert!(matches!(err, Error::InvalidFormat { .. }));
}

#[cfg(feature = "alloc")]
#[test]
fn encode_attribute_roundtrip() {
    use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: CoreByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::scalar();
    let raw = 99u32.to_le_bytes();
    let bytes = encode_attribute("count", &dt, &shape, &raw).unwrap();
    assert_eq!(bytes[0], 3); // version 3
    // name_size includes null terminator per HDF5 spec §IV.A.2.m v3.
    assert_eq!(LittleEndian::read_u16(&bytes[2..4]), 6); // "count".len() + 1 (null)
    assert_eq!(&bytes[9..14], b"count");
    assert_eq!(bytes[14], 0); // null terminator
    assert_eq!(LittleEndian::read_u32(&bytes[bytes.len() - 4..]), 99);
}
