//! Write-state and superblock emission tests.

use super::super::*;
use crate::constants::HDF5_MAGIC;
use crate::property_list::FileCreationProps;
use byteorder::{ByteOrder as _, LittleEndian};

#[test]
fn write_state_allocate() {
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    assert_eq!(state.eof, 0);
    let a1 = state.allocate(16);
    assert_eq!(a1, 0);
    assert_eq!(state.eof, 16);
    let a2 = state.allocate(32);
    assert_eq!(a2, 16);
    assert_eq!(state.eof, 48);
}

#[test]
fn write_state_allocate_aligned() {
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    state.eof = 5;
    let a = state.allocate_aligned(10);
    assert_eq!(a, 8); // 5 aligned up to 8
    assert_eq!(state.eof, 18);
}

#[test]
fn chunk_size_encoding_boundaries() {
    assert_eq!(chunk_size_encoding(0), (1, 0x00));
    assert_eq!(chunk_size_encoding(255), (1, 0x00));
    assert_eq!(chunk_size_encoding(256), (2, 0x01));
    assert_eq!(chunk_size_encoding(65535), (2, 0x01));
    assert_eq!(chunk_size_encoding(65536), (4, 0x02));
    #[cfg(target_pointer_width = "64")]
    assert_eq!(chunk_size_encoding(1usize << 32), (8, 0x03));
    #[cfg(target_pointer_width = "32")]
    assert_eq!(chunk_size_encoding(usize::MAX), (4, 0x02));
}

#[cfg(feature = "alloc")]
#[test]
fn write_superblock_roundtrip() {
    let mut cursor = consus_io::MemCursor::new();
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    write_superblock(&mut cursor, &mut state, 0x100).unwrap();

    let data = cursor.as_bytes();
    assert_eq!(&data[0..8], &HDF5_MAGIC);
    assert_eq!(data[8], 2); // version
    assert_eq!(data[9], 8); // offset size
    assert_eq!(data[10], 8); // length size
    // Root group address at offset 12 + 3*8 = 36
    assert_eq!(LittleEndian::read_u64(&data[36..44]), 0x100);
}
