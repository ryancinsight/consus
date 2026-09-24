//! Chunk B-tree index emission tests.

use super::super::*;
use crate::property_list::FileCreationProps;
use byteorder::{ByteOrder as _, LittleEndian};

#[cfg(feature = "alloc")]
#[test]
fn write_chunk_btree_v1_leaf_header() {
    let mut cursor = consus_io::MemCursor::new();
    let props = FileCreationProps::default();
    let mut state = WriteState::new(props);
    state.eof = 64;

    let entries = vec![ChunkIndexEntry {
        chunk_offsets: vec![0, 0],
        filter_mask: 0,
        chunk_size: 16,
        chunk_address: 0x2000,
    }];

    // chunk_dims=[2,2] → ndims=3, key_size=32, K=32, max_entries=64
    // node_size = header(24) + 65*32 + 64*8 = 24 + 2080 + 512 = 2616
    let expected_node_size: usize = 24 + 65 * 32 + 64 * 8;
    let addr =
        write_chunk_btree_v1(&mut cursor, &mut state, &[2, 2], &entries, &[4, 4], 4).unwrap();
    let bytes = cursor.as_bytes();
    assert_eq!(&bytes[addr as usize..addr as usize + 4], b"TREE");
    assert_eq!(bytes[addr as usize + 4], 1);
    assert_eq!(bytes[addr as usize + 5], 0);
    assert_eq!(
        LittleEndian::read_u16(&bytes[addr as usize + 6..addr as usize + 8]),
        1
    );
    // Verify full fixed-size node is written.
    assert_eq!(
        cursor.as_bytes().len() - addr as usize,
        expected_node_size,
        "B-tree node must be the full K=32 fixed size"
    );
}
