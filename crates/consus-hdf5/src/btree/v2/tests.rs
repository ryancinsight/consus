//! Unit tests for the B-tree v2 leaves.

use super::*;
use crate::address::ParseContext;
use consus_core::Error;

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
        let ctx = ParseContext::new(8, 8);
        let leaf = BTreeV2LeafNode::parse(&cursor, 0, &header, 3, &ctx).unwrap();

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
        let ctx = ParseContext::new(8, 8);
        let err = BTreeV2LeafNode::parse(&cursor, 0, &header, 0, &ctx).unwrap_err();
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
        file_data[leaf_offset + 10..leaf_offset + 14].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);

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
