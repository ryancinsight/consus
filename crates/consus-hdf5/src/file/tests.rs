//! Unit tests for the file module leaves.

use super::*;
use crate::constants::HDF5_MAGIC;
use byteorder::{ByteOrder, LittleEndian};
use consus_io::MemCursor;

/// Build a minimal v2 HDF5 file image and open it.
fn make_minimal_hdf5() -> Vec<u8> {
    let mut data = vec![0u8; 4096];
    // Superblock at offset 0
    data[0..8].copy_from_slice(&HDF5_MAGIC);
    data[8] = 2; // version
    data[9] = 8; // offset size
    data[10] = 8; // length size
    data[11] = 0; // consistency flags
    LittleEndian::write_u64(&mut data[12..20], 0); // base address
    LittleEndian::write_u64(&mut data[20..28], u64::MAX); // extension
    LittleEndian::write_u64(&mut data[28..36], 4096); // EOF
    LittleEndian::write_u64(&mut data[36..44], 96); // root group OH
    data
}

#[test]
fn open_minimal_file() {
    let data = make_minimal_hdf5();
    let cursor = MemCursor::from_bytes(data);
    let file = Hdf5File::open(cursor).expect("must open");
    assert_eq!(file.superblock().version, 2);
    assert_eq!(file.superblock().root_group_address, 96);
    assert_eq!(file.context().offset_size, 8);
    assert_eq!(file.context().length_size, 8);
}

#[test]
fn reject_non_hdf5() {
    let cursor = MemCursor::from_bytes(vec![0u8; 4096]);
    assert!(Hdf5File::open(cursor).is_err());
}

#[test]
fn fixed_array_entry_count_beyond_budget_is_rejected() {
    let mut data = make_minimal_hdf5();
    data.resize(512, 0);

    // FAHD at offset 128: an 8-byte entry size and a hostile element
    // count. The count must be rejected before the FADB read/allocation.
    data[128..132].copy_from_slice(b"FAHD");
    data[134] = 8;
    data[136..144].copy_from_slice(&u64::MAX.to_le_bytes());
    data[144..152].copy_from_slice(&256u64.to_le_bytes());

    let file = Hdf5File::open(MemCursor::from_bytes(data)).unwrap();
    let error = file
        .read_v4_chunk_farray_entries(128, &[1], &[1])
        .expect_err("u64::MAX FAHD element count must not be allocated");
    assert!(matches!(error, consus_core::Error::ResourceLimit { .. }));
}

#[test]
fn v1_chunk_btree_record_region_is_budgeted_before_read() {
    let ctx = ParseContext::with_budget(8, 8, consus_core::ParseBudget::new(32, 4, 8));
    let error = checked_v1_chunk_btree_data_size(&ctx, 1, 1)
        .expect_err("record region beyond byte budget must be rejected");
    assert!(matches!(error, consus_core::Error::ResourceLimit { .. }));
}
