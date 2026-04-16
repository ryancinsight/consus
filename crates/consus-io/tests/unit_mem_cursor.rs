//! Comprehensive tests for MemCursor implementation.
//!
//! Tests cover:
//! - Construction from Vec<u8>
//! - Grow/shrink behavior via set_len
//! - Position tracking through read/write operations
//! - Clone semantics and zero-copy guarantees

use consus_io::{Length, MemCursor, ReadAt, Truncate, WriteAt};

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Construction Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Creating an empty cursor yields zero-length buffer.
#[test]
fn new_creates_empty_buffer() {
    let cursor = MemCursor::new();
    assert_eq!(cursor.byte_len(), 0);
    assert!(cursor.is_empty());
    assert_eq!(cursor.as_bytes(), &[]);
}

/// Default trait creates an empty cursor identical to new().
#[test]
fn default_matches_new() {
    let cursor = MemCursor::default();
    assert_eq!(cursor.byte_len(), 0);
    assert!(cursor.is_empty());
}

/// Cursor with pre-allocated capacity has zero length but non-zero capacity.
#[test]
fn with_capacity_reserves_without_initializing() {
    let cursor = MemCursor::with_capacity(1024);
    assert_eq!(cursor.byte_len(), 0);
    assert!(cursor.is_empty());
    // Capacity may be >= requested value, but we cannot test capacity directly.
    // Writing up to capacity should not reallocate.
    let mut cursor = cursor;
    let data = vec![0xABu8; 512];
    cursor
        .write_at(0, &data)
        .expect("write within capacity must succeed");
    assert_eq!(cursor.byte_len(), 512);
}

/// from_bytes creates cursor with exact content.
#[test]
fn from_bytes_preserves_content() {
    let original = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
    let cursor = MemCursor::from_bytes(original.clone());
    assert_eq!(cursor.byte_len(), original.len());
    assert_eq!(cursor.as_bytes(), original.as_slice());
}

/// from_bytes with empty vector yields empty cursor.
#[test]
fn from_bytes_empty() {
    let cursor = MemCursor::from_bytes(vec![]);
    assert_eq!(cursor.byte_len(), 0);
    assert!(cursor.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Grow/Shrink Behavior (set_len)
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// set_len to larger size zero-fills the extension.
#[test]
fn set_len_extends_with_zeros() {
    let mut cursor = MemCursor::from_bytes(vec![0x11, 0x22]);
    cursor.set_len(5).expect("set_len must succeed");

    assert_eq!(cursor.byte_len(), 5);
    assert_eq!(cursor.as_bytes(), &[0x11, 0x22, 0x00, 0x00, 0x00]);
}

/// set_len to smaller size truncates content.
#[test]
fn set_len_truncates_content() {
    let mut cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    cursor.set_len(3).expect("set_len must succeed");

    assert_eq!(cursor.byte_len(), 3);
    assert_eq!(cursor.as_bytes(), &[1, 2, 3]);
}

/// set_len to same size is a no-op.
#[test]
fn set_len_same_size_preserves_content() {
    let original = vec![0xAA, 0xBB, 0xCC, 0xDD];
    let mut cursor = MemCursor::from_bytes(original.clone());
    cursor.set_len(4).expect("set_len must succeed");

    assert_eq!(cursor.byte_len(), 4);
    assert_eq!(cursor.as_bytes(), original.as_slice());
}

/// set_len to zero clears the buffer.
#[test]
fn set_len_zero_empties_buffer() {
    let mut cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);
    cursor.set_len(0).expect("set_len must succeed");

    assert_eq!(cursor.byte_len(), 0);
    assert!(cursor.is_empty());
}

/// Sequential set_len operations compound correctly.
#[test]
fn set_len_sequential_operations() {
    let mut cursor = MemCursor::new();

    // Extend to 10
    cursor.set_len(10).expect("extend must succeed");
    assert_eq!(cursor.byte_len(), 10);

    // Extend further to 20
    cursor.set_len(20).expect("extend must succeed");
    assert_eq!(cursor.byte_len(), 20);

    // Truncate back to 5
    cursor.set_len(5).expect("truncate must succeed");
    assert_eq!(cursor.byte_len(), 5);

    // Extend again to 15
    cursor.set_len(15).expect("extend must succeed");
    assert_eq!(cursor.byte_len(), 15);
}

/// Large set_len extension zero-fills correctly.
#[test]
fn set_len_large_extension() {
    let mut cursor = MemCursor::from_bytes(vec![0xFF]);
    cursor.set_len(1000).expect("large set_len must succeed");

    assert_eq!(cursor.byte_len(), 1000);
    assert_eq!(cursor.as_bytes()[0], 0xFF);

    // Check zero-fill at various positions
    for i in 1..1000 {
        assert_eq!(cursor.as_bytes()[i], 0, "position {} should be zero", i);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// ReadAt Implementation Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Read at offset 0 returns full buffer.
#[test]
fn read_at_zero_returns_full_buffer() {
    let data = vec![10, 20, 30, 40, 50];
    let cursor = MemCursor::from_bytes(data.clone());

    let mut buf = vec![0u8; 5];
    cursor.read_at(0, &mut buf).expect("read must succeed");

    assert_eq!(buf, data);
}

/// Read at non-zero offset returns correct slice.
#[test]
fn read_at_offset_returns_correct_slice() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let cursor = MemCursor::from_bytes(data.clone());

    let mut buf = [0u8; 3];
    cursor.read_at(5, &mut buf).expect("read must succeed");

    assert_eq!(buf, [6, 7, 8]);
}

/// Read at last valid position returns remaining bytes.
#[test]
fn read_at_last_position() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 1];
    cursor.read_at(4, &mut buf).expect("read must succeed");

    assert_eq!(buf, [5]);
}

/// Partial read from middle of buffer.
#[test]
fn read_partial_from_middle() {
    let data: Vec<u8> = (0..100).collect();
    let cursor = MemCursor::from_bytes(data);

    let mut buf = [0u8; 10];
    cursor.read_at(45, &mut buf).expect("read must succeed");

    let expected: Vec<u8> = (45..55).collect();
    assert_eq!(buf.as_slice(), expected.as_slice());
}

/// Read beyond buffer bounds returns BufferTooSmall error.
#[test]
fn read_beyond_bounds_returns_error() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    let mut buf = [0u8; 2];
    let err = cursor.read_at(2, &mut buf).expect_err("should fail");

    match err {
        consus_core::Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 4);
            assert_eq!(provided, 3);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Read starting beyond buffer length returns error.
#[test]
fn read_starting_beyond_length_returns_error() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    let mut buf = [0u8; 1];
    let err = cursor.read_at(10, &mut buf).expect_err("should fail");

    match err {
        consus_core::Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 11);
            assert_eq!(provided, 3);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Zero-length read at any position succeeds.
#[test]
fn read_zero_length_succeeds_at_any_position() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);
    let mut buf = [];

    cursor
        .read_at(0, &mut buf)
        .expect("zero-length read at 0 must succeed");
    cursor
        .read_at(3, &mut buf)
        .expect("zero-length read at len must succeed");
    cursor
        .read_at(1000, &mut buf)
        .expect("zero-length read beyond len must succeed");
}

/// Read with exact buffer size succeeds.
#[test]
fn read_exact_buffer_size() {
    let data = vec![0xAB; 256];
    let cursor = MemCursor::from_bytes(data.clone());

    let mut buf = vec![0u8; 256];
    cursor.read_at(0, &mut buf).expect("read must succeed");

    assert_eq!(buf, data);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// WriteAt Implementation Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Write at offset 0 fills from beginning.
#[test]
fn write_at_zero_fills_from_beginning() {
    let mut cursor = MemCursor::with_capacity(10);

    cursor
        .write_at(0, &[1, 2, 3, 4, 5])
        .expect("write must succeed");

    assert_eq!(cursor.byte_len(), 5);
    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5]);
}

/// Write at non-zero offset extends buffer with zero-fill.
#[test]
fn write_at_offset_extends_with_zero_fill() {
    let mut cursor = MemCursor::new();

    cursor
        .write_at(5, &[0xAA, 0xBB])
        .expect("write must succeed");

    assert_eq!(cursor.byte_len(), 7);
    assert_eq!(cursor.as_bytes()[..5], [0, 0, 0, 0, 0]);
    assert_eq!(cursor.as_bytes()[5..], [0xAA, 0xBB]);
}

/// Write overwriting existing content preserves surrounding bytes.
#[test]
fn write_overwrite_preserves_surrounding() {
    let mut cursor = MemCursor::from_bytes(vec![0xAA; 10]);

    cursor
        .write_at(3, &[0x11, 0x22, 0x33])
        .expect("write must succeed");

    assert_eq!(cursor.byte_len(), 10);
    assert_eq!(&cursor.as_bytes()[..3], &[0xAA; 3]);
    assert_eq!(&cursor.as_bytes()[3..6], &[0x11, 0x22, 0x33]);
    assert_eq!(&cursor.as_bytes()[6..], &[0xAA; 4]);
}

/// Multiple sequential writes build correct buffer.
#[test]
fn write_sequential_builds_correct_buffer() {
    let mut cursor = MemCursor::new();

    cursor.write_at(0, &[1, 2]).expect("write 1");
    cursor.write_at(2, &[3, 4]).expect("write 2");
    cursor.write_at(4, &[5, 6]).expect("write 3");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5, 6]);
}

/// Out-of-order writes produce correct final state.
#[test]
fn write_out_of_order() {
    let mut cursor = MemCursor::new();

    cursor.write_at(6, &[7, 8]).expect("write at end");
    cursor.write_at(0, &[1, 2]).expect("write at start");
    cursor.write_at(2, &[3, 4, 5, 6]).expect("write middle");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

/// Zero-length write is a no-op.
#[test]
fn write_zero_length_no_op() {
    let mut cursor = MemCursor::from_bytes(vec![1, 2, 3]);
    let len_before = cursor.byte_len();

    cursor
        .write_at(0, &[])
        .expect("zero-length write must succeed");
    cursor
        .write_at(100, &[])
        .expect("zero-length write beyond bounds must succeed");

    assert_eq!(cursor.byte_len(), len_before);
    assert_eq!(cursor.as_bytes(), &[1, 2, 3]);
}

/// Write beyond current length auto-extends correctly.
#[test]
fn write_beyond_length_auto_extends() {
    let mut cursor = MemCursor::from_bytes(vec![0xFF]);

    cursor
        .write_at(100, &[0xAA])
        .expect("write far beyond must succeed");

    assert_eq!(cursor.byte_len(), 101);
    assert_eq!(cursor.as_bytes()[0], 0xFF);
    // Bytes [1..100) should be zero-filled
    for i in 1..100 {
        assert_eq!(cursor.as_bytes()[i], 0);
    }
    assert_eq!(cursor.as_bytes()[100], 0xAA);
}

/// Large write succeeds.
#[test]
fn write_large_buffer() {
    let mut cursor = MemCursor::new();
    let data = vec![0x42u8; 10_000];

    cursor.write_at(0, &data).expect("large write must succeed");

    assert_eq!(cursor.byte_len(), 10_000);
    for i in 0..10_000 {
        assert_eq!(cursor.as_bytes()[i], 0x42);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Position Tracking Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Position tracking via sequential reads.
#[test]
fn position_tracking_sequential_reads() {
    let data: Vec<u8> = (0..20).collect();
    let cursor = MemCursor::from_bytes(data);

    // Read [0..5)
    let mut buf1 = [0u8; 5];
    cursor.read_at(0, &mut buf1).expect("read 1");
    assert_eq!(buf1, [0, 1, 2, 3, 4]);

    // Read [5..10)
    let mut buf2 = [0u8; 5];
    cursor.read_at(5, &mut buf2).expect("read 2");
    assert_eq!(buf2, [5, 6, 7, 8, 9]);

    // Read [15..20)
    let mut buf3 = [0u8; 5];
    cursor.read_at(15, &mut buf3).expect("read 3");
    assert_eq!(buf3, [15, 16, 17, 18, 19]);
}

/// Position tracking via sequential writes.
#[test]
fn position_tracking_sequential_writes() {
    let mut cursor = MemCursor::new();

    cursor.write_at(0, &[10, 11, 12]).expect("write 1");
    assert_eq!(cursor.byte_len(), 3);

    cursor.write_at(3, &[13, 14, 15]).expect("write 2");
    assert_eq!(cursor.byte_len(), 6);

    cursor.write_at(6, &[16, 17, 18]).expect("write 3");
    assert_eq!(cursor.byte_len(), 9);

    assert_eq!(cursor.as_bytes(), &[10, 11, 12, 13, 14, 15, 16, 17, 18]);
}

/// Interleaved reads and writes maintain consistency.
#[test]
fn position_interleaved_read_write() {
    let mut cursor = MemCursor::from_bytes(vec![0xAA; 10]);

    // Write at position 2
    cursor.write_at(2, &[0x11, 0x22]).expect("write");

    // Read from position 0
    let mut buf = [0u8; 5];
    cursor.read_at(0, &mut buf).expect("read");
    assert_eq!(buf, [0xAA, 0xAA, 0x11, 0x22, 0xAA]);

    // Write at position 7
    cursor.write_at(7, &[0x33]).expect("write 2");

    // Read from position 5
    let mut buf2 = [0u8; 5];
    cursor.read_at(5, &mut buf2).expect("read 2");
    assert_eq!(buf2, [0xAA, 0xAA, 0x33, 0xAA, 0xAA]);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Clone Semantics Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Clone creates independent copy.
#[test]
fn clone_creates_independent_copy() {
    let original = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);
    let clone = original.clone();

    assert_eq!(original.as_bytes(), clone.as_bytes());
}

/// Modifying clone does not affect original.
#[test]
fn clone_modification_independence() {
    let original = MemCursor::from_bytes(vec![0xAA; 5]);
    let mut clone = original.clone();

    clone.write_at(0, &[0x00, 0x00]).expect("write to clone");

    assert_eq!(original.as_bytes(), &[0xAA; 5]);
    assert_eq!(clone.as_bytes()[..2], [0x00, 0x00]);
}

/// Clone of empty cursor works correctly.
#[test]
fn clone_empty_cursor() {
    let original = MemCursor::new();
    let clone = original.clone();

    assert!(clone.is_empty());
    assert_eq!(clone.byte_len(), 0);
}

/// Clone preserves exact byte content.
#[test]
fn clone_preserves_exact_content() {
    let data: Vec<u8> = (0..=255).collect();
    let original = MemCursor::from_bytes(data.clone());
    let clone = original.clone();

    assert_eq!(clone.as_bytes(), data.as_slice());
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Zero-Copy Semantics Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// as_bytes returns reference without copy.
#[test]
fn as_bytes_zero_copy_reference() {
    let data = vec![1, 2, 3, 4, 5];
    let cursor = MemCursor::from_bytes(data);

    let bytes = cursor.as_bytes();
    assert_eq!(bytes.len(), 5);
    assert_eq!(bytes, &[1, 2, 3, 4, 5]);
}

/// as_bytes_mut allows in-place modification.
#[test]
fn as_bytes_mut_allows_modification() {
    let mut cursor = MemCursor::from_bytes(vec![0, 0, 0, 0, 0]);

    {
        let bytes = cursor.as_bytes_mut();
        bytes[0] = 1;
        bytes[2] = 3;
        bytes[4] = 5;
    }

    assert_eq!(cursor.as_bytes(), &[1, 0, 3, 0, 5]);
}

/// into_bytes consumes cursor without copy.
#[test]
fn into_bytes_consumes_without_copy() {
    let original_data = vec![10, 20, 30, 40, 50];
    let cursor = MemCursor::from_bytes(original_data.clone());

    let recovered = cursor.into_bytes();
    assert_eq!(recovered, original_data);
}

/// from_bytes + into_bytes round-trip.
#[test]
fn from_into_round_trip() {
    let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let cursor = MemCursor::from_bytes(original.clone());
    let recovered = cursor.into_bytes();
    assert_eq!(recovered, original);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Length Trait Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Length trait returns correct u64 value.
#[test]
fn length_trait_returns_u64() {
    let cursor = MemCursor::from_bytes(vec![0u8; 256]);
    let len = Length::len(&cursor).expect("length must succeed");
    assert_eq!(len, 256u64);
}

/// is_empty returns true for empty cursor.
#[test]
fn is_empty_trait_true() {
    let cursor = MemCursor::new();
    assert!(Length::is_empty(&cursor).expect("is_empty must succeed"));
}

/// is_empty returns false for non-empty cursor.
#[test]
fn is_empty_trait_false() {
    let cursor = MemCursor::from_bytes(vec![1]);
    assert!(!Length::is_empty(&cursor).expect("is_empty must succeed"));
}

/// Length updates after write.
#[test]
fn length_updates_after_write() {
    let mut cursor = MemCursor::new();
    assert_eq!(Length::len(&cursor).unwrap(), 0);

    cursor.write_at(0, &[1, 2, 3]).expect("write");
    assert_eq!(Length::len(&cursor).unwrap(), 3);

    cursor.write_at(10, &[4, 5]).expect("write beyond");
    assert_eq!(Length::len(&cursor).unwrap(), 12);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Truncate Trait Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Truncate trait reduces size.
#[test]
fn truncate_reduces_size() {
    let mut cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);
    Truncate::set_len(&mut cursor, 2).expect("truncate must succeed");

    assert_eq!(Length::len(&cursor).unwrap(), 2);
    assert_eq!(cursor.as_bytes(), &[1, 2]);
}

/// Truncate trait increases size with zero-fill.
#[test]
fn truncate_increases_size() {
    let mut cursor = MemCursor::from_bytes(vec![0xFF, 0xFF]);
    Truncate::set_len(&mut cursor, 5).expect("truncate must succeed");

    assert_eq!(Length::len(&cursor).unwrap(), 5);
    assert_eq!(cursor.as_bytes(), &[0xFF, 0xFF, 0, 0, 0]);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Edge Cases and Boundary Conditions
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Read of entire buffer contents.
#[test]
fn read_entire_buffer() {
    let data: Vec<u8> = (0..=255).collect();
    let cursor = MemCursor::from_bytes(data.clone());

    let mut buf = vec![0u8; 256];
    cursor.read_at(0, &mut buf).expect("full read must succeed");

    assert_eq!(buf, data);
}

/// Single byte read/write at various positions.
#[test]
fn single_byte_operations() {
    let mut cursor = MemCursor::new();

    for i in 0u8..=255 {
        cursor.write_at(i as u64, &[i]).expect("single byte write");
    }

    assert_eq!(cursor.byte_len(), 256);

    for i in 0u8..=255 {
        let mut buf = [0u8; 1];
        cursor
            .read_at(i as u64, &mut buf)
            .expect("single byte read");
        assert_eq!(buf[0], i);
    }
}

/// Very large cursor operations.
#[test]
fn large_cursor_operations() {
    let size = 100_000;
    let mut cursor = MemCursor::with_capacity(size);

    let data = vec![0xABu8; size];
    cursor.write_at(0, &data).expect("large write");

    assert_eq!(cursor.byte_len(), size);

    let mut buf = vec![0u8; size];
    cursor.read_at(0, &mut buf).expect("large read");

    assert_eq!(buf.len(), size);
    assert!(buf.iter().all(|&b| b == 0xAB));
}
