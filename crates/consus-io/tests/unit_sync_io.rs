//! Comprehensive tests for sync I/O traits: ReadAt and WriteAt.
//!
//! Tests cover:
//! - ReadAt implementation for MemCursor
//! - WriteAt implementation for MemCursor
//! - Read/write at various offsets
//! - Bounds checking (read beyond EOF)
//! - Zero-copy semantics
//!
//! These tests run WITHOUT the async-io feature enabled.

use consus_core::Error;
use consus_io::{Length, MemCursor, ReadAt, Truncate, WriteAt};

// ═══════════════════════════════════════════════════════════════════════════════════════════
// ReadAt Trait Contract Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// ReadAt reads exact number of bytes at specified offset.
#[test]
fn read_at_reads_exact_bytes() {
    let cursor = MemCursor::from_bytes(vec![10, 20, 30, 40, 50]);

    let mut buf = [0u8; 3];
    cursor.read_at(1, &mut buf).expect("read must succeed");

    assert_eq!(buf, [20, 30, 40]);
}

/// ReadAt at offset 0 reads from start.
#[test]
fn read_at_zero_offset() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 2];
    cursor.read_at(0, &mut buf).expect("read must succeed");

    assert_eq!(buf, [1, 2]);
}

/// ReadAt at last valid offset reads remaining bytes.
#[test]
fn read_at_last_valid_offset() {
    let cursor = MemCursor::from_bytes(vec![100, 101, 102, 103, 104]);

    let mut buf = [0u8; 1];
    cursor.read_at(4, &mut buf).expect("read must succeed");

    assert_eq!(buf, [104]);
}

/// ReadAt reads full buffer in one call.
#[test]
fn read_at_full_buffer() {
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let cursor = MemCursor::from_bytes(data.clone());

    let mut buf = vec![0u8; 4];
    cursor.read_at(0, &mut buf).expect("read must succeed");

    assert_eq!(buf, data);
}

/// ReadAt with zero-length buffer succeeds at any offset.
#[test]
fn read_at_zero_length_succeeds() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);
    let mut buf = [];

    cursor
        .read_at(0, &mut buf)
        .expect("zero-length read at 0 must succeed");
    cursor
        .read_at(3, &mut buf)
        .expect("zero-length read at len must succeed");
    cursor
        .read_at(100, &mut buf)
        .expect("zero-length read beyond len must succeed");
}

/// ReadAt beyond buffer length returns BufferTooSmall.
#[test]
fn read_at_beyond_length_error() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    let mut buf = [0u8; 2];
    let err = cursor.read_at(2, &mut buf).expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 4);
            assert_eq!(provided, 3);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// ReadAt starting past end returns BufferTooSmall.
#[test]
fn read_at_starting_past_end() {
    let cursor = MemCursor::from_bytes(vec![1, 2]);

    let mut buf = [0u8; 1];
    let err = cursor.read_at(5, &mut buf).expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 6);
            assert_eq!(provided, 2);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// ReadAt on empty cursor fails for non-zero length.
#[test]
fn read_at_empty_cursor() {
    let cursor = MemCursor::new();

    let mut buf = [0u8; 1];
    let err = cursor.read_at(0, &mut buf).expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 1);
            assert_eq!(provided, 0);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// ReadAt validates buffer content at various offsets.
#[test]
fn read_at_various_offsets() {
    let data: Vec<u8> = (0..=255).collect();
    let cursor = MemCursor::from_bytes(data);

    // Read from start
    let mut buf1 = [0u8; 10];
    cursor.read_at(0, &mut buf1).expect("read at 0");
    assert_eq!(buf1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // Read from middle
    let mut buf2 = [0u8; 10];
    cursor.read_at(100, &mut buf2).expect("read at 100");
    assert_eq!(buf2, [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]);

    // Read from near end
    let mut buf3 = [0u8; 10];
    cursor.read_at(246, &mut buf3).expect("read at 246");
    assert_eq!(buf3, [246, 247, 248, 249, 250, 251, 252, 253, 254, 255]);
}

/// ReadAt multiple sequential reads advance correctly.
#[test]
fn read_at_sequential_reads() {
    let cursor = MemCursor::from_bytes(vec![10, 20, 30, 40, 50, 60, 70, 80]);

    let mut buf1 = [0u8; 2];
    cursor.read_at(0, &mut buf1).expect("read 1");
    assert_eq!(buf1, [10, 20]);

    let mut buf2 = [0u8; 2];
    cursor.read_at(2, &mut buf2).expect("read 2");
    assert_eq!(buf2, [30, 40]);

    let mut buf3 = [0u8; 2];
    cursor.read_at(4, &mut buf3).expect("read 3");
    assert_eq!(buf3, [50, 60]);

    let mut buf4 = [0u8; 2];
    cursor.read_at(6, &mut buf4).expect("read 4");
    assert_eq!(buf4, [70, 80]);
}

/// ReadAt does not modify cursor contents.
#[test]
fn read_at_does_not_modify_cursor() {
    let original = vec![0xAA, 0xBB, 0xCC, 0xDD];
    let cursor = MemCursor::from_bytes(original.clone());

    let mut buf = [0u8; 2];
    cursor.read_at(1, &mut buf).expect("read");

    assert_eq!(cursor.as_bytes(), original.as_slice());
}

/// ReadAt with large buffer.
#[test]
fn read_at_large_buffer() {
    let size = 10_000;
    let data: Vec<u8> = (0..=255).cycle().take(size).collect();
    let cursor = MemCursor::from_bytes(data.clone());

    let mut buf = vec![0u8; size];
    cursor
        .read_at(0, &mut buf)
        .expect("large read must succeed");

    assert_eq!(buf, data);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// WriteAt Trait Contract Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// WriteAt writes exact bytes at specified offset.
#[test]
fn write_at_writes_exact_bytes() {
    let mut cursor = MemCursor::with_capacity(10);

    cursor
        .write_at(0, &[1, 2, 3, 4, 5])
        .expect("write must succeed");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5]);
}

/// WriteAt at offset 0 writes from start.
#[test]
fn write_at_zero_offset() {
    let mut cursor = MemCursor::new();

    cursor
        .write_at(0, &[0xDE, 0xAD, 0xBE, 0xEF])
        .expect("write must succeed");

    assert_eq!(cursor.as_bytes(), &[0xDE, 0xAD, 0xBE, 0xEF]);
}

/// WriteAt at non-zero offset extends buffer.
#[test]
fn write_at_offset_extends_buffer() {
    let mut cursor = MemCursor::new();

    cursor.write_at(5, &[0xAA]).expect("write must succeed");

    assert_eq!(cursor.byte_len(), 6);
    assert_eq!(cursor.as_bytes()[..5], [0, 0, 0, 0, 0]);
    assert_eq!(cursor.as_bytes()[5], 0xAA);
}

/// WriteAt overwrites existing content.
#[test]
fn write_at_overwrites_content() {
    let mut cursor = MemCursor::from_bytes(vec![0xAA; 10]);

    cursor
        .write_at(3, &[0x11, 0x22, 0x33])
        .expect("write must succeed");

    assert_eq!(cursor.byte_len(), 10);
    assert_eq!(&cursor.as_bytes()[..3], &[0xAA; 3]);
    assert_eq!(&cursor.as_bytes()[3..6], &[0x11, 0x22, 0x33]);
    assert_eq!(&cursor.as_bytes()[6..], &[0xAA; 4]);
}

/// WriteAt zero-length succeeds at any offset.
#[test]
fn write_at_zero_length_succeeds() {
    let mut cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    cursor
        .write_at(0, &[])
        .expect("zero-length write at 0 must succeed");
    cursor
        .write_at(100, &[])
        .expect("zero-length write beyond len must succeed");

    assert_eq!(cursor.byte_len(), 3);
}

/// WriteAt extends buffer beyond current length.
#[test]
fn write_at_extends_buffer() {
    let mut cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    cursor
        .write_at(10, &[0xFF])
        .expect("write beyond must succeed");

    assert_eq!(cursor.byte_len(), 11);
    assert_eq!(cursor.as_bytes()[..3], [1, 2, 3]);
    // Bytes [3..10) should be zero
    for i in 3..10 {
        assert_eq!(cursor.as_bytes()[i], 0);
    }
    assert_eq!(cursor.as_bytes()[10], 0xFF);
}

/// WriteAt multiple sequential writes.
#[test]
fn write_at_sequential_writes() {
    let mut cursor = MemCursor::new();

    cursor.write_at(0, &[1, 2]).expect("write 1");
    assert_eq!(cursor.as_bytes(), &[1, 2]);

    cursor.write_at(2, &[3, 4]).expect("write 2");
    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4]);

    cursor.write_at(4, &[5, 6]).expect("write 3");
    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5, 6]);
}

/// WriteAt out-of-order writes.
#[test]
fn write_at_out_of_order() {
    let mut cursor = MemCursor::new();

    cursor.write_at(6, &[7, 8]).expect("write at end");
    cursor.write_at(2, &[3, 4, 5, 6]).expect("write middle");
    cursor.write_at(0, &[1, 2]).expect("write start");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

/// WriteAt with large buffer.
#[test]
fn write_at_large_buffer() {
    let mut cursor = MemCursor::new();
    let data = vec![0x42u8; 50_000];

    cursor.write_at(0, &data).expect("large write must succeed");

    assert_eq!(cursor.byte_len(), 50_000);
    assert!(cursor.as_bytes().iter().all(|&b| b == 0x42));
}

/// WriteAt preserves surrounding content.
#[test]
fn write_at_preserves_surrounding() {
    let mut cursor = MemCursor::from_bytes(vec![0xFF; 20]);

    cursor
        .write_at(8, &[0xAA, 0xBB, 0xCC])
        .expect("write middle");

    assert_eq!(cursor.byte_len(), 20);
    assert_eq!(&cursor.as_bytes()[..8], &[0xFF; 8]);
    assert_eq!(&cursor.as_bytes()[8..11], &[0xAA, 0xBB, 0xCC]);
    assert_eq!(&cursor.as_bytes()[11..], &[0xFF; 9]);
}

/// WriteAt at exact end of buffer.
#[test]
fn write_at_exact_end() {
    let mut cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    cursor.write_at(3, &[4, 5]).expect("write at end");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5]);
}

/// WriteAt far beyond current length.
#[test]
fn write_at_far_beyond_length() {
    let mut cursor = MemCursor::from_bytes(vec![1]);

    cursor
        .write_at(1000, &[0xFF])
        .expect("write far beyond must succeed");

    assert_eq!(cursor.byte_len(), 1001);
    assert_eq!(cursor.as_bytes()[0], 1);
    // Check zero-fill at various positions
    assert_eq!(cursor.as_bytes()[500], 0);
    assert_eq!(cursor.as_bytes()[999], 0);
    assert_eq!(cursor.as_bytes()[1000], 0xFF);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// ReadAt + WriteAt Integration Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Write then read round-trip at identical offset.
#[test]
fn write_then_read_roundtrip() {
    let mut cursor = MemCursor::new();
    let payload = b"consus-io positioned I/O test";

    cursor.write_at(0, payload).expect("write must succeed");
    assert_eq!(cursor.byte_len(), payload.len());

    let mut buf = vec![0u8; payload.len()];
    cursor.read_at(0, &mut buf).expect("read must succeed");

    assert_eq!(&buf, payload);
}

/// Interleaved read/write operations.
#[test]
fn interleaved_read_write() {
    let mut cursor = MemCursor::from_bytes(vec![0xAA; 10]);

    // Read initial state
    let mut buf1 = [0u8; 3];
    cursor.read_at(0, &mut buf1).expect("read 1");
    assert_eq!(buf1, [0xAA, 0xAA, 0xAA]);

    // Write at position 2
    cursor.write_at(2, &[0x11, 0x22]).expect("write 1");

    // Read from start to see modification
    let mut buf2 = [0u8; 5];
    cursor.read_at(0, &mut buf2).expect("read 2");
    assert_eq!(buf2, [0xAA, 0xAA, 0x11, 0x22, 0xAA]);

    // Write at position 0
    cursor.write_at(0, &[0x00]).expect("write 2");

    // Read entire buffer
    let mut buf3 = vec![0u8; 10];
    cursor.read_at(0, &mut buf3).expect("read 3");
    assert_eq!(
        buf3,
        [0x00, 0xAA, 0x11, 0x22, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA]
    );
}

/// Multiple writes to same region.
#[test]
fn multiple_writes_same_region() {
    let mut cursor = MemCursor::from_bytes(vec![0; 10]);

    cursor.write_at(2, &[1, 2, 3]).expect("write 1");
    cursor.write_at(2, &[10, 20]).expect("overwrite");
    cursor.write_at(4, &[30]).expect("partial overwrite");

    let mut buf = vec![0u8; 10];
    cursor.read_at(0, &mut buf).expect("read");
    assert_eq!(buf, [0, 0, 10, 20, 30, 0, 0, 0, 0, 0]);
}

/// Read and write at various offsets verify content.
#[test]
fn read_write_various_offsets() {
    let mut cursor = MemCursor::new();

    // Build buffer with known pattern
    for i in 0..100u8 {
        cursor.write_at(i as u64, &[i]).expect("write");
    }

    // Verify all bytes
    for i in 0..100u8 {
        let mut buf = [0u8; 1];
        cursor.read_at(i as u64, &mut buf).expect("read");
        assert_eq!(buf[0], i, "byte at position {}", i);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Bounds Checking Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Read exactly at buffer boundary.
#[test]
fn read_at_exact_boundary() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 1];
    cursor.read_at(4, &mut buf).expect("read at last position");
    assert_eq!(buf, [5]);
}

/// Read one byte beyond boundary fails.
#[test]
fn read_one_byte_beyond_boundary() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 1];
    let err = cursor.read_at(5, &mut buf).expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 6);
            assert_eq!(provided, 5);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Read spanning beyond boundary fails.
#[test]
fn read_spanning_beyond_boundary() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 3];
    let err = cursor.read_at(3, &mut buf).expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 6);
            assert_eq!(provided, 5);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Read at position zero on empty cursor.
#[test]
fn read_at_zero_empty_cursor() {
    let cursor = MemCursor::new();

    let mut buf = [0u8; 1];
    let err = cursor.read_at(0, &mut buf).expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 1);
            assert_eq!(provided, 0);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Bounds error includes correct required and provided values.
#[test]
fn bounds_error_includes_correct_values() {
    let cursor = MemCursor::from_bytes(vec![0u8; 100]);

    // Try to read 50 bytes starting at position 60
    let mut buf = vec![0u8; 50];
    let err = cursor.read_at(60, &mut buf).expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 110); // 60 + 50
            assert_eq!(provided, 100);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Zero-length read beyond bounds succeeds.
#[test]
fn zero_length_read_beyond_bounds() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    let mut buf = [];
    cursor
        .read_at(1000, &mut buf)
        .expect("zero-length read beyond bounds must succeed");
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Zero-Copy Semantics Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// as_bytes provides direct reference to underlying data.
#[test]
fn as_bytes_zero_copy() {
    let data = vec![10, 20, 30, 40, 50];
    let cursor = MemCursor::from_bytes(data);

    let bytes = cursor.as_bytes();
    assert_eq!(bytes.len(), 5);

    // Verify content matches
    assert_eq!(bytes, &[10, 20, 30, 40, 50]);
}

/// as_bytes_mut allows in-place modification.
#[test]
fn as_bytes_mut_direct_modification() {
    let mut cursor = MemCursor::from_bytes(vec![0, 0, 0, 0, 0]);

    {
        let bytes = cursor.as_bytes_mut();
        bytes[0] = 1;
        bytes[2] = 3;
        bytes[4] = 5;
    }

    assert_eq!(cursor.as_bytes(), &[1, 0, 3, 0, 5]);
}

/// into_bytes consumes cursor and returns original Vec.
#[test]
fn into_bytes_consumes_cursor() {
    let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let cursor = MemCursor::from_bytes(original.clone());

    let recovered = cursor.into_bytes();
    assert_eq!(recovered, original);
}

/// ReadAt copies bytes into caller's buffer.
#[test]
fn read_at_copies_to_caller_buffer() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 3];
    cursor.read_at(1, &mut buf).expect("read");

    // Buffer now contains copy of bytes
    assert_eq!(buf, [2, 3, 4]);
}

/// WriteAt copies bytes from caller's buffer.
#[test]
fn write_at_copies_from_caller_buffer() {
    let mut cursor = MemCursor::new();

    let data = vec![10, 20, 30];
    cursor.write_at(0, &data).expect("write");

    // Modify original data
    let mut data = data;
    data[0] = 99;

    // Cursor should not be affected
    assert_eq!(cursor.as_bytes(), &[10, 20, 30]);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Flush Operation Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Flush on MemCursor succeeds as no-op.
#[test]
fn flush_succeeds() {
    let mut cursor = MemCursor::new();
    cursor.flush().expect("flush must succeed");
}

/// Flush after write succeeds.
#[test]
fn flush_after_write() {
    let mut cursor = MemCursor::new();
    cursor.write_at(0, &[1, 2, 3]).expect("write");
    cursor.flush().expect("flush must succeed");

    // Data should still be there
    let mut buf = [0u8; 3];
    cursor.read_at(0, &mut buf).expect("read");
    assert_eq!(buf, [1, 2, 3]);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Object Safety Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// ReadAt is object-safe and can be used as dyn trait.
#[test]
fn read_at_is_object_safe() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);
    let _reader: &dyn ReadAt = &cursor;
}

/// WriteAt is object-safe and can be used as dyn trait.
#[test]
fn write_at_is_object_safe() {
    let mut cursor = MemCursor::new();
    let _writer: &mut dyn WriteAt = &mut cursor;
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Edge Cases
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Single byte operations.
#[test]
fn single_byte_read_write() {
    let mut cursor = MemCursor::new();

    // Write single bytes
    cursor.write_at(0, &[42]).expect("single byte write");

    // Read single byte
    let mut buf = [0u8; 1];
    cursor.read_at(0, &mut buf).expect("single byte read");
    assert_eq!(buf[0], 42);
}

/// Very large single operation.
#[test]
fn very_large_single_operation() {
    let size = 1_000_000;
    let mut cursor = MemCursor::new();

    let write_data = vec![0xABu8; size];
    cursor.write_at(0, &write_data).expect("large write");

    let mut read_data = vec![0u8; size];
    cursor.read_at(0, &mut read_data).expect("large read");

    assert_eq!(read_data.len(), size);
    assert!(read_data.iter().all(|&b| b == 0xAB));
}

/// Position overflow would return error.
#[test]
fn position_overflow_handling() {
    let cursor = MemCursor::from_bytes(vec![1, 2, 3]);

    // Position that would cause overflow when added to buffer length
    let mut buf = [0u8; 1];
    // This should fail due to bounds check, not overflow
    let err = cursor.read_at(u64::MAX, &mut buf).expect_err("should fail");

    match err {
        Error::Overflow => {
            // Overflow detected
        }
        Error::BufferTooSmall { .. } => {
            // Bounds check caught it first - also acceptable
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

/// Empty operations on empty cursor.
#[test]
fn empty_cursor_empty_operations() {
    let cursor = MemCursor::new();

    // Zero-length read on empty cursor
    let mut buf = [];
    cursor
        .read_at(0, &mut buf)
        .expect("empty read on empty cursor");
}

/// Write and read pattern verification.
#[test]
fn write_read_pattern_verification() {
    let mut cursor = MemCursor::new();

    // Write a pattern
    let pattern: Vec<u8> = (0..=255).cycle().take(1000).collect();
    cursor.write_at(0, &pattern).expect("write pattern");

    // Read back and verify
    let mut buf = vec![0u8; 1000];
    cursor.read_at(0, &mut buf).expect("read pattern");

    assert_eq!(buf.len(), 1000);
    for (i, &byte) in buf.iter().enumerate() {
        assert_eq!(byte, (i % 256) as u8, "position {}", i);
    }
}
