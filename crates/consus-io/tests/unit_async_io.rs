//! Comprehensive tests for async I/O traits: AsyncReadAt and AsyncWriteAt.
//!
//! Tests cover:
//! - AsyncReadAt implementation for AsyncMemCursor
//! - AsyncWriteAt implementation for AsyncMemCursor
//! - Tokio runtime integration
//! - Async bounds checking and error handling
//!
//! These tests are feature-gated with `#[cfg(feature = "async-io")]`.

#![cfg(feature = "async-io")]

use consus_core::Error;
use consus_io::{AsyncLength, AsyncMemCursor, AsyncReadAt, AsyncTruncate, AsyncWriteAt};

// ═══════════════════════════════════════════════════════════════════════════════════════════
// AsyncReadAt Trait Contract Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// AsyncReadAt reads exact bytes at specified offset.
#[tokio::test]
async fn async_read_at_reads_exact_bytes() {
    let cursor = AsyncMemCursor::from_bytes(vec![10, 20, 30, 40, 50]);

    let mut buf = [0u8; 3];
    cursor
        .read_at(1, &mut buf)
        .await
        .expect("async read must succeed");

    assert_eq!(buf, [20, 30, 40]);
}

/// AsyncReadAt at offset 0 reads from start.
#[tokio::test]
async fn async_read_at_zero_offset() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 2];
    cursor
        .read_at(0, &mut buf)
        .await
        .expect("async read must succeed");

    assert_eq!(buf, [1, 2]);
}

/// AsyncReadAt at last valid offset reads remaining bytes.
#[tokio::test]
async fn async_read_at_last_valid_offset() {
    let cursor = AsyncMemCursor::from_bytes(vec![100, 101, 102, 103, 104]);

    let mut buf = [0u8; 1];
    cursor
        .read_at(4, &mut buf)
        .await
        .expect("async read must succeed");

    assert_eq!(buf, [104]);
}

/// AsyncReadAt reads full buffer in one call.
#[tokio::test]
async fn async_read_at_full_buffer() {
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let cursor = AsyncMemCursor::from_bytes(data.clone());

    let mut buf = vec![0u8; 4];
    cursor
        .read_at(0, &mut buf)
        .await
        .expect("async read must succeed");

    assert_eq!(buf, data);
}

/// AsyncReadAt with zero-length buffer succeeds at any offset.
#[tokio::test]
async fn async_read_at_zero_length_succeeds() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3]);
    let mut buf = [];

    cursor
        .read_at(0, &mut buf)
        .await
        .expect("zero-length async read at 0 must succeed");
    cursor
        .read_at(3, &mut buf)
        .await
        .expect("zero-length async read at len must succeed");
    cursor
        .read_at(100, &mut buf)
        .await
        .expect("zero-length async read beyond len must succeed");
}

/// AsyncReadAt beyond buffer length returns BufferTooSmall.
#[tokio::test]
async fn async_read_at_beyond_length_error() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3]);

    let mut buf = [0u8; 2];
    let err = cursor.read_at(2, &mut buf).await.expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 4);
            assert_eq!(provided, 3);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// AsyncReadAt starting past end returns BufferTooSmall.
#[tokio::test]
async fn async_read_at_starting_past_end() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2]);

    let mut buf = [0u8; 1];
    let err = cursor.read_at(5, &mut buf).await.expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 6);
            assert_eq!(provided, 2);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// AsyncReadAt on empty cursor fails for non-zero length.
#[tokio::test]
async fn async_read_at_empty_cursor() {
    let cursor = AsyncMemCursor::new();

    let mut buf = [0u8; 1];
    let err = cursor.read_at(0, &mut buf).await.expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 1);
            assert_eq!(provided, 0);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// AsyncReadAt validates buffer content at various offsets.
#[tokio::test]
async fn async_read_at_various_offsets() {
    let data: Vec<u8> = (0..=255).collect();
    let cursor = AsyncMemCursor::from_bytes(data);

    // Read from start
    let mut buf1 = [0u8; 10];
    cursor.read_at(0, &mut buf1).await.expect("read at 0");
    assert_eq!(buf1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // Read from middle
    let mut buf2 = [0u8; 10];
    cursor.read_at(100, &mut buf2).await.expect("read at 100");
    assert_eq!(buf2, [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]);

    // Read from near end
    let mut buf3 = [0u8; 10];
    cursor.read_at(246, &mut buf3).await.expect("read at 246");
    assert_eq!(buf3, [246, 247, 248, 249, 250, 251, 252, 253, 254, 255]);
}

/// AsyncReadAt multiple sequential reads advance correctly.
#[tokio::test]
async fn async_read_at_sequential_reads() {
    let cursor = AsyncMemCursor::from_bytes(vec![10, 20, 30, 40, 50, 60, 70, 80]);

    let mut buf1 = [0u8; 2];
    cursor.read_at(0, &mut buf1).await.expect("read 1");
    assert_eq!(buf1, [10, 20]);

    let mut buf2 = [0u8; 2];
    cursor.read_at(2, &mut buf2).await.expect("read 2");
    assert_eq!(buf2, [30, 40]);

    let mut buf3 = [0u8; 2];
    cursor.read_at(4, &mut buf3).await.expect("read 3");
    assert_eq!(buf3, [50, 60]);

    let mut buf4 = [0u8; 2];
    cursor.read_at(6, &mut buf4).await.expect("read 4");
    assert_eq!(buf4, [70, 80]);
}

/// AsyncReadAt does not modify cursor contents.
#[tokio::test]
async fn async_read_at_does_not_modify_cursor() {
    let original = vec![0xAA, 0xBB, 0xCC, 0xDD];
    let cursor = AsyncMemCursor::from_bytes(original.clone());

    let mut buf = [0u8; 2];
    cursor.read_at(1, &mut buf).await.expect("read");

    assert_eq!(cursor.as_bytes(), original.as_slice());
}

/// AsyncReadAt with large buffer.
#[tokio::test]
async fn async_read_at_large_buffer() {
    let size = 10_000;
    let data: Vec<u8> = (0..=255).cycle().take(size).collect();
    let cursor = AsyncMemCursor::from_bytes(data.clone());

    let mut buf = vec![0u8; size];
    cursor
        .read_at(0, &mut buf)
        .await
        .expect("large async read must succeed");

    assert_eq!(buf, data);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// AsyncWriteAt Trait Contract Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// AsyncWriteAt writes exact bytes at specified offset.
#[tokio::test]
async fn async_write_at_writes_exact_bytes() {
    let mut cursor = AsyncMemCursor::with_capacity(10);

    cursor
        .write_at(0, &[1, 2, 3, 4, 5])
        .await
        .expect("async write must succeed");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5]);
}

/// AsyncWriteAt at offset 0 writes from start.
#[tokio::test]
async fn async_write_at_zero_offset() {
    let mut cursor = AsyncMemCursor::new();

    cursor
        .write_at(0, &[0xDE, 0xAD, 0xBE, 0xEF])
        .await
        .expect("async write must succeed");

    assert_eq!(cursor.as_bytes(), &[0xDE, 0xAD, 0xBE, 0xEF]);
}

/// AsyncWriteAt at non-zero offset extends buffer.
#[tokio::test]
async fn async_write_at_offset_extends_buffer() {
    let mut cursor = AsyncMemCursor::new();

    cursor
        .write_at(5, &[0xAA])
        .await
        .expect("async write must succeed");

    assert_eq!(cursor.byte_len(), 6);
    assert_eq!(cursor.as_bytes()[..5], [0, 0, 0, 0, 0]);
    assert_eq!(cursor.as_bytes()[5], 0xAA);
}

/// AsyncWriteAt overwrites existing content.
#[tokio::test]
async fn async_write_at_overwrites_content() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![0xAA; 10]);

    cursor
        .write_at(3, &[0x11, 0x22, 0x33])
        .await
        .expect("async write must succeed");

    assert_eq!(cursor.byte_len(), 10);
    assert_eq!(&cursor.as_bytes()[..3], &[0xAA; 3]);
    assert_eq!(&cursor.as_bytes()[3..6], &[0x11, 0x22, 0x33]);
    assert_eq!(&cursor.as_bytes()[6..], &[0xAA; 4]);
}

/// AsyncWriteAt zero-length succeeds at any offset.
#[tokio::test]
async fn async_write_at_zero_length_succeeds() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3]);

    cursor
        .write_at(0, &[])
        .await
        .expect("zero-length async write at 0 must succeed");
    cursor
        .write_at(100, &[])
        .await
        .expect("zero-length async write beyond len must succeed");

    assert_eq!(cursor.byte_len(), 3);
}

/// AsyncWriteAt extends buffer beyond current length.
#[tokio::test]
async fn async_write_at_extends_buffer() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3]);

    cursor
        .write_at(10, &[0xFF])
        .await
        .expect("async write beyond must succeed");

    assert_eq!(cursor.byte_len(), 11);
    assert_eq!(cursor.as_bytes()[..3], [1, 2, 3]);
    // Bytes [3..10) should be zero
    for i in 3..10 {
        assert_eq!(cursor.as_bytes()[i], 0);
    }
    assert_eq!(cursor.as_bytes()[10], 0xFF);
}

/// AsyncWriteAt multiple sequential writes.
#[tokio::test]
async fn async_write_at_sequential_writes() {
    let mut cursor = AsyncMemCursor::new();

    cursor.write_at(0, &[1, 2]).await.expect("write 1");
    assert_eq!(cursor.as_bytes(), &[1, 2]);

    cursor.write_at(2, &[3, 4]).await.expect("write 2");
    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4]);

    cursor.write_at(4, &[5, 6]).await.expect("write 3");
    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5, 6]);
}

/// AsyncWriteAt out-of-order writes.
#[tokio::test]
async fn async_write_at_out_of_order() {
    let mut cursor = AsyncMemCursor::new();

    cursor.write_at(6, &[7, 8]).await.expect("write at end");
    cursor
        .write_at(2, &[3, 4, 5, 6])
        .await
        .expect("write middle");
    cursor.write_at(0, &[1, 2]).await.expect("write start");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

/// AsyncWriteAt with large buffer.
#[tokio::test]
async fn async_write_at_large_buffer() {
    let mut cursor = AsyncMemCursor::new();
    let data = vec![0x42u8; 50_000];

    cursor
        .write_at(0, &data)
        .await
        .expect("large async write must succeed");

    assert_eq!(cursor.byte_len(), 50_000);
    assert!(cursor.as_bytes().iter().all(|&b| b == 0x42));
}

/// AsyncWriteAt preserves surrounding content.
#[tokio::test]
async fn async_write_at_preserves_surrounding() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![0xFF; 20]);

    cursor
        .write_at(8, &[0xAA, 0xBB, 0xCC])
        .await
        .expect("write middle");

    assert_eq!(cursor.byte_len(), 20);
    assert_eq!(&cursor.as_bytes()[..8], &[0xFF; 8]);
    assert_eq!(&cursor.as_bytes()[8..11], &[0xAA, 0xBB, 0xCC]);
    assert_eq!(&cursor.as_bytes()[11..], &[0xFF; 9]);
}

/// AsyncWriteAt at exact end of buffer.
#[tokio::test]
async fn async_write_at_exact_end() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3]);

    cursor.write_at(3, &[4, 5]).await.expect("write at end");

    assert_eq!(cursor.as_bytes(), &[1, 2, 3, 4, 5]);
}

/// AsyncWriteAt far beyond current length.
#[tokio::test]
async fn async_write_at_far_beyond_length() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![1]);

    cursor
        .write_at(1000, &[0xFF])
        .await
        .expect("write far beyond must succeed");

    assert_eq!(cursor.byte_len(), 1001);
    assert_eq!(cursor.as_bytes()[0], 1);
    // Check zero-fill at various positions
    assert_eq!(cursor.as_bytes()[500], 0);
    assert_eq!(cursor.as_bytes()[999], 0);
    assert_eq!(cursor.as_bytes()[1000], 0xFF);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// AsyncReadAt + AsyncWriteAt Integration Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Async write then read round-trip at identical offset.
#[tokio::test]
async fn async_write_then_read_roundtrip() {
    let mut cursor = AsyncMemCursor::new();
    let payload = b"async consus-io positioned I/O test";

    cursor
        .write_at(0, payload)
        .await
        .expect("async write must succeed");
    assert_eq!(cursor.byte_len(), payload.len());

    let mut buf = vec![0u8; payload.len()];
    cursor
        .read_at(0, &mut buf)
        .await
        .expect("async read must succeed");

    assert_eq!(&buf, payload);
}

/// Async interleaved read/write operations.
#[tokio::test]
async fn async_interleaved_read_write() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![0xAA; 10]);

    // Read initial state
    let mut buf1 = [0u8; 3];
    cursor.read_at(0, &mut buf1).await.expect("read 1");
    assert_eq!(buf1, [0xAA, 0xAA, 0xAA]);

    // Write at position 2
    cursor.write_at(2, &[0x11, 0x22]).await.expect("write 1");

    // Read from start to see modification
    let mut buf2 = [0u8; 5];
    cursor.read_at(0, &mut buf2).await.expect("read 2");
    assert_eq!(buf2, [0xAA, 0xAA, 0x11, 0x22, 0xAA]);

    // Write at position 0
    cursor.write_at(0, &[0x00]).await.expect("write 2");

    // Read entire buffer
    let mut buf3 = vec![0u8; 10];
    cursor.read_at(0, &mut buf3).await.expect("read 3");
    assert_eq!(
        buf3,
        [0x00, 0xAA, 0x11, 0x22, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA]
    );
}

/// Multiple async writes to same region.
#[tokio::test]
async fn async_multiple_writes_same_region() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![0; 10]);

    cursor.write_at(2, &[1, 2, 3]).await.expect("write 1");
    cursor.write_at(2, &[10, 20]).await.expect("overwrite");
    cursor.write_at(4, &[30]).await.expect("partial overwrite");

    let mut buf = vec![0u8; 10];
    cursor.read_at(0, &mut buf).await.expect("read");
    assert_eq!(buf, [0, 0, 10, 20, 30, 0, 0, 0, 0, 0]);
}

/// Async read and write at various offsets verify content.
#[tokio::test]
async fn async_read_write_various_offsets() {
    let mut cursor = AsyncMemCursor::new();

    // Build buffer with known pattern
    for i in 0..100u8 {
        cursor.write_at(i as u64, &[i]).await.expect("write");
    }

    // Verify all bytes
    for i in 0..100u8 {
        let mut buf = [0u8; 1];
        cursor.read_at(i as u64, &mut buf).await.expect("read");
        assert_eq!(buf[0], i, "byte at position {}", i);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Async Bounds Checking Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Async read exactly at buffer boundary.
#[tokio::test]
async fn async_read_at_exact_boundary() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 1];
    cursor
        .read_at(4, &mut buf)
        .await
        .expect("read at last position");
    assert_eq!(buf, [5]);
}

/// Async read one byte beyond boundary fails.
#[tokio::test]
async fn async_read_one_byte_beyond_boundary() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 1];
    let err = cursor.read_at(5, &mut buf).await.expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 6);
            assert_eq!(provided, 5);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Async read spanning beyond boundary fails.
#[tokio::test]
async fn async_read_spanning_beyond_boundary() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3, 4, 5]);

    let mut buf = [0u8; 3];
    let err = cursor.read_at(3, &mut buf).await.expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 6);
            assert_eq!(provided, 5);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Async bounds error includes correct required and provided values.
#[tokio::test]
async fn async_bounds_error_includes_correct_values() {
    let cursor = AsyncMemCursor::from_bytes(vec![0u8; 100]);

    // Try to read 50 bytes starting at position 60
    let mut buf = vec![0u8; 50];
    let err = cursor.read_at(60, &mut buf).await.expect_err("should fail");

    match err {
        Error::BufferTooSmall { required, provided } => {
            assert_eq!(required, 110); // 60 + 50
            assert_eq!(provided, 100);
        }
        other => panic!("expected BufferTooSmall, got: {:?}", other),
    }
}

/// Async zero-length read beyond bounds succeeds.
#[tokio::test]
async fn async_zero_length_read_beyond_bounds() {
    let cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3]);

    let mut buf = [];
    cursor
        .read_at(1000, &mut buf)
        .await
        .expect("zero-length async read beyond bounds must succeed");
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Async Flush Operation Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Async flush on AsyncMemCursor succeeds as no-op.
#[tokio::test]
async fn async_flush_succeeds() {
    let mut cursor = AsyncMemCursor::new();
    cursor.flush().await.expect("async flush must succeed");
}

/// Async flush after write succeeds.
#[tokio::test]
async fn async_flush_after_write() {
    let mut cursor = AsyncMemCursor::new();
    cursor.write_at(0, &[1, 2, 3]).await.expect("write");
    cursor.flush().await.expect("async flush must succeed");

    // Data should still be there
    let mut buf = [0u8; 3];
    cursor.read_at(0, &mut buf).await.expect("read");
    assert_eq!(buf, [1, 2, 3]);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// AsyncLength Trait Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// AsyncLength trait returns correct u64 value.
#[tokio::test]
async fn async_length_trait_returns_u64() {
    let cursor = AsyncMemCursor::from_bytes(vec![0u8; 256]);
    let len = AsyncLength::len(&cursor)
        .await
        .expect("async length must succeed");
    assert_eq!(len, 256u64);
}

/// Async is_empty returns true for empty cursor.
#[tokio::test]
async fn async_is_empty_trait_true() {
    let cursor = AsyncMemCursor::new();
    assert!(
        AsyncLength::is_empty(&cursor)
            .await
            .expect("async is_empty must succeed")
    );
}

/// Async is_empty returns false for non-empty cursor.
#[tokio::test]
async fn async_is_empty_trait_false() {
    let cursor = AsyncMemCursor::from_bytes(vec![1]);
    assert!(
        !AsyncLength::is_empty(&cursor)
            .await
            .expect("async is_empty must succeed")
    );
}

/// Async length updates after write.
#[tokio::test]
async fn async_length_updates_after_write() {
    let mut cursor = AsyncMemCursor::new();
    assert_eq!(AsyncLength::len(&cursor).await.unwrap(), 0);

    cursor.write_at(0, &[1, 2, 3]).await.expect("write");
    assert_eq!(AsyncLength::len(&cursor).await.unwrap(), 3);

    cursor.write_at(10, &[4, 5]).await.expect("write beyond");
    assert_eq!(AsyncLength::len(&cursor).await.unwrap(), 12);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// AsyncTruncate Trait Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Async truncate trait reduces size.
#[tokio::test]
async fn async_truncate_reduces_size() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![1, 2, 3, 4, 5]);
    AsyncTruncate::set_len(&mut cursor, 2)
        .await
        .expect("async truncate must succeed");

    assert_eq!(AsyncLength::len(&cursor).await.unwrap(), 2);
    assert_eq!(cursor.as_bytes(), &[1, 2]);
}

/// Async truncate trait increases size with zero-fill.
#[tokio::test]
async fn async_truncate_increases_size() {
    let mut cursor = AsyncMemCursor::from_bytes(vec![0xFF, 0xFF]);
    AsyncTruncate::set_len(&mut cursor, 5)
        .await
        .expect("async truncate must succeed");

    assert_eq!(AsyncLength::len(&cursor).await.unwrap(), 5);
    assert_eq!(cursor.as_bytes(), &[0xFF, 0xFF, 0, 0, 0]);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Tokio Runtime Integration Tests
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Multiple concurrent reads on same cursor (tokio spawn).
#[tokio::test]
async fn tokio_concurrent_reads() {
    let cursor = AsyncMemCursor::from_bytes(vec![42u8; 1000]);
    let cursor = std::sync::Arc::new(cursor);

    let mut handles = vec![];

    for i in 0..10 {
        let cursor_clone = std::sync::Arc::clone(&cursor);
        let handle = tokio::spawn(async move {
            let offset = i * 50;
            let mut buf = [0u8; 50];
            cursor_clone
                .read_at(offset, &mut buf)
                .await
                .expect("concurrent read must succeed");
            buf
        });
        handles.push(handle);
    }

    // Wait for all reads to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.expect("task must complete"))
        .collect();

    // Verify all reads returned correct data
    for buf in results {
        assert_eq!(buf, [42u8; 50]);
    }
}

/// Sequential async operations in tokio runtime.
#[tokio::test]
async fn tokio_sequential_operations() {
    let mut cursor = AsyncMemCursor::new();

    // Sequential writes
    for i in 0u8..100 {
        cursor.write_at(i as u64, &[i]).await.expect("write");
    }

    // Sequential reads
    for i in 0u8..100 {
        let mut buf = [0u8; 1];
        cursor.read_at(i as u64, &mut buf).await.expect("read");
        assert_eq!(buf[0], i);
    }
}

/// Async operations with tokio::select! pattern.
#[tokio::test]
async fn tokio_select_pattern() {
    use tokio::time::{Duration, timeout};

    let mut cursor = AsyncMemCursor::new();
    cursor.write_at(0, &[1, 2, 3, 4, 5]).await.expect("write");

    let read_future = async {
        let mut buf = [0u8; 5];
        cursor.read_at(0, &mut buf).await.expect("read");
        buf
    };

    let result = timeout(Duration::from_millis(100), read_future)
        .await
        .expect("read should complete immediately");

    assert_eq!(result, [1, 2, 3, 4, 5]);
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Edge Cases
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Async single byte operations.
#[tokio::test]
async fn async_single_byte_read_write() {
    let mut cursor = AsyncMemCursor::new();

    // Write single bytes
    cursor.write_at(0, &[42]).await.expect("single byte write");

    // Read single byte
    let mut buf = [0u8; 1];
    cursor.read_at(0, &mut buf).await.expect("single byte read");
    assert_eq!(buf[0], 42);
}

/// Async very large single operation.
#[tokio::test]
async fn async_very_large_single_operation() {
    let size = 1_000_000;
    let mut cursor = AsyncMemCursor::new();

    let write_data = vec![0xABu8; size];
    cursor
        .write_at(0, &write_data)
        .await
        .expect("large async write");

    let mut read_data = vec![0u8; size];
    cursor
        .read_at(0, &mut read_data)
        .await
        .expect("large async read");

    assert_eq!(read_data.len(), size);
    assert!(read_data.iter().all(|&b| b == 0xAB));
}

/// Async empty operations on empty cursor.
#[tokio::test]
async fn async_empty_cursor_empty_operations() {
    let cursor = AsyncMemCursor::new();

    // Zero-length read on empty cursor
    let mut buf = [];
    cursor
        .read_at(0, &mut buf)
        .await
        .expect("empty async read on empty cursor");
}

/// Async write and read pattern verification.
#[tokio::test]
async fn async_write_read_pattern_verification() {
    let mut cursor = AsyncMemCursor::new();

    // Write a pattern
    let pattern: Vec<u8> = (0..=255).cycle().take(1000).collect();
    cursor
        .write_at(0, &pattern)
        .await
        .expect("async write pattern");

    // Read back and verify
    let mut buf = vec![0u8; 1000];
    cursor
        .read_at(0, &mut buf)
        .await
        .expect("async read pattern");

    assert_eq!(buf.len(), 1000);
    for (i, &byte) in buf.iter().enumerate() {
        assert_eq!(byte, (i % 256) as u8, "position {}", i);
    }
}

/// AsyncMemCursor is Send + Sync.
#[tokio::test]
async fn async_cursor_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AsyncMemCursor>();
}

/// AsyncMemCursor can be used across await points.
#[tokio::test]
async fn async_cursor_across_await_points() {
    let mut cursor = AsyncMemCursor::new();

    cursor.write_at(0, &[1, 2, 3]).await.expect("write");

    // Yield to the runtime
    tokio::task::yield_now().await;

    // Cursor state is preserved
    let mut buf = [0u8; 3];
    cursor.read_at(0, &mut buf).await.expect("read");
    assert_eq!(buf, [1, 2, 3]);
}
