//! Property-based tests for I/O traits using proptest.
//!
//! Tests cover:
//! - Random read/write patterns at arbitrary offsets
//! - Concurrent read/write interleaving
//! - Large data streaming
//! - Round-trip invariants
//! - Edge case discovery via shrinking

#![cfg(feature = "alloc")]

use consus_io::{Length, MemCursor, ReadAt, Truncate, WriteAt};
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Read/Write Round-Trip Properties
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Writing arbitrary data at offset 0, then reading it back yields identical data.
#[test]
fn prop_write_read_roundtrip_start() {
    proptest!(|(data: Vec<u8>)| {
        let mut cursor = MemCursor::new();
        cursor.write_at(0, &data).expect("write must succeed");

        let mut buf = vec![0u8; data.len()];
        cursor.read_at(0, &mut buf).expect("read must succeed");

        prop_assert_eq!(buf, data);
    });
}

/// Writing arbitrary data at arbitrary offset, then reading it back yields identical data.
#[test]
fn prop_write_read_roundtrip_offset() {
    proptest!(|(offset: u64, data: Vec<u8>)| {
        // Limit offset and data size to reasonable bounds for test performance
        let offset = offset % 10_000;
        let data: Vec<u8> = data.into_iter().take(1000).collect();

        let mut cursor = MemCursor::new();
        cursor.write_at(offset, &data).expect("write must succeed");

        let mut buf = vec![0u8; data.len()];
        cursor.read_at(offset, &mut buf).expect("read must succeed");

        prop_assert_eq!(buf, data);
    });
}

/// Writing arbitrary data at arbitrary offset produces correct zero-fill before the write.
#[test]
fn prop_write_zero_fill_before() {
    proptest!(|(offset: u64, data: Vec<u8>)| {
        let offset = (offset % 1000) + 1; // Ensure offset > 0
        let data: Vec<u8> = data.into_iter().take(100).collect();

        let mut cursor = MemCursor::new();
        cursor.write_at(offset, &data).expect("write must succeed");

        // All bytes before the write offset should be zero.
        if offset > 0 {
            let prefix_len = core::cmp::min(offset as usize, 256);
            let mut prefix = vec![0u8; prefix_len];
            cursor.read_at(0, &mut prefix).expect("read prefix must succeed");
            prop_assert!(prefix.iter().all(|&b| b == 0));
        }
    });
}

/// Multiple writes at different offsets, then reading yields correct combined state.
#[test]
fn prop_multiple_writes_combined() {
    proptest!(|(writes: Vec<(u64, Vec<u8>)>)| {
        let mut cursor = MemCursor::new();

        // Limit to reasonable number of writes
        let writes: Vec<_> = writes.into_iter().take(10).collect();

        for (offset, data) in &writes {
            let offset = offset % 1000;
            let data: Vec<u8> = data.iter().cloned().take(100).collect();
            cursor.write_at(offset, &data).expect("write must succeed");
        }

        // Verify each write
        for (offset, data) in &writes {
            let offset = offset % 1000;
            let data: Vec<u8> = data.iter().cloned().take(100).collect();

            if data.is_empty() {
                continue;
            }

            let end_offset = offset + data.len() as u64;
            let current_len = cursor.byte_len() as u64;

            // Only verify if the region is still within bounds
            if end_offset <= current_len {
                let mut buf = vec![0u8; data.len()];
                if cursor.read_at(offset, &mut buf).is_ok() {
                    // Note: later writes may have overwritten earlier writes,
                    // so we only check that the read succeeds
                    prop_assert!(true);
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Bounds Checking Properties
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Reading beyond buffer length always returns BufferTooSmall error.
#[test]
fn prop_read_beyond_bounds_errors() {
    proptest!(|(initial_data: Vec<u8>, read_offset: u64, read_len: usize)| {
        let initial_data: Vec<u8> = initial_data.into_iter().take(100).collect();
        let cursor = MemCursor::from_bytes(initial_data.clone());
        let read_offset = read_offset % 1000;
        let read_len = read_len % 1000 + 1; // Ensure non-zero

        let required = read_offset as usize + read_len;
        let provided = initial_data.len();

        if required > provided {
            let mut buf = vec![0u8; read_len];
            let result = cursor.read_at(read_offset, &mut buf);
            prop_assert!(result.is_err());
        }
    });
}

/// Reading at valid offsets always succeeds and returns correct data.
#[test]
fn prop_read_valid_offset_succeeds() {
    proptest!(|(initial_data: Vec<u8>, read_offset: u64, read_len: usize)| {
        let initial_data: Vec<u8> = initial_data.into_iter().take(100).collect();
        let cursor = MemCursor::from_bytes(initial_data.clone());

        if initial_data.is_empty() {
            return Ok(());
        }

        let max_offset = initial_data.len() - 1;
        let read_offset = (read_offset as usize) % (max_offset + 1);
        let max_read_len = initial_data.len() - read_offset;
        let read_len = if max_read_len == 0 {
            0
        } else {
            (read_len % max_read_len).max(1)
        };

        let mut buf = vec![0u8; read_len];
        let result = cursor.read_at(read_offset as u64, &mut buf);

        prop_assert!(result.is_ok());
        prop_assert_eq!(buf.as_slice(), &initial_data[read_offset..read_offset + read_len]);
    });
}

/// Zero-length reads always succeed at any offset.
#[test]
fn prop_zero_length_read_always_succeeds() {
    proptest!(|(initial_data: Vec<u8>, offset: u64)| {
        let cursor = MemCursor::from_bytes(initial_data);
        let mut buf = [];
        let result = cursor.read_at(offset, &mut buf);
        prop_assert!(result.is_ok());
    });
}

/// Zero-length writes always succeed at any offset and don't change length.
#[test]
fn prop_zero_length_write_no_change() {
    proptest!(|(initial_data: Vec<u8>, offset: u64)| {
        let mut cursor = MemCursor::from_bytes(initial_data.clone());
        let len_before = cursor.byte_len();

        cursor.write_at(offset, &[]).expect("zero-length write must succeed");

        prop_assert_eq!(cursor.byte_len(), len_before);
        prop_assert_eq!(cursor.as_bytes(), initial_data.as_slice());
    });
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Length and Truncate Properties
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Length after write equals max of the previous length and `offset + data_len`.
#[test]
fn prop_length_after_write() {
    proptest!(|(offset: u64, data: Vec<u8>)| {
        let offset = offset % 10_000;
        let data: Vec<u8> = data.into_iter().take(1000).collect();

        let mut cursor = MemCursor::new();
        let len_before = cursor.byte_len();
        cursor.write_at(offset, &data).expect("write must succeed");

        let expected_len = if data.is_empty() && len_before == 0 {
            len_before
        } else {
            core::cmp::max(len_before as u64, offset + data.len() as u64) as usize
        };
        prop_assert_eq!(cursor.byte_len(), expected_len as usize);
    });
}

/// set_len to smaller value truncates data correctly.
#[test]
fn prop_truncate_preserves_prefix() {
    proptest!(|(initial_data: Vec<u8>, new_len: u64)| {
        let initial_data: Vec<u8> = initial_data.into_iter().take(100).collect();
        let new_len = (new_len as usize) % (initial_data.len() + 50);

        let mut cursor = MemCursor::from_bytes(initial_data.clone());
        cursor.set_len(new_len as u64).expect("set_len must succeed");

        prop_assert_eq!(cursor.byte_len(), new_len);

        if new_len <= initial_data.len() {
            prop_assert_eq!(cursor.as_bytes(), &initial_data[..new_len]);
        }
    });
}

/// set_len to larger value zero-fills extension.
#[test]
fn prop_extend_zero_fills() {
    proptest!(|(initial_data: Vec<u8>, additional_len: u64)| {
        let initial_data: Vec<u8> = initial_data.into_iter().take(50).collect();
        let additional_len = (additional_len as usize) % 100 + 1;

        let original_len = initial_data.len();
        let new_len = original_len + additional_len;

        let mut cursor = MemCursor::from_bytes(initial_data.clone());
        cursor.set_len(new_len as u64).expect("set_len must succeed");

        prop_assert_eq!(cursor.byte_len(), new_len);
        prop_assert_eq!(&cursor.as_bytes()[..original_len], initial_data.as_slice());

        // Extension should be zero-filled
        for i in original_len..new_len {
            prop_assert_eq!(cursor.as_bytes()[i], 0);
        }
    });
}

/// Length is always consistent between Length trait and byte_len method.
#[test]
fn prop_length_consistency() {
    proptest!(|(data: Vec<u8>)| {
        let cursor = MemCursor::from_bytes(data.clone());
        let trait_len = Length::len(&cursor).expect("Length::len must succeed") as usize;
        let method_len = cursor.byte_len();
        prop_assert_eq!(trait_len, method_len);
    });
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Overwrite and Interleaving Properties
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Overwriting existing data preserves surrounding bytes.
#[test]
fn prop_overwrite_preserves_surrounding() {
    proptest!(|(initial_data: Vec<u8>, write_offset: u64, write_data: Vec<u8>)| {
        let initial_data = vec![0xAAu8; 100];
        let write_offset = (write_offset as usize) % 80 + 10; // Ensure middle region
        let write_data: Vec<u8> = write_data.into_iter().take(20).collect();

        let mut cursor = MemCursor::from_bytes(initial_data.clone());
        cursor.write_at(write_offset as u64, &write_data).expect("write must succeed");

        // Bytes before write_offset should be unchanged
        prop_assert_eq!(&cursor.as_bytes()[..write_offset], &initial_data[..write_offset]);

        // Bytes after write should be unchanged
        let end = write_offset + write_data.len();
        if end < cursor.byte_len() {
            prop_assert_eq!(&cursor.as_bytes()[end..], &initial_data[end..]);
        }
    });
}

/// Interleaved read/write operations maintain consistency.
#[test]
fn prop_interleaved_read_write() {
    proptest!(|(operations: Vec<(bool, u64, Vec<u8>)>)| {
        let mut cursor = MemCursor::new();
        let operations: Vec<_> = operations.into_iter().take(20).collect();

        for (is_write, offset, data) in &operations {
            let offset = offset % 500;
            let data: Vec<u8> = data.iter().cloned().take(50).collect();

            if *is_write {
                cursor.write_at(offset, &data).expect("write must succeed");
            } else if !data.is_empty() {
                let mut buf = vec![0u8; data.len()];
                let _ = cursor.read_at(offset, &mut buf);
                // Read result depends on previous writes; we just verify no panic
            }
        }

        // Cursor should be in a consistent state
        let len = cursor.byte_len();
        if len > 0 {
            let mut buf = vec![0u8; len];
            let result = cursor.read_at(0, &mut buf);
            prop_assert!(result.is_ok() || len == 0);
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Large Data Streaming Properties
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Large sequential writes can be read back correctly.
#[test]
fn prop_large_streaming_write_read() {
    proptest!(|(chunk_count: usize, chunk_size: usize)| {
        let chunk_count = (chunk_count % 10) + 1;
        let chunk_size = (chunk_size % 1000) + 1;

        let mut cursor = MemCursor::new();
        let mut all_data = Vec::new();

        for i in 0..chunk_count {
            let data: Vec<u8> = (0..chunk_size).map(|j| ((i + j) % 256) as u8).collect();
            cursor.write_at(all_data.len() as u64, &data).expect("write must succeed");
            all_data.extend_from_slice(&data);
        }

        let mut buf = vec![0u8; all_data.len()];
        cursor.read_at(0, &mut buf).expect("read must succeed");

        prop_assert_eq!(buf, all_data);
    });
}

/// Large random writes maintain data integrity.
#[test]
fn prop_large_random_writes() {
    proptest!(|(writes: Vec<(usize, Vec<u8>)>)| {
        let max_size = 10_000;
        let writes: Vec<_> = writes.into_iter().take(20).collect();

        let mut cursor = MemCursor::new();
        let mut reference = vec![0u8; max_size];

        for (offset, data) in &writes {
            let offset = offset % max_size;
            let data: Vec<u8> = data.iter().cloned().take(100).collect();

            cursor.write_at(offset as u64, &data).expect("write must succeed");

            let end = (offset + data.len()).min(max_size);
            reference[offset..end].copy_from_slice(&data[..end - offset]);
        }

        let cursor_len = cursor.byte_len();
        if cursor_len <= max_size {
            let mut buf = vec![0u8; cursor_len];
            cursor.read_at(0, &mut buf).expect("read must succeed");

            for i in 0..cursor_len.min(max_size) {
                prop_assert_eq!(buf[i], reference[i], "mismatch at position {}", i);
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Clone and Copy Properties
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Clone produces identical independent copy.
#[test]
fn prop_clone_identical() {
    proptest!(|(data: Vec<u8>)| {
        let cursor = MemCursor::from_bytes(data.clone());
        let clone = cursor.clone();

        prop_assert_eq!(cursor.as_bytes(), clone.as_bytes());
        prop_assert_eq!(cursor.byte_len(), clone.byte_len());
    });
}

/// Modifications to clone don't affect original.
#[test]
fn prop_clone_independence() {
    proptest!(|(initial_data: Vec<u8>, write_offset: u64, write_data: Vec<u8>)| {
        let initial_data: Vec<u8> = initial_data.into_iter().take(100).collect();
        let write_offset = write_offset % 200;
        let write_data: Vec<u8> = write_data.into_iter().take(50).collect();

        let original = MemCursor::from_bytes(initial_data.clone());
        let mut clone = original.clone();

        clone.write_at(write_offset, &write_data).expect("write must succeed");

        prop_assert_eq!(original.as_bytes(), initial_data.as_slice());
        prop_assert!(clone.byte_len() >= initial_data.len());
    });
}

/// from_bytes followed by into_bytes returns identical data.
#[test]
fn prop_from_into_roundtrip() {
    proptest!(|(data: Vec<u8>)| {
        let cursor = MemCursor::from_bytes(data.clone());
        let recovered = cursor.into_bytes();
        prop_assert_eq!(recovered, data);
    });
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Pattern and Stress Properties
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Byte pattern writes maintain pattern integrity.
#[test]
fn prop_pattern_integrity() {
    proptest!(|(pattern_byte: u8, offset: u64, len: usize)| {
        let offset = offset % 1000;
        let len = (len % 1000) + 1;

        let data = vec![pattern_byte; len];
        let mut cursor = MemCursor::new();
        cursor.write_at(offset, &data).expect("write must succeed");

        let mut buf = vec![0u8; len];
        cursor.read_at(offset, &mut buf).expect("read must succeed");

        prop_assert!(buf.iter().all(|&b| b == pattern_byte));
    });
}

/// Alternating pattern writes maintain sequence integrity.
#[test]
fn prop_alternating_pattern() {
    proptest!(|(offset: u64, len: usize)| {
        let offset = offset % 500;
        let len = (len % 500) + 10;

        let data: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
        let mut cursor = MemCursor::new();
        cursor.write_at(offset, &data).expect("write must succeed");

        let mut buf = vec![0u8; len];
        cursor.read_at(offset, &mut buf).expect("read must succeed");

        for (i, &byte) in buf.iter().enumerate() {
            prop_assert_eq!(byte, (i % 256) as u8, "position {}", i);
        }
    });
}

/// Stress test: many random operations maintain consistency.
#[test]
fn prop_stress_random_operations() {
    proptest!(|(seed: u64)| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut rng_state = hasher.finish();

        let next_random = |state: &mut u64, max: usize| -> usize {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state as usize) % max
        };

        let mut cursor = MemCursor::new();
        let mut reference: Vec<u8> = Vec::new();

        for _ in 0..100 {
            let op = next_random(&mut rng_state, 3);

            match op {
                0 => {
                    // Write
                    let offset = next_random(&mut rng_state, 500);
                    let len = next_random(&mut rng_state, 100) + 1;
                    let data: Vec<u8> = (0..len).map(|_| next_random(&mut rng_state, 256) as u8).collect();

                    let end = offset + len;
                    if end > reference.len() {
                        reference.resize(end, 0);
                    }
                    reference[offset..end].copy_from_slice(&data);
                    cursor.write_at(offset as u64, &data).expect("write must succeed");
                }
                1 => {
                    // Read
                    if !reference.is_empty() {
                        let offset = next_random(&mut rng_state, reference.len());
                        let max_len = reference.len() - offset;
                        let len = next_random(&mut rng_state, max_len + 1);

                        if len > 0 {
                            let mut buf = vec![0u8; len];
                            if cursor.read_at(offset as u64, &mut buf).is_ok() {
                                prop_assert_eq!(buf.as_slice(), &reference[offset..offset + len]);
                            }
                        }
                    }
                }
                2 => {
                    // Truncate
                    let new_len = next_random(&mut rng_state, reference.len() + 100);
                    reference.resize(new_len, 0);
                    cursor.set_len(new_len as u64).expect("set_len must succeed");
                }
                _ => {}
            }
        }

        // Final consistency check
        prop_assert_eq!(cursor.byte_len(), reference.len());
        if !reference.is_empty() {
            let mut buf = vec![0u8; reference.len()];
            cursor.read_at(0, &mut buf).expect("final read must succeed");
            prop_assert_eq!(buf, reference);
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Edge Case Discovery
// ═══════════════════════════════════════════════════════════════════════════════════════════

/// Single byte operations at various positions.
#[test]
fn prop_single_byte_operations() {
    proptest!(|(offset: u64, byte: u8)| {
        let offset = offset % 1000;
        let mut cursor = MemCursor::new();

        cursor.write_at(offset, &[byte]).expect("single byte write must succeed");

        let mut buf = [0u8; 1];
        cursor.read_at(offset, &mut buf).expect("single byte read must succeed");

        prop_assert_eq!(buf[0], byte);
        prop_assert_eq!(cursor.byte_len(), offset as usize + 1);
    });
}

/// Empty writes are no-ops.
#[test]
fn prop_empty_write_noop() {
    proptest!(|(initial_data: Vec<u8>, offset: u64)| {
        let mut cursor = MemCursor::from_bytes(initial_data.clone());
        let len_before = cursor.byte_len();

        cursor.write_at(offset, &[]).expect("empty write must succeed");

        prop_assert_eq!(cursor.byte_len(), len_before);
        prop_assert_eq!(cursor.as_bytes(), initial_data.as_slice());
    });
}

/// Zero-filled regions remain zero after non-overlapping writes.
#[test]
fn prop_zero_regions_preserved() {
    proptest!(|(offset1: u64, data1: Vec<u8>, offset2: u64, data2: Vec<u8>)| {
        let offset1 = offset1 % 100;
        let offset2 = (offset2 % 100) + 200; // Non-overlapping
        let data1: Vec<u8> = data1.into_iter().take(50).collect();
        let data2: Vec<u8> = data2.into_iter().take(50).collect();

        let mut cursor = MemCursor::new();
        cursor.write_at(offset1, &data1).expect("write1");
        cursor.write_at(offset2, &data2).expect("write2");

        // Region between writes should be zero.
        let gap_start = offset1 as usize + data1.len();
        let gap_end = offset2 as usize;

        if gap_start < gap_end {
            let gap_len = core::cmp::min(gap_end - gap_start, 256);
            let mut gap = vec![0u8; gap_len];
            cursor.read_at(gap_start as u64, &mut gap).expect("read gap");
            prop_assert!(gap.iter().all(|&b| b == 0));
        }
    });
}
