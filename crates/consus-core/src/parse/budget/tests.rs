// Test code is exempt from the production panic policy: an `unwrap` here
// asserts, it does not ship.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use super::*;

#[test]
fn plausible_byte_length_passes_through() {
    let budget = ParseBudget::default();
    assert_eq!(budget.checked_bytes(4096, "node").unwrap(), 4096);
    assert_eq!(
        budget
            .checked_bytes(u64::try_from(DEFAULT_MAX_ALLOC_BYTES).unwrap(), "node")
            .unwrap(),
        DEFAULT_MAX_ALLOC_BYTES
    );
}

#[test]
fn hostile_byte_length_is_a_typed_error() {
    let budget = ParseBudget::default();
    let error = budget.checked_bytes(u64::MAX, "node").unwrap_err();
    assert!(matches!(
        error,
        Error::ResourceLimit {
            what: "node",
            requested: u64::MAX,
            ..
        }
    ));
}

/// The element footprint, not just the count, is bounded: 2^20 elements of
/// 4 KiB each is a modest count and a 4 GiB allocation.
#[test]
fn element_footprint_is_bounded_independently_of_count() {
    let budget = ParseBudget::default();
    assert!(budget.checked_elements(1_048_576, 4096, "records").is_err());
    assert_eq!(
        budget.checked_elements(1024, 4096, "records").unwrap(),
        1024
    );
}

#[test]
fn capacity_hint_clamps_instead_of_rejecting() {
    let budget = ParseBudget::default();
    assert_eq!(budget.capacity_hint(10, 24), 10);
    assert_eq!(
        budget.capacity_hint(u64::MAX, 24),
        DEFAULT_MAX_ALLOC_BYTES / 24
    );
    // A zero element size must not divide by zero.
    assert_eq!(budget.capacity_hint(u64::MAX, 0), DEFAULT_MAX_ALLOC_BYTES);
}

#[test]
fn descend_stops_at_the_depth_ceiling() {
    let budget = ParseBudget::new(1024, 1024, 3);
    assert_eq!(budget.descend(0, "nesting").unwrap(), 1);
    assert_eq!(budget.descend(2, "nesting").unwrap(), 3);
    let error = budget.descend(3, "nesting").unwrap_err();
    assert!(matches!(
        error,
        Error::ResourceLimit {
            what: "nesting",
            requested: 4,
            limit: 3
        }
    ));
}

#[cfg(feature = "alloc")]
#[test]
fn zeroed_returns_an_error_rather_than_aborting() {
    let budget = ParseBudget::default();
    assert_eq!(budget.zeroed(8, "buffer").unwrap(), vec![0u8; 8]);
    assert!(budget.zeroed(u64::MAX, "buffer").is_err());
}

#[cfg(feature = "alloc")]
#[test]
fn vec_with_capacity_bounds_by_element_footprint() {
    let budget = ParseBudget::default();
    let reserved: Vec<u64> = budget.vec_with_capacity(16, "records").unwrap();
    assert!(reserved.capacity() >= 16);
    assert!(
        budget
            .vec_with_capacity::<u64>(u64::MAX, "records")
            .is_err()
    );
}
#[cfg(feature = "std")]
#[test]
fn read_bounded_rejects_output_above_budget() {
    let budget = ParseBudget::new(4, 4, 1);
    let mut reader = std::io::Cursor::new([1u8, 2, 3, 4, 5]);
    let error = budget
        .read_bounded(&mut reader, 0, "decompressed output")
        .unwrap_err();
    assert!(matches!(error, Error::ResourceLimit { .. }));
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_clamps_a_hint_above_the_budget() {
    let budget = ParseBudget::new(16, 4, 1);
    let mut reader = std::io::Cursor::new([1u8, 2, 3, 4, 5]);
    let decoded = budget
        .read_bounded(&mut reader, 1 << 20, "decompressed output")
        .unwrap();
    assert_eq!(decoded, [1, 2, 3, 4, 5]);
    assert!(decoded.capacity() <= 16);
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_refuses_data_above_the_budget_whatever_the_hint() {
    let budget = ParseBudget::new(4, 4, 1);
    for hint in [0, 4, 1 << 20, usize::MAX] {
        let mut reader = std::io::Cursor::new([1u8, 2, 3, 4, 5]);
        let error = budget
            .read_bounded(&mut reader, hint, "decompressed output")
            .unwrap_err();
        assert!(
            matches!(
                error,
                Error::ResourceLimit {
                    what: "decompressed output",
                    requested: 5,
                    limit: 4
                }
            ),
            "hint {hint}: {error:?}"
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_decodes_under_a_zero_hint_and_one_equal_to_the_budget() {
    let budget = ParseBudget::new(8, 4, 1);
    for hint in [0, 8] {
        let mut reader = std::io::Cursor::new([1u8, 2, 3, 4]);
        let decoded = budget
            .read_bounded(&mut reader, hint, "decompressed output")
            .unwrap();
        assert_eq!(decoded, [1, 2, 3, 4], "hint {hint}");
    }
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_decodes_a_stream_exactly_at_the_budget() {
    let budget = ParseBudget::new(8, 4, 1);
    for hint in [0, 3, 8, 1 << 20] {
        let mut reader = std::io::Cursor::new([1u8, 2, 3, 4, 5, 6, 7, 8]);
        let decoded = budget
            .read_bounded(&mut reader, hint, "decompressed output")
            .unwrap();
        assert_eq!(decoded, [1, 2, 3, 4, 5, 6, 7, 8], "hint {hint}");
    }
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_checks_growth_after_the_first_read() {
    struct OneByteReads(std::vec::IntoIter<u8>);
    impl std::io::Read for OneByteReads {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            match (self.0.next(), buf.first_mut()) {
                (Some(byte), Some(slot)) => {
                    *slot = byte;
                    Ok(1)
                }
                _ => Ok(0),
            }
        }
    }
    let budget = ParseBudget::new(4, 4, 1);
    let mut reader = OneByteReads(vec![1u8, 2, 3, 4, 5, 6].into_iter());
    let error = budget
        .read_bounded(&mut reader, 0, "decompressed output")
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::ResourceLimit {
                what: "decompressed output",
                requested: 5,
                limit: 4
            }
        ),
        "{error:?}"
    );
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_reports_the_length_read_when_the_budget_is_first_crossed() {
    let budget = ParseBudget::new(4, 4, 1);
    let mut reader = std::io::Cursor::new(vec![0u8; 1 << 20]);
    let error = budget
        .read_bounded(&mut reader, 0, "decompressed output")
        .unwrap_err();
    let Error::ResourceLimit {
        requested, limit, ..
    } = error
    else {
        panic!("expected ResourceLimit, got {error:?}");
    };
    assert_eq!(limit, 4);
    assert!((5..=4 + 8192).contains(&requested), "requested {requested}");
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_reserves_the_budget_for_a_hint_above_it() {
    let budget = ParseBudget::new(16, 4, 1);
    let mut reader = std::io::Cursor::new([1u8, 2, 3]);
    let decoded = budget
        .read_bounded(&mut reader, 1 << 20, "decompressed output")
        .unwrap();
    assert_eq!(decoded, [1, 2, 3]);
    assert_eq!(decoded.capacity(), 16);
}

#[cfg(feature = "std")]
#[test]
fn read_bounded_grows_from_the_data_when_the_hint_cannot_be_reserved() {
    let budget = ParseBudget::new(usize::MAX, 4, 1);
    let mut reader = std::io::Cursor::new([1u8, 2, 3, 4, 5]);
    let decoded = budget
        .read_bounded(&mut reader, usize::MAX, "decompressed output")
        .unwrap();
    assert_eq!(decoded, [1, 2, 3, 4, 5]);
}
