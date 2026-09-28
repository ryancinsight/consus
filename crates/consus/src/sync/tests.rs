use super::*;
use consus_core::{ByteOrder, Datatype, Shape};
use consus_io::MemCursor;
use core::num::NonZeroUsize;

fn f64_datatype() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(64).expect("non-zero"),
        byte_order: ByteOrder::LittleEndian,
    }
}

#[test]
fn byte_view_reports_zero_copy_state() {
    let borrowed = ByteView::Borrowed(&[1, 2, 3]);
    let owned = ByteView::Owned(vec![1, 2, 3]);

    assert!(borrowed.is_zero_copy());
    assert!(!owned.is_zero_copy());
    assert_eq!(borrowed.as_slice(), &[1, 2, 3]);
    assert_eq!(owned.as_slice(), &[1, 2, 3]);
}

#[test]
fn typed_view_validates_element_multiple() {
    let err = TypedByteView::new(ByteView::Owned(vec![0u8; 3]), f64_datatype())
        .expect_err("3 bytes cannot represent whole f64 elements");

    match err {
        Error::DatatypeMismatch { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn selection_byte_len_for_all_selection() {
    let shape = Shape::fixed(&[2, 2]);
    let bytes =
        selection_byte_len(&f64_datatype(), &shape, &Selection::All).expect("selection bytes");

    assert_eq!(bytes, 32);
}

#[test]
fn read_ranges_reads_expected_bytes() {
    let reader = MemCursor::from_bytes((0u8..16).collect());
    let ranges = [IoRange::new(0, 4), IoRange::new(8, 4)];

    let result = read_ranges(&reader, &ranges).expect("range reads");

    assert_eq!(result.len(), 2);
    assert_eq!(result.get(0), Some(&[0u8, 1, 2, 3][..]));
    assert_eq!(result.get(1), Some(&[8u8, 9, 10, 11][..]));
    assert_eq!(
        result.iter().collect::<Vec<_>>(),
        vec![&[0u8, 1, 2, 3][..], &[8u8, 9, 10, 11][..]]
    );
}

#[cfg(feature = "std")]
#[test]
fn par_read_ranges_matches_sequential_reads() {
    let reader = std::sync::Arc::new(MemCursor::from_bytes((0u8..64).collect()));
    let ranges = [
        IoRange::new(0, 4),
        IoRange::new(8, 4),
        IoRange::new(16, 4),
        IoRange::new(40, 6),
    ];

    let parallel = par_read_ranges(reader.clone(), &ranges).expect("parallel range reads");
    let sequential = read_ranges(reader.as_ref(), &ranges).expect("sequential range reads");

    assert_eq!(parallel, sequential);
    assert_eq!(parallel.len(), ranges.len());
    assert_eq!(parallel.get(3), Some(&[40u8, 41, 42, 43, 44, 45][..]));
}

#[test]
fn parallelism_threshold_disables_small_reads() {
    let policy = Parallelism::new().min_len(1024).partitions(8);

    assert_eq!(policy.partitions_for_len(128), 1);
    assert_eq!(policy.partitions_for_len(4096), 8);
}

#[cfg(feature = "atlas-themis")]
#[test]
fn default_parallelism_matches_themis_topology() {
    let expected = themis::CpuTopology::detect()
        .map(|topology| topology.logical_processors())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .max(1);

    assert_eq!(Parallelism::new().configured_partitions(), expected);
}

#[test]
fn partition_range_covers_total_length() {
    let ranges = partition_range(10, 3).expect("partition");

    assert_eq!(
        ranges,
        vec![IoRange::new(0, 4), IoRange::new(4, 3), IoRange::new(7, 3)]
    );
}
