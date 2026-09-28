//! Before/after evidence for ATLAS-ARCH-008: `read_ranges` returning one
//! flat `RangeBytes` buffer instead of a `Vec<Vec<u8>>` per call.
//!
//! `read_ranges_jagged` is the pre-conversion baseline (one heap allocation
//! per range, matching the removed implementation); `read_ranges` is the
//! current library function.

use consus::{IoRange, read_ranges};
use consus_io::{MemCursor, ReadAt};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Pre-conversion baseline: one heap-allocated `Vec<u8>` per range.
fn read_ranges_jagged(reader: &MemCursor, ranges: &[IoRange]) -> Vec<Vec<u8>> {
    ranges
        .iter()
        .map(|range| {
            let mut buffer = vec![0u8; range.len];
            reader
                .read_at(range.offset, &mut buffer)
                .expect("range read");
            buffer
        })
        .collect()
}

fn range_read_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("consus_read_ranges");
    let total = 4 * 1024 * 1024usize;
    let data: Vec<u8> = (0..total).map(|i| (i & 0xFF) as u8).collect();
    let cursor = MemCursor::from_bytes(data);

    for range_len in [256usize, 4096usize] {
        let stride = range_len * 2;
        let count = total / stride;
        let ranges: Vec<IoRange> = (0..count)
            .map(|i| IoRange::new((i * stride) as u64, range_len))
            .collect();
        group.throughput(Throughput::Bytes((count * range_len) as u64));

        group.bench_with_input(
            BenchmarkId::new("jagged_vec_of_vec", range_len),
            &ranges,
            |b, ranges| {
                b.iter(|| black_box(read_ranges_jagged(black_box(&cursor), black_box(ranges))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("flat_range_bytes", range_len),
            &ranges,
            |b, ranges| {
                b.iter(|| {
                    black_box(read_ranges(black_box(&cursor), black_box(ranges)).expect("read"))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, range_read_benchmark);
criterion_main!(benches);
