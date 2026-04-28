use consus_arrow::column_values_to_arrow;
use consus_parquet::ColumnValues;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn bench_bridge_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_values_to_arrow_i32");
    for &n in &[1_000usize, 10_000usize, 100_000usize] {
        let values = ColumnValues::Int32((0..n as i32).collect());
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| column_values_to_arrow(&values));
        });
    }
    group.finish();
}

fn bench_bridge_double(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_values_to_arrow_double");
    for &n in &[1_000usize, 10_000usize, 100_000usize] {
        let values = ColumnValues::Double((0..n).map(|i| i as f64).collect());
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| column_values_to_arrow(&values));
        });
    }
    group.finish();
}

fn bench_bridge_byte_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_values_to_arrow_byte_array");
    for &n in &[1_000usize, 10_000usize] {
        let values =
            ColumnValues::ByteArray((0..n).map(|i: usize| i.to_le_bytes().to_vec()).collect());
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| column_values_to_arrow(&values));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bridge_i32,
    bench_bridge_double,
    bench_bridge_byte_array
);
criterion_main!(benches);
