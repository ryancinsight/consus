//! Criterion benchmarks for `consus-parquet` write and read throughput.
//!
//! ## Measured operations
//!
//! | Benchmark group          | Parameter | Operation                            |
//! |--------------------------|-----------|--------------------------------------|
//! | `parquet_write_i32`      | row count | `ParquetWriter::write_dataset_bytes` |
//! | `parquet_read_i32`       | row count | `ParquetReader::read_column_chunk`   |
//!
//! ## Analytical basis
//!
//! INT32 PLAIN encoding: each value occupies exactly 4 bytes.
//! Expected write throughput scales linearly with row count.
//! Expected read throughput is dominated by Thrift footer decode + PLAIN
//! value decode, both O(n).

use consus_core::Result;
use consus_parquet::{
    CellValue, ColumnChunkDescriptor, FieldDescriptor, FieldId, ParquetDatasetDescriptor,
    ParquetPhysicalType, ParquetReader, ParquetWriter, RowGroupDescriptor, RowSource, RowValue,
    SchemaDescriptor,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

// ── dataset helpers ──────────────────────────────────────────────────────────

/// Build a single-column required INT32 dataset descriptor for `n` rows.
///
/// Schema: one required INT32 field "col" (FieldId 1).
/// One row group containing all `n` rows.
fn i32_dataset(n: usize) -> ParquetDatasetDescriptor {
    let schema = SchemaDescriptor::new(vec![FieldDescriptor::required(
        FieldId::new(1),
        "col",
        ParquetPhysicalType::Int32,
    )]);
    ParquetDatasetDescriptor::new(
        schema,
        vec![
            RowGroupDescriptor::new(
                n,
                vec![ColumnChunkDescriptor::new(FieldId::new(1), n, 1).unwrap()],
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

// ── RowSource ────────────────────────────────────────────────────────────────

struct I32Rows(Vec<i32>);

impl RowSource for I32Rows {
    fn row_count(&self) -> usize {
        self.0.len()
    }

    fn row(&self, i: usize) -> Result<RowValue<'_>> {
        Ok(RowValue::new(vec![CellValue::Int32(self.0[i])]))
    }
}

// ── benchmark functions ──────────────────────────────────────────────────────

fn bench_write_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_write_i32");
    for &n in &[1_000usize, 10_000usize, 100_000usize] {
        let dataset = i32_dataset(n);
        let rows = I32Rows((0..n as i32).collect());
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                ParquetWriter::new()
                    .write_dataset_bytes(&dataset, &rows)
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_read_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_read_i32");
    for &n in &[1_000usize, 10_000usize, 100_000usize] {
        let dataset = i32_dataset(n);
        let rows = I32Rows((0..n as i32).collect());
        let bytes = ParquetWriter::new()
            .write_dataset_bytes(&dataset, &rows)
            .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let reader = ParquetReader::new(&bytes).unwrap();
                reader.read_column_chunk(0, 0).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_write_i32, bench_read_i32);
criterion_main!(benches);
