//! Extended writer tests: multi-row-group splitting, SNAPPY/ZSTD/LZ4 roundtrips,
//! optional/repeated flat column roundtrips, and proptest suites.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::dataset::{ColumnChunkDescriptor, ParquetDatasetDescriptor, RowGroupDescriptor};
use crate::encoding::column::{ColumnValues, ColumnValuesWithLevels};
use crate::reader::{ColumnPageDecoder, ParquetReader};
use crate::schema::field::{FieldDescriptor, FieldId, SchemaDescriptor};
use crate::schema::logical::Repetition;

use crate::schema::physical::ParquetPhysicalType;
use crate::writer::{CellValue, ParquetWriter, RowSource, RowValue};

use super::encode_bool_column_plain;
use crate::encoding::plain::decode_plain_boolean;

// ── helpers ─────────────────────────────────────────────────────────────────

/// Build a single-column INT32 required dataset with `n` rows in one row group.
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

/// Concatenate INT32 values from all row groups in column 0.
fn read_all_i32(bytes: &[u8]) -> Vec<i32> {
    let reader = ParquetReader::new(bytes).unwrap();
    let num_groups = reader.metadata().row_groups.len();
    let mut all: Vec<i32> = Vec::new();
    let mut rg = 0;
    while rg < num_groups {
        if let ColumnValues::Int32(v) = reader.read_column_chunk(rg, 0).unwrap() {
            all.extend(v);
        }
        rg += 1;
    }
    all
}

/// Build a single-column dataset with an arbitrary `FieldDescriptor` and `n` rows.
fn single_column_dataset(field: FieldDescriptor, n: usize) -> ParquetDatasetDescriptor {
    let fid = field.id();
    let schema = SchemaDescriptor::new(vec![field]);
    ParquetDatasetDescriptor::new(
        schema,
        vec![
            RowGroupDescriptor::new(n, vec![ColumnChunkDescriptor::new(fid, n, 1).unwrap()])
                .unwrap(),
        ],
    )
    .unwrap()
}

// ── RowSource implementations ────────────────────────────────────────────────

struct I32Rows(Vec<i32>);

impl RowSource for I32Rows {
    fn row_count(&self) -> usize {
        self.0.len()
    }
    fn row(&self, idx: usize) -> consus_core::Result<RowValue<'_>> {
        Ok(RowValue::new(vec![CellValue::Int32(self.0[idx])]))
    }
}

/// Row source for an optional INT32 column: `None` → `CellValue::Null`.
struct OptionalI32Rows(Vec<Option<i32>>);

impl RowSource for OptionalI32Rows {
    fn row_count(&self) -> usize {
        self.0.len()
    }
    fn row(&self, idx: usize) -> consus_core::Result<RowValue<'_>> {
        let cell = match self.0[idx] {
            Some(v) => CellValue::Int32(v),
            None => CellValue::Null,
        };
        Ok(RowValue::new(vec![cell]))
    }
}

/// Row source for a repeated INT32 column: each row is `CellValue::Repeated`.
struct RepeatedI32Rows(Vec<Vec<i32>>);

impl RowSource for RepeatedI32Rows {
    fn row_count(&self) -> usize {
        self.0.len()
    }
    fn row(&self, idx: usize) -> consus_core::Result<RowValue<'_>> {
        let items: Vec<CellValue<'_>> = self.0[idx].iter().map(|&v| CellValue::Int32(v)).collect();
        Ok(RowValue::new(vec![CellValue::Repeated(items)]))
    }
}

// ── optional flat column roundtrip tests ─────────────────────────────────────

/// Invariant: write [Some(1), None, Some(3)] to an optional INT32 column,
/// read back with levels; non-null values = [1, 3], def_levels = [1, 0, 1].
#[test]
fn optional_i32_null_roundtrip_non_null_values_and_def_levels() {
    let field = FieldDescriptor::optional(FieldId::new(1), "x", ParquetPhysicalType::Int32, None);
    let dataset = single_column_dataset(field, 3);
    let rows = OptionalI32Rows(vec![Some(1), None, Some(3)]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let result: ColumnValuesWithLevels = reader.read_column_chunk_with_levels(0, 0).unwrap();

    // Non-null values: [1, 3]
    assert_eq!(result.non_null_count(), 2);
    assert!(
        matches!(&result.values, ColumnValues::Int32(v) if *v == alloc::vec![1i32, 3]),
        "expected Int32([1, 3]), got {:?}",
        result.values
    );

    // Definition levels: [1, 0, 1] (defined, null, defined)
    assert_eq!(result.max_def_level, 1);
    assert_eq!(result.def_levels, alloc::vec![1i32, 0, 1]);

    // Repetition levels: empty (max_rep = 0)
    assert_eq!(result.max_rep_level, 0);
    assert!(result.rep_levels.is_empty());
}

/// All-null optional INT32: write [None, None] → non-null values = [], def_levels = [0, 0].
#[test]
fn optional_i32_all_null_roundtrip() {
    let field = FieldDescriptor::optional(FieldId::new(1), "x", ParquetPhysicalType::Int32, None);
    let dataset = single_column_dataset(field, 2);
    let rows = OptionalI32Rows(vec![None, None]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let result = reader.read_column_chunk_with_levels(0, 0).unwrap();

    assert_eq!(result.non_null_count(), 0);
    assert_eq!(result.total_count(), 2);
    assert_eq!(result.def_levels, alloc::vec![0i32, 0]);
    assert!(result.rep_levels.is_empty());
}

/// All-defined optional INT32: [Some(7), Some(-1)] → behaves like required column.
#[test]
fn optional_i32_no_null_roundtrip() {
    let field = FieldDescriptor::optional(FieldId::new(1), "x", ParquetPhysicalType::Int32, None);
    let dataset = single_column_dataset(field, 2);
    let rows = OptionalI32Rows(vec![Some(7), Some(-1)]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let result = reader.read_column_chunk_with_levels(0, 0).unwrap();

    assert_eq!(result.non_null_count(), 2);
    assert!(matches!(&result.values, ColumnValues::Int32(v) if *v == alloc::vec![7i32, -1]));
    assert_eq!(result.def_levels, alloc::vec![1i32, 1]);
}

/// `read_column_chunk` (non-levels API) on an optional column returns only non-null values.
#[test]
fn optional_i32_flat_read_returns_non_null_only() {
    let field = FieldDescriptor::optional(FieldId::new(1), "x", ParquetPhysicalType::Int32, None);
    let dataset = single_column_dataset(field, 4);
    let rows = OptionalI32Rows(vec![Some(10), None, Some(20), None]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let v = reader.read_column_chunk(0, 0).unwrap();
    assert!(
        matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![10i32, 20]),
        "expected Int32([10, 20]), got {:?}",
        v
    );
}

// ── repeated flat column roundtrip tests ─────────────────────────────────────

/// Invariant: write [[10, 20], [], [30]] to a repeated INT32 column.
/// Non-null values = [10, 20, 30]; rep_levels = [0, 1, 0, 0]; def_levels = [1, 1, 0, 1].
#[test]
fn repeated_i32_roundtrip_values_and_levels() {
    let field = FieldDescriptor::repeated(FieldId::new(1), "xs", ParquetPhysicalType::Int32);
    let dataset = single_column_dataset(field, 3);
    let rows = RepeatedI32Rows(vec![vec![10, 20], vec![], vec![30]]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let result = reader.read_column_chunk_with_levels(0, 0).unwrap();

    // Non-null leaf values
    assert!(
        matches!(&result.values, ColumnValues::Int32(v) if *v == alloc::vec![10i32, 20, 30]),
        "expected Int32([10, 20, 30]), got {:?}",
        result.values
    );

    // Total count = 4 logical entries (2 + 1 empty-list + 1)
    assert_eq!(result.total_count(), 4);

    // Rep levels: [0, 1, 0, 0]
    // Row1: first=0, second rep=1; Row2: empty=0; Row3: first=0
    assert_eq!(result.max_rep_level, 1);
    assert_eq!(result.rep_levels, alloc::vec![0i32, 1, 0, 0]);

    // Def levels: [1, 1, 0, 1]
    assert_eq!(result.max_def_level, 1);
    assert_eq!(result.def_levels, alloc::vec![1i32, 1, 0, 1]);
}

/// Single-item repeated column across multiple rows: [[5], [6], [7]].
#[test]
fn repeated_i32_single_item_per_row_roundtrip() {
    let field = FieldDescriptor::repeated(FieldId::new(1), "xs", ParquetPhysicalType::Int32);
    let dataset = single_column_dataset(field, 3);
    let rows = RepeatedI32Rows(vec![vec![5], vec![6], vec![7]]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let result = reader.read_column_chunk_with_levels(0, 0).unwrap();

    assert!(matches!(&result.values, ColumnValues::Int32(v) if *v == alloc::vec![5i32, 6, 7]));
    // All first occurrences → rep_levels all 0
    assert_eq!(result.rep_levels, alloc::vec![0i32, 0, 0]);
    assert_eq!(result.def_levels, alloc::vec![1i32, 1, 1]);
}

/// `read_column_chunk` (non-levels API) on a repeated column returns only defined values.
#[test]
fn repeated_i32_flat_read_returns_defined_only() {
    let field = FieldDescriptor::repeated(FieldId::new(1), "xs", ParquetPhysicalType::Int32);
    let dataset = single_column_dataset(field, 2);
    let rows = RepeatedI32Rows(vec![vec![1, 2, 3], vec![]]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let v = reader.read_column_chunk(0, 0).unwrap();
    assert!(
        matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![1i32, 2, 3]),
        "expected Int32([1, 2, 3]), got {:?}",
        v
    );
}

// ── multi-row-group value-semantic tests ─────────────────────────────────────

#[test]
fn multi_row_group_even_split_ten_values_two_groups() {
    // 10 values, row_group_size=5 → 2 groups of 5 each.
    let values: Vec<i32> = (1..=10).collect();
    let dataset = i32_dataset(10);
    let bytes = ParquetWriter::new()
        .with_row_group_size(5)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(
        reader.metadata().row_groups.len(),
        2,
        "expected 2 row groups"
    );
    assert_eq!(reader.metadata().row_groups[0].num_rows, 5);
    assert_eq!(reader.metadata().row_groups[1].num_rows, 5);

    let all = read_all_i32(&bytes);
    assert_eq!(all, values);
}

#[test]
fn multi_row_group_uneven_split_seven_values_three_groups() {
    // 7 values, row_group_size=3 → groups [3, 3, 1].
    let values: Vec<i32> = (10..=16).collect();
    let dataset = i32_dataset(7);
    let bytes = ParquetWriter::new()
        .with_row_group_size(3)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(
        reader.metadata().row_groups.len(),
        3,
        "expected 3 row groups"
    );
    assert_eq!(reader.metadata().row_groups[0].num_rows, 3);
    assert_eq!(reader.metadata().row_groups[1].num_rows, 3);
    assert_eq!(reader.metadata().row_groups[2].num_rows, 1);

    let all = read_all_i32(&bytes);
    assert_eq!(all, values);
}

#[test]
fn multi_row_group_size_larger_than_row_count_gives_one_group() {
    // row_group_size=100, 3 values → 1 group.
    let values = vec![7i32, -3, 42];
    let dataset = i32_dataset(3);
    let bytes = ParquetWriter::new()
        .with_row_group_size(100)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().row_groups.len(), 1);
    assert_eq!(reader.metadata().row_groups[0].num_rows, 3);
    assert_eq!(read_all_i32(&bytes), values);
}

#[test]
fn multi_row_group_exact_multiple_of_group_size() {
    // 6 values, row_group_size=2 → 3 groups of 2.
    let values: Vec<i32> = vec![100, 200, 300, 400, 500, 600];
    let dataset = i32_dataset(6);
    let bytes = ParquetWriter::new()
        .with_row_group_size(2)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().row_groups.len(), 3);
    assert_eq!(reader.metadata().row_groups[0].num_rows, 2);
    assert_eq!(reader.metadata().row_groups[1].num_rows, 2);
    assert_eq!(reader.metadata().row_groups[2].num_rows, 2);
    assert_eq!(read_all_i32(&bytes), values);
}

#[test]
fn default_writer_produces_single_row_group() {
    // Default (no row_group_size) → 1 group containing all rows.
    let values: Vec<i32> = vec![1, 2, 3, 4, 5];
    let dataset = i32_dataset(5);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().row_groups.len(), 1);
    assert_eq!(reader.metadata().num_rows, 5);
    assert_eq!(read_all_i32(&bytes), values);
}

#[test]
fn with_row_group_size_zero_produces_single_group() {
    // row_group_size=0 is treated as unlimited → 1 group.
    let values: Vec<i32> = vec![9, 8, 7];
    let dataset = i32_dataset(3);
    let bytes = ParquetWriter::new()
        .with_row_group_size(0)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().row_groups.len(), 1);
    assert_eq!(read_all_i32(&bytes), values);
}

// ── SNAPPY / ZSTD / LZ4 compressed writer roundtrip tests ───────────────────

#[cfg(feature = "snappy")]
#[test]
fn writer_snappy_roundtrip_i32_three_values() {
    use crate::encoding::compression::CompressionCodec;
    let values = vec![42i32, -1, 0];
    let dataset = i32_dataset(3);
    let bytes = ParquetWriter::new()
        .with_compression(CompressionCodec::Snappy)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();
    assert_eq!(read_all_i32(&bytes), values);
}

#[cfg(feature = "zstd")]
#[test]
fn writer_zstd_roundtrip_i32_three_values() {
    use crate::encoding::compression::CompressionCodec;
    let values = vec![1000i32, -9999, i32::MAX];
    let dataset = i32_dataset(3);
    let bytes = ParquetWriter::new()
        .with_compression(CompressionCodec::Zstd)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();
    assert_eq!(read_all_i32(&bytes), values);
}

#[cfg(feature = "lz4")]
#[test]
fn writer_lz4_raw_roundtrip_i32_three_values() {
    use crate::encoding::compression::CompressionCodec;
    let values = vec![55i32, -55, 0];
    let dataset = i32_dataset(3);
    let bytes = ParquetWriter::new()
        .with_compression(CompressionCodec::Lz4Raw)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();
    assert_eq!(read_all_i32(&bytes), values);
}

#[cfg(feature = "lz4")]
#[test]
fn writer_lz4_roundtrip_i32_three_values() {
    use crate::encoding::compression::CompressionCodec;
    let values = vec![1i32, 2, 3];
    let dataset = i32_dataset(3);
    let bytes = ParquetWriter::new()
        .with_compression(CompressionCodec::Lz4)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();
    assert_eq!(read_all_i32(&bytes), values);
}

// ── Optional i32 proptest ────────────────────────────────────────────────────

proptest! {
    /// For any vector of optional i32 values containing at least one element,
    /// write → read back with levels recovers exactly the non-null values and
    /// produces def_levels in {0,1} with 1 at every non-null position.
    ///
    /// Invariant:
    ///   let non_null = values.iter().filter_map(|x| *x).collect::<Vec<_>>();
    ///   let expected_def = values.iter().map(|x| if x.is_some() { 1 } else { 0 }).collect();
    ///   read_column_chunk_with_levels(write(values)) == (non_null, expected_def)
    #[test]
    fn prop_optional_i32_roundtrip(
        values in proptest::collection::vec(
            proptest::option::of(i32::MIN..=i32::MAX),
            1..=30,
        )
    ) {
        let n = values.len();
        let field = FieldDescriptor::optional(
            FieldId::new(1),
            "x",
            ParquetPhysicalType::Int32,
            None,
        );
        let dataset = single_column_dataset(field, n);
        let rows = OptionalI32Rows(values.clone());
        let bytes = ParquetWriter::new()
            .write_dataset_bytes(&dataset, &rows)
            .unwrap();

        let reader = ParquetReader::new(&bytes).unwrap();
        let result = reader.read_column_chunk_with_levels(0, 0).unwrap();

        let expected_non_null: Vec<i32> = values.iter().filter_map(|x| *x).collect();
        let expected_def: Vec<i32> = values.iter().map(|x| if x.is_some() { 1 } else { 0 }).collect();

        prop_assert!(
            matches!(&result.values, ColumnValues::Int32(v) if *v == expected_non_null),
            "non-null values mismatch: expected {:?}, got {:?}",
            expected_non_null,
            result.values,
        );
        prop_assert_eq!(&result.def_levels, &expected_def);
        prop_assert!(result.rep_levels.is_empty());
    }
}

// ── Boolean bit-packing proptest ─────────────────────────────────────────────

proptest! {
    /// For any non-empty boolean vector, encode_bool_column_plain followed by
    /// decode_plain_boolean recovers the original vector exactly.
    ///
    /// Invariant: ∀ bools: Vec<bool>, |bools| ≥ 1:
    ///   decode_plain_boolean(encode_bool_column_plain(bools), |bools|) == bools
    #[test]
    fn prop_bool_encode_decode_roundtrip(
        bools in proptest::collection::vec(proptest::bool::ANY, 1..=64)
    ) {
        let n = bools.len();
        let encoded = encode_bool_column_plain(&bools);
        let decoded = decode_plain_boolean(&encoded, n).unwrap();
        prop_assert_eq!(decoded, bools);
    }
}

// ── Multi-row-group writer proptest ──────────────────────────────────────────

proptest! {
    /// For any non-empty i32 vector and any row_group_size ≥ 1, writing with
    /// multi-row-group splitting and reading back yields the original values.
    ///
    /// Invariant: ∀ values: Vec<i32>, m ≥ 1:
    ///   read_all_i32(write(values, row_group_size=m)) == values
    #[test]
    fn prop_multi_row_group_i32_roundtrip(
        values in proptest::collection::vec(i32::MIN..=i32::MAX, 1..=50),
        row_group_size in 1usize..=20,
    ) {
        let n = values.len();
        let dataset = i32_dataset(n);
        let bytes = ParquetWriter::new()
            .with_row_group_size(row_group_size)
            .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
            .unwrap();
        let all = read_all_i32(&bytes);
        prop_assert_eq!(all, values);
    }
}

// ── Dremel nested-column roundtrip tests ─────────────────────────────────────

/// Invariant: write 3 rows of (x: i32, y: i64) into a required group field "point";
/// the writer lowers this to two leaf column chunks ("point/x" and "point/y") each
/// with max_rep=0, max_def=0.  Both are decoded as plain required columns.
///
/// ParquetReader::new fails for this schema because the file has 2 column chunks
/// but the schema has 1 top-level field; we parse the footer manually instead.
#[test]
fn nested_required_group_two_leaves_roundtrip() {
    use crate::encoding::compression::CompressionCodec;

    let group_field = FieldDescriptor::group(
        FieldId::new(1),
        "point",
        Repetition::Required,
        vec![
            FieldDescriptor::required(FieldId::new(2), "x", ParquetPhysicalType::Int32),
            FieldDescriptor::required(
                FieldId::new(3),
                "y",
                crate::schema::physical::ParquetPhysicalType::Int64,
            ),
        ],
    );
    let dataset = single_column_dataset(group_field, 3);

    struct PointRows(alloc::vec::Vec<(i32, i64)>);
    impl RowSource for PointRows {
        fn row_count(&self) -> usize {
            self.0.len()
        }
        fn row(&self, idx: usize) -> consus_core::Result<RowValue<'_>> {
            let (x, y) = self.0[idx];
            Ok(RowValue::new(vec![CellValue::Group(vec![
                CellValue::Int32(x),
                CellValue::Int64(y),
            ])]))
        }
    }

    let rows = PointRows(vec![(1, 10), (2, 20), (3, 30)]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    // Parse footer manually: ParquetReader::new would fail because the written
    // file has 2 leaf column chunks but the schema has only 1 top-level field.
    let prelude = crate::wire::validate_footer_prelude(&bytes).unwrap();
    let meta = crate::wire::metadata::decode_file_metadata(&bytes, &prelude).unwrap();
    assert_eq!(meta.row_groups.len(), 1);
    assert_eq!(
        meta.row_groups[0].columns.len(),
        2,
        "expected 2 leaf column chunks (point/x and point/y)"
    );

    // Leaf 0: "point/x" — required Int32, max_rep=0, max_def=0
    let chunk_x = meta.row_groups[0].columns[0].meta_data.as_ref().unwrap();
    let sx = chunk_x.data_page_offset as usize;
    let ex = sx + chunk_x.total_compressed_size as usize;
    let mut dec_x = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let rx = dec_x.decode_pages_with_levels(&bytes[sx..ex]).unwrap();

    // Leaf 1: "point/y" — required Int64, max_rep=0, max_def=0
    let chunk_y = meta.row_groups[0].columns[1].meta_data.as_ref().unwrap();
    let sy = chunk_y.data_page_offset as usize;
    let ey = sy + chunk_y.total_compressed_size as usize;
    let mut dec_y = ColumnPageDecoder::new(
        crate::schema::physical::ParquetPhysicalType::Int64,
        CompressionCodec::Uncompressed,
        0,
        0,
    );
    let ry = dec_y.decode_pages_with_levels(&bytes[sy..ey]).unwrap();

    // x values: [1, 2, 3] — no levels for required columns
    assert!(
        matches!(&rx.values, ColumnValues::Int32(v) if *v == alloc::vec![1i32, 2, 3]),
        "expected Int32([1, 2, 3]), got {:?}",
        rx.values
    );
    assert!(
        rx.def_levels.is_empty(),
        "required column must have no def levels"
    );
    assert!(
        rx.rep_levels.is_empty(),
        "required column must have no rep levels"
    );

    // y values: [10, 20, 30]
    assert!(
        matches!(&ry.values, crate::encoding::column::ColumnValues::Int64(v) if *v == alloc::vec![10i64, 20, 30]),
        "expected Int64([10, 20, 30]), got {:?}",
        ry.values
    );
    assert!(
        ry.def_levels.is_empty(),
        "required column must have no def levels"
    );
    assert!(
        ry.rep_levels.is_empty(),
        "required column must have no rep levels"
    );
}

/// Invariant: write [Some(v=5), None, Some(v=7)] to schema
/// `message { optional group payload { required int32 v; } }`.
///
/// The leaf "payload/v" has max_rep=0, max_def=1 (optional group adds 1).
/// ParquetReader::new succeeds (1 leaf chunk == 1 top-level field) but
/// read_column_chunk_with_levels would fail on is_nested(); we use
/// ColumnPageDecoder directly.
///
/// Expected: values=[5,7], def_levels=[1,0,1], rep_levels=[].
#[test]
fn nested_optional_group_roundtrip() {
    use crate::encoding::compression::CompressionCodec;

    let field = FieldDescriptor::group(
        FieldId::new(1),
        "payload",
        Repetition::Optional,
        vec![FieldDescriptor::required(
            FieldId::new(2),
            "v",
            ParquetPhysicalType::Int32,
        )],
    );
    let dataset = single_column_dataset(field, 3);

    struct GroupOptRows(alloc::vec::Vec<Option<i32>>);
    impl RowSource for GroupOptRows {
        fn row_count(&self) -> usize {
            self.0.len()
        }
        fn row(&self, idx: usize) -> consus_core::Result<RowValue<'_>> {
            let cell = match self.0[idx] {
                Some(v) => CellValue::Group(vec![CellValue::Int32(v)]),
                None => CellValue::Null,
            };
            Ok(RowValue::new(vec![cell]))
        }
    }

    let rows = GroupOptRows(vec![Some(5), None, Some(7)]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    // ParquetReader::new succeeds (1 leaf chunk == 1 schema field).
    let reader = ParquetReader::new(&bytes).unwrap();
    let chunk = reader.metadata().row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    let start = chunk.data_page_offset as usize;
    let end = start + chunk.total_compressed_size as usize;

    // max_rep=0, max_def=1: optional group with required leaf.
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        1,
    );
    let result = dec.decode_pages_with_levels(&bytes[start..end]).unwrap();

    assert!(
        matches!(&result.values, ColumnValues::Int32(v) if *v == alloc::vec![5i32, 7]),
        "expected Int32([5, 7]), got {:?}",
        result.values
    );
    assert_eq!(
        result.def_levels,
        alloc::vec![1i32, 0, 1],
        "def_levels must be [1, 0, 1]"
    );
    assert!(
        result.rep_levels.is_empty(),
        "max_rep=0: no rep levels expected"
    );
}

/// Invariant: write [[{v=1},{v=2}], [{v=3}]] to schema
/// `message { repeated group items { required int32 v; } }`.
///
/// The leaf "items/v" has max_rep=1, max_def=1.
/// Expected: values=[1,2,3], rep_levels=[0,1,0], def_levels=[1,1,1].
#[test]
fn nested_repeated_group_roundtrip() {
    use crate::encoding::compression::CompressionCodec;

    let field = FieldDescriptor::group(
        FieldId::new(1),
        "items",
        Repetition::Repeated,
        vec![FieldDescriptor::required(
            FieldId::new(2),
            "v",
            ParquetPhysicalType::Int32,
        )],
    );
    let dataset = single_column_dataset(field, 2);

    struct RepeatedGroupRows(alloc::vec::Vec<alloc::vec::Vec<i32>>);
    impl RowSource for RepeatedGroupRows {
        fn row_count(&self) -> usize {
            self.0.len()
        }
        fn row(&self, idx: usize) -> consus_core::Result<RowValue<'_>> {
            let items: alloc::vec::Vec<CellValue<'_>> = self.0[idx]
                .iter()
                .map(|&v| CellValue::Group(vec![CellValue::Int32(v)]))
                .collect();
            Ok(RowValue::new(vec![CellValue::Repeated(items)]))
        }
    }

    let rows = RepeatedGroupRows(vec![vec![1, 2], vec![3]]);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &rows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let chunk = reader.metadata().row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    let start = chunk.data_page_offset as usize;
    let end = start + chunk.total_compressed_size as usize;

    // max_rep=1, max_def=1: repeated group with required leaf.
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        1,
        1,
    );
    let result = dec.decode_pages_with_levels(&bytes[start..end]).unwrap();

    assert!(
        matches!(&result.values, ColumnValues::Int32(v) if *v == alloc::vec![1i32, 2, 3]),
        "expected Int32([1, 2, 3]), got {:?}",
        result.values
    );
    assert_eq!(
        result.rep_levels,
        alloc::vec![0i32, 1, 0],
        "rep_levels must be [0, 1, 0]"
    );
    assert_eq!(
        result.def_levels,
        alloc::vec![1i32, 1, 1],
        "def_levels must be [1, 1, 1]"
    );
}

/// Invariant: write [a=None, a=Some({x=None}), a=Some({x=Some(42)})] to schema
/// `message { optional group a { optional int32 x; } }`.
///
/// The leaf "a/x" has max_rep=0, max_def=2:
///   def=0 → a is absent
///   def=1 → a is present but x is null
///   def=2 → both a and x are present
///
/// Expected: values=[42], def_levels=[0,1,2], rep_levels=[].
#[test]
fn deeply_nested_optional_in_optional_group_roundtrip() {
    use crate::encoding::compression::CompressionCodec;

    let field = FieldDescriptor::group(
        FieldId::new(1),
        "a",
        Repetition::Optional,
        vec![FieldDescriptor::optional(
            FieldId::new(2),
            "x",
            ParquetPhysicalType::Int32,
            None,
        )],
    );
    let dataset = single_column_dataset(field, 3);

    struct DeepOptRows;
    impl RowSource for DeepOptRows {
        fn row_count(&self) -> usize {
            3
        }
        fn row(&self, idx: usize) -> consus_core::Result<RowValue<'_>> {
            let cell = match idx {
                0 => CellValue::Null,                              // a absent: def=0
                1 => CellValue::Group(vec![CellValue::Null]),      // a present, x null: def=1
                _ => CellValue::Group(vec![CellValue::Int32(42)]), // a present, x=42: def=2
            };
            Ok(RowValue::new(vec![cell]))
        }
    }

    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &DeepOptRows)
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    let chunk = reader.metadata().row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    let start = chunk.data_page_offset as usize;
    let end = start + chunk.total_compressed_size as usize;

    // max_rep=0, max_def=2: optional group a with optional leaf x.
    let mut dec = ColumnPageDecoder::new(
        ParquetPhysicalType::Int32,
        CompressionCodec::Uncompressed,
        0,
        2,
    );
    let result = dec.decode_pages_with_levels(&bytes[start..end]).unwrap();

    assert!(
        matches!(&result.values, ColumnValues::Int32(v) if *v == alloc::vec![42i32]),
        "expected Int32([42]), got {:?}",
        result.values
    );
    assert_eq!(
        result.def_levels,
        alloc::vec![0i32, 1, 2],
        "def_levels must be [0, 1, 2]"
    );
    assert!(
        result.rep_levels.is_empty(),
        "max_rep=0: no rep levels expected"
    );
}

// ── Multi-page column chunk splitting tests ───────────────────────────────────

/// Invariant: 6 INT32 values [1..=6] with page_row_limit=3 produces 2 pages per
/// column chunk. Reading back via ParquetReader yields the original 6 values in
/// order, with num_rows=6 in the single row group and num_values=6 in column 0.
#[test]
fn multi_page_i32_two_pages_data_roundtrip() {
    let values: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let dataset = i32_dataset(6);
    let bytes = ParquetWriter::new()
        .with_page_row_limit(3)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(
        reader.metadata().row_groups.len(),
        1,
        "expected 1 row group"
    );
    assert_eq!(
        reader.metadata().row_groups[0].num_rows,
        6,
        "row group must contain 6 rows"
    );
    let col_meta = reader.metadata().row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    assert_eq!(
        col_meta.num_values, 6,
        "num_values must aggregate across all pages"
    );

    let all = read_all_i32(&bytes);
    assert_eq!(all, values, "round-tripped values must match exactly");
}

/// Invariant: 6 INT32 values [10..=15] with page_row_limit=2 produces 3 pages
/// (2+2+2 rows). Reading back yields [10, 11, 12, 13, 14, 15] in order.
#[test]
fn multi_page_three_pages_all_values_preserved() {
    let values: Vec<i32> = (10..=15).collect();
    let dataset = i32_dataset(6);
    let bytes = ParquetWriter::new()
        .with_page_row_limit(2)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let all = read_all_i32(&bytes);
    assert_eq!(all, values, "all 6 values must survive 3-page split");

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().row_groups[0].num_rows, 6);
    let col_meta = reader.metadata().row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    assert_eq!(col_meta.num_values, 6);
}

/// Invariant: 5 INT32 values [100, 200, 300, 400, 500] with page_row_limit=3
/// produces 2 pages (3 rows + 2 rows). Reading back yields the original 5 values.
#[test]
fn multi_page_uneven_split_last_page_smaller() {
    let values: Vec<i32> = vec![100, 200, 300, 400, 500];
    let dataset = i32_dataset(5);
    let bytes = ParquetWriter::new()
        .with_page_row_limit(3)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let all = read_all_i32(&bytes);
    assert_eq!(all, values, "uneven last page must preserve all values");

    let reader = ParquetReader::new(&bytes).unwrap();
    let col_meta = reader.metadata().row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    // Total num_values across both pages must equal 5.
    assert_eq!(col_meta.num_values, 5);
}

/// Invariant: page_row_limit larger than the total row count degrades to a single
/// page per column chunk — identical behavior to no limit.
#[test]
fn multi_page_limit_larger_than_rows_gives_one_page() {
    let values: Vec<i32> = vec![7, 8, 9];
    let dataset = i32_dataset(3);
    let bytes = ParquetWriter::new()
        .with_page_row_limit(100)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let all = read_all_i32(&bytes);
    assert_eq!(all, values);

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().row_groups.len(), 1);
    assert_eq!(reader.metadata().row_groups[0].num_rows, 3);
    let col_meta = reader.metadata().row_groups[0].columns[0]
        .meta_data
        .as_ref()
        .unwrap();
    assert_eq!(col_meta.num_values, 3);
}

/// Invariant: 12 values [1..=12] with rg_size=6 and page_row_limit=3 produces
/// 2 row groups each containing 2 pages per column chunk. All 12 values are
/// recovered in order.
#[test]
fn multi_page_combined_with_multi_row_group() {
    let values: Vec<i32> = (1..=12).collect();
    let dataset = i32_dataset(12);
    let bytes = ParquetWriter::new()
        .with_row_group_size(6)
        .with_page_row_limit(3)
        .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
        .unwrap();

    let reader = ParquetReader::new(&bytes).unwrap();
    assert_eq!(
        reader.metadata().row_groups.len(),
        2,
        "expected 2 row groups of 6"
    );
    assert_eq!(reader.metadata().row_groups[0].num_rows, 6);
    assert_eq!(reader.metadata().row_groups[1].num_rows, 6);

    // Each row group has 6 rows split into 2 pages of 3; num_values per chunk = 6.
    for rg_idx in 0..2 {
        let col_meta = reader.metadata().row_groups[rg_idx].columns[0]
            .meta_data
            .as_ref()
            .unwrap();
        assert_eq!(
            col_meta.num_values, 6,
            "row group {rg_idx}: num_values must aggregate both pages"
        );
    }

    let all = read_all_i32(&bytes);
    assert_eq!(all, values, "all 12 values must be recovered in order");
}

// ── Multi-page proptest ───────────────────────────────────────────────────────

proptest! {
    /// For any non-empty i32 vector and any page_row_limit ≥ 1, writing with
    /// multi-page splitting and reading back via ParquetReader yields the original
    /// values in exact order.
    ///
    /// Invariant: ∀ values: Vec<i32>, p ≥ 1:
    ///   read_all_i32(write(values, page_row_limit=p)) == values
    #[test]
    fn prop_multi_page_i32_roundtrip(
        values in proptest::collection::vec(proptest::num::i32::ANY, 1..=60usize),
        page_limit in 1usize..=20usize,
    ) {
        let n = values.len();
        let dataset = i32_dataset(n);
        let bytes = ParquetWriter::new()
            .with_page_row_limit(page_limit)
            .write_dataset_bytes(&dataset, &I32Rows(values.clone()))
            .unwrap();
        let read_values = read_all_i32(&bytes);
        prop_assert_eq!(read_values, values);
    }
}
