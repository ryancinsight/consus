//! Proptest-based roundtrip properties for `ParquetReader`.
//!
//! ## Invariants verified
//!
//! 1. `∀ vals: Vec<i32>, |vals| ∈ [1,100]`: write→read recovers exact values.
//! 2. `∀ vals: Vec<f64 (NORMAL)>, |vals| ∈ [1,50]`: write→read recovers exact values.
//! 3. `∀ vals: Vec<bool>, |vals| ∈ [1,128]`: write→read recovers exact values.
//! 4. `∀ vals: Vec<Vec<u8>>, |vals| ∈ [1,30], |each| ∈ [0,16]`: write→read recovers exact values.
//! 5. `∀ n ∈ [1,30], (i32[], f64[])`: two-column write→read recovers both columns exactly.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use crate::dataset::{ColumnChunkDescriptor, ParquetDatasetDescriptor, RowGroupDescriptor};
use crate::encoding::column::ColumnValues;
use crate::reader::ParquetReader;
use crate::schema::field::{FieldDescriptor, FieldId, SchemaDescriptor};
use crate::schema::physical::ParquetPhysicalType;
use crate::writer::{CellValue, ParquetWriter, RowSource, RowValue};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build a single-column required dataset with `n` rows in one row group.
fn make_dataset(physical_type: ParquetPhysicalType, n: usize) -> ParquetDatasetDescriptor {
    let schema = SchemaDescriptor::new(vec![FieldDescriptor::required(
        FieldId::new(1),
        "col",
        physical_type,
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

// ── properties ───────────────────────────────────────────────────────────────

proptest! {
    /// Property 1 — INT32 roundtrip.
    ///
    /// Invariant: ∀ vals: Vec<i32>, |vals| ∈ [1,100]:
    ///   read_column_chunk(0,0) == vals  AND  num_rows == |vals|
    #[test]
    fn prop_reader_i32_roundtrip(
        vals in proptest::collection::vec(i32::MIN..=i32::MAX, 1..=100)
    ) {
        struct Rows(Vec<i32>);
        impl RowSource for Rows {
            fn row_count(&self) -> usize { self.0.len() }
            fn row(&self, i: usize) -> consus_core::Result<RowValue<'_>> {
                Ok(RowValue::new(vec![CellValue::Int32(self.0[i])]))
            }
        }
        let n = vals.len();
        let dataset = make_dataset(ParquetPhysicalType::Int32, n);
        let bytes = ParquetWriter::new()
            .write_dataset_bytes(&dataset, &Rows(vals.clone()))
            .unwrap();
        let reader = ParquetReader::new(&bytes).unwrap();
        prop_assert_eq!(reader.metadata().num_rows, n as i64);
        let col = reader.read_column_chunk(0, 0).unwrap();
        prop_assert_eq!(col.len(), n);
        let ColumnValues::Int32(got) = col else {
            return Err(TestCaseError::fail("expected Int32 column"));
        };
        prop_assert_eq!(got, vals);
    }

    /// Property 2 — DOUBLE roundtrip (finite values only).
    ///
    /// `proptest::num::f64::NORMAL` produces finite, non-NaN, non-Inf values so
    /// bitwise equality holds after the PLAIN encode/decode cycle.
    ///
    /// Invariant: ∀ vals: Vec<f64 (NORMAL)>, |vals| ∈ [1,50]:
    ///   read_column_chunk(0,0) == vals
    #[test]
    fn prop_reader_f64_roundtrip(
        vals in proptest::collection::vec(proptest::num::f64::NORMAL, 1..=50)
    ) {
        struct Rows(Vec<f64>);
        impl RowSource for Rows {
            fn row_count(&self) -> usize { self.0.len() }
            fn row(&self, i: usize) -> consus_core::Result<RowValue<'_>> {
                Ok(RowValue::new(vec![CellValue::Double(self.0[i])]))
            }
        }
        let n = vals.len();
        let dataset = make_dataset(ParquetPhysicalType::Double, n);
        let bytes = ParquetWriter::new()
            .write_dataset_bytes(&dataset, &Rows(vals.clone()))
            .unwrap();
        let reader = ParquetReader::new(&bytes).unwrap();
        let col = reader.read_column_chunk(0, 0).unwrap();
        prop_assert_eq!(col.len(), n);
        let ColumnValues::Double(got) = col else {
            return Err(TestCaseError::fail("expected Double column"));
        };
        prop_assert_eq!(got, vals);
    }

    /// Property 3 — BOOLEAN roundtrip.
    ///
    /// Invariant: ∀ vals: Vec<bool>, |vals| ∈ [1,128]:
    ///   read_column_chunk(0,0) == vals
    #[test]
    fn prop_reader_bool_roundtrip(
        vals in proptest::collection::vec(proptest::bool::ANY, 1..=128)
    ) {
        struct Rows(Vec<bool>);
        impl RowSource for Rows {
            fn row_count(&self) -> usize { self.0.len() }
            fn row(&self, i: usize) -> consus_core::Result<RowValue<'_>> {
                Ok(RowValue::new(vec![CellValue::Boolean(self.0[i])]))
            }
        }
        let n = vals.len();
        let dataset = make_dataset(ParquetPhysicalType::Boolean, n);
        let bytes = ParquetWriter::new()
            .write_dataset_bytes(&dataset, &Rows(vals.clone()))
            .unwrap();
        let reader = ParquetReader::new(&bytes).unwrap();
        let col = reader.read_column_chunk(0, 0).unwrap();
        prop_assert_eq!(col.len(), n);
        let ColumnValues::Boolean(got) = col else {
            return Err(TestCaseError::fail("expected Boolean column"));
        };
        prop_assert_eq!(got, vals);
    }

    /// Property 4 — BYTE_ARRAY roundtrip.
    ///
    /// Each element is 0–16 arbitrary bytes; the column has 1–30 elements.
    ///
    /// Invariant: ∀ vals: Vec<Vec<u8>>, |vals| ∈ [1,30], |each| ∈ [0,16]:
    ///   read_column_chunk(0,0) == vals
    #[test]
    fn prop_reader_byte_array_roundtrip(
        vals in proptest::collection::vec(
            proptest::collection::vec(any::<u8>(), 0..=16),
            1..=30
        )
    ) {
        struct Rows(Vec<Vec<u8>>);
        impl RowSource for Rows {
            fn row_count(&self) -> usize { self.0.len() }
            fn row(&self, i: usize) -> consus_core::Result<RowValue<'_>> {
                Ok(RowValue::new(vec![CellValue::ByteArray(&self.0[i])]))
            }
        }
        let n = vals.len();
        let dataset = make_dataset(ParquetPhysicalType::ByteArray, n);
        let bytes_out = ParquetWriter::new()
            .write_dataset_bytes(&dataset, &Rows(vals.clone()))
            .unwrap();
        let reader = ParquetReader::new(&bytes_out).unwrap();
        let col = reader.read_column_chunk(0, 0).unwrap();
        prop_assert_eq!(col.len(), n);
        let ColumnValues::ByteArray(got) = col else {
            return Err(TestCaseError::fail("expected ByteArray column"));
        };
        prop_assert_eq!(got, vals);
    }

    /// Property 5 — two-column (INT32, DOUBLE) roundtrip.
    ///
    /// Both vectors are trimmed to `min(|ints|, |doubles|)` so they share a
    /// single row count. Both columns are recovered independently and verified
    /// against their respective input slices.
    ///
    /// Invariant: ∀ n = min(|ints|,|doubles|) ∈ [1,30]:
    ///   col0 == ints[..n]  AND  col1 == doubles[..n]  AND  num_rows == n
    #[test]
    fn prop_reader_two_column_i32_f64_roundtrip(
        ints in proptest::collection::vec(any::<i32>(), 1..=30),
        doubles in proptest::collection::vec(proptest::num::f64::NORMAL, 1..=30),
    ) {
        let n = ints.len().min(doubles.len());
        let ints = &ints[..n];
        let doubles = &doubles[..n];

        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "x", ParquetPhysicalType::Int32),
            FieldDescriptor::required(FieldId::new(2), "y", ParquetPhysicalType::Double),
        ]);
        let dataset = ParquetDatasetDescriptor::new(
            schema,
            vec![RowGroupDescriptor::new(
                n,
                vec![
                    ColumnChunkDescriptor::new(FieldId::new(1), n, 1).unwrap(),
                    ColumnChunkDescriptor::new(FieldId::new(2), n, 1).unwrap(),
                ],
            )
            .unwrap()],
        )
        .unwrap();

        struct TwoColRows {
            ints: Vec<i32>,
            doubles: Vec<f64>,
        }
        impl RowSource for TwoColRows {
            fn row_count(&self) -> usize { self.ints.len() }
            fn row(&self, i: usize) -> consus_core::Result<RowValue<'_>> {
                Ok(RowValue::new(vec![
                    CellValue::Int32(self.ints[i]),
                    CellValue::Double(self.doubles[i]),
                ]))
            }
        }

        let rows = TwoColRows {
            ints: ints.to_vec(),
            doubles: doubles.to_vec(),
        };
        let out = ParquetWriter::new().write_dataset_bytes(&dataset, &rows).unwrap();
        let reader = ParquetReader::new(&out).unwrap();
        prop_assert_eq!(reader.metadata().num_rows, n as i64);

        let col0 = reader.read_column_chunk(0, 0).unwrap();
        let col1 = reader.read_column_chunk(0, 1).unwrap();

        let ColumnValues::Int32(got_ints) = col0 else {
            return Err(TestCaseError::fail("expected Int32 column 0"));
        };
        let ColumnValues::Double(got_doubles) = col1 else {
            return Err(TestCaseError::fail("expected Double column 1"));
        };

        prop_assert_eq!(got_ints, ints);
        prop_assert_eq!(got_doubles, doubles);
    }
}
