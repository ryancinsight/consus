//! End-to-end integration tests: ParquetWriter → ParquetReader → column_values_to_arrow.
//!
//! ## Specification
//!
//! Pipeline under test for each scenario:
//! 1. Build a `ParquetDatasetDescriptor` and a `RowSource`.
//! 2. Serialize via `ParquetWriter::write_dataset_bytes` to a `Vec<u8>`.
//! 3. Deserialize via `ParquetReader::new`.
//! 4. Read each column via `reader.read_column_chunk(0, col_ordinal)`.
//! 5. Materialize via `column_values_to_arrow`.
//! 6. Assert structural (`len`, `element_width` / `offsets`) and byte-level
//!    (`values`) correctness against analytically derived expectations.
//!
//! ## Mathematical derivations
//!
//! | Type        | PLAIN encoding                              | Arrow output                     |
//! |-------------|---------------------------------------------|----------------------------------|
//! | INT32       | 4-byte LE per value                         | FixedWidth, element_width=4, LE  |
//! | INT64       | 8-byte LE per value                         | FixedWidth, element_width=8, LE  |
//! | DOUBLE      | 8-byte IEEE 754 LE per value                | FixedWidth, element_width=8, LE  |
//! | BYTE_ARRAY  | [4-byte LE len][raw bytes] per value        | VariableWidth, monotone offsets  |
//! | BOOLEAN     | bit-packed LSB-first, ⌈n/8⌉ bytes          | FixedWidth, element_width=1      |
//!
//! BOOLEAN round-trip detail:
//!   Writer packs [true,false,true,true] into byte 0 = bit0|bit2|bit3 = 0b00001101 = 0x0D.
//!   Reader unpacks back to Vec<bool>: [true, false, true, true].
//!   column_values_to_arrow maps true→0x01, false→0x00 (one byte per bool).
//!   Final bytes: [0x01, 0x00, 0x01, 0x01].

use consus_arrow::{ArrayData, column_values_to_arrow};
use consus_core::Result;
use consus_parquet::{
    CellValue, ColumnChunkDescriptor, ColumnValues, FieldDescriptor, FieldId,
    ParquetDatasetDescriptor, ParquetPhysicalType, ParquetReader, ParquetWriter,
    RowGroupDescriptor, RowSource, RowValue, SchemaDescriptor,
};

// ── extraction helpers ───────────────────────────────────────────────────────

/// Extract `(len, element_width, values)` from a `FixedWidth` array.
/// Panics if the array is `VariableWidth`.
fn fixed_parts(arr: &consus_arrow::ArrowArray) -> (usize, usize, &[u8]) {
    match &arr.data {
        ArrayData::FixedWidth {
            len,
            element_width,
            values,
            ..
        } => (*len, *element_width, values),
        _ => panic!("expected FixedWidth, got VariableWidth"),
    }
}

/// Extract `(len, offsets, values)` from a `VariableWidth` array.
/// Panics if the array is `FixedWidth`.
fn var_parts(arr: &consus_arrow::ArrowArray) -> (usize, &[usize], &[u8]) {
    match &arr.data {
        ArrayData::VariableWidth {
            len,
            offsets,
            values,
            ..
        } => (*len, offsets, values),
        _ => panic!("expected VariableWidth, got FixedWidth"),
    }
}

// ── dataset descriptor helpers ───────────────────────────────────────────────

/// Build a single-column, single-row-group `ParquetDatasetDescriptor`.
///
/// Schema: one required field with `FieldId(1)` named "col" of the given
/// physical type. One row group covering all `row_count` rows.
fn single_column_dataset(pt: ParquetPhysicalType, row_count: usize) -> ParquetDatasetDescriptor {
    let schema = SchemaDescriptor::new(vec![FieldDescriptor::required(
        FieldId::new(1),
        "col",
        pt,
    )]);
    ParquetDatasetDescriptor::new(
        schema,
        vec![RowGroupDescriptor::new(
            row_count,
            vec![ColumnChunkDescriptor::new(FieldId::new(1), row_count, 1).unwrap()],
        )
        .unwrap()],
    )
    .unwrap()
}

// ── test 1: INT32 three values ───────────────────────────────────────────────

struct I32ThreeRows;

impl RowSource for I32ThreeRows {
    fn row_count(&self) -> usize {
        3
    }

    fn row(&self, idx: usize) -> Result<RowValue<'_>> {
        let v = [10i32, 20, 30][idx];
        Ok(RowValue::new(vec![CellValue::Int32(v)]))
    }
}

#[test]
fn e2e_i32_three_values_pipeline() {
    // Analytical derivation:
    // INT32 PLAIN: each value v → v.to_le_bytes() (4 bytes, little-endian).
    //   10i32 → [0x0A, 0x00, 0x00, 0x00]
    //   20i32 → [0x14, 0x00, 0x00, 0x00]
    //   30i32 → [0x1E, 0x00, 0x00, 0x00]
    // Concatenated: 12 bytes.
    // column_values_to_arrow: FixedWidth, len=3, element_width=4.
    let dataset = single_column_dataset(ParquetPhysicalType::Int32, 3);
    let file_bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &I32ThreeRows)
        .unwrap();

    let reader = ParquetReader::new(&file_bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 3);

    let col_values: ColumnValues = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(col_values.len(), 3);

    let arr = column_values_to_arrow(&col_values);
    let (len, element_width, data_bytes) = fixed_parts(&arr);

    assert_eq!(len, 3);
    assert_eq!(element_width, 4);
    assert_eq!(data_bytes.len(), 12);
    assert_eq!(
        data_bytes,
        &[
            10u8, 0, 0, 0, // 10i32 LE
            20u8, 0, 0, 0, // 20i32 LE
            30u8, 0, 0, 0, // 30i32 LE
        ]
    );
}

// ── test 2: INT64 two values ─────────────────────────────────────────────────

struct I64TwoRows;

impl RowSource for I64TwoRows {
    fn row_count(&self) -> usize {
        2
    }

    fn row(&self, idx: usize) -> Result<RowValue<'_>> {
        let v = [i64::MAX, -1i64][idx];
        Ok(RowValue::new(vec![CellValue::Int64(v)]))
    }
}

#[test]
fn e2e_i64_two_values_pipeline() {
    // Analytical derivation:
    // INT64 PLAIN: each value v → v.to_le_bytes() (8 bytes, little-endian).
    //   i64::MAX = 0x7FFFFFFFFFFFFFFF → LE: [FF,FF,FF,FF,FF,FF,FF,7F]
    //   -1i64    = 0xFFFFFFFFFFFFFFFF → LE: [FF,FF,FF,FF,FF,FF,FF,FF]
    // column_values_to_arrow: FixedWidth, len=2, element_width=8.
    let dataset = single_column_dataset(ParquetPhysicalType::Int64, 2);
    let file_bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &I64TwoRows)
        .unwrap();

    let reader = ParquetReader::new(&file_bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 2);

    let col_values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(col_values.len(), 2);

    let arr = column_values_to_arrow(&col_values);
    let (len, element_width, data_bytes) = fixed_parts(&arr);

    assert_eq!(len, 2);
    assert_eq!(element_width, 8);
    assert_eq!(data_bytes.len(), 16);
    assert_eq!(&data_bytes[0..8], &i64::MAX.to_le_bytes());
    assert_eq!(&data_bytes[8..16], &(-1i64).to_le_bytes());
}

// ── test 3: DOUBLE two values ────────────────────────────────────────────────

struct DoubleTwoRows;

impl RowSource for DoubleTwoRows {
    fn row_count(&self) -> usize {
        2
    }

    fn row(&self, idx: usize) -> Result<RowValue<'_>> {
        let v = [1.5f64, -0.25f64][idx];
        Ok(RowValue::new(vec![CellValue::Double(v)]))
    }
}

#[test]
fn e2e_double_two_values_pipeline() {
    // Analytical derivation (IEEE 754 double-precision, little-endian):
    //   1.5f64   = 0x3FF8000000000000 → LE: [00,00,00,00,00,00,F8,3F]
    //   -0.25f64 = 0xBFD0000000000000 → LE: [00,00,00,00,00,00,D0,BF]
    // column_values_to_arrow: FixedWidth, len=2, element_width=8.
    let dataset = single_column_dataset(ParquetPhysicalType::Double, 2);
    let file_bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &DoubleTwoRows)
        .unwrap();

    let reader = ParquetReader::new(&file_bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 2);

    let col_values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(col_values.len(), 2);

    let arr = column_values_to_arrow(&col_values);
    let (len, element_width, data_bytes) = fixed_parts(&arr);

    assert_eq!(len, 2);
    assert_eq!(element_width, 8);
    assert_eq!(data_bytes.len(), 16);
    assert_eq!(&data_bytes[0..8], &1.5f64.to_le_bytes());
    assert_eq!(&data_bytes[8..16], &(-0.25f64).to_le_bytes());
}

// ── test 4: BYTE_ARRAY two values ────────────────────────────────────────────

struct ByteArrayTwoRows;

impl RowSource for ByteArrayTwoRows {
    fn row_count(&self) -> usize {
        2
    }

    fn row(&self, idx: usize) -> Result<RowValue<'_>> {
        let data: &[u8] = if idx == 0 { b"hello" } else { b"world" };
        Ok(RowValue::new(vec![CellValue::ByteArray(data)]))
    }
}

#[test]
fn e2e_byte_array_two_values_pipeline() {
    // Analytical derivation:
    // BYTE_ARRAY PLAIN encodes each value as [4-byte LE length][raw bytes].
    //   "hello": [05,00,00,00][68,65,6C,6C,6F]  — 9 bytes total
    //   "world": [05,00,00,00][77,6F,72,6C,64]  — 9 bytes total
    // Reader strips the length prefixes and reconstructs Vec<Vec<u8>>.
    // column_values_to_arrow produces VariableWidth:
    //   len     = 2
    //   offsets = [0, 5, 10]  (monotone, len+1 entries)
    //   values  = b"helloworld" (10 bytes)
    let dataset = single_column_dataset(ParquetPhysicalType::ByteArray, 2);
    let file_bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &ByteArrayTwoRows)
        .unwrap();

    let reader = ParquetReader::new(&file_bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 2);

    let col_values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(col_values.len(), 2);

    let arr = column_values_to_arrow(&col_values);
    let (len, offsets, payload) = var_parts(&arr);

    assert_eq!(len, 2);
    assert_eq!(offsets, &[0usize, 5, 10]);
    assert_eq!(payload, b"helloworld");
}

// ── test 5: BOOLEAN four values ──────────────────────────────────────────────

struct BoolFourRows;

impl RowSource for BoolFourRows {
    fn row_count(&self) -> usize {
        4
    }

    fn row(&self, idx: usize) -> Result<RowValue<'_>> {
        let v = [true, false, true, true][idx];
        Ok(RowValue::new(vec![CellValue::Boolean(v)]))
    }
}

#[test]
fn e2e_boolean_four_values_pipeline() {
    // Analytical derivation:
    // PLAIN BOOLEAN: bit-packed LSB-first, ⌈4/8⌉ = 1 byte.
    //   value[0]=true  → bit 0 of byte 0 = 1
    //   value[1]=false → bit 1 of byte 0 = 0
    //   value[2]=true  → bit 2 of byte 0 = 1
    //   value[3]=true  → bit 3 of byte 0 = 1
    //   byte 0 = 0b00001101 = 0x0D
    // Reader unpacks bits back to Vec<bool>: [true, false, true, true].
    // column_values_to_arrow: FixedWidth, element_width=1.
    //   true  → 0x01, false → 0x00
    //   bytes = [0x01, 0x00, 0x01, 0x01]
    let dataset = single_column_dataset(ParquetPhysicalType::Boolean, 4);
    let file_bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &BoolFourRows)
        .unwrap();

    let reader = ParquetReader::new(&file_bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 4);

    let col_values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(col_values.len(), 4);

    let arr = column_values_to_arrow(&col_values);
    let (len, element_width, data_bytes) = fixed_parts(&arr);

    assert_eq!(len, 4);
    assert_eq!(element_width, 1);
    assert_eq!(data_bytes, &[1u8, 0u8, 1u8, 1u8]);
}

// ── test 6: two-column INT32 + DOUBLE pipeline ───────────────────────────────

struct TwoColumnRows;

impl RowSource for TwoColumnRows {
    fn row_count(&self) -> usize {
        2
    }

    fn row(&self, idx: usize) -> Result<RowValue<'_>> {
        match idx {
            0 => Ok(RowValue::new(vec![
                CellValue::Int32(7),
                CellValue::Double(3.14),
            ])),
            1 => Ok(RowValue::new(vec![
                CellValue::Int32(-1),
                CellValue::Double(-2.718),
            ])),
            _ => unreachable!("row index {} exceeds row_count=2", idx),
        }
    }
}

#[test]
fn e2e_two_column_int32_double_pipeline() {
    // Analytical derivation:
    // Schema: field "a" INT32 (FieldId 1), field "b" DOUBLE (FieldId 2); 2 rows.
    //
    // Column 0 (INT32):
    //   row 0: 7i32    → [07,00,00,00]
    //   row 1: -1i32   → [FF,FF,FF,FF]
    //   FixedWidth, len=2, element_width=4, 8 bytes total.
    //
    // Column 1 (DOUBLE):
    //   row 0: 3.14f64   = 0x40091EB851EB851F → LE 8 bytes
    //   row 1: -2.718f64 = 0xC005C28F5C28F5C3 → LE 8 bytes
    //   FixedWidth, len=2, element_width=8, 16 bytes total.
    let schema = SchemaDescriptor::new(vec![
        FieldDescriptor::required(FieldId::new(1), "a", ParquetPhysicalType::Int32),
        FieldDescriptor::required(FieldId::new(2), "b", ParquetPhysicalType::Double),
    ]);
    let dataset = ParquetDatasetDescriptor::new(
        schema,
        vec![RowGroupDescriptor::new(
            2,
            vec![
                ColumnChunkDescriptor::new(FieldId::new(1), 2, 1).unwrap(),
                ColumnChunkDescriptor::new(FieldId::new(2), 2, 1).unwrap(),
            ],
        )
        .unwrap()],
    )
    .unwrap();

    let file_bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &TwoColumnRows)
        .unwrap();

    let reader = ParquetReader::new(&file_bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 2);
    assert_eq!(reader.dataset().column_count(), 2);

    // ── column 0: INT32 [7, -1] ──────────────────────────────────────────────
    let col0_values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(col0_values.len(), 2);

    let arr0 = column_values_to_arrow(&col0_values);
    let (len0, ew0, bytes0) = fixed_parts(&arr0);

    assert_eq!(len0, 2);
    assert_eq!(ew0, 4);
    assert_eq!(bytes0.len(), 8);
    assert_eq!(&bytes0[0..4], &7i32.to_le_bytes());
    assert_eq!(&bytes0[4..8], &(-1i32).to_le_bytes());

    // ── column 1: DOUBLE [3.14, -2.718] ─────────────────────────────────────
    let col1_values = reader.read_column_chunk(0, 1).unwrap();
    assert_eq!(col1_values.len(), 2);

    let arr1 = column_values_to_arrow(&col1_values);
    let (len1, ew1, bytes1) = fixed_parts(&arr1);

    assert_eq!(len1, 2);
    assert_eq!(ew1, 8);
    assert_eq!(bytes1.len(), 16);
    assert_eq!(&bytes1[0..8], &3.14f64.to_le_bytes());
    assert_eq!(&bytes1[8..16], &(-2.718f64).to_le_bytes());
}
