use super::*;
use consus_core::{Error, Result};

use crate::dataset::ParquetDatasetDescriptor;
use crate::schema::field::{FieldDescriptor, FieldId, SchemaDescriptor};
use crate::schema::logical::Repetition;
use crate::schema::physical::ParquetPhysicalType;

#[test]
fn writer_plan_handles_nested_columns() {
    let schema = SchemaDescriptor::new(vec![FieldDescriptor::group(
        FieldId::new(1),
        "root",
        Repetition::Required,
        vec![
            FieldDescriptor::required(FieldId::new(2), "a", ParquetPhysicalType::Int32),
            FieldDescriptor::optional(FieldId::new(3), "b", ParquetPhysicalType::Int64, None),
        ],
    )]);
    let dataset = ParquetDatasetDescriptor::new(
        schema,
        vec![
            crate::dataset::RowGroupDescriptor::new(
                1,
                vec![crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), 1, 1).unwrap()],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let plan = ParquetWriter::new().plan(&dataset).unwrap();
    assert_eq!(plan.leaves().len(), 2);
    assert_eq!(
        plan.leaves()[0].path(),
        &["root".to_string(), "a".to_string()]
    );
    assert_eq!(
        plan.leaves()[1].path(),
        &["root".to_string(), "b".to_string()]
    );
}

#[test]
fn writer_rejects_row_count_mismatch() {
    struct EmptyRows;
    impl RowSource for EmptyRows {
        fn row_count(&self) -> usize {
            0
        }
        fn row(&self, _: usize) -> Result<RowValue<'_>> {
            Err(Error::InvalidFormat {
                message: String::from("unreachable"),
            })
        }
    }

    let schema = SchemaDescriptor::new(vec![FieldDescriptor::required(
        FieldId::new(1),
        "x",
        ParquetPhysicalType::Int32,
    )]);
    let dataset = ParquetDatasetDescriptor::new(
        schema,
        vec![
            crate::dataset::RowGroupDescriptor::new(
                1,
                vec![crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), 1, 1).unwrap()],
            )
            .unwrap(),
        ],
    )
    .unwrap();

    let err = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &EmptyRows)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidFormat { .. }));
}

#[test]
fn row_value_tracks_columns() {
    let row = RowValue::new(vec![CellValue::Int32(7), CellValue::Null]);
    assert_eq!(row.len(), 2);
    assert!(!row.is_empty());
}

#[test]
fn footer_roundtrip_metadata_and_trailer() {
    let schema = SchemaDescriptor::new(vec![FieldDescriptor::required(
        FieldId::new(1),
        "x",
        ParquetPhysicalType::Int32,
    )]);
    let dataset = ParquetDatasetDescriptor::new(
        schema,
        vec![
            crate::dataset::RowGroupDescriptor::new(
                1,
                vec![crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), 1, 1).unwrap()],
            )
            .unwrap(),
        ],
    )
    .unwrap();

    struct OneRow;
    impl RowSource for OneRow {
        fn row_count(&self) -> usize {
            1
        }
        fn row(&self, _: usize) -> Result<RowValue<'_>> {
            Ok(RowValue::new(vec![CellValue::Int32(7)]))
        }
    }

    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &OneRow)
        .unwrap();

    assert_eq!(&bytes[0..4], b"PAR1");
    let trailer_len =
        u32::from_le_bytes(bytes[bytes.len() - 8..bytes.len() - 4].try_into().unwrap());
    assert!(trailer_len > 0);
    assert_eq!(&bytes[bytes.len() - 4..], b"PAR1");
}

// ── End-to-end writer → reader roundtrip tests ────────────────────────

/// Build a single-column single-row-group ParquetDatasetDescriptor.
fn make_single_column_dataset(
    physical_type: ParquetPhysicalType,
    row_count: usize,
) -> crate::dataset::ParquetDatasetDescriptor {
    let schema = SchemaDescriptor::new(vec![FieldDescriptor::required(
        FieldId::new(1),
        "col",
        physical_type,
    )]);
    crate::dataset::ParquetDatasetDescriptor::new(
        schema,
        vec![
            crate::dataset::RowGroupDescriptor::new(
                row_count,
                vec![
                    crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), row_count, 1)
                        .unwrap(),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn writer_reader_roundtrip_i32_three_values() {
    // Analytical derivation: 3 × INT32 values [10, 20, 30].
    // DataPage v1, PLAIN encoding, UNCOMPRESSED.
    // read_column_chunk must return ColumnValues::Int32([10, 20, 30]).
    struct Rows;
    impl RowSource for Rows {
        fn row_count(&self) -> usize {
            3
        }
        fn row(&self, idx: usize) -> Result<RowValue<'_>> {
            let v = [10i32, 20, 30][idx];
            Ok(RowValue::new(vec![CellValue::Int32(v)]))
        }
    }

    let dataset = make_single_column_dataset(ParquetPhysicalType::Int32, 3);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &Rows)
        .unwrap();

    let reader = crate::reader::ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 3);
    let values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(values.len(), 3);
    assert!(
        matches!(&values, crate::encoding::column::ColumnValues::Int32(v) if *v == alloc::vec![10, 20, 30])
    );
}

#[test]
fn writer_reader_roundtrip_double_two_values() {
    // Analytical derivation: 2 × DOUBLE values [1.5, -0.25].
    // PLAIN encoding: 8-byte LE IEEE 754.
    // 1.5  = 3FF8000000000000 LE: 00 00 00 00 00 00 F8 3F
    // -0.25= BFD0000000000000 LE: 00 00 00 00 00 00 D0 BF
    struct Rows;
    impl RowSource for Rows {
        fn row_count(&self) -> usize {
            2
        }
        fn row(&self, idx: usize) -> Result<RowValue<'_>> {
            let v = [1.5f64, -0.25][idx];
            Ok(RowValue::new(vec![CellValue::Double(v)]))
        }
    }

    let dataset = make_single_column_dataset(ParquetPhysicalType::Double, 2);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &Rows)
        .unwrap();

    let reader = crate::reader::ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 2);
    let values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(values.len(), 2);
    assert!(
        matches!(&values, crate::encoding::column::ColumnValues::Double(v) if *v == alloc::vec![1.5f64, -0.25])
    );
}

#[test]
fn writer_reader_roundtrip_byte_array_two_values() {
    // Analytical derivation: 2 × BYTE_ARRAY values ["hello", "world"].
    // PLAIN encoding: 4-byte LE length prefix + raw bytes.
    struct Rows;
    impl RowSource for Rows {
        fn row_count(&self) -> usize {
            2
        }
        fn row(&self, idx: usize) -> Result<RowValue<'_>> {
            let data: &[u8] = if idx == 0 { b"hello" } else { b"world" };
            Ok(RowValue::new(vec![CellValue::ByteArray(data)]))
        }
    }

    let dataset = make_single_column_dataset(ParquetPhysicalType::ByteArray, 2);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &Rows)
        .unwrap();

    let reader = crate::reader::ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 2);
    let values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(values.len(), 2);
    assert!(
        matches!(&values, crate::encoding::column::ColumnValues::ByteArray(v)
            if *v == alloc::vec![b"hello".to_vec(), b"world".to_vec()])
    );
}

#[test]
fn writer_reader_roundtrip_boolean_four_values() {
    // Analytical derivation: 4 BOOLEAN values [true, false, true, true].
    // PLAIN BOOLEAN: bit-packed LSB-first.
    // Byte 0: bit0=1, bit1=0, bit2=1, bit3=1 = 0x0D (13)
    // 1 byte total (⌈4/8⌉=1).
    struct Rows;
    impl RowSource for Rows {
        fn row_count(&self) -> usize {
            4
        }
        fn row(&self, idx: usize) -> Result<RowValue<'_>> {
            let v = [true, false, true, true][idx];
            Ok(RowValue::new(vec![CellValue::Boolean(v)]))
        }
    }

    let dataset = make_single_column_dataset(ParquetPhysicalType::Boolean, 4);
    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &Rows)
        .unwrap();

    let reader = crate::reader::ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 4);
    let values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(values.len(), 4);
    assert!(
        matches!(&values, crate::encoding::column::ColumnValues::Boolean(v)
            if *v == alloc::vec![true, false, true, true])
    );
}

#[test]
fn writer_reader_roundtrip_two_columns() {
    // Two-column schema: x:INT32, y:DOUBLE; 2 rows.
    // Row 0: x=7, y=3.125
    // Row 1: x=42, y=-1.0
    struct Rows;
    impl RowSource for Rows {
        fn row_count(&self) -> usize {
            2
        }
        fn row(&self, idx: usize) -> Result<RowValue<'_>> {
            let (xi, yf): (i32, f64) = if idx == 0 { (7, 3.125) } else { (42, -1.0) };
            Ok(RowValue::new(vec![
                CellValue::Int32(xi),
                CellValue::Double(yf),
            ]))
        }
    }

    let schema = SchemaDescriptor::new(vec![
        FieldDescriptor::required(FieldId::new(1), "x", ParquetPhysicalType::Int32),
        FieldDescriptor::required(FieldId::new(2), "y", ParquetPhysicalType::Double),
    ]);
    let dataset = crate::dataset::ParquetDatasetDescriptor::new(
        schema,
        vec![
            crate::dataset::RowGroupDescriptor::new(
                2,
                vec![
                    crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), 2, 1).unwrap(),
                    crate::dataset::ColumnChunkDescriptor::new(FieldId::new(2), 2, 1).unwrap(),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();

    let bytes = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &Rows)
        .unwrap();

    let reader = crate::reader::ParquetReader::new(&bytes).unwrap();
    assert_eq!(reader.metadata().num_rows, 2);
    assert_eq!(reader.dataset().column_count(), 2);

    let x_vals = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(x_vals.len(), 2);
    assert!(
        matches!(&x_vals, crate::encoding::column::ColumnValues::Int32(v) if *v == alloc::vec![7, 42])
    );

    let y_vals = reader.read_column_chunk(0, 1).unwrap();
    assert_eq!(y_vals.len(), 2);
    assert!(
        matches!(&y_vals, crate::encoding::column::ColumnValues::Double(v) if *v == alloc::vec![3.125, -1.0])
    );
}

#[test]
fn compress_page_values_uncompressed_passthrough() {
    use crate::encoding::compression::{CompressionCodec, compress_page_values};
    let data = alloc::vec![1u8, 2, 3, 4];
    let out = compress_page_values(&data, CompressionCodec::Uncompressed).unwrap();
    assert_eq!(out, data);
}

#[test]
fn compress_page_values_brotli_returns_unsupported() {
    use crate::encoding::compression::{CompressionCodec, compress_page_values};
    let err = compress_page_values(&[], CompressionCodec::Brotli).unwrap_err();
    assert!(matches!(err, consus_core::Error::UnsupportedFeature { .. }));
}

#[cfg(feature = "gzip")]
#[test]
fn writer_gzip_roundtrip_i32_three_values() {
    // Analytical: 3 × INT32 [42, -1, 0] written with GZIP, read back with ParquetReader.
    // ParquetReader::read_column_chunk decompresses GZIP automatically.
    use crate::encoding::compression::CompressionCodec;
    struct Rows;
    impl RowSource for Rows {
        fn row_count(&self) -> usize {
            3
        }
        fn row(&self, idx: usize) -> Result<RowValue<'_>> {
            Ok(RowValue::new(vec![CellValue::Int32([42i32, -1, 0][idx])]))
        }
    }
    let dataset = make_single_column_dataset(ParquetPhysicalType::Int32, 3);
    let bytes = ParquetWriter::new()
        .with_compression(CompressionCodec::Gzip)
        .write_dataset_bytes(&dataset, &Rows)
        .unwrap();
    let reader = crate::reader::ParquetReader::new(&bytes).unwrap();
    let values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(values.len(), 3);
    assert!(matches!(
        &values,
        crate::encoding::column::ColumnValues::Int32(v) if *v == alloc::vec![42i32, -1, 0]
    ));
}

#[cfg(feature = "gzip")]
#[test]
fn writer_gzip_roundtrip_byte_array() {
    // Analytical: 2 × BYTE_ARRAY ["foo", "baz"] written with GZIP.
    use crate::encoding::compression::CompressionCodec;
    struct Rows;
    impl RowSource for Rows {
        fn row_count(&self) -> usize {
            2
        }
        fn row(&self, idx: usize) -> Result<RowValue<'_>> {
            let data: &[u8] = if idx == 0 { b"foo" } else { b"baz" };
            Ok(RowValue::new(vec![CellValue::ByteArray(data)]))
        }
    }
    let dataset = make_single_column_dataset(ParquetPhysicalType::ByteArray, 2);
    let bytes = ParquetWriter::new()
        .with_compression(CompressionCodec::Gzip)
        .write_dataset_bytes(&dataset, &Rows)
        .unwrap();
    let reader = crate::reader::ParquetReader::new(&bytes).unwrap();
    let values = reader.read_column_chunk(0, 0).unwrap();
    assert_eq!(values.len(), 2);
    assert!(matches!(
        &values,
        crate::encoding::column::ColumnValues::ByteArray(v)
            if *v == alloc::vec![b"foo".to_vec(), b"baz".to_vec()]
    ));
}

#[test]
fn writer_null_in_required_column_returns_error() {
    // Null values in required columns must produce InvalidFormat.
    struct NullRow;
    impl RowSource for NullRow {
        fn row_count(&self) -> usize {
            1
        }
        fn row(&self, _: usize) -> Result<RowValue<'_>> {
            Ok(RowValue::new(vec![CellValue::Null]))
        }
    }

    let dataset = make_single_column_dataset(ParquetPhysicalType::Int32, 1);
    let err = ParquetWriter::new()
        .write_dataset_bytes(&dataset, &NullRow)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidFormat { .. }));
}
