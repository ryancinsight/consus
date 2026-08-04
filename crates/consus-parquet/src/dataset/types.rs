use alloc::{boxed::Box, string::String, vec::Vec};

use consus_core::{CompoundField, Datatype, Error, Result, Shape};

use crate::conversion::parquet_field_to_core;
use crate::schema::{FieldDescriptor, FieldId, SchemaDescriptor};

/// Canonical storage classification for a Parquet column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnStorage {
    /// Fixed-width physical storage.
    FixedWidth { bytes_per_value: usize },
    /// Variable-width physical storage.
    VariableWidth,
    /// Nested group storage represented by child fields.
    Nested,
}

/// Canonical descriptor for one top-level Parquet column.
#[derive(Debug, Clone, PartialEq)]
pub struct ParquetColumnDescriptor {
    field: FieldDescriptor,
    datatype: Datatype,
    storage: ColumnStorage,
    shape: Shape,
}

impl ParquetColumnDescriptor {
    /// Build a canonical column descriptor from a top-level schema field.
    pub fn from_field(field: &FieldDescriptor, row_count: usize) -> Result<Self> {
        field.validate()?;
        let datatype = canonicalize_top_level_field(field)?;
        let storage = if field.is_group() {
            ColumnStorage::Nested
        } else if field.is_repeated() || datatype.is_variable_length() {
            ColumnStorage::VariableWidth
        } else if let Some(width) = datatype.element_size() {
            ColumnStorage::FixedWidth {
                bytes_per_value: width,
            }
        } else {
            ColumnStorage::VariableWidth
        };
        let shape = Shape::fixed(&[row_count]);
        Ok(Self {
            field: field.clone(),
            datatype,
            storage,
            shape,
        })
    }

    /// Stable field identifier.
    #[must_use]
    pub fn field_id(&self) -> FieldId {
        self.field.id()
    }

    /// Field name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.field.name()
    }

    /// Canonical datatype.
    #[must_use]
    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    /// Canonical storage classification.
    #[must_use]
    pub fn storage(&self) -> ColumnStorage {
        self.storage
    }

    /// Canonical one-dimensional column shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Source schema field.
    #[must_use]
    pub fn field(&self) -> &FieldDescriptor {
        &self.field
    }

    /// Whether the column is nested.
    #[must_use]
    pub fn is_nested(&self) -> bool {
        matches!(self.storage, ColumnStorage::Nested)
    }
}

/// Physical metadata for one column chunk inside one row group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnChunkDescriptor {
    field_id: FieldId,
    row_count: usize,
    byte_len: usize,
}

impl ColumnChunkDescriptor {
    /// Create a chunk descriptor with exact row and byte counts.
    pub fn new(field_id: FieldId, row_count: usize, byte_len: usize) -> Result<Self> {
        if row_count == 0 {
            return Err(Error::InvalidFormat {
                message: String::from("parquet column chunk row_count must be positive"),
            });
        }
        Ok(Self {
            field_id,
            row_count,
            byte_len,
        })
    }

    /// Stable field identifier.
    #[must_use]
    pub fn field_id(&self) -> FieldId {
        self.field_id
    }

    /// Number of rows covered by the chunk.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Physical byte length of the chunk payload.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

/// Canonical row-group descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGroupDescriptor {
    row_count: usize,
    column_chunks: Vec<ColumnChunkDescriptor>,
}

impl RowGroupDescriptor {
    /// Create a row-group descriptor.
    pub fn new(row_count: usize, column_chunks: Vec<ColumnChunkDescriptor>) -> Result<Self> {
        if row_count == 0 {
            return Err(Error::InvalidFormat {
                message: String::from("parquet row group row_count must be positive"),
            });
        }
        if column_chunks.is_empty() {
            return Err(Error::InvalidFormat {
                message: String::from("parquet row group must contain column chunks"),
            });
        }
        let mut i = 0;
        while i < column_chunks.len() {
            if column_chunks[i].row_count != row_count {
                return Err(Error::InvalidFormat {
                    message: String::from(
                        "parquet row group chunk row_count must equal row group row_count",
                    ),
                });
            }
            i += 1;
        }
        Ok(Self {
            row_count,
            column_chunks,
        })
    }

    /// Number of rows in the row group.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Borrow the column chunks in schema order.
    #[must_use]
    pub fn column_chunks(&self) -> &[ColumnChunkDescriptor] {
        &self.column_chunks
    }

    /// Sum of physical bytes across all chunks.
    #[must_use]
    pub fn total_byte_len(&self) -> usize {
        self.column_chunks
            .iter()
            .map(ColumnChunkDescriptor::byte_len)
            .sum()
    }
}

/// Canonical validated Parquet dataset descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ParquetDatasetDescriptor {
    schema: SchemaDescriptor,
    columns: Vec<ParquetColumnDescriptor>,
    row_groups: Vec<RowGroupDescriptor>,
    total_rows: usize,
}

impl ParquetDatasetDescriptor {
    /// Build a validated dataset descriptor from schema and row groups.
    pub fn new(schema: SchemaDescriptor, row_groups: Vec<RowGroupDescriptor>) -> Result<Self> {
        schema.validate()?;
        if row_groups.is_empty() {
            return Err(Error::InvalidFormat {
                message: String::from("parquet dataset must contain at least one row group"),
            });
        }

        let total_rows = row_groups
            .iter()
            .try_fold(0usize, |acc, group| acc.checked_add(group.row_count()))
            .ok_or(Error::Overflow)?;

        let mut columns = Vec::with_capacity(schema.field_count());
        let mut i = 0;
        while i < schema.fields().len() {
            columns.push(ParquetColumnDescriptor::from_field(
                &schema.fields()[i],
                total_rows,
            )?);
            i += 1;
        }

        let expected_columns = schema.field_count();
        let mut group_index = 0;
        while group_index < row_groups.len() {
            let group = &row_groups[group_index];
            if group.column_chunks.len() != expected_columns {
                return Err(Error::InvalidFormat {
                    message: String::from(
                        "parquet row group chunk count must equal schema field count",
                    ),
                });
            }

            let mut chunk_index = 0;
            while chunk_index < group.column_chunks.len() {
                if group.column_chunks[chunk_index].field_id() != schema.fields()[chunk_index].id()
                {
                    return Err(Error::InvalidFormat {
                        message: String::from(
                            "parquet row group chunk field order must match schema order",
                        ),
                    });
                }
                chunk_index += 1;
            }

            group_index += 1;
        }

        Ok(Self {
            schema,
            columns,
            row_groups,
            total_rows,
        })
    }

    /// Borrow the authoritative schema.
    #[must_use]
    pub fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    /// Borrow the canonical top-level columns.
    #[must_use]
    pub fn columns(&self) -> &[ParquetColumnDescriptor] {
        &self.columns
    }

    /// Borrow the row groups.
    #[must_use]
    pub fn row_groups(&self) -> &[RowGroupDescriptor] {
        &self.row_groups
    }

    /// Total number of rows across all row groups.
    #[must_use]
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Number of top-level columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Borrow a column by name.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&ParquetColumnDescriptor> {
        self.columns.iter().find(|column| column.name() == name)
    }

    /// Total physical bytes across all row groups.
    #[must_use]
    pub fn total_byte_len(&self) -> usize {
        self.row_groups
            .iter()
            .map(RowGroupDescriptor::total_byte_len)
            .sum()
    }
}

fn canonicalize_top_level_field(field: &FieldDescriptor) -> Result<Datatype> {
    let base = canonicalize_field(field)?;
    if field.is_repeated() {
        Ok(Datatype::VarLen {
            base: Box::new(base),
        })
    } else {
        Ok(base)
    }
}

fn canonicalize_field(field: &FieldDescriptor) -> Result<Datatype> {
    if field.is_group() {
        let mut fields = Vec::with_capacity(field.children().len());
        let mut offset = 0usize;
        let mut fixed_size = true;

        let mut i = 0;
        while i < field.children().len() {
            let child = &field.children()[i];
            let child_datatype = canonicalize_top_level_field(child)?;
            let child_size = child_datatype.element_size();
            if let Some(size) = child_size {
                offset = offset.checked_add(size).ok_or(Error::Overflow)?;
            } else {
                fixed_size = false;
            }
            fields.push(CompoundField {
                name: child.name().to_owned(),
                datatype: child_datatype,
                offset: if fixed_size {
                    offset
                        .checked_sub(child_size.unwrap_or(0))
                        .ok_or(Error::Overflow)?
                } else {
                    0
                },
            });
            i += 1;
        }

        Ok(Datatype::Compound {
            fields,
            size: if fixed_size { offset } else { 0 },
        })
    } else {
        Ok(parquet_field_to_core(field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{LogicalType, ParquetPhysicalType, Repetition};

    #[test]
    fn dataset_descriptor_computes_total_rows_and_columns() {
        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "id", ParquetPhysicalType::Int64),
            FieldDescriptor::optional(
                FieldId::new(2),
                "name",
                ParquetPhysicalType::ByteArray,
                Some(LogicalType::String),
            ),
        ]);

        let row_groups = vec![
            RowGroupDescriptor::new(
                3,
                vec![
                    ColumnChunkDescriptor::new(FieldId::new(1), 3, 24).unwrap(),
                    ColumnChunkDescriptor::new(FieldId::new(2), 3, 17).unwrap(),
                ],
            )
            .unwrap(),
            RowGroupDescriptor::new(
                2,
                vec![
                    ColumnChunkDescriptor::new(FieldId::new(1), 2, 16).unwrap(),
                    ColumnChunkDescriptor::new(FieldId::new(2), 2, 11).unwrap(),
                ],
            )
            .unwrap(),
        ];

        let dataset = ParquetDatasetDescriptor::new(schema, row_groups).unwrap();
        assert_eq!(dataset.total_rows(), 5);
        assert_eq!(dataset.column_count(), 2);
        assert_eq!(dataset.total_byte_len(), 68);
        assert_eq!(
            dataset
                .column("id")
                .unwrap()
                .shape()
                .current_dims()
                .as_slice(),
            &[5]
        );
        assert!(matches!(
            dataset.column("id").unwrap().storage(),
            ColumnStorage::FixedWidth { bytes_per_value: 8 }
        ));
        assert!(matches!(
            dataset.column("name").unwrap().storage(),
            ColumnStorage::VariableWidth
        ));
    }

    #[test]
    fn dataset_descriptor_rejects_chunk_count_mismatch() {
        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "x", ParquetPhysicalType::Int32),
            FieldDescriptor::required(FieldId::new(2), "y", ParquetPhysicalType::Int32),
        ]);

        let row_groups = vec![
            RowGroupDescriptor::new(
                4,
                vec![ColumnChunkDescriptor::new(FieldId::new(1), 4, 16).unwrap()],
            )
            .unwrap(),
        ];

        let err = ParquetDatasetDescriptor::new(schema, row_groups).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat { .. }));
    }

    #[test]
    fn dataset_descriptor_rejects_chunk_field_order_mismatch() {
        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "x", ParquetPhysicalType::Int32),
            FieldDescriptor::required(FieldId::new(2), "y", ParquetPhysicalType::Int32),
        ]);

        let row_groups = vec![
            RowGroupDescriptor::new(
                4,
                vec![
                    ColumnChunkDescriptor::new(FieldId::new(2), 4, 16).unwrap(),
                    ColumnChunkDescriptor::new(FieldId::new(1), 4, 16).unwrap(),
                ],
            )
            .unwrap(),
        ];

        let err = ParquetDatasetDescriptor::new(schema, row_groups).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat { .. }));
    }

    #[test]
    fn nested_group_column_maps_to_nested_storage() {
        let schema = SchemaDescriptor::new(vec![FieldDescriptor::group(
            FieldId::new(1),
            "point",
            Repetition::Required,
            vec![
                FieldDescriptor::required(FieldId::new(2), "x", ParquetPhysicalType::Float),
                FieldDescriptor::required(FieldId::new(3), "y", ParquetPhysicalType::Float),
            ],
        )]);

        let row_groups = vec![
            RowGroupDescriptor::new(
                3,
                vec![ColumnChunkDescriptor::new(FieldId::new(1), 3, 24).unwrap()],
            )
            .unwrap(),
        ];

        let dataset = ParquetDatasetDescriptor::new(schema, row_groups).unwrap();
        let column = dataset.column("point").unwrap();
        assert!(column.is_nested());
        assert!(matches!(column.storage(), ColumnStorage::Nested));
        match column.datatype() {
            Datatype::Compound { fields, size } => {
                assert_eq!(*size, 8);
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[0].offset, 0);
                assert_eq!(fields[1].name, "y");
                assert_eq!(fields[1].offset, 4);
            }
            other => panic!("expected Compound datatype, got {other:?}"),
        }
    }

    #[test]
    fn repeated_scalar_column_maps_to_varlen_datatype_and_variable_storage() {
        let schema = SchemaDescriptor::new(vec![FieldDescriptor::repeated(
            FieldId::new(1),
            "samples",
            ParquetPhysicalType::Int32,
        )]);

        let row_groups = vec![
            RowGroupDescriptor::new(
                4,
                vec![ColumnChunkDescriptor::new(FieldId::new(1), 4, 32).unwrap()],
            )
            .unwrap(),
        ];

        let dataset = ParquetDatasetDescriptor::new(schema, row_groups).unwrap();
        let column = dataset.column("samples").unwrap();
        assert!(matches!(column.storage(), ColumnStorage::VariableWidth));
        match column.datatype() {
            Datatype::VarLen { base } => {
                assert!(matches!(
                    base.as_ref(),
                    Datatype::Integer { bits, signed: true, .. } if bits.get() == 32
                ));
            }
            other => panic!("expected VarLen datatype, got {other:?}"),
        }
    }

    #[test]
    fn repeated_group_column_maps_to_varlen_compound_datatype() {
        let schema = SchemaDescriptor::new(vec![FieldDescriptor::group(
            FieldId::new(1),
            "points",
            Repetition::Repeated,
            vec![
                FieldDescriptor::required(FieldId::new(2), "x", ParquetPhysicalType::Float),
                FieldDescriptor::required(FieldId::new(3), "y", ParquetPhysicalType::Float),
            ],
        )]);

        let row_groups = vec![
            RowGroupDescriptor::new(
                2,
                vec![ColumnChunkDescriptor::new(FieldId::new(1), 2, 16).unwrap()],
            )
            .unwrap(),
        ];

        let dataset = ParquetDatasetDescriptor::new(schema, row_groups).unwrap();
        let column = dataset.column("points").unwrap();
        assert!(column.is_nested());
        assert!(matches!(column.storage(), ColumnStorage::Nested));
        match column.datatype() {
            Datatype::VarLen { base } => match base.as_ref() {
                Datatype::Compound { fields, size } => {
                    assert_eq!(*size, 8);
                    assert_eq!(fields.len(), 2);
                    assert_eq!(fields[0].name, "x");
                    assert_eq!(fields[0].offset, 0);
                    assert_eq!(fields[1].name, "y");
                    assert_eq!(fields[1].offset, 4);
                }
                other => panic!("expected Compound base datatype, got {other:?}"),
            },
            other => panic!("expected VarLen datatype, got {other:?}"),
        }
    }
}
