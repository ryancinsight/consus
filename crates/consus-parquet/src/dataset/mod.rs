//! Canonical Parquet dataset and projection model.
//!
//! ## Specification
//!
//! This module defines the authoritative in-memory dataset descriptors used
//! by `consus-parquet` before wire-level decoding is introduced.
//!
//! A dataset descriptor is composed of:
//! - one validated `SchemaDescriptor`
//! - one or more row groups
//! - one column descriptor per top-level schema field
//! - exact row counts and byte counts for each row group and column chunk
//!
//! ## Invariants
//!
//! - `columns.len() == schema.field_count()`
//! - `row_groups.len() > 0`
//! - `total_rows = sum(row_group.row_count)`
//! - each row group contains exactly one chunk per top-level schema field
//! - each projected column exists in the source schema
//! - projection preserves source field order
//!
//! ## Non-goals
//!
//! - No Parquet wire decoding is implemented here.
//! - No fabricated payload values are stored here.
//! - No public API claims file-read support.
//!
//! ## Architecture
//!
//! ```text
//! dataset/
//! +-- ParquetColumnDescriptor   # Canonical top-level column metadata
//! +-- ColumnChunkDescriptor     # Per-row-group physical chunk metadata
//! +-- RowGroupDescriptor        # Row-group row count and chunk set
//! +-- ParquetDatasetDescriptor  # Whole-dataset validated descriptor
//! +-- ColumnProjection          # One projected column
//! +-- ParquetProjection         # Ordered projected dataset view
//! ```

use alloc::{boxed::Box, string::String, vec::Vec};

use consus_core::{CompoundField, Datatype, Error, Result, Shape};

use crate::conversion::parquet_field_to_core;
use crate::schema::logical::Repetition;
use crate::schema::physical::ParquetPhysicalType;
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

    /// Build an ordered projection over a subset of top-level columns.
    pub fn project(&self, names: &[&str]) -> Result<ParquetProjection> {
        if names.is_empty() {
            return Err(Error::InvalidFormat {
                message: String::from("parquet projection must contain at least one column"),
            });
        }

        let mut projected = Vec::with_capacity(names.len());
        let mut i = 0;
        while i < self.columns.len() {
            if names.iter().any(|name| *name == self.columns[i].name()) {
                projected.push(ColumnProjection {
                    ordinal: i,
                    column: self.columns[i].clone(),
                });
            }
            i += 1;
        }

        if projected.len() != names.len() {
            return Err(Error::InvalidFormat {
                message: String::from("parquet projection references an unknown column"),
            });
        }

        Ok(ParquetProjection {
            source_total_rows: self.total_rows,
            columns: projected,
        })
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

/// One projected column with its source ordinal.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnProjection {
    ordinal: usize,
    column: ParquetColumnDescriptor,
}

impl ColumnProjection {
    /// Source ordinal in the dataset schema.
    #[must_use]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Borrow the projected column descriptor.
    #[must_use]
    pub fn column(&self) -> &ParquetColumnDescriptor {
        &self.column
    }
}

/// Ordered projected dataset view.
#[derive(Debug, Clone, PartialEq)]
pub struct ParquetProjection {
    source_total_rows: usize,
    columns: Vec<ColumnProjection>,
}

impl ParquetProjection {
    /// Borrow projected columns in source schema order.
    #[must_use]
    pub fn columns(&self) -> &[ColumnProjection] {
        &self.columns
    }

    /// Number of projected columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Total rows inherited from the source dataset.
    #[must_use]
    pub fn total_rows(&self) -> usize {
        self.source_total_rows
    }
}

/// Reconstruct a `SchemaDescriptor` from a flat list of Parquet `SchemaElement`s.
///
/// ## Specification
///
/// Parquet stores the schema as a flat pre-order DFS list of nodes.
/// The first element is always the root node (a group with `num_children` set).
/// Each group node's children immediately follow it in the list.
///
/// ## Invariants
///
/// - The root node must have `num_children >= 1`.
/// - Each child is either a leaf (type_ is Some) or a group (num_children is Some and > 0).
/// - Leaf nodes map to `FieldDescriptor` scalar fields.
/// - Group nodes map to `FieldDescriptor::group` with recursively parsed children.
/// - Schemas containing nested groups return `Ok(...)` with group fields.
///
/// ## Field ID assignment
///
/// Field IDs are taken from `schema_element.field_id` when present.
/// If absent, sequential IDs are assigned starting from 1.
#[cfg(feature = "alloc")]
pub fn schema_elements_to_schema(
    elements: &[crate::wire::metadata::SchemaElement],
) -> Result<SchemaDescriptor> {
    if elements.is_empty() {
        return Err(Error::InvalidFormat {
            message: String::from("schema elements list must not be empty"),
        });
    }
    let num_children = elements[0].num_children.unwrap_or(0) as usize;
    if num_children == 0 {
        return Err(Error::InvalidFormat {
            message: String::from("parquet schema root must have at least one child field"),
        });
    }
    let mut id_seq: u32 = 1;
    let (fields, _consumed) = parse_fields(elements, 1, num_children, &mut id_seq)?;
    Ok(SchemaDescriptor::new(fields))
}

/// Recursive pre-order DFS parser for the flat Parquet schema element list.
///
/// Returns the parsed fields and the total number of `SchemaElement` entries consumed
/// (including all descendants of group nodes).
///
/// ## Parameters
///
/// - `elements`: full flat element list
/// - `pos`: index of the first element to consume in this call
/// - `count`: number of fields to parse at this nesting level
/// - `id_seq`: auto-increment counter used when `field_id` is absent
#[cfg(feature = "alloc")]
fn parse_fields(
    elements: &[crate::wire::metadata::SchemaElement],
    mut pos: usize,
    count: usize,
    id_seq: &mut u32,
) -> Result<(Vec<FieldDescriptor>, usize)> {
    let mut fields = Vec::with_capacity(count);
    let mut total_consumed: usize = 0;
    let mut i = 0;
    while i < count {
        if pos >= elements.len() {
            return Err(Error::InvalidFormat {
                message: String::from("parquet schema element list is truncated"),
            });
        }
        let elem = &elements[pos];
        pos += 1;
        total_consumed += 1;

        let field_id = if let Some(fid) = elem.field_id {
            FieldId::new(fid as u32)
        } else {
            let id = FieldId::new(*id_seq);
            *id_seq += 1;
            id
        };

        let name: String = elem.name.clone();

        let repetition = match elem.repetition_type {
            Some(0) => Repetition::Required,
            Some(1) => Repetition::Optional,
            Some(2) => Repetition::Repeated,
            _ => Repetition::Optional,
        };

        let child_count = elem.num_children.unwrap_or(0) as usize;
        if child_count > 0 {
            let (children, children_consumed) = parse_fields(elements, pos, child_count, id_seq)?;
            pos += children_consumed;
            total_consumed += children_consumed;
            fields.push(FieldDescriptor::group(field_id, name, repetition, children));
        } else {
            let physical_type = match elem.type_ {
                Some(0) => ParquetPhysicalType::Boolean,
                Some(1) => ParquetPhysicalType::Int32,
                Some(2) => ParquetPhysicalType::Int64,
                Some(3) => ParquetPhysicalType::Int96,
                Some(4) => ParquetPhysicalType::Float,
                Some(5) => ParquetPhysicalType::Double,
                Some(6) => ParquetPhysicalType::ByteArray,
                Some(7) => {
                    ParquetPhysicalType::FixedLenByteArray(elem.type_length.unwrap_or(0) as usize)
                }
                _ => ParquetPhysicalType::ByteArray,
            };
            let field = match repetition {
                Repetition::Required => FieldDescriptor::required(field_id, name, physical_type),
                Repetition::Optional => {
                    FieldDescriptor::optional(field_id, name, physical_type, None)
                }
                Repetition::Repeated => FieldDescriptor::repeated(field_id, name, physical_type),
            };
            fields.push(field);
        }
        i += 1;
    }
    Ok((fields, total_consumed))
}

/// Build a `ParquetDatasetDescriptor` from decoded Parquet wire metadata.
///
/// ## Specification
///
/// Maps `FileMetadata` to `ParquetDatasetDescriptor` using the following rules:
///
/// 1. Schema is reconstructed from the flat `FileMetadata.schema` list using
///    `schema_elements_to_schema`.
/// 2. For each `RowGroupMetadata`, one `RowGroupDescriptor` is built.
/// 3. Each `ColumnChunkMetadata` maps to one `ColumnChunkDescriptor` using:
///    - `field_id`: taken from the corresponding schema field by position
///    - `row_count`: taken from `meta_data.num_values as usize`
///    - `byte_len`: taken from `meta_data.total_compressed_size as usize`
/// 4. Column chunks are matched to schema fields by their position in the
///    `RowGroupMetadata.columns` list (i.e. columns[i] maps to schema.fields()[i]).
///
/// ## Constraints
///
/// - `meta.row_groups` must be non-empty.
/// - Each `ColumnChunkMetadata` must have `meta_data` present.
/// - The number of columns per row group must equal the number of top-level
///   schema fields (i.e. flat schemas only; nested group columns are not yet
///   supported via this bridge).
///
/// ## Invariants
///
/// - `result.total_rows() == sum(rg.num_rows for rg in meta.row_groups)`
/// - `result.column_count() == schema.field_count()`
#[cfg(feature = "alloc")]
pub fn dataset_from_file_metadata(
    meta: &crate::wire::metadata::FileMetadata,
) -> Result<ParquetDatasetDescriptor> {
    let schema = schema_elements_to_schema(&meta.schema)?;

    let mut row_groups = Vec::with_capacity(meta.row_groups.len());
    let mut rg_idx = 0;
    while rg_idx < meta.row_groups.len() {
        let rg = &meta.row_groups[rg_idx];
        if rg.columns.len() != schema.field_count() {
            return Err(Error::InvalidFormat {
                message: String::from("row group column count does not match schema field count"),
            });
        }
        let mut column_chunks = Vec::with_capacity(rg.columns.len());
        let mut col_idx = 0;
        while col_idx < rg.columns.len() {
            let col = &rg.columns[col_idx];
            let field_id = schema.fields()[col_idx].id();
            let meta_data = col.meta_data.as_ref().ok_or_else(|| Error::InvalidFormat {
                message: String::from("column chunk meta_data is absent"),
            })?;
            // Use the row group's logical row count, not num_values.
            // For repeated columns, num_values counts Dremel entries (which
            // may exceed num_rows), so RowGroupDescriptor validation would
            // fail if we used num_values here.
            let row_count = rg.num_rows as usize;
            let raw_byte_len = meta_data.total_compressed_size as usize;
            let byte_len = if raw_byte_len == 0 { 1 } else { raw_byte_len };
            column_chunks.push(ColumnChunkDescriptor::new(field_id, row_count, byte_len)?);
            col_idx += 1;
        }
        row_groups.push(RowGroupDescriptor::new(
            rg.num_rows as usize,
            column_chunks,
        )?);
        rg_idx += 1;
    }

    ParquetDatasetDescriptor::new(schema, row_groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldDescriptor, LogicalType, ParquetPhysicalType, Repetition};

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
    fn projection_preserves_source_schema_order() {
        let schema = SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "a", ParquetPhysicalType::Int32),
            FieldDescriptor::required(FieldId::new(2), "b", ParquetPhysicalType::Int64),
            FieldDescriptor::required(FieldId::new(3), "c", ParquetPhysicalType::Double),
        ]);

        let row_groups = vec![
            RowGroupDescriptor::new(
                2,
                vec![
                    ColumnChunkDescriptor::new(FieldId::new(1), 2, 8).unwrap(),
                    ColumnChunkDescriptor::new(FieldId::new(2), 2, 16).unwrap(),
                    ColumnChunkDescriptor::new(FieldId::new(3), 2, 16).unwrap(),
                ],
            )
            .unwrap(),
        ];

        let dataset = ParquetDatasetDescriptor::new(schema, row_groups).unwrap();
        let projection = dataset.project(&["c", "a"]).unwrap();

        assert_eq!(projection.total_rows(), 2);
        assert_eq!(projection.column_count(), 2);
        assert_eq!(projection.columns()[0].ordinal(), 0);
        assert_eq!(projection.columns()[0].column().name(), "a");
        assert_eq!(projection.columns()[1].ordinal(), 2);
        assert_eq!(projection.columns()[1].column().name(), "c");
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

    #[cfg(feature = "alloc")]
    #[test]
    fn schema_elements_to_schema_flat() {
        use crate::wire::metadata::SchemaElement;
        let elements = vec![
            SchemaElement {
                name: "schema".into(),
                num_children: Some(2),
                type_: None,
                repetition_type: None,
                field_id: None,
                type_length: None,
                converted_type: None,
                scale: None,
                precision: None,
            },
            SchemaElement {
                name: "id".into(),
                num_children: None,
                type_: Some(2),
                repetition_type: Some(0),
                field_id: Some(1),
                type_length: None,
                converted_type: None,
                scale: None,
                precision: None,
            },
            SchemaElement {
                name: "name".into(),
                num_children: None,
                type_: Some(6),
                repetition_type: Some(1),
                field_id: Some(2),
                type_length: None,
                converted_type: None,
                scale: None,
                precision: None,
            },
        ];
        let schema = schema_elements_to_schema(&elements).unwrap();
        assert_eq!(schema.field_count(), 2);
        assert_eq!(schema.fields()[0].name(), "id");
        assert_eq!(schema.fields()[1].name(), "name");
        assert!(schema.fields()[0].is_required());
        assert!(schema.fields()[1].is_optional());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn dataset_from_file_metadata_roundtrip() {
        use crate::wire::metadata::{
            ColumnChunkMetadata, ColumnMetadata, FileMetadata, RowGroupMetadata, SchemaElement,
        };
        let meta = FileMetadata {
            version: 2,
            schema: vec![
                SchemaElement {
                    name: "schema".into(),
                    num_children: Some(1),
                    type_: None,
                    repetition_type: None,
                    field_id: None,
                    type_length: None,
                    converted_type: None,
                    scale: None,
                    precision: None,
                },
                SchemaElement {
                    name: "x".into(),
                    num_children: None,
                    type_: Some(1),
                    repetition_type: Some(0),
                    field_id: Some(1),
                    type_length: None,
                    converted_type: None,
                    scale: None,
                    precision: None,
                },
            ],
            num_rows: 5,
            row_groups: vec![RowGroupMetadata {
                columns: vec![ColumnChunkMetadata {
                    file_path: None,
                    file_offset: 4,
                    meta_data: Some(ColumnMetadata {
                        type_: 1,
                        encodings: vec![0],
                        path_in_schema: vec!["x".into()],
                        codec: 0,
                        num_values: 5,
                        total_uncompressed_size: 20,
                        total_compressed_size: 20,
                        data_page_offset: 4,
                        index_page_offset: None,
                        dictionary_page_offset: None,
                    }),
                }],
                total_byte_size: 20,
                num_rows: 5,
                file_offset: None,
                total_compressed_size: None,
                ordinal: None,
            }],
            key_value_metadata: vec![],
            created_by: None,
        };

        let dataset = dataset_from_file_metadata(&meta).unwrap();
        assert_eq!(dataset.total_rows(), 5);
        assert_eq!(dataset.column_count(), 1);
        assert_eq!(dataset.columns()[0].name(), "x");
    }
}
