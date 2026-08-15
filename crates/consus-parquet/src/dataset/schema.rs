use alloc::{string::String, vec::Vec};

use consus_core::{Error, Result};

use crate::schema::logical::Repetition;
use crate::schema::physical::ParquetPhysicalType;
use crate::schema::{FieldDescriptor, FieldId, SchemaDescriptor};

use super::types::{ColumnChunkDescriptor, ParquetDatasetDescriptor, RowGroupDescriptor};

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
    let (fields, _consumed) = parse_fields(elements, 1, num_children, &mut id_seq, 0)?;
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
/// - `depth`: nesting level already entered; a hostile schema can chain
///   single-child group nodes arbitrarily deep (each ~1 input element), so the
///   descent is bounded against the [`ParseBudget`] ceiling before recursing.
fn parse_fields(
    elements: &[crate::wire::metadata::SchemaElement],
    mut pos: usize,
    count: usize,
    id_seq: &mut u32,
    depth: u16,
) -> Result<(Vec<FieldDescriptor>, usize)> {
    let depth = consus_core::ParseBudget::DEFAULT.descend(depth, "parquet schema group depth")?;
    let available = elements.len().saturating_sub(pos);
    if count > available {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "parquet schema child count {count} exceeds remaining elements {available}"
            ),
        });
    }
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
            let (children, children_consumed) =
                parse_fields(elements, pos, child_count, id_seq, depth)?;
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
///    `RowGroupMetadata.columns` list (i.e. `columns\[i\]` maps to
///    `schema.fields()\[i\]`).
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
    use crate::wire::metadata::{
        ColumnChunkMetadata, ColumnMetadata, FileMetadata, RowGroupMetadata, SchemaElement,
    };

    #[test]
    fn schema_elements_to_schema_flat() {
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

    #[test]
    fn schema_elements_reject_child_count_beyond_flat_list() {
        let elements = vec![SchemaElement {
            name: "schema".into(),
            num_children: Some(i32::MAX),
            type_: None,
            repetition_type: None,
            field_id: None,
            type_length: None,
            converted_type: None,
            scale: None,
            precision: None,
        }];

        let error = schema_elements_to_schema(&elements).unwrap_err();

        match error {
            Error::InvalidFormat { message } => {
                assert!(message.contains("schema child count"));
                assert!(message.contains("remaining elements 0"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn dataset_from_file_metadata_roundtrip() {
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

    fn group_element(name: &str, num_children: i32) -> SchemaElement {
        SchemaElement {
            name: name.into(),
            num_children: Some(num_children),
            type_: None,
            repetition_type: None,
            field_id: None,
            type_length: None,
            converted_type: None,
            scale: None,
            precision: None,
        }
    }

    fn leaf_element(name: &str) -> SchemaElement {
        SchemaElement {
            name: name.into(),
            num_children: None,
            type_: Some(6),
            repetition_type: Some(0),
            field_id: Some(1),
            type_length: None,
            converted_type: None,
            scale: None,
            precision: None,
        }
    }

    /// A schema whose group nodes chain single children beyond the depth
    /// ceiling must be rejected, not recursed until the stack overflows.
    #[test]
    fn deeply_nested_group_chain_is_rejected_by_the_depth_ceiling() {
        let depth = usize::from(consus_core::ParseBudget::DEFAULT.max_depth);
        // Root + a chain of single-child groups, then one leaf.
        let mut elements = vec![group_element("root", 1)];
        for i in 0..depth {
            elements.push(group_element(&format!("g{i}"), 1));
        }
        elements.push(leaf_element("leaf"));
        let result = schema_elements_to_schema(&elements);
        assert!(
            matches!(
                result,
                Err(Error::ResourceLimit {
                    what: "parquet schema group depth",
                    ..
                })
            ),
            "a deep group chain must be rejected as a depth resource limit, got {result:?}"
        );
    }

    /// A shallow group nesting (well within the ceiling) still parses.
    #[test]
    fn shallow_group_nesting_still_parses() {
        let elements = vec![
            group_element("root", 1),
            group_element("mid", 1),
            leaf_element("leaf"),
        ];
        let schema = schema_elements_to_schema(&elements).unwrap();
        assert_eq!(schema.field_count(), 1);
    }
}
