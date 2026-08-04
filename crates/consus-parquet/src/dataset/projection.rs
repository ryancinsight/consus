use alloc::{string::String, vec::Vec};

use consus_core::{Error, Result};

use super::types::{ParquetColumnDescriptor, ParquetDatasetDescriptor};

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

impl ParquetDatasetDescriptor {
    /// Build an ordered projection over a subset of top-level columns.
    pub fn project(&self, names: &[&str]) -> Result<ParquetProjection> {
        if names.is_empty() {
            return Err(Error::InvalidFormat {
                message: String::from("parquet projection must contain at least one column"),
            });
        }

        let mut projected = Vec::with_capacity(names.len());
        let mut i = 0;
        while i < self.columns().len() {
            if names.iter().any(|name| *name == self.columns()[i].name()) {
                projected.push(ColumnProjection {
                    ordinal: i,
                    column: self.columns()[i].clone(),
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
            source_total_rows: self.total_rows(),
            columns: projected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{ColumnChunkDescriptor, RowGroupDescriptor};
    use crate::schema::{FieldDescriptor, FieldId, ParquetPhysicalType, SchemaDescriptor};

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
}
