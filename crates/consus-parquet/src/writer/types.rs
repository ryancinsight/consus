use alloc::{string::String, vec::Vec};

use consus_core::Result;

use crate::schema::field::{FieldId, SchemaDescriptor};
use crate::schema::logical::Repetition;
use crate::schema::physical::ParquetPhysicalType;

/// Canonical output sink for writer emission.
pub trait ByteSink {
    /// Append bytes to the sink.
    fn write_all(&mut self, bytes: &[u8]) -> Result<()>;
}

impl ByteSink for Vec<u8> {
    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.extend_from_slice(bytes);
        Ok(())
    }
}

/// Logical row source for a canonical Parquet writer.
pub trait RowSource {
    /// Number of logical rows in the source.
    fn row_count(&self) -> usize;

    /// Borrow one row as a sequence of top-level column values.
    fn row(&self, index: usize) -> Result<RowValue<'_>>;
}

/// Canonical row representation used by the writer.
#[derive(Debug, Clone, PartialEq)]
pub struct RowValue<'a> {
    columns: Vec<CellValue<'a>>,
}

impl<'a> RowValue<'a> {
    /// Construct a row from ordered column cells.
    #[must_use]
    pub fn new(columns: Vec<CellValue<'a>>) -> Self {
        Self { columns }
    }

    /// Borrow the row cells in schema order.
    #[must_use]
    pub fn columns(&self) -> &[CellValue<'a>] {
        &self.columns
    }

    /// Number of cells in the row.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Returns `true` if the row contains no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

/// Canonical scalar or nested cell value.
#[derive(Debug, Clone, PartialEq)]
pub enum CellValue<'a> {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Int96([u8; 12]),
    Float(f32),
    Double(f64),
    ByteArray(&'a [u8]),
    FixedLenByteArray(&'a [u8]),
    Group(Vec<CellValue<'a>>),
    Repeated(Vec<CellValue<'a>>),
}

/// Lowered leaf-column plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafColumnPlan {
    pub(super) field_id: FieldId,
    pub(super) path: Vec<String>,
    pub(super) physical_type: ParquetPhysicalType,
    pub(super) repetition: Repetition,
    pub(super) max_rep_level: i32,
    pub(super) max_def_level: i32,
    /// Index into `schema.fields()` (and into `row.columns()`) for this leaf.
    pub(super) top_field_idx: usize,
}

impl LeafColumnPlan {
    /// Stable field identifier.
    #[must_use]
    pub fn field_id(&self) -> FieldId {
        self.field_id
    }

    /// Full schema path from root to leaf.
    #[must_use]
    pub fn path(&self) -> &[String] {
        &self.path
    }

    /// Physical leaf type.
    #[must_use]
    pub fn physical_type(&self) -> ParquetPhysicalType {
        self.physical_type
    }

    /// Repetition kind at the leaf.
    #[must_use]
    pub fn repetition(&self) -> Repetition {
        self.repetition
    }

    /// Maximum repetition level.
    #[must_use]
    pub fn max_rep_level(&self) -> i32 {
        self.max_rep_level
    }

    /// Maximum definition level.
    #[must_use]
    pub fn max_def_level(&self) -> i32 {
        self.max_def_level
    }

    /// Index of the top-level schema field this leaf descends from.
    ///
    /// Matches `schema.fields()[top_field_idx]` and `row.columns()[top_field_idx]`.
    #[must_use]
    pub fn top_field_idx(&self) -> usize {
        self.top_field_idx
    }
}

/// Canonical row-group write plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WritePlan {
    pub(super) schema: SchemaDescriptor,
    pub(super) leaves: Vec<LeafColumnPlan>,
}

impl WritePlan {
    /// Borrow the schema being written.
    #[must_use]
    pub fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    /// Borrow the lowered leaf plans.
    #[must_use]
    pub fn leaves(&self) -> &[LeafColumnPlan] {
        &self.leaves
    }
}
