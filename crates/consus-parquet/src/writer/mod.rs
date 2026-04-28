//! Canonical Parquet writer and wire encoders.
//!
//! This module replaces the prior writer scaffold with a real, testable
//! encoding implementation for footer metadata and page headers. It also
//! provides the canonical write-planning surface used by the rest of the
//! crate.
//!
//! ## Scope
//!
//! - Thrift compact binary encoding for footer metadata structs
//! - Thrift compact binary encoding for page headers
//! - Canonical writer-side schema lowering over nested/group fields
//! - Canonical row-source row/value model for future page emission
//! - Complete file assembly helpers for `PAR1` trailer emission
//!
//! ## Invariants
//!
//! - Schema order is preserved.
//! - Nested/group fields lower to deterministic leaf paths.
//! - Encoded footer metadata round-trips through the existing decoders.
//! - Encoded page headers round-trip through the existing decoders.
//! - Trailer length and magic are emitted in the Parquet format order.
//!
//! ## Non-goals
//!
//! - This module does not fabricate row payload values.
//! - This module does not clone type-specific writer APIs.
//! - Unsupported row-to-page synthesis is reported explicitly.
//!
//! ## Architecture
//!
//! ```text
//! writer/
//! ├── ParquetWriter           # Canonical writer entry point
//! ├── WritePlan               # Lowered schema + row-group emission plan
//! ├── RowSource               # Canonical row iteration trait
//! ├── Thrift encoder          # Footer metadata serialization
//! ├── Page-header encoder     # Page header serialization
//! └── File assembly           # Payload + footer + trailer emission
//! ```

use alloc::{format, string::String, vec, vec::Vec};

use consus_core::{Error, Result};

use crate::dataset::ParquetDatasetDescriptor;
use crate::schema::field::{FieldDescriptor, FieldId, SchemaDescriptor};
use crate::schema::logical::Repetition;
use crate::schema::physical::ParquetPhysicalType;
use crate::wire::metadata::{
    ColumnChunkMetadata, ColumnMetadata, FileMetadata, KeyValue, RowGroupMetadata, SchemaElement,
};
use crate::wire::page::{
    DataPageHeader, DataPageHeaderV2, DictionaryPageHeader, PageHeader, PageType,
};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

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
    field_id: FieldId,
    path: Vec<String>,
    physical_type: ParquetPhysicalType,
    repetition: Repetition,
    max_rep_level: i32,
    max_def_level: i32,
    /// Index into `schema.fields()` (and into `row.columns()`) for this leaf.
    top_field_idx: usize,
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
    schema: SchemaDescriptor,
    leaves: Vec<LeafColumnPlan>,
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

/// Canonical Parquet writer.
pub struct ParquetWriter {
    compression: crate::encoding::compression::CompressionCodec,
    /// Maximum rows per row group. `None` = all rows in one group.
    row_group_size: Option<usize>,
    /// Maximum rows per data page within a column chunk. `None` = one page per column chunk.
    page_row_limit: Option<usize>,
}

impl ParquetWriter {
    /// Construct a writer with uncompressed pages.
    #[must_use]
    pub fn new() -> Self {
        Self {
            compression: crate::encoding::compression::CompressionCodec::Uncompressed,
            row_group_size: None,
            page_row_limit: None,
        }
    }

    /// Set page compression codec.
    #[must_use]
    pub fn with_compression(
        mut self,
        compression: crate::encoding::compression::CompressionCodec,
    ) -> Self {
        self.compression = compression;
        self
    }

    /// Set the maximum number of rows per row group.
    ///
    /// `n = 0` reverts to unlimited (all rows in a single row group).
    /// Default: unlimited.
    #[must_use]
    pub fn with_row_group_size(mut self, n: usize) -> Self {
        self.row_group_size = if n == 0 { None } else { Some(n) };
        self
    }

    /// Set the maximum number of rows per data page within a column chunk.
    ///
    /// `0` reverts to unlimited (single page per column chunk).
    /// Default: unlimited.
    #[must_use]
    pub fn with_page_row_limit(mut self, limit: usize) -> Self {
        self.page_row_limit = if limit == 0 { None } else { Some(limit) };
        self
    }

    /// Lower a dataset into a write plan.
    pub fn plan(&self, dataset: &ParquetDatasetDescriptor) -> Result<WritePlan> {
        let mut leaves = Vec::new();
        let mut i = 0usize;
        while i < dataset.columns().len() {
            let column = &dataset.columns()[i];
            lower_column(&mut leaves, column.field(), Vec::new(), 0, 0, i)?;
            i += 1;
        }
        Ok(WritePlan {
            schema: dataset.schema().clone(),
            leaves,
        })
    }

    /// Write a full dataset to an output sink.
    pub fn write_dataset<S: ByteSink>(
        &self,
        dataset: &ParquetDatasetDescriptor,
        rows: &impl RowSource,
        sink: &mut S,
    ) -> Result<()> {
        let plan = self.plan(dataset)?;
        if rows.row_count() != dataset.total_rows() {
            return Err(Error::InvalidFormat {
                message: String::from(
                    "parquet writer row source length must equal dataset row count",
                ),
            });
        }

        let file = build_file_bytes(
            self.compression,
            dataset,
            &plan,
            rows,
            self.row_group_size,
            self.page_row_limit,
        )?;
        sink.write_all(&file)?;
        Ok(())
    }

    /// Build file bytes directly.
    pub fn write_dataset_bytes(
        &self,
        dataset: &ParquetDatasetDescriptor,
        rows: &impl RowSource,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.write_dataset(dataset, rows, &mut out)?;
        Ok(out)
    }
}

fn lower_column(
    leaves: &mut Vec<LeafColumnPlan>,
    field: &FieldDescriptor,
    path: Vec<String>,
    rep: i32,
    def: i32,
    top_field_idx: usize,
) -> Result<()> {
    let mut next_path = path;

    if field.is_group() {
        next_path.push(field.name().to_owned());
        let (child_rep, child_def) = match field.repetition() {
            Repetition::Required => (rep, def),
            Repetition::Optional => (rep, def + 1),
            Repetition::Repeated => (rep + 1, def + 1),
        };
        let mut i = 0usize;
        while i < field.children().len() {
            lower_column(
                leaves,
                &field.children()[i],
                next_path.clone(),
                child_rep,
                child_def,
                top_field_idx,
            )?;
            i += 1;
        }
        return Ok(());
    }

    next_path.push(field.name().to_owned());

    leaves.push(LeafColumnPlan {
        field_id: field.id(),
        path: next_path,
        physical_type: field.physical_type(),
        repetition: field.repetition(),
        max_rep_level: rep
            + if field.repetition() == Repetition::Repeated {
                1
            } else {
                0
            },
        max_def_level: def
            + if field.repetition() == Repetition::Optional
                || field.repetition() == Repetition::Repeated
            {
                1
            } else {
                0
            },
        top_field_idx,
    });
    Ok(())
}

/// Encode one non-Boolean `CellValue` into PLAIN bytes, appending to `out`.
///
/// PLAIN encoding per physical type:
/// - INT32: 4 bytes little-endian
/// - INT64: 8 bytes little-endian
/// - INT96: 12 raw bytes
/// - FLOAT: 4 bytes little-endian IEEE 754
/// - DOUBLE: 8 bytes little-endian IEEE 754
/// - BYTE_ARRAY: 4-byte LE u32 length prefix + raw bytes
/// - FIXED_LEN_BYTE_ARRAY: raw bytes (width from physical type)
///
/// Boolean values are handled separately via `encode_bool_column_plain`
/// because PLAIN BOOLEAN encoding is bit-packed across the entire page.
fn encode_cell_plain(cell: &CellValue<'_>, out: &mut Vec<u8>) -> Result<()> {
    match cell {
        CellValue::Boolean(_) => Err(Error::InvalidFormat {
            message: String::from(
                "parquet: Boolean values must be encoded via encode_bool_column_plain",
            ),
        }),
        CellValue::Int32(v) => {
            out.extend_from_slice(&v.to_le_bytes());
            Ok(())
        }
        CellValue::Int64(v) => {
            out.extend_from_slice(&v.to_le_bytes());
            Ok(())
        }
        CellValue::Int96(arr) => {
            out.extend_from_slice(arr);
            Ok(())
        }
        CellValue::Float(v) => {
            out.extend_from_slice(&v.to_le_bytes());
            Ok(())
        }
        CellValue::Double(v) => {
            out.extend_from_slice(&v.to_le_bytes());
            Ok(())
        }
        CellValue::ByteArray(bytes) => {
            let len = bytes.len() as u32;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(bytes);
            Ok(())
        }
        CellValue::FixedLenByteArray(bytes) => {
            out.extend_from_slice(bytes);
            Ok(())
        }
        CellValue::Null => Err(Error::InvalidFormat {
            message: String::from("parquet: Null value in required column"),
        }),
        CellValue::Group(_) | CellValue::Repeated(_) => Err(Error::UnsupportedFeature {
            feature: String::from(
                "parquet: nested/repeated cell values not yet supported in writer",
            ),
        }),
    }
}

/// Bit-pack boolean values LSB-first per the Parquet PLAIN BOOLEAN encoding.
///
/// value[i] occupies bit `i % 8` of byte `i / 8`. Returns `⌈count / 8⌉` bytes.
fn encode_bool_column_plain(bools: &[bool]) -> Vec<u8> {
    let byte_count = bools.len().saturating_add(7) / 8;
    let mut out = vec![0u8; byte_count];
    let mut i = 0usize;
    while i < bools.len() {
        if bools[i] {
            out[i / 8] |= 1u8 << (i % 8);
        }
        i += 1;
    }
    out
}

/// Recursively traverse a `CellValue` tree following the Dremel algorithm,
/// encoding rep/def levels and values directly into output buffers.
///
/// ## Dremel Algorithm
///
/// For each logical entry in the column (one per leaf repetition unit), this
/// function appends exactly one `(rep_level, def_level)` pair, and for
/// non-null entries, one encoded value.  Accumulating all entries across all
/// rows produces the complete Dremel column encoding.
///
/// ## Parameters
///
/// - `cell`        — current value in the tree, at `field`
/// - `field`       — schema descriptor for the field that `cell` belongs to
/// - `sub_path`    — remaining path segments after `field.name()` to navigate
///                   (`leaf.path()[1..]` for the initial call on a top-level field)
/// - `first_rep`   — repetition level to assign to the FIRST entry this call produces
/// - `rep_so_far`  — number of `Repeated` ancestors above (not including) this field
/// - `def_above`   — accumulated definition level contributed by ancestors above
/// - `rep_levels`, `def_levels`, `val_bytes`, `bool_vals` — output accumulators
/// - `is_bool`     — true when the leaf physical type is `Boolean`
///
/// ## Formal invariant
///
/// After all rows are processed, `rep_levels.len() == def_levels.len()` and each
/// position carries exactly one level pair.  The value count equals the number of
/// positions where `def_level == max_def_level`.
fn traverse_dremel_into(
    cell: &CellValue<'_>,
    field: &FieldDescriptor,
    sub_path: &[String],
    first_rep: i32,
    rep_so_far: i32,
    def_above: i32,
    rep_levels: &mut Vec<i32>,
    def_levels: &mut Vec<i32>,
    val_bytes: &mut Vec<u8>,
    bool_vals: &mut Vec<bool>,
    is_bool: bool,
) -> Result<()> {
    if sub_path.is_empty() {
        // ── At the leaf: encode based on field repetition and cell value ──────
        match field.repetition() {
            Repetition::Required => {
                if let CellValue::Null = cell {
                    return Err(Error::InvalidFormat {
                        message: format!(
                            "parquet: Null value in required leaf field '{}'",
                            field.name()
                        ),
                    });
                }
                rep_levels.push(first_rep);
                def_levels.push(def_above);
                if is_bool {
                    match cell {
                        CellValue::Boolean(b) => bool_vals.push(*b),
                        _ => {
                            return Err(Error::InvalidFormat {
                                message: format!(
                                    "parquet: expected Boolean at required leaf field '{}'",
                                    field.name()
                                ),
                            });
                        }
                    }
                } else {
                    encode_cell_plain(cell, val_bytes)?;
                }
            }
            Repetition::Optional => {
                if let CellValue::Null = cell {
                    rep_levels.push(first_rep);
                    def_levels.push(def_above);
                } else {
                    rep_levels.push(first_rep);
                    def_levels.push(def_above + 1);
                    if is_bool {
                        match cell {
                            CellValue::Boolean(b) => bool_vals.push(*b),
                            _ => {
                                return Err(Error::InvalidFormat {
                                    message: format!(
                                        "parquet: expected Boolean at optional leaf field '{}'",
                                        field.name()
                                    ),
                                });
                            }
                        }
                    } else {
                        encode_cell_plain(cell, val_bytes)?;
                    }
                }
            }
            Repetition::Repeated => {
                let this_rep = rep_so_far + 1;
                match cell {
                    CellValue::Repeated(items) if items.is_empty() => {
                        // Empty list: one null entry (rep=first_rep, def=def_above).
                        rep_levels.push(first_rep);
                        def_levels.push(def_above);
                    }
                    CellValue::Repeated(items) => {
                        let mut i = 0usize;
                        while i < items.len() {
                            let rep = if i == 0 { first_rep } else { this_rep };
                            rep_levels.push(rep);
                            def_levels.push(def_above + 1);
                            if is_bool {
                                match &items[i] {
                                    CellValue::Boolean(b) => bool_vals.push(*b),
                                    _ => {
                                        return Err(Error::InvalidFormat {
                                            message: format!(
                                                "parquet: expected Boolean item in repeated leaf field '{}'",
                                                field.name()
                                            ),
                                        });
                                    }
                                }
                            } else {
                                encode_cell_plain(&items[i], val_bytes)?;
                            }
                            i += 1;
                        }
                    }
                    _ => {
                        return Err(Error::InvalidFormat {
                            message: format!(
                                "parquet: expected Repeated cell at repeated leaf field '{}'",
                                field.name()
                            ),
                        });
                    }
                }
            }
        }
        return Ok(());
    }

    // ── Not at the leaf: navigate into a group ────────────────────────────────
    let next_name = &sub_path[0];
    let next_sub_path = &sub_path[1..];

    // Locate the child field by name.
    let mut child_idx = 0usize;
    while child_idx < field.children().len() {
        if field.children()[child_idx].name() == next_name.as_str() {
            break;
        }
        child_idx += 1;
    }
    if child_idx >= field.children().len() {
        return Err(Error::InvalidFormat {
            message: format!(
                "parquet: field '{}' has no child named '{}'",
                field.name(),
                next_name
            ),
        });
    }
    let child_field = &field.children()[child_idx];
    // Propagate rep_so_far: add 1 if this field itself is Repeated.
    let child_rep_so_far = rep_so_far + if field.is_repeated() { 1 } else { 0 };

    match field.repetition() {
        Repetition::Required => match cell {
            CellValue::Group(children) => {
                let child_cell = children
                    .get(child_idx)
                    .ok_or_else(|| Error::InvalidFormat {
                        message: format!(
                            "parquet: required group '{}' has fewer children than schema",
                            field.name()
                        ),
                    })?;
                traverse_dremel_into(
                    child_cell,
                    child_field,
                    next_sub_path,
                    first_rep,
                    child_rep_so_far,
                    def_above,
                    rep_levels,
                    def_levels,
                    val_bytes,
                    bool_vals,
                    is_bool,
                )
            }
            _ => Err(Error::InvalidFormat {
                message: format!(
                    "parquet: expected Group cell at required field '{}'",
                    field.name()
                ),
            }),
        },

        Repetition::Optional => match cell {
            CellValue::Null => {
                // Entire optional group is absent: one null entry.
                rep_levels.push(first_rep);
                def_levels.push(def_above);
                Ok(())
            }
            CellValue::Group(children) => {
                let child_cell = children
                    .get(child_idx)
                    .ok_or_else(|| Error::InvalidFormat {
                        message: format!(
                            "parquet: optional group '{}' has fewer children than schema",
                            field.name()
                        ),
                    })?;
                traverse_dremel_into(
                    child_cell,
                    child_field,
                    next_sub_path,
                    first_rep,
                    child_rep_so_far,
                    def_above + 1,
                    rep_levels,
                    def_levels,
                    val_bytes,
                    bool_vals,
                    is_bool,
                )
            }
            _ => Err(Error::InvalidFormat {
                message: format!(
                    "parquet: expected Group or Null at optional field '{}'",
                    field.name()
                ),
            }),
        },

        Repetition::Repeated => {
            let this_rep = rep_so_far + 1;
            match cell {
                CellValue::Repeated(items) if items.is_empty() => {
                    // Empty repeated group: one null entry.
                    rep_levels.push(first_rep);
                    def_levels.push(def_above);
                    Ok(())
                }
                CellValue::Repeated(items) => {
                    let mut i = 0usize;
                    while i < items.len() {
                        let item_rep = if i == 0 { first_rep } else { this_rep };
                        match &items[i] {
                            CellValue::Group(children) => {
                                let child_cell = children.get(child_idx).ok_or_else(|| {
                                    Error::InvalidFormat {
                                        message: format!(
                                            "parquet: repeated group '{}' item {} has fewer children than schema",
                                            field.name(),
                                            i
                                        ),
                                    }
                                })?;
                                traverse_dremel_into(
                                    child_cell,
                                    child_field,
                                    next_sub_path,
                                    item_rep,
                                    child_rep_so_far,
                                    def_above + 1,
                                    rep_levels,
                                    def_levels,
                                    val_bytes,
                                    bool_vals,
                                    is_bool,
                                )?;
                            }
                            _ => {
                                return Err(Error::InvalidFormat {
                                    message: format!(
                                        "parquet: expected Group item in repeated field '{}'",
                                        field.name()
                                    ),
                                });
                            }
                        }
                        i += 1;
                    }
                    Ok(())
                }
                _ => Err(Error::InvalidFormat {
                    message: format!(
                        "parquet: expected Repeated cell at repeated field '{}'",
                        field.name()
                    ),
                }),
            }
        }
    }
}

/// Encoded output for one leaf column in one row group.
///
/// Separates level bytes from value bytes so `build_file_bytes` can pass the
/// full payload to compression and populate `DataPageHeader` correctly.
struct EncodedLeafColumn {
    /// RLE level prefix (rep_levels || def_levels) + PLAIN value bytes.
    /// For required columns, this is value bytes only.
    payload: Vec<u8>,
    /// Total value count, including null / empty-list positions.
    num_values: i32,
    /// Thrift Encoding discriminant for definition levels: 3=RLE, 0=none.
    def_level_encoding: i32,
    /// Thrift Encoding discriminant for repetition levels: 3=RLE, 0=none.
    rep_level_encoding: i32,
}

/// Encode `levels` as a RLE/bit-packing hybrid byte string (Parquet encoding 3).
///
/// Uses pure RLE runs (no bit-packed groups) for simplicity.
/// Each run: unsigned varint (run_len << 1), then value as `ceil(bit_width / 8)` LE bytes.
/// If `bit_width == 0` or `levels.is_empty()`, returns empty Vec.
fn encode_rle_hybrid(levels: &[i32], bit_width: u8) -> Vec<u8> {
    if bit_width == 0 || levels.is_empty() {
        return Vec::new();
    }
    let value_bytes = (bit_width as usize + 7) / 8;
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0usize;
    while i < levels.len() {
        let val = levels[i];
        let mut run_len = 1usize;
        while i + run_len < levels.len() && levels[i + run_len] == val {
            run_len += 1;
        }
        // RLE run header: (run_len << 1) | 0
        encode_unsigned_varint((run_len as u64) << 1, &mut out);
        // Value: LE bytes, low byte first
        let val_u64 = val as u64;
        for k in 0..value_bytes {
            out.push(((val_u64 >> (k * 8)) & 0xFF) as u8);
        }
        i += run_len;
    }
    out
}

/// Encode `levels` for a DataPage v1 level section.
///
/// Returns an empty Vec when `max_level == 0` (required/no-level column).
/// Otherwise returns: 4-byte LE u32 byte count || RLE hybrid bytes.
fn encode_levels_for_page_v1(levels: &[i32], max_level: i32) -> Vec<u8> {
    if max_level == 0 || levels.is_empty() {
        return Vec::new();
    }
    let bit_width = crate::encoding::levels::level_bit_width(max_level);
    let rle = encode_rle_hybrid(levels, bit_width);
    let len = rle.len() as u32;
    let mut out = Vec::with_capacity(4 + rle.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&rle);
    out
}

/// Map `ParquetPhysicalType` to its Thrift Type enum discriminant.
///
/// Per parquet.thrift Type enum:
/// BOOLEAN=0, INT32=1, INT64=2, INT96=3, FLOAT=4, DOUBLE=5,
/// BYTE_ARRAY=6, FIXED_LEN_BYTE_ARRAY=7.
fn physical_type_discriminant(t: ParquetPhysicalType) -> i32 {
    match t {
        ParquetPhysicalType::Boolean => 0,
        ParquetPhysicalType::Int32 => 1,
        ParquetPhysicalType::Int64 => 2,
        ParquetPhysicalType::Int96 => 3,
        ParquetPhysicalType::Float => 4,
        ParquetPhysicalType::Double => 5,
        ParquetPhysicalType::ByteArray => 6,
        ParquetPhysicalType::FixedLenByteArray(_) => 7,
    }
}

/// Encode PLAIN bytes (with Dremel level prefixes when needed) for each leaf column
/// for rows `[row_start, row_end)`.
///
/// Returns one [`EncodedLeafColumn`] per leaf in `plan` order.
///
/// ## Level encoding
///
/// Uses [`traverse_dremel_into`] for all cases, which handles arbitrarily nested
/// schemas via recursive CellValue traversal.  The level sections included in the
/// payload are determined by `max_rep_level` and `max_def_level`:
///
/// | max_rep | max_def | payload layout                            |
/// |---------|---------|-------------------------------------------|
/// | 0       | 0       | value bytes only                          |
/// | 0       | > 0     | RLE def-level section + value bytes       |
/// | > 0     | any     | RLE rep-section + RLE def-section + values|
///
/// ## Errors
///
/// - `InvalidFormat` — Null in required column, wrong cell type for schema, or
///   group/cell structure inconsistent with schema.
fn encode_leaf_columns(
    plan: &WritePlan,
    rows: &impl RowSource,
    row_start: usize,
    row_end: usize,
) -> Result<Vec<EncodedLeafColumn>> {
    let leaf_count = plan.leaves().len();
    let mut result: Vec<EncodedLeafColumn> = Vec::with_capacity(leaf_count);

    let mut col_idx = 0usize;
    while col_idx < leaf_count {
        let leaf = &plan.leaves()[col_idx];
        let top_field = &plan.schema.fields()[leaf.top_field_idx()];
        // sub_path: path segments below the top-level field name.
        // leaf.path()[0] == top_field.name(); leaf.path()[1..] navigates into children.
        let sub_path: &[String] = &leaf.path()[1..];
        let is_bool = leaf.physical_type() == ParquetPhysicalType::Boolean;

        let mut rep_levels: Vec<i32> = Vec::new();
        let mut def_levels: Vec<i32> = Vec::new();
        let mut val_bytes: Vec<u8> = Vec::new();
        let mut bool_vals: Vec<bool> = Vec::new();

        let mut row_idx = row_start;
        while row_idx < row_end {
            let row = rows.row(row_idx)?;
            let top_cell =
                row.columns()
                    .get(leaf.top_field_idx())
                    .ok_or_else(|| Error::InvalidFormat {
                        message: format!(
                            "parquet: row {} has no column {}",
                            row_idx,
                            leaf.top_field_idx()
                        ),
                    })?;
            traverse_dremel_into(
                top_cell,
                top_field,
                sub_path,
                0, // first_rep = 0 at the start of each new top-level row
                0, // rep_so_far = 0 at the top level
                0, // def_above = 0 initially
                &mut rep_levels,
                &mut def_levels,
                &mut val_bytes,
                &mut bool_vals,
                is_bool,
            )?;
            row_idx += 1;
        }

        let num_values = rep_levels.len() as i32;
        let value_section: Vec<u8> = if is_bool {
            encode_bool_column_plain(&bool_vals)
        } else {
            val_bytes
        };

        let (payload, def_level_encoding, rep_level_encoding) = if leaf.max_rep_level() > 0 {
            let rep_section = encode_levels_for_page_v1(&rep_levels, leaf.max_rep_level());
            let def_section = encode_levels_for_page_v1(&def_levels, leaf.max_def_level());
            let mut p = rep_section;
            p.extend_from_slice(&def_section);
            p.extend_from_slice(&value_section);
            (p, 3i32, 3i32)
        } else if leaf.max_def_level() > 0 {
            let def_section = encode_levels_for_page_v1(&def_levels, leaf.max_def_level());
            let mut p = def_section;
            p.extend_from_slice(&value_section);
            (p, 3i32, 0i32)
        } else {
            (value_section, 0i32, 0i32)
        };

        result.push(EncodedLeafColumn {
            payload,
            num_values,
            def_level_encoding,
            rep_level_encoding,
        });

        col_idx += 1;
    }
    Ok(result)
}

fn build_file_bytes(
    codec: crate::encoding::compression::CompressionCodec,
    dataset: &ParquetDatasetDescriptor,
    plan: &WritePlan,
    rows: &impl RowSource,
    row_group_size: Option<usize>,
    page_row_limit: Option<usize>,
) -> Result<Vec<u8>> {
    let schema_elements = build_schema_elements(dataset.schema());
    let row_count = rows.row_count();
    let leaf_count = plan.leaves().len();

    // effective_group_size: None or 0 → all rows in one group (row_count.max(1)
    // prevents a zero divisor when row_count == 0, ensuring exactly one group).
    let effective_group_size = match row_group_size {
        None | Some(0) => row_count.max(1),
        Some(n) => n,
    };

    let mut file: Vec<u8> = Vec::new();
    file.extend_from_slice(PARQUET_MAGIC);

    let mut all_row_groups: Vec<RowGroupMetadata> = Vec::new();

    // Always execute at least once to guarantee ≥ 1 row group in the output,
    // even when row_count == 0.
    let mut group_start = 0usize;
    loop {
        let group_end = (group_start + effective_group_size).min(row_count);
        let group_rows = group_end - group_start;

        // Build page ranges within this row group.
        // When group_rows == 0 (empty row count edge case), emit one empty page to
        // preserve the invariant of ≥ 1 page per column chunk.
        let page_ranges: Vec<(usize, usize)> = if group_rows == 0 {
            vec![(group_start, group_end)]
        } else {
            let effective_page_rows = match page_row_limit {
                Some(p) if p > 0 => p,
                _ => group_rows, // None or 0 → all rows in one page
            };
            let mut ranges = Vec::new();
            let mut ps = group_start;
            while ps < group_end {
                let pe = (ps + effective_page_rows).min(group_end);
                ranges.push((ps, pe));
                ps = pe;
            }
            ranges
        };

        // Encode all leaves for each page range; transpose to [leaf_idx][page_idx].
        let mut pages_by_column: Vec<Vec<EncodedLeafColumn>> =
            (0..leaf_count).map(|_| Vec::new()).collect();
        for &(ps, pe) in &page_ranges {
            let page_cols = encode_leaf_columns(plan, rows, ps, pe)?;
            for (leaf_idx, enc) in page_cols.into_iter().enumerate() {
                pages_by_column[leaf_idx].push(enc);
            }
        }

        // Emit all pages for each column contiguously, then record ColumnChunkMetadata.
        // Invariants:
        //   data_page_offset  = byte offset of the FIRST page header for this column chunk.
        //   total_uncompressed_size = Σ (page_header_bytes + uncompressed_payload) over all pages.
        //   total_compressed_size   = Σ (page_header_bytes + compressed_payload)   over all pages.
        //   num_values              = Σ enc.num_values over all pages.
        let mut column_chunks: Vec<ColumnChunkMetadata> = Vec::with_capacity(leaf_count);
        let mut col_idx = 0usize;
        while col_idx < leaf_count {
            let leaf = &plan.leaves()[col_idx];
            let data_page_offset = file.len() as i64; // offset of the FIRST page
            let mut total_uncompressed_size: i64 = 0;
            let mut total_compressed_size: i64 = 0;
            let mut total_num_values: i32 = 0;

            for enc in &pages_by_column[col_idx] {
                let compressed_bytes =
                    crate::encoding::compression::compress_page_values(&enc.payload, codec)?;
                let plain_size = enc.payload.len() as i32;
                let compressed_size = compressed_bytes.len() as i32;

                let dph = DataPageHeader {
                    num_values: enc.num_values,
                    encoding: 0, // PLAIN values
                    definition_level_encoding: enc.def_level_encoding,
                    repetition_level_encoding: enc.rep_level_encoding,
                };
                let ph = PageHeader {
                    type_: PageType::DataPage,
                    uncompressed_page_size: plain_size,
                    compressed_page_size: compressed_size,
                    crc: None,
                    data_page_header: Some(dph),
                    dictionary_page_header: None,
                    data_page_header_v2: None,
                };

                let mut hdr_bytes: Vec<u8> = Vec::new();
                encode_page_header(&ph, &mut hdr_bytes);

                total_uncompressed_size += (hdr_bytes.len() + enc.payload.len()) as i64;
                total_compressed_size += (hdr_bytes.len() + compressed_bytes.len()) as i64;
                total_num_values += enc.num_values;
                file.extend_from_slice(&hdr_bytes);
                file.extend_from_slice(&compressed_bytes);
            }

            let col_meta = ColumnMetadata {
                type_: physical_type_discriminant(leaf.physical_type()),
                encodings: vec![0], // PLAIN
                path_in_schema: leaf.path().to_vec(),
                codec: codec as i32,
                num_values: total_num_values as i64,
                total_uncompressed_size,
                total_compressed_size,
                data_page_offset,
                index_page_offset: None,
                dictionary_page_offset: None,
            };
            column_chunks.push(ColumnChunkMetadata {
                file_path: None,
                file_offset: data_page_offset,
                meta_data: Some(col_meta),
            });
            col_idx += 1;
        }

        let total_rg_bytes: i64 = column_chunks
            .iter()
            .map(|c| c.meta_data.as_ref().map_or(0, |m| m.total_compressed_size))
            .sum();

        all_row_groups.push(RowGroupMetadata {
            columns: column_chunks,
            total_byte_size: total_rg_bytes,
            num_rows: group_rows as i64,
            file_offset: None,
            total_compressed_size: Some(total_rg_bytes),
            ordinal: None,
        });

        group_start = group_end;
        if group_start >= row_count {
            break;
        }
    }

    let metadata = FileMetadata {
        version: 2,
        schema: schema_elements,
        num_rows: row_count as i64,
        row_groups: all_row_groups,
        key_value_metadata: Vec::new(),
        created_by: Some(String::from("consus-parquet")),
    };

    let mut footer: Vec<u8> = Vec::new();
    encode_file_metadata(&metadata, &mut footer);

    file.extend_from_slice(&footer);
    let footer_len = footer.len() as u32;
    file.extend_from_slice(&footer_len.to_le_bytes());
    file.extend_from_slice(PARQUET_MAGIC);

    Ok(file)
}

fn encode_unsigned_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn zigzag_i16(value: i16) -> u64 {
    ((value << 1) ^ (value >> 15)) as u16 as u64
}

fn zigzag_i32(value: i32) -> u64 {
    ((value << 1) ^ (value >> 31)) as u32 as u64
}

fn zigzag_i64(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn encode_stop(out: &mut Vec<u8>) {
    out.push(0x00);
}

fn encode_list_header(elem_type: u8, len: usize, out: &mut Vec<u8>) {
    if len <= 14 {
        out.push(((len as u8) << 4) | (elem_type & 0x0F));
    } else {
        out.push(0xF0 | (elem_type & 0x0F));
        encode_unsigned_varint(len as u64, out);
    }
}

// ── Thrift compact binary field emitters with correct relative-delta tracking ──
//
// Thrift compact binary encodes each field header as:
//   high nibble = delta from previous field ID in this struct
//   low  nibble = type code
// `last` must be initialized to 0 at the start of each struct and is updated
// by every field emitter so that consecutive optional fields produce the right
// relative delta even when earlier fields were omitted.

#[inline]
fn field_header(field_id: i16, type_code: u8, last: &mut i16, out: &mut Vec<u8>) {
    let delta = (field_id - *last) as u8;
    *last = field_id;
    out.push((delta << 4) | (type_code & 0x0F));
}

#[inline]
fn field_i32(field_id: i16, value: i32, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x05, last, out);
    encode_unsigned_varint(zigzag_i32(value), out);
}

#[inline]
fn field_i64(field_id: i16, value: i64, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x06, last, out);
    encode_unsigned_varint(zigzag_i64(value), out);
}

#[inline]
fn field_i16(field_id: i16, value: i16, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x04, last, out);
    encode_unsigned_varint(zigzag_i16(value), out);
}

#[inline]
fn field_binary(field_id: i16, bytes: &[u8], last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x08, last, out);
    encode_unsigned_varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

#[inline]
fn field_bool(field_id: i16, value: bool, last: &mut i16, out: &mut Vec<u8>) {
    let tc = if value { 0x01 } else { 0x02 };
    let delta = (field_id - *last) as u8;
    *last = field_id;
    out.push((delta << 4) | tc);
}

#[inline]
fn field_list(field_id: i16, elem_type: u8, count: usize, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x09, last, out);
    encode_list_header(elem_type, count, out);
}

/// Emit the struct field header (type=0x0C). The caller writes struct content
/// then `encode_stop`.
#[inline]
fn field_struct_header(field_id: i16, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x0C, last, out);
}

fn encode_schema_element(element: &SchemaElement, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    if let Some(t) = element.type_ {
        field_i32(1, t, &mut last, out);
    }
    if let Some(tl) = element.type_length {
        field_i32(2, tl, &mut last, out);
    }
    if let Some(r) = element.repetition_type {
        field_i32(3, r, &mut last, out);
    }
    field_binary(4, element.name.as_bytes(), &mut last, out);
    if let Some(nc) = element.num_children {
        field_i32(5, nc, &mut last, out);
    }
    if let Some(ct) = element.converted_type {
        field_i32(6, ct, &mut last, out);
    }
    if let Some(scale) = element.scale {
        field_i32(7, scale, &mut last, out);
    }
    if let Some(precision) = element.precision {
        field_i32(8, precision, &mut last, out);
    }
    if let Some(fid) = element.field_id {
        field_i32(9, fid, &mut last, out);
    }
    encode_stop(out);
}

fn encode_key_value(kv: &KeyValue, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_binary(1, kv.key.as_bytes(), &mut last, out);
    if let Some(value) = &kv.value {
        field_binary(2, value.as_bytes(), &mut last, out);
    }
    encode_stop(out);
}

fn encode_column_metadata(meta: &ColumnMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, meta.type_, &mut last, out);
    if !meta.encodings.is_empty() {
        field_list(2, 0x05, meta.encodings.len(), &mut last, out);
        for encoding in &meta.encodings {
            encode_unsigned_varint(zigzag_i32(*encoding), out);
        }
    }
    if !meta.path_in_schema.is_empty() {
        field_list(3, 0x08, meta.path_in_schema.len(), &mut last, out);
        for path in &meta.path_in_schema {
            encode_unsigned_varint(path.len() as u64, out);
            out.extend_from_slice(path.as_bytes());
        }
    }
    field_i32(4, meta.codec, &mut last, out);
    field_i64(5, meta.num_values, &mut last, out);
    field_i64(6, meta.total_uncompressed_size, &mut last, out);
    field_i64(7, meta.total_compressed_size, &mut last, out);
    // Field 8 does not exist in parquet.thrift ColumnMetaData; field 9 is
    // data_page_offset, so delta from field 7 is 2.
    field_i64(9, meta.data_page_offset, &mut last, out);
    if let Some(value) = meta.index_page_offset {
        field_i64(10, value, &mut last, out);
    }
    if let Some(value) = meta.dictionary_page_offset {
        field_i64(11, value, &mut last, out);
    }
    encode_stop(out);
}

fn encode_column_chunk_metadata(meta: &ColumnChunkMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    if let Some(file_path) = &meta.file_path {
        field_binary(1, file_path.as_bytes(), &mut last, out);
    }
    field_i64(2, meta.file_offset, &mut last, out);
    if let Some(inner) = &meta.meta_data {
        field_struct_header(3, &mut last, out);
        encode_column_metadata(inner, out);
    }
    encode_stop(out);
}

fn encode_row_group_metadata(meta: &RowGroupMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    if !meta.columns.is_empty() {
        field_list(1, 0x0C, meta.columns.len(), &mut last, out);
        for column in &meta.columns {
            encode_column_chunk_metadata(column, out);
        }
    }
    field_i64(2, meta.total_byte_size, &mut last, out);
    field_i64(3, meta.num_rows, &mut last, out);
    // Field 4 does not exist in parquet.thrift RowGroup; field 5 is
    // file_offset, so delta from field 3 is 2.
    if let Some(value) = meta.file_offset {
        field_i64(5, value, &mut last, out);
    }
    if let Some(value) = meta.total_compressed_size {
        field_i64(6, value, &mut last, out);
    }
    if let Some(value) = meta.ordinal {
        field_i16(7, value, &mut last, out);
    }
    encode_stop(out);
}

fn encode_file_metadata(meta: &FileMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, meta.version, &mut last, out);
    field_list(2, 0x0C, meta.schema.len(), &mut last, out);
    for element in &meta.schema {
        encode_schema_element(element, out);
    }
    field_i64(3, meta.num_rows, &mut last, out);
    field_list(4, 0x0C, meta.row_groups.len(), &mut last, out);
    for row_group in &meta.row_groups {
        encode_row_group_metadata(row_group, out);
    }
    if !meta.key_value_metadata.is_empty() {
        field_list(5, 0x0C, meta.key_value_metadata.len(), &mut last, out);
        for kv in &meta.key_value_metadata {
            encode_key_value(kv, out);
        }
    }
    if let Some(created_by) = &meta.created_by {
        field_binary(6, created_by.as_bytes(), &mut last, out);
    }
    encode_stop(out);
}

fn encode_page_header(header: &PageHeader, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.type_ as i32, &mut last, out);
    field_i32(2, header.uncompressed_page_size, &mut last, out);
    field_i32(3, header.compressed_page_size, &mut last, out);
    if let Some(crc) = header.crc {
        field_i32(4, crc, &mut last, out);
    }
    match header.type_ {
        PageType::DataPage => {
            if let Some(dph) = &header.data_page_header {
                field_struct_header(5, &mut last, out);
                encode_data_page_header(dph, out);
            }
        }
        PageType::DictionaryPage => {
            if let Some(dph) = &header.dictionary_page_header {
                field_struct_header(7, &mut last, out);
                encode_dictionary_page_header(dph, out);
            }
        }
        PageType::DataPageV2 => {
            if let Some(dph) = &header.data_page_header_v2 {
                field_struct_header(8, &mut last, out);
                encode_data_page_header_v2(dph, out);
            }
        }
        PageType::IndexPage => {}
    }
    encode_stop(out);
}

fn encode_data_page_header(header: &DataPageHeader, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.num_values, &mut last, out);
    field_i32(2, header.encoding, &mut last, out);
    field_i32(3, header.definition_level_encoding, &mut last, out);
    field_i32(4, header.repetition_level_encoding, &mut last, out);
    encode_stop(out);
}

fn encode_dictionary_page_header(header: &DictionaryPageHeader, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.num_values, &mut last, out);
    field_i32(2, header.encoding, &mut last, out);
    if let Some(sorted) = header.is_sorted {
        field_bool(3, sorted, &mut last, out);
    }
    encode_stop(out);
}

fn encode_data_page_header_v2(header: &DataPageHeaderV2, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.num_values, &mut last, out);
    field_i32(2, header.num_nulls, &mut last, out);
    field_i32(3, header.num_rows, &mut last, out);
    field_i32(4, header.encoding, &mut last, out);
    field_i32(5, header.definition_levels_byte_length, &mut last, out);
    field_i32(6, header.repetition_levels_byte_length, &mut last, out);
    if let Some(compressed) = header.is_compressed {
        field_bool(7, compressed, &mut last, out);
    }
    encode_stop(out);
}

fn build_schema_elements(schema: &SchemaDescriptor) -> Vec<SchemaElement> {
    let mut elements = Vec::new();
    elements.push(SchemaElement {
        type_: None,
        type_length: None,
        repetition_type: Some(0),
        name: String::from("schema"),
        num_children: Some(schema.field_count() as i32),
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
    });

    let mut i = 0usize;
    while i < schema.fields().len() {
        push_schema_element(&schema.fields()[i], &mut elements);
        i += 1;
    }

    elements
}

fn push_schema_element(field: &FieldDescriptor, out: &mut Vec<SchemaElement>) {
    let repetition_type = match field.repetition() {
        Repetition::Required => Some(0),
        Repetition::Optional => Some(1),
        Repetition::Repeated => Some(2),
    };

    if field.is_group() {
        out.push(SchemaElement {
            type_: None,
            type_length: None,
            repetition_type,
            name: field.name().to_owned(),
            num_children: Some(field.children().len() as i32),
            converted_type: None,
            scale: None,
            precision: None,
            field_id: Some(field.id().get() as i32),
        });
        let mut i = 0usize;
        while i < field.children().len() {
            push_schema_element(&field.children()[i], out);
            i += 1;
        }
    } else {
        let (type_, type_length) = match field.physical_type() {
            ParquetPhysicalType::Boolean => (Some(0), None),
            ParquetPhysicalType::Int32 => (Some(1), None),
            ParquetPhysicalType::Int64 => (Some(2), None),
            ParquetPhysicalType::Int96 => (Some(3), None),
            ParquetPhysicalType::Float => (Some(4), None),
            ParquetPhysicalType::Double => (Some(5), None),
            ParquetPhysicalType::ByteArray => (Some(6), None),
            ParquetPhysicalType::FixedLenByteArray(width) => (Some(7), Some(width as i32)),
        };

        out.push(SchemaElement {
            type_,
            type_length,
            repetition_type,
            name: field.name().to_owned(),
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: Some(field.id().get() as i32),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    vec![
                        crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), 1, 1).unwrap(),
                    ],
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
                    vec![
                        crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), 1, 1).unwrap(),
                    ],
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
                    vec![
                        crate::dataset::ColumnChunkDescriptor::new(FieldId::new(1), 1, 1).unwrap(),
                    ],
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
        // Row 0: x=7, y=3.14
        // Row 1: x=42, y=-1.0
        struct Rows;
        impl RowSource for Rows {
            fn row_count(&self) -> usize {
                2
            }
            fn row(&self, idx: usize) -> Result<RowValue<'_>> {
                let (xi, yf): (i32, f64) = if idx == 0 { (7, 3.14) } else { (42, -1.0) };
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
            matches!(&y_vals, crate::encoding::column::ColumnValues::Double(v) if *v == alloc::vec![3.14, -1.0])
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
}

#[cfg(test)]
mod tests_extra;
