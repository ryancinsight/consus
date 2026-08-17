use alloc::{format, string::String, vec, vec::Vec};

use consus_core::{Error, Result};

use crate::dataset::ParquetDatasetDescriptor;
use crate::schema::field::FieldDescriptor;
use crate::schema::logical::Repetition;
use crate::schema::physical::ParquetPhysicalType;
use crate::wire::metadata::{ColumnChunkMetadata, ColumnMetadata, FileMetadata, RowGroupMetadata};
use crate::wire::page::{DataPageHeader, PageHeader, PageType};

use super::thrift::{
    build_schema_elements, encode_file_metadata, encode_page_header, encode_unsigned_varint,
};
use super::types::*;

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Canonical Parquet writer.
pub struct ParquetWriter {
    compression: crate::encoding::compression::CompressionCodec,
    /// Maximum rows per row group. `None` = all rows in one group.
    row_group_size: Option<usize>,
    /// Maximum rows per data page within a column chunk. `None` = one page per column chunk.
    page_row_limit: Option<usize>,
}

impl Default for ParquetWriter {
    fn default() -> Self {
        Self::new()
    }
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
/// `value[i]` occupies bit `i % 8` of byte `i / 8`. Returns `⌈count / 8⌉` bytes.
pub(super) fn encode_bool_column_plain(bools: &[bool]) -> Vec<u8> {
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
    let value_bytes = (bit_width as usize).div_ceil(8);
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
        let top_field = &plan.schema().fields()[leaf.top_field_idx()];
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
