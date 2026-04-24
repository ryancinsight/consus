//! FITS ASCII and binary table descriptors and raw row access.
//!
//! ## Scope
//!
//! This module is the authoritative table-domain boundary for `consus-fits`.
//! It defines:
//! - ASCII table descriptors derived from FITS header keywords
//! - binary table descriptors derived from FITS header keywords
//! - row/column metadata for standard table extensions
//! - raw row-oriented access over FITS data-unit spans
//!
//! ## FITS invariants
//!
//! For standard table extensions:
//! - `XTENSION = 'TABLE'` denotes an ASCII table
//! - `XTENSION = 'BINTABLE'` denotes a binary table
//! - `NAXIS1` is the row length in bytes
//! - `NAXIS2` is the number of rows
//! - `TFIELDS` is the number of columns
//! - the logical payload size is `NAXIS1 * NAXIS2`
//!
//! FITS stores table payloads in row-major record order. The data unit is padded
//! with zero bytes to the next 2880-byte boundary.
//!
//! ## Architectural role
//!
//! This module depends on the authoritative `header`, `types`, and
//! `datastructure` modules. It does not duplicate FITS keyword parsing or
//! blocking math.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::{Datatype, Error, Result, Selection, Shape};
use consus_io::ReadAt;

use crate::datastructure::FitsDataSpan;
use crate::header::{FitsHeader, HeaderValue};
use crate::types::{binary_format_element_size, parse_binary_format, tform_to_datatype};

pub(crate) mod decode;

#[cfg(feature = "alloc")]
pub use decode::FitsColumnValue;

/// FITS table column descriptor.
///
/// This type is the single source of truth for per-column metadata extracted
/// from `TTYPEn`, `TFORMn`, `TUNITn`, and related standard keywords.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsTableColumn {
    index: usize,
    name: Option<String>,
    format: String,
    datatype: Datatype,
    byte_width: usize,
    col_offset: usize,
    unit: Option<String>,
    display: Option<String>,
    null: Option<String>,
    scale: Option<f64>,
    zero: Option<f64>,
}

#[cfg(feature = "alloc")]
impl FitsTableColumn {
    /// Construct a table column descriptor from canonical fields.
    ///
    /// The `datatype` and `byte_width` fields are derived from `format` via
    /// [`tform_to_datatype`] and [`binary_format_element_size`] for binary
    /// table columns. For ASCII table columns, callers should pass a
    /// `Datatype::FixedString` with the column width and `byte_width` equal
    /// to the column width.
    pub fn new(
        index: usize,
        name: Option<String>,
        format: String,
        datatype: Datatype,
        byte_width: usize,
        col_offset: usize,
        unit: Option<String>,
        display: Option<String>,
        null: Option<String>,
        scale: Option<f64>,
        zero: Option<f64>,
    ) -> Self {
        Self {
            index,
            name,
            format,
            datatype,
            byte_width,
            col_offset,
            unit,
            display,
            null,
            scale,
            zero,
        }
    }

    /// Construct a binary table column descriptor from a `TFORMn` value.
    ///
    /// Derives `datatype` via [`tform_to_datatype`] and `byte_width` as
    /// `repeat * binary_format_element_size(code)` from [`parse_binary_format`].
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if the TFORM string is not a valid
    /// FITS binary table format code.
    pub fn from_binary_tform(
        index: usize,
        name: Option<String>,
        tform: String,
        unit: Option<String>,
        display: Option<String>,
        null: Option<String>,
        scale: Option<f64>,
        zero: Option<f64>,
    ) -> Result<Self> {
        let datatype = tform_to_datatype(&tform)?;
        let (repeat, code) = parse_binary_format(tform.trim_end())?;
        let byte_width = binary_format_element_size(code) * repeat;
        Ok(Self {
            index,
            name,
            format: tform,
            datatype,
            byte_width,
            col_offset: 0,
            unit,
            display,
            null,
            scale,
            zero,
        })
    }

    /// Return the 1-based FITS column index.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Return the optional column name from `TTYPEn`.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Return the FITS column format token from `TFORMn`.
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Return the canonical [`Datatype`] derived from `TFORMn`.
    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    /// Return the per-column byte width derived from `TFORMn`.
    pub const fn byte_width(&self) -> usize {
        self.byte_width
    }

    /// Return the per-column byte offset from the start of the row.
    pub const fn col_offset(&self) -> usize {
        self.col_offset
    }

    /// Set the per-column byte offset. Used during descriptor construction.
    pub(crate) fn set_col_offset(&mut self, offset: usize) {
        self.col_offset = offset;
    }

    /// Return the optional unit string from `TUNITn`.
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// Return the optional display format from `TDISPn`.
    pub fn display(&self) -> Option<&str> {
        self.display.as_deref()
    }

    /// Return the optional null sentinel from `TNULLn`.
    pub fn null(&self) -> Option<&str> {
        self.null.as_deref()
    }

    /// Return the optional scale factor from `TSCALn`.
    pub const fn scale(&self) -> Option<f64> {
        self.scale
    }

    /// Return the optional zero-point offset from `TZEROn`.
    pub const fn zero(&self) -> Option<f64> {
        self.zero
    }
}

/// Common FITS table descriptor state shared by ASCII and binary tables.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
struct FitsTableDescriptorCore {
    row_len: usize,
    rows: usize,
    columns: Vec<FitsTableColumn>,
    heap_size: usize,
}

#[cfg(feature = "alloc")]
impl FitsTableDescriptorCore {
    fn new(row_len: usize, rows: usize, columns: Vec<FitsTableColumn>, heap_size: usize) -> Self {
        Self {
            row_len,
            rows,
            columns,
            heap_size,
        }
    }

    fn row_len(&self) -> usize {
        self.row_len
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> &[FitsTableColumn] {
        &self.columns
    }

    fn heap_size(&self) -> usize {
        self.heap_size
    }

    fn shape(&self) -> Shape {
        Shape::fixed(&[self.rows])
    }

    fn logical_data_len(&self) -> Result<usize> {
        self.row_len
            .checked_mul(self.rows)
            .and_then(|rows_bytes| rows_bytes.checked_add(self.heap_size))
            .ok_or(Error::Overflow)
    }

    fn is_empty(&self) -> bool {
        self.rows == 0 && self.heap_size == 0
    }
}

/// FITS ASCII table descriptor.
///
/// ASCII tables are identified by `XTENSION = 'TABLE'`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsAsciiTableDescriptor {
    core: FitsTableDescriptorCore,
}

#[cfg(feature = "alloc")]
impl FitsAsciiTableDescriptor {
    /// Parse an ASCII table descriptor from a FITS header.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` when required table keywords are missing
    /// or semantically invalid.
    pub fn from_header(header: &FitsHeader) -> Result<Self> {
        validate_xtension(header, "TABLE")?;
        let core = parse_table_core(header, false)?;
        Ok(Self { core })
    }

    /// Return the row length in bytes.
    pub fn row_len(&self) -> usize {
        self.core.row_len()
    }

    /// Return the number of rows.
    pub fn rows(&self) -> usize {
        self.core.rows()
    }

    /// Return the parsed column descriptors.
    pub fn columns(&self) -> &[FitsTableColumn] {
        self.core.columns()
    }

    /// Return the canonical dataset shape.
    pub fn shape(&self) -> Shape {
        self.core.shape()
    }

    /// Return the logical payload size in bytes.
    pub fn logical_data_len(&self) -> Result<usize> {
        self.core.logical_data_len()
    }

    /// Return whether the table payload is empty.
    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }
}

/// FITS binary table descriptor.
///
/// Binary tables are identified by `XTENSION = 'BINTABLE'`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsBinaryTableDescriptor {
    core: FitsTableDescriptorCore,
}

#[cfg(feature = "alloc")]
impl FitsBinaryTableDescriptor {
    /// Parse a binary table descriptor from a FITS header.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` when required table keywords are missing,
    /// semantically invalid, or when the sum of per-column byte widths does
    /// not equal `NAXIS1`.
    pub fn from_header(header: &FitsHeader) -> Result<Self> {
        validate_xtension(header, "BINTABLE")?;
        let core = parse_table_core(header, true)?;
        let naxis1 = core.row_len();
        let computed_row_len: usize = core.columns().iter().map(|c| c.byte_width()).sum();
        if computed_row_len != naxis1 {
            return invalid_format("FITS BINTABLE column byte widths do not sum to NAXIS1");
        }
        Ok(Self { core })
    }

    /// Return the row length in bytes.
    pub fn row_len(&self) -> usize {
        self.core.row_len()
    }

    /// Return the number of rows.
    pub fn rows(&self) -> usize {
        self.core.rows()
    }

    /// Return the parsed column descriptors.
    pub fn columns(&self) -> &[FitsTableColumn] {
        self.core.columns()
    }

    /// Return the canonical dataset shape.
    pub fn shape(&self) -> Shape {
        self.core.shape()
    }

    /// Return the binary-table heap size in bytes from `PCOUNT`.
    pub fn heap_size(&self) -> usize {
        self.core.heap_size()
    }

    /// Return the logical payload size in bytes, including any heap.
    pub fn logical_data_len(&self) -> Result<usize> {
        self.core.logical_data_len()
    }

    /// Return whether the table payload is empty.
    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }
}

/// Unified FITS table descriptor.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum FitsTableDescriptor {
    /// ASCII table extension descriptor.
    Ascii(FitsAsciiTableDescriptor),
    /// Binary table extension descriptor.
    Binary(FitsBinaryTableDescriptor),
}

#[cfg(feature = "alloc")]
impl FitsTableDescriptor {
    /// Parse a table descriptor from a FITS header by inspecting `XTENSION`.
    pub fn from_header(header: &FitsHeader) -> Result<Self> {
        let xtension = parse_required_string(header, "XTENSION")?.trim_end();
        match xtension {
            "TABLE" => FitsAsciiTableDescriptor::from_header(header).map(Self::Ascii),
            "BINTABLE" => FitsBinaryTableDescriptor::from_header(header).map(Self::Binary),
            _ => invalid_format("unsupported FITS table XTENSION value"),
        }
    }

    /// Return the row length in bytes.
    pub fn row_len(&self) -> usize {
        match self {
            Self::Ascii(value) => value.row_len(),
            Self::Binary(value) => value.row_len(),
        }
    }

    /// Return the number of rows.
    pub fn rows(&self) -> usize {
        match self {
            Self::Ascii(value) => value.rows(),
            Self::Binary(value) => value.rows(),
        }
    }

    /// Return the parsed column descriptors.
    pub fn columns(&self) -> &[FitsTableColumn] {
        match self {
            Self::Ascii(value) => value.columns(),
            Self::Binary(value) => value.columns(),
        }
    }

    /// Return the canonical dataset shape.
    pub fn shape(&self) -> Shape {
        match self {
            Self::Ascii(value) => value.shape(),
            Self::Binary(value) => value.shape(),
        }
    }

    /// Return the logical payload size in bytes.
    pub fn logical_data_len(&self) -> Result<usize> {
        match self {
            Self::Ascii(value) => value.logical_data_len(),
            Self::Binary(value) => value.logical_data_len(),
        }
    }

    /// Return whether this is an ASCII table descriptor.
    pub fn is_ascii(&self) -> bool {
        matches!(self, Self::Ascii(_))
    }

    /// Return whether this is a binary table descriptor.
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary(_))
    }
}

/// Raw FITS table view over a data-unit span.
///
/// This type provides row-oriented raw-byte access through `consus-io::ReadAt`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsTableData {
    descriptor: FitsTableDescriptor,
    span: FitsDataSpan,
}

#[cfg(feature = "alloc")]
impl FitsTableData {
    /// Construct a raw table view from a descriptor and data span.
    pub const fn new(descriptor: FitsTableDescriptor, span: FitsDataSpan) -> Self {
        Self { descriptor, span }
    }

    /// Return the table descriptor.
    pub const fn descriptor(&self) -> &FitsTableDescriptor {
        &self.descriptor
    }

    /// Return the data-unit span.
    pub const fn span(&self) -> FitsDataSpan {
        self.span
    }

    /// Read the entire logical table payload into `buf`.
    pub fn read_all<R: ReadAt>(&self, reader: &R, buf: &mut [u8]) -> Result<usize> {
        let logical_len = self.descriptor.logical_data_len()?;
        if buf.len() < logical_len {
            return Err(Error::BufferTooSmall {
                required: logical_len,
                provided: buf.len(),
            });
        }
        reader.read_at(self.span.offset(), &mut buf[..logical_len])?;
        Ok(logical_len)
    }

    /// Read a single row into `buf`.
    ///
    /// ## Errors
    ///
    /// Returns:
    /// - `Error::SelectionOutOfBounds` if `row_index >= rows`
    /// - `Error::BufferTooSmall` if `buf` is smaller than one row
    pub fn read_row<R: ReadAt>(&self, reader: &R, row_index: usize, buf: &mut [u8]) -> Result<()> {
        let row_len = self.descriptor.row_len();
        let rows = self.descriptor.rows();
        if row_index >= rows {
            return Err(Error::SelectionOutOfBounds);
        }
        if buf.len() < row_len {
            return Err(Error::BufferTooSmall {
                required: row_len,
                provided: buf.len(),
            });
        }
        let row_offset = row_index.checked_mul(row_len).ok_or(Error::Overflow)?;
        let absolute_offset = self
            .span
            .offset()
            .checked_add(u64::try_from(row_offset).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        reader.read_at(absolute_offset, &mut buf[..row_len])?;
        Ok(())
    }

    /// Read a raw selection from the table payload.
    ///
    /// Current support is intentionally strict:
    /// - `Selection::All` reads the full logical payload
    /// - `Selection::None` reads zero bytes
    /// - contiguous 1-D hyperslabs over rows are supported
    ///
    /// Point selections and non-contiguous hyperslabs are rejected.
    pub fn read_selection<R: ReadAt>(
        &self,
        reader: &R,
        selection: &Selection,
        buf: &mut [u8],
    ) -> Result<usize> {
        match selection {
            Selection::All => self.read_all(reader, buf),
            Selection::None => Ok(0),
            Selection::Points(_) => Err(Error::UnsupportedFeature {
                #[cfg(feature = "alloc")]
                feature: "FITS table point selection is not implemented".into(),
            }),
            Selection::Hyperslab(hyperslab) => {
                if hyperslab.rank() != 1 {
                    return Err(Error::UnsupportedFeature {
                        #[cfg(feature = "alloc")]
                        feature: "FITS table hyperslab rank must be 1".into(),
                    });
                }
                let dim = hyperslab.dims[0];
                if dim.stride != 1 || dim.block != 1 {
                    return Err(Error::UnsupportedFeature {
                        #[cfg(feature = "alloc")]
                        feature: "FITS table hyperslab must be contiguous rows".into(),
                    });
                }
                let rows = self.descriptor.rows();
                if dim.start > rows || dim.count > rows.saturating_sub(dim.start) {
                    return Err(Error::SelectionOutOfBounds);
                }
                let row_len = self.descriptor.row_len();
                let byte_len = dim.count.checked_mul(row_len).ok_or(Error::Overflow)?;
                if buf.len() < byte_len {
                    return Err(Error::BufferTooSmall {
                        required: byte_len,
                        provided: buf.len(),
                    });
                }
                let byte_offset = dim.start.checked_mul(row_len).ok_or(Error::Overflow)?;
                let absolute_offset = self
                    .span
                    .offset()
                    .checked_add(u64::try_from(byte_offset).map_err(|_| Error::Overflow)?)
                    .ok_or(Error::Overflow)?;
                reader.read_at(absolute_offset, &mut buf[..byte_len])?;
                Ok(byte_len)
            }
        }
    }

    /// Decode all column cells in one row.
    ///
    /// Returns a `Vec<FitsColumnValue>` with one entry per column, in column order.
    /// Dispatches to the binary or ASCII decoder based on the table kind.
    #[cfg(feature = "alloc")]
    pub fn decode_row<R: ReadAt>(&self, reader: &R, row_index: usize) -> Result<alloc::vec::Vec<decode::FitsColumnValue>> {
        let row_len = self.descriptor.row_len();
        let mut row_buf = alloc::vec![0u8; row_len];
        self.read_row(reader, row_index, &mut row_buf)?;
        let is_binary = self.descriptor.is_binary();
        self.descriptor
            .columns()
            .iter()
            .map(|col| {
                if is_binary {
                    decode::decode_binary_column(&row_buf, col)
                } else {
                    decode::decode_ascii_column(&row_buf, col)
                }
            })
            .collect()
    }

    /// Decode one column across all rows.
    ///
    /// Returns a `Vec<FitsColumnValue>` with one entry per row.
    /// `col_index` is 0-based.
    #[cfg(feature = "alloc")]
    pub fn decode_column<R: ReadAt>(&self, reader: &R, col_index: usize) -> Result<alloc::vec::Vec<decode::FitsColumnValue>> {
        let columns = self.descriptor.columns();
        if col_index >= columns.len() {
            return Err(Error::SelectionOutOfBounds);
        }
        let col = columns[col_index].clone();
        let is_binary = self.descriptor.is_binary();
        let rows = self.descriptor.rows();
        let row_len = self.descriptor.row_len();
        let mut row_buf = alloc::vec![0u8; row_len];
        let mut result = alloc::vec::Vec::with_capacity(rows);
        for row_idx in 0..rows {
            self.read_row(reader, row_idx, &mut row_buf)?;
            let value = if is_binary {
                decode::decode_binary_column(&row_buf, &col)?
            } else {
                decode::decode_ascii_column(&row_buf, &col)?
            };
            result.push(value);
        }
        Ok(result)
    }
}

#[cfg(feature = "alloc")]
fn parse_table_core(header: &FitsHeader, binary: bool) -> Result<FitsTableDescriptorCore> {
    let row_len = parse_required_non_negative_integer(header, "NAXIS1")?;
    let rows = parse_required_non_negative_integer(header, "NAXIS2")?;
    let fields = parse_required_non_negative_integer(header, "TFIELDS")?;
    let heap_size = if binary {
        parse_optional_non_negative_integer(header, "PCOUNT")?.unwrap_or(0)
    } else {
        0
    };
    let mut columns = Vec::with_capacity(fields);
    for index in 1..=fields {
        columns.push(parse_column(header, index, binary)?);
    }
    // Assign col_offset for each column.
    if binary {
        // Binary tables: column offsets are prefix sums (FITS section 7.3, no gaps).
        let mut offset = 0usize;
        for col in &mut columns {
            col.set_col_offset(offset);
            offset = offset.checked_add(col.byte_width()).ok_or(Error::Overflow)?;
        }
    } else {
        // ASCII tables: use TBCOLn (1-based) if present, fall back to prefix sums.
        let mut prefix_offset = 0usize;
        for (i, col) in columns.iter_mut().enumerate() {
            let keyword = indexed_keyword("TBCOL", i + 1)?;
            let tbcol_opt = parse_optional_non_negative_integer(header, &keyword)?;
            let col_offset = match tbcol_opt {
                Some(tbcol) if tbcol > 0 => tbcol - 1,
                _ => prefix_offset,
            };
            col.set_col_offset(col_offset);
            prefix_offset = prefix_offset.checked_add(col.byte_width()).ok_or(Error::Overflow)?;
        }
    }
    Ok(FitsTableDescriptorCore::new(
        row_len, rows, columns, heap_size,
    ))
}

/// Extract the column width from an ASCII TFORM string.
///
/// ASCII TFORM values use formats like `A8`, `I10`, `E16.7`, `F12.5`, etc.
/// The width is the numeric portion before any decimal point.
///
/// ## Derivation
///
/// FITS Standard 4.0 §7.2: ASCII table TFORM is `<rTa<n>.<m>` where `<n>`
/// is the column width in characters. For this function, `<n>` is extracted
/// as the digits before the optional decimal point.
fn parse_ascii_column_width(tform: &str) -> usize {
    let trimmed = tform.trim_end();
    // Skip the format letter (A, I, E, F, D, etc.) to find the width digits.
    let digits_start = trimmed
        .char_indices()
        .find(|(_, c)| c.is_ascii_alphabetic())
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    let rest = &trimmed[digits_start..];
    // Take digits up to an optional decimal point.
    let width_str: &str = rest.split('.').next().unwrap_or(rest);
    width_str.parse::<usize>().unwrap_or(0)
}

#[cfg(feature = "alloc")]
fn parse_column(header: &FitsHeader, index: usize, binary: bool) -> Result<FitsTableColumn> {
    let tform = parse_required_string(header, indexed_keyword("TFORM", index)?.as_str())?
        .trim_end()
        .to_owned();
    let name = parse_optional_string(header, indexed_keyword("TTYPE", index)?.as_str())?
        .map(str::trim_end)
        .map(ToOwned::to_owned);
    let unit = parse_optional_string(header, indexed_keyword("TUNIT", index)?.as_str())?
        .map(str::trim_end)
        .map(ToOwned::to_owned);
    let display = parse_optional_string(header, indexed_keyword("TDISP", index)?.as_str())?
        .map(str::trim_end)
        .map(ToOwned::to_owned);
    let null = parse_optional_string(header, indexed_keyword("TNULL", index)?.as_str())?
        .map(str::trim_end)
        .map(ToOwned::to_owned);
    let scale = parse_optional_real(header, indexed_keyword("TSCAL", index)?.as_str())?;
    let zero = parse_optional_real(header, indexed_keyword("TZERO", index)?.as_str())?;

    if binary {
        FitsTableColumn::from_binary_tform(index, name, tform, unit, display, null, scale, zero)
    } else {
        let width = parse_ascii_column_width(&tform);
        Ok(FitsTableColumn::new(
            index,
            name,
            tform,
            Datatype::FixedString {
                length: width,
                encoding: consus_core::StringEncoding::Ascii,
            },
            width,
            0,
            unit,
            display,
            null,
            scale,
            zero,
        ))
    }
}

#[cfg(feature = "alloc")]
fn validate_xtension(header: &FitsHeader, expected: &str) -> Result<()> {
    let xtension = parse_required_string(header, "XTENSION")?.trim_end();
    if xtension == expected {
        Ok(())
    } else {
        invalid_format("unexpected FITS XTENSION value for table descriptor")
    }
}

#[cfg(feature = "alloc")]
fn parse_required_non_negative_integer(header: &FitsHeader, keyword: &str) -> Result<usize> {
    parse_optional_non_negative_integer(header, keyword)?.ok_or_else(|| Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: alloc::format!("missing required FITS table keyword: {keyword}"),
    })
}

#[cfg(feature = "alloc")]
fn parse_optional_non_negative_integer(
    header: &FitsHeader,
    keyword: &str,
) -> Result<Option<usize>> {
    let Some(value) = parse_optional_integer(header, keyword)? else {
        return Ok(None);
    };
    if value < 0 {
        return invalid_format("FITS table keyword must be a non-negative integer");
    }
    usize::try_from(value)
        .map(Some)
        .map_err(|_| Error::Overflow)
}

#[cfg(feature = "alloc")]
fn parse_optional_integer(header: &FitsHeader, keyword: &str) -> Result<Option<i64>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };
    match card.value() {
        Some(HeaderValue::Integer(value)) => value.to_i64().map(Some),
        Some(_) => invalid_format("FITS table keyword must contain an integer value"),
        None => invalid_format("FITS table keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn parse_optional_real(header: &FitsHeader, keyword: &str) -> Result<Option<f64>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };
    match card.value() {
        Some(HeaderValue::Integer(value)) => Ok(Some(value.to_i64()? as f64)),
        Some(HeaderValue::Real(value)) => value.to_f64().map(Some),
        Some(_) => invalid_format("FITS table keyword must contain a numeric value"),
        None => invalid_format("FITS table keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn parse_required_string<'a>(header: &'a FitsHeader, keyword: &str) -> Result<&'a str> {
    parse_optional_string(header, keyword)?.ok_or_else(|| Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: alloc::format!("missing required FITS table keyword: {keyword}"),
    })
}

#[cfg(feature = "alloc")]
fn parse_optional_string<'a>(header: &'a FitsHeader, keyword: &str) -> Result<Option<&'a str>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };
    match card.value() {
        Some(HeaderValue::String(value)) => Ok(Some(value.as_str())),
        Some(_) => invalid_format("FITS table keyword must contain a string value"),
        None => invalid_format("FITS table keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn indexed_keyword(prefix: &str, index: usize) -> Result<String> {
    if index == 0 {
        return invalid_format("FITS indexed keywords are 1-based");
    }
    Ok(alloc::format!("{prefix}{index}"))
}

#[cfg(feature = "alloc")]
fn invalid_format<T>(message: &str) -> Result<T> {
    Err(Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.into(),
    })
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::{Hyperslab, HyperslabDim};
    use consus_io::MemCursor;

    use crate::datastructure::FitsBlockAlignment;
    use crate::file::parse_extension_header_bytes;

    fn card(text: &str) -> [u8; 80] {
        assert!(text.len() <= 80);
        let mut raw = [b' '; 80];
        raw[..text.len()].copy_from_slice(text.as_bytes());
        raw
    }

    fn header_bytes(cards: &[&str]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for text in cards {
            bytes.extend_from_slice(&card(text));
        }
        let padded_len = FitsBlockAlignment::padded_len(bytes.len());
        bytes.resize(padded_len, b' ');
        bytes
    }

    #[test]
    fn parses_ascii_table_descriptor() {
        let bytes = header_bytes(&[
            "XTENSION= 'TABLE   '",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 24",
            "NAXIS2  = 3",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 2",
            "TTYPE1  = 'NAME    '",
            "TFORM1  = 'A8      '",
            "TTYPE2  = 'VALUE   '",
            "TFORM2  = 'E16.7   '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsAsciiTableDescriptor::from_header(&header).unwrap();

        assert_eq!(descriptor.row_len(), 24);
        assert_eq!(descriptor.rows(), 3);
        assert_eq!(descriptor.columns().len(), 2);
        assert_eq!(descriptor.columns()[0].name(), Some("NAME"));
        assert_eq!(descriptor.columns()[0].format(), "A8");
        assert_eq!(descriptor.columns()[1].name(), Some("VALUE"));
        assert_eq!(descriptor.columns()[1].format(), "E16.7");
        assert_eq!(descriptor.logical_data_len().unwrap(), 72);
    }

    #[test]
    fn parses_binary_table_descriptor_with_heap() {
        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 8",
            "NAXIS2  = 4",
            "PCOUNT  = 16",
            "GCOUNT  = 1",
            "TFIELDS = 2",
            "TTYPE1  = 'X       '",
            "TFORM1  = '1J      '",
            "TTYPE2  = 'Y       '",
            "TFORM2  = '1E      '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsBinaryTableDescriptor::from_header(&header).unwrap();

        assert_eq!(descriptor.row_len(), 8);
        assert_eq!(descriptor.rows(), 4);
        assert_eq!(descriptor.heap_size(), 16);
        assert_eq!(descriptor.columns().len(), 2);
        assert_eq!(descriptor.columns()[0].format(), "1J");
        assert_eq!(descriptor.columns()[1].format(), "1E");
        assert_eq!(descriptor.logical_data_len().unwrap(), 48);
    }

    #[test]
    fn unified_descriptor_dispatches_on_xtension() {
        let ascii_bytes = header_bytes(&[
            "XTENSION= 'TABLE   '",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 8",
            "NAXIS2  = 2",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TFORM1  = 'A8      '",
            "END",
        ]);
        let ascii_header = parse_extension_header_bytes(&ascii_bytes).unwrap();
        let ascii = FitsTableDescriptor::from_header(&ascii_header).unwrap();
        assert!(ascii.is_ascii());
        assert!(!ascii.is_binary());

        let binary_bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 4",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TFORM1  = '1J      '",
            "END",
        ]);
        let binary_header = parse_extension_header_bytes(&binary_bytes).unwrap();
        let binary = FitsTableDescriptor::from_header(&binary_header).unwrap();
        assert!(binary.is_binary());
        assert!(!binary.is_ascii());
    }

    #[test]
    fn reads_single_row() {
        let descriptor = FitsTableDescriptor::Ascii(FitsAsciiTableDescriptor {
            core: FitsTableDescriptorCore::new(
                4,
                3,
                vec![FitsTableColumn::new(
                    1,
                    None,
                    "A4".into(),
                    Datatype::FixedString {
                        length: 4,
                        encoding: consus_core::StringEncoding::Ascii,
                    },
                    4,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                )],
                0,
            ),
        });
        let span = FitsDataSpan::new(0, 12).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(b"AAAABBBBCCCC".to_vec());
        let mut row = [0u8; 4];
        table.read_row(&reader, 1, &mut row).unwrap();
        assert_eq!(&row, b"BBBB");
    }

    #[test]
    fn reads_contiguous_row_hyperslab() {
        let descriptor = FitsTableDescriptor::Binary(FitsBinaryTableDescriptor {
            core: FitsTableDescriptorCore::new(
                2,
                4,
                vec![FitsTableColumn::new(
                    1,
                    None,
                    "1I".into(),
                    Datatype::Integer {
                        bits: core::num::NonZeroUsize::new(16).unwrap(),
                        byte_order: consus_core::ByteOrder::BigEndian,
                        signed: true,
                    },
                    2,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                )],
                0,
            ),
        });
        let span = FitsDataSpan::new(0, 8).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let selection = Selection::Hyperslab(Hyperslab::new(&[HyperslabDim {
            start: 1,
            stride: 1,
            count: 2,
            block: 1,
        }]));
        let mut buf = [0u8; 4];
        let read = table.read_selection(&reader, &selection, &mut buf).unwrap();
        assert_eq!(read, 4);
        assert_eq!(buf, [3, 4, 5, 6]);
    }

    #[test]
    fn rejects_wrong_xtension_for_ascii_descriptor() {
        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 4",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TFORM1  = '1J      '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let error = FitsAsciiTableDescriptor::from_header(&header).unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }

    #[test]
    fn binary_column_datatype_matches_tform() {
        use consus_core::ByteOrder;
        use core::num::NonZeroUsize;

        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 12",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 2",
            "TTYPE1  = 'X '   ",
            "TFORM1  = '1J ' ",
            "TTYPE2  = 'Y '   ",
            "TFORM2  = '1D ' ",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsBinaryTableDescriptor::from_header(&header).unwrap();
        let cols = descriptor.columns();

        assert_eq!(
            cols[0].datatype(),
            &Datatype::Integer {
                bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
                signed: true,
            }
        );
        assert_eq!(cols[0].byte_width(), 4);

        assert_eq!(
            cols[1].datatype(),
            &Datatype::Float {
                bits: NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );
        assert_eq!(cols[1].byte_width(), 8);
    }

    #[test]
    fn binary_column_array_tform_produces_array_datatype() {
        use consus_core::ByteOrder;
        use core::num::NonZeroUsize;

        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 12",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TTYPE1  = 'VEC ' ",
            "TFORM1  = '3E '   ",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsBinaryTableDescriptor::from_header(&header).unwrap();
        let col = &descriptor.columns()[0];

        match col.datatype() {
            Datatype::Array { base, dims } => {
                assert_eq!(
                    base.as_ref(),
                    &Datatype::Float {
                        bits: NonZeroUsize::new(32).unwrap(),
                        byte_order: ByteOrder::BigEndian,
                    }
                );
                assert_eq!(dims.as_slice(), &[3]);
            }
            other => panic!("expected Array datatype, got {other:?}"),
        }
        assert_eq!(col.byte_width(), 12);
    }

    #[test]
    fn binary_table_rejects_naxis1_mismatch() {
        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 99",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TTYPE1  = 'X '   ",
            "TFORM1  = '1J ' ",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let result = FitsBinaryTableDescriptor::from_header(&header);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidFormat { .. }));
    }

    #[test]
    fn ascii_column_datatype_is_fixed_string() {
        let bytes = header_bytes(&[
            "XTENSION= 'TABLE '",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 24",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 2",
            "TTYPE1  = 'NAME ' ",
            "TFORM1  = 'A8 '    ",
            "TTYPE2  = 'VALUE '",
            "TFORM2  = 'E16.7 '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsAsciiTableDescriptor::from_header(&header).unwrap();
        let cols = descriptor.columns();

        assert_eq!(
            cols[0].datatype(),
            &Datatype::FixedString {
                length: 8,
                encoding: consus_core::StringEncoding::Ascii,
            }
        );
        assert_eq!(cols[0].byte_width(), 8);

        assert_eq!(
            cols[1].datatype(),
            &Datatype::FixedString {
                length: 16,
                encoding: consus_core::StringEncoding::Ascii,
            }
        );
        assert_eq!(cols[1].byte_width(), 16);
    }

    #[test]
    fn binary_column_complex_and_descriptor_types() {
        use consus_core::ByteOrder;
        use core::num::NonZeroUsize;

        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 16",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 2",
            "TTYPE1  = 'CPLX ' ",
            "TFORM1  = '1C '    ",
            "TTYPE2  = 'DESC ' ",
            "TFORM2  = '1P '    ",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsBinaryTableDescriptor::from_header(&header).unwrap();
        let cols = descriptor.columns();

        assert_eq!(
            cols[0].datatype(),
            &Datatype::Complex {
                component_bits: NonZeroUsize::new(32).unwrap(),
                byte_order: ByteOrder::BigEndian,
            }
        );
        assert_eq!(cols[0].byte_width(), 8);

        match cols[1].datatype() {
            Datatype::Compound { size, .. } => {
                assert_eq!(*size, 8);
            }
            other => panic!("expected Compound datatype for P descriptor, got {other:?}"),
        }
        assert_eq!(cols[1].byte_width(), 8);
    }
    #[test]
    fn decode_row_binary_table_two_columns() {
        // Binary table: 2 columns -- 1J (i32) and 1E (f32), 3 rows.
        // Row layout: [i32 be][f32 be] = 8 bytes per row.
        // Row 0: i32=10, f32=1.5
        // Row 1: i32=20, f32=2.5
        // Row 2: i32=30, f32=3.5
        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 8",
            "NAXIS2  = 3",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 2",
            "TTYPE1  = 'IDX     '",
            "TFORM1  = '1J      '",
            "TTYPE2  = 'VAL     '",
            "TFORM2  = '1E      '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsTableDescriptor::from_header(&header).unwrap();

        let mut data: Vec<u8> = Vec::new();
        for (idx, val) in [(10_i32, 1.5_f32), (20_i32, 2.5_f32), (30_i32, 3.5_f32)] {
            data.extend_from_slice(&idx.to_be_bytes());
            data.extend_from_slice(&val.to_be_bytes());
        }

        let span = FitsDataSpan::new(0, 24).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(data);

        // Decode row 1: i32=20, f32=2.5
        let row_vals = table.decode_row(&reader, 1).unwrap();
        assert_eq!(row_vals.len(), 2);
        assert_eq!(row_vals[0], FitsColumnValue::Int32(20));
        assert_eq!(row_vals[1], FitsColumnValue::Float32(2.5));

        // Decode column 0 (i32) across all 3 rows: 10, 20, 30
        let col0_vals = table.decode_column(&reader, 0).unwrap();
        assert_eq!(col0_vals.len(), 3);
        assert_eq!(col0_vals[0], FitsColumnValue::Int32(10));
        assert_eq!(col0_vals[1], FitsColumnValue::Int32(20));
        assert_eq!(col0_vals[2], FitsColumnValue::Int32(30));

        // Decode column 1 (f32) across all 3 rows: 1.5, 2.5, 3.5
        let col1_vals = table.decode_column(&reader, 1).unwrap();
        assert_eq!(col1_vals.len(), 3);
        assert_eq!(col1_vals[0], FitsColumnValue::Float32(1.5));
        assert_eq!(col1_vals[1], FitsColumnValue::Float32(2.5));
        assert_eq!(col1_vals[2], FitsColumnValue::Float32(3.5));
    }

    #[test]
    fn decode_row_ascii_table_two_columns() {
        // ASCII table: 2 columns -- A6 (name) and I8 (integer value), 2 rows.
        // Row width = 6 + 8 = 14. No TBCOLn -- prefix sum used.
        // Row 0: "Alpha " + "     100"
        // Row 1: "Beta  " + "     200"
        let bytes = header_bytes(&[
            "XTENSION= 'TABLE   '",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 14",
            "NAXIS2  = 2",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 2",
            "TTYPE1  = 'NAME    '",
            "TFORM1  = 'A6      '",
            "TTYPE2  = 'VALUE   '",
            "TFORM2  = 'I8      '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsTableDescriptor::from_header(&header).unwrap();

        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(b"Alpha      100");
        data.extend_from_slice(b"Beta       200");

        let span = FitsDataSpan::new(0, 28).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(data);

        // Decode row 0
        let row_vals = table.decode_row(&reader, 0).unwrap();
        assert_eq!(row_vals.len(), 2);
        assert_eq!(row_vals[0], FitsColumnValue::Chars("Alpha".to_owned()));
        assert_eq!(row_vals[1], FitsColumnValue::Int64(100));

        // Decode column 1 (I8) across rows: 100, 200
        let col1_vals = table.decode_column(&reader, 1).unwrap();
        assert_eq!(col1_vals.len(), 2);
        assert_eq!(col1_vals[0], FitsColumnValue::Int64(100));
        assert_eq!(col1_vals[1], FitsColumnValue::Int64(200));
    }

    #[test]
    fn decode_column_out_of_bounds() {
        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 4",
            "NAXIS2  = 1",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TTYPE1  = 'X       '",
            "TFORM1  = '1J      '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        let descriptor = FitsTableDescriptor::from_header(&header).unwrap();
        let span = FitsDataSpan::new(0, 4).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(vec![0u8; 4]);
        let err = table.decode_column(&reader, 99).unwrap_err();
        assert!(matches!(err, Error::SelectionOutOfBounds));
    }

}