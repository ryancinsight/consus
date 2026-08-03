//! FITS table descriptors and column metadata.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::{Datatype, Error, Result, Shape};

#[cfg(feature = "alloc")]
use crate::header::FitsHeader;
#[cfg(feature = "alloc")]
use crate::types::{binary_format_element_size, parse_binary_format, tform_to_datatype};

#[cfg(feature = "alloc")]
use super::parse::{invalid_format, parse_required_string, parse_table_core, validate_xtension};

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
pub(super) struct FitsTableDescriptorCore {
    row_len: usize,
    rows: usize,
    columns: Vec<FitsTableColumn>,
    heap_size: usize,
}

#[cfg(feature = "alloc")]
impl FitsTableDescriptorCore {
    pub(super) fn new(
        row_len: usize,
        rows: usize,
        columns: Vec<FitsTableColumn>,
        heap_size: usize,
    ) -> Self {
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;
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
}
