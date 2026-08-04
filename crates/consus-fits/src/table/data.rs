//! FITS table data-unit access.

#[cfg(feature = "alloc")]
use consus_core::{Error, Result, Selection};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;

#[cfg(feature = "alloc")]
use crate::datastructure::FitsDataSpan;

#[cfg(feature = "alloc")]
use super::{FitsTableDescriptor, decode};

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
    pub fn decode_row<R: ReadAt>(
        &self,
        reader: &R,
        row_index: usize,
    ) -> Result<alloc::vec::Vec<decode::FitsColumnValue>> {
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
    pub fn decode_column<R: ReadAt>(
        &self,
        reader: &R,
        col_index: usize,
    ) -> Result<alloc::vec::Vec<decode::FitsColumnValue>> {
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::{borrow::ToOwned, vec::Vec};

    use super::*;
    use crate::datastructure::{FitsBlockAlignment, FitsDataSpan};
    use crate::file::types::parse_extension_header_bytes;
    use crate::table::{FitsColumnValue, FitsTableDescriptor};
    use consus_core::{Error, Hyperslab, HyperslabDim};
    use consus_io::MemCursor;

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

    fn table_descriptor(cards: &[&str]) -> FitsTableDescriptor {
        let header = parse_extension_header_bytes(&header_bytes(cards)).unwrap();
        FitsTableDescriptor::from_header(&header).unwrap()
    }

    #[test]
    fn reads_single_row() {
        let descriptor = table_descriptor(&[
            "XTENSION= 'TABLE   '",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 4",
            "NAXIS2  = 3",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TFORM1  = 'A4      '",
            "END",
        ]);
        let span = FitsDataSpan::new(0, 12).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(b"AAAABBBBCCCC".to_vec());
        let mut row = [0u8; 4];
        table.read_row(&reader, 1, &mut row).unwrap();
        assert_eq!(&row, b"BBBB");
    }

    #[test]
    fn reads_contiguous_row_hyperslab() {
        let descriptor = table_descriptor(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  = 8",
            "NAXIS   = 2",
            "NAXIS1  = 2",
            "NAXIS2  = 4",
            "PCOUNT  = 0",
            "GCOUNT  = 1",
            "TFIELDS = 1",
            "TFORM1  = '1I      '",
            "END",
        ]);
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
    fn decode_row_binary_table_two_columns() {
        let descriptor = table_descriptor(&[
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

        let mut data: Vec<u8> = Vec::new();
        for (idx, val) in [(10_i32, 1.5_f32), (20_i32, 2.5_f32), (30_i32, 3.5_f32)] {
            data.extend_from_slice(&idx.to_be_bytes());
            data.extend_from_slice(&val.to_be_bytes());
        }

        let span = FitsDataSpan::new(0, 24).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(data);

        let row_vals = table.decode_row(&reader, 1).unwrap();
        assert_eq!(row_vals.len(), 2);
        assert_eq!(row_vals[0], FitsColumnValue::Int32(20));
        assert_eq!(row_vals[1], FitsColumnValue::Float32(2.5));

        let col0_vals = table.decode_column(&reader, 0).unwrap();
        assert_eq!(col0_vals.len(), 3);
        assert_eq!(col0_vals[0], FitsColumnValue::Int32(10));
        assert_eq!(col0_vals[1], FitsColumnValue::Int32(20));
        assert_eq!(col0_vals[2], FitsColumnValue::Int32(30));

        let col1_vals = table.decode_column(&reader, 1).unwrap();
        assert_eq!(col1_vals.len(), 3);
        assert_eq!(col1_vals[0], FitsColumnValue::Float32(1.5));
        assert_eq!(col1_vals[1], FitsColumnValue::Float32(2.5));
        assert_eq!(col1_vals[2], FitsColumnValue::Float32(3.5));
    }

    #[test]
    fn decode_row_ascii_table_two_columns() {
        let descriptor = table_descriptor(&[
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

        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(b"Alpha      100");
        data.extend_from_slice(b"Beta       200");

        let span = FitsDataSpan::new(0, 28).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(data);

        let row_vals = table.decode_row(&reader, 0).unwrap();
        assert_eq!(row_vals.len(), 2);
        assert_eq!(row_vals[0], FitsColumnValue::Chars("Alpha".to_owned()));
        assert_eq!(row_vals[1], FitsColumnValue::Int64(100));

        let col1_vals = table.decode_column(&reader, 1).unwrap();
        assert_eq!(col1_vals.len(), 2);
        assert_eq!(col1_vals[0], FitsColumnValue::Int64(100));
        assert_eq!(col1_vals[1], FitsColumnValue::Int64(200));
    }

    #[test]
    fn decode_column_out_of_bounds() {
        let descriptor = table_descriptor(&[
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
        let span = FitsDataSpan::new(0, 4).unwrap();
        let table = FitsTableData::new(descriptor, span);
        let reader = MemCursor::from_bytes(vec![0u8; 4]);
        let err = table.decode_column(&reader, 99).unwrap_err();
        assert!(matches!(err, Error::SelectionOutOfBounds));
    }
}
