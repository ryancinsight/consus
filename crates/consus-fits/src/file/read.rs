use consus_core::{Datatype, Error, FileRead, NodeType, Result, Selection, Shape};
use consus_io::{Length, ReadAt};

use crate::datastructure::FitsDataSpan;
use crate::hdu::{FitsHdu, FitsHduPayload};
use crate::image::FitsImageDescriptor;
use crate::table::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor};

use super::types::{invalid_format, parse_dataset_path, DatasetPath, FitsFile, FITS_FORMAT_NAME};

#[cfg(feature = "alloc")]
impl<IO> FileRead for FitsFile<IO>
where
    IO: ReadAt + Length,
{
    fn format(&self) -> &str {
        FITS_FORMAT_NAME
    }

    fn exists(&self, path: &str) -> Result<bool> {
        if path == "/" {
            return Ok(true);
        }

        match parse_dataset_path(path) {
            Ok(DatasetPath::Primary) => Ok(self.primary_hdu().is_some()),
            Ok(DatasetPath::Hdu(index)) => Ok(self.hdu(index).is_some()),
            Err(_) => Ok(false),
        }
    }

    fn node_type_at(&self, path: &str) -> Result<NodeType> {
        if path == "/" {
            return Ok(NodeType::Group);
        }

        self.hdu_at_path(path)?;
        Ok(NodeType::Dataset)
    }

    fn num_children_at(&self, path: &str) -> Result<usize> {
        if path == "/" {
            return Ok(self.hdu_count());
        }

        self.hdu_at_path(path)?;
        invalid_format("FITS HDU payloads are datasets, not groups")
    }

    fn dataset_datatype(&self, path: &str) -> Result<Datatype> {
        let hdu = self.hdu_at_path(path)?;
        datatype_for_hdu(hdu)
    }

    fn dataset_shape(&self, path: &str) -> Result<Shape> {
        let hdu = self.hdu_at_path(path)?;
        shape_for_hdu(hdu)
    }

    fn read_dataset_raw(&self, path: &str, selection: &Selection, buf: &mut [u8]) -> Result<usize> {
        let hdu = self.hdu_at_path(path)?;
        read_hdu_payload(self.io(), hdu, selection, buf)
    }
}

#[cfg(feature = "alloc")]
fn datatype_for_hdu(hdu: &FitsHdu) -> Result<Datatype> {
    match hdu.payload() {
        FitsHduPayload::Image(image) => Ok(image.bitpix().to_datatype()),
        FitsHduPayload::AsciiTable(table) => opaque_row_datatype(table.row_len()),
        FitsHduPayload::BinaryTable(table) => opaque_row_datatype(table.row_len()),
    }
}

#[cfg(feature = "alloc")]
fn shape_for_hdu(hdu: &FitsHdu) -> Result<Shape> {
    match hdu.payload() {
        FitsHduPayload::Image(image) => Ok(image.shape().clone()),
        FitsHduPayload::AsciiTable(table) => Ok(table.shape()),
        FitsHduPayload::BinaryTable(table) => Ok(table.shape()),
    }
}

#[cfg(feature = "alloc")]
fn opaque_row_datatype(row_len: usize) -> Result<Datatype> {
    if row_len == 0 {
        return invalid_format("FITS table row length must be positive for dataset projection");
    }

    Ok(Datatype::Opaque {
        size: row_len,
        #[cfg(feature = "alloc")]
        tag: Some("fits-row".into()),
    })
}

#[cfg(feature = "alloc")]
pub(super) fn read_hdu_payload<IO: ReadAt>(
    io: &IO,
    hdu: &FitsHdu,
    selection: &Selection,
    buf: &mut [u8],
) -> Result<usize> {
    match hdu.payload() {
        FitsHduPayload::Image(image) => {
            read_image_payload(io, image, hdu.data_span(), selection, buf)
        }
        FitsHduPayload::AsciiTable(table) => {
            read_ascii_table_payload(io, table, hdu.data_span(), selection, buf)
        }
        FitsHduPayload::BinaryTable(table) => {
            read_binary_table_payload(io, table, hdu.data_span(), selection, buf)
        }
    }
}

#[cfg(feature = "alloc")]
fn read_image_payload<IO: ReadAt>(
    io: &IO,
    image: &FitsImageDescriptor,
    span: FitsDataSpan,
    selection: &Selection,
    buf: &mut [u8],
) -> Result<usize> {
    let logical_len = image.logical_data_len()?;
    match selection {
        Selection::All => {
            if buf.len() < logical_len {
                return Err(Error::BufferTooSmall {
                    required: logical_len,
                    provided: buf.len(),
                });
            }
            io.read_at(span.offset(), &mut buf[..logical_len])?;
            Ok(logical_len)
        }
        Selection::None => Ok(0),
        Selection::Hyperslab(_) | Selection::Points(_) => Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: "FITS image partial selection is not implemented".into(),
        }),
    }
}

#[cfg(feature = "alloc")]
fn read_ascii_table_payload<IO: ReadAt>(
    io: &IO,
    table: &FitsAsciiTableDescriptor,
    span: FitsDataSpan,
    selection: &Selection,
    buf: &mut [u8],
) -> Result<usize> {
    read_table_payload(io, table.row_len(), table.rows(), span, selection, buf)
}

#[cfg(feature = "alloc")]
fn read_binary_table_payload<IO: ReadAt>(
    io: &IO,
    table: &FitsBinaryTableDescriptor,
    span: FitsDataSpan,
    selection: &Selection,
    buf: &mut [u8],
) -> Result<usize> {
    read_table_payload_with_heap(
        io,
        table.row_len(),
        table.rows(),
        table.heap_size(),
        span,
        selection,
        buf,
    )
}

#[cfg(feature = "alloc")]
fn read_table_payload<IO: ReadAt>(
    io: &IO,
    row_len: usize,
    rows: usize,
    span: FitsDataSpan,
    selection: &Selection,
    buf: &mut [u8],
) -> Result<usize> {
    read_table_payload_with_heap(io, row_len, rows, 0, span, selection, buf)
}

#[cfg(feature = "alloc")]
fn read_table_payload_with_heap<IO: ReadAt>(
    io: &IO,
    row_len: usize,
    rows: usize,
    heap_size: usize,
    span: FitsDataSpan,
    selection: &Selection,
    buf: &mut [u8],
) -> Result<usize> {
    let rows_bytes = row_len.checked_mul(rows).ok_or(Error::Overflow)?;
    let logical_len = rows_bytes.checked_add(heap_size).ok_or(Error::Overflow)?;

    match selection {
        Selection::All => {
            if buf.len() < logical_len {
                return Err(Error::BufferTooSmall {
                    required: logical_len,
                    provided: buf.len(),
                });
            }
            io.read_at(span.offset(), &mut buf[..logical_len])?;
            Ok(logical_len)
        }
        Selection::None => Ok(0),
        Selection::Points(_) => Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: "FITS table point selection is not implemented".into(),
        }),
        Selection::Hyperslab(hyperslab) => {
            if heap_size != 0 {
                return Err(Error::UnsupportedFeature {
                    #[cfg(feature = "alloc")]
                    feature: "FITS binary table heap partial selection is not implemented".into(),
                });
            }

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

            if dim.start > rows || dim.count > rows.saturating_sub(dim.start) {
                return Err(Error::SelectionOutOfBounds);
            }

            let byte_offset = dim.start.checked_mul(row_len).ok_or(Error::Overflow)?;
            let byte_len = dim.count.checked_mul(row_len).ok_or(Error::Overflow)?;
            if buf.len() < byte_len {
                return Err(Error::BufferTooSmall {
                    required: byte_len,
                    provided: buf.len(),
                });
            }

            let absolute_offset = span
                .offset()
                .checked_add(u64::try_from(byte_offset).map_err(|_| Error::Overflow)?)
                .ok_or(Error::Overflow)?;
            io.read_at(absolute_offset, &mut buf[..byte_len])?;
            Ok(byte_len)
        }
    }
}
