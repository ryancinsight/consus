#[cfg(feature = "alloc")]
use alloc::vec;

use consus_core::{Error, FileWrite, Result, Selection};
use consus_io::{Length, ReadAt, WriteAt};

use crate::datastructure::FitsDataSpan;
use crate::hdu::{FitsHdu, FitsHduPayload};
use crate::image::FitsImageDescriptor;
use crate::table::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor};

use super::types::FitsFile;

#[cfg(feature = "alloc")]
impl<IO> FileWrite for FitsFile<IO>
where
    IO: ReadAt + WriteAt + Length,
{
    fn flush(&mut self) -> Result<()> {
        self.io_mut().flush()
    }

    fn create_group(&mut self, path: &str) -> Result<()> {
        if path == "/" {
            return Ok(());
        }

        Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: "FITS does not support hierarchical group creation".into(),
        })
    }

    fn write_dataset_raw(&mut self, path: &str, selection: &Selection, data: &[u8]) -> Result<()> {
        let hdu = self.hdu_at_path(path)?.clone();
        write_hdu_payload(self.io_mut(), &hdu, selection, data)
    }
}

#[cfg(feature = "alloc")]
pub(super) fn write_hdu_payload<IO: WriteAt>(
    io: &mut IO,
    hdu: &FitsHdu,
    selection: &Selection,
    data: &[u8],
) -> Result<()> {
    match hdu.payload() {
        FitsHduPayload::Image(image) => {
            write_image_payload(io, image, hdu.data_span(), selection, data)
        }
        FitsHduPayload::AsciiTable(table) => {
            write_ascii_table_payload(io, table, hdu.data_span(), selection, data)
        }
        FitsHduPayload::BinaryTable(table) => {
            write_binary_table_payload(io, table, hdu.data_span(), selection, data)
        }
    }
}

#[cfg(feature = "alloc")]
fn write_image_payload<IO: WriteAt>(
    io: &mut IO,
    image: &FitsImageDescriptor,
    span: FitsDataSpan,
    selection: &Selection,
    data: &[u8],
) -> Result<()> {
    let logical_len = image.logical_data_len()?;
    match selection {
        Selection::All => {
            if data.len() < logical_len {
                return Err(Error::BufferTooSmall {
                    required: logical_len,
                    provided: data.len(),
                });
            }
            io.write_at(span.offset(), &data[..logical_len])?;
            write_zero_padding(io, span)?;
            Ok(())
        }
        Selection::None => Ok(()),
        Selection::Hyperslab(_) | Selection::Points(_) => Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: "FITS image partial write selection is not implemented".into(),
        }),
    }
}

#[cfg(feature = "alloc")]
fn write_ascii_table_payload<IO: WriteAt>(
    io: &mut IO,
    table: &FitsAsciiTableDescriptor,
    span: FitsDataSpan,
    selection: &Selection,
    data: &[u8],
) -> Result<()> {
    write_table_payload(io, table.row_len(), table.rows(), 0, span, selection, data)
}

#[cfg(feature = "alloc")]
fn write_binary_table_payload<IO: WriteAt>(
    io: &mut IO,
    table: &FitsBinaryTableDescriptor,
    span: FitsDataSpan,
    selection: &Selection,
    data: &[u8],
) -> Result<()> {
    write_table_payload(
        io,
        table.row_len(),
        table.rows(),
        table.heap_size(),
        span,
        selection,
        data,
    )
}

#[cfg(feature = "alloc")]
fn write_table_payload<IO: WriteAt>(
    io: &mut IO,
    row_len: usize,
    rows: usize,
    heap_size: usize,
    span: FitsDataSpan,
    selection: &Selection,
    data: &[u8],
) -> Result<()> {
    let rows_bytes = row_len.checked_mul(rows).ok_or(Error::Overflow)?;
    let logical_len = rows_bytes.checked_add(heap_size).ok_or(Error::Overflow)?;

    match selection {
        Selection::All => {
            if data.len() < logical_len {
                return Err(Error::BufferTooSmall {
                    required: logical_len,
                    provided: data.len(),
                });
            }
            io.write_at(span.offset(), &data[..logical_len])?;
            write_zero_padding(io, span)?;
            Ok(())
        }
        Selection::None => Ok(()),
        Selection::Points(_) => Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: "FITS table point write selection is not implemented".into(),
        }),
        Selection::Hyperslab(hyperslab) => {
            if heap_size != 0 {
                return Err(Error::UnsupportedFeature {
                    #[cfg(feature = "alloc")]
                    feature: "FITS binary table heap partial write is not implemented".into(),
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
            if data.len() < byte_len {
                return Err(Error::BufferTooSmall {
                    required: byte_len,
                    provided: data.len(),
                });
            }

            let absolute_offset = span
                .offset()
                .checked_add(u64::try_from(byte_offset).map_err(|_| Error::Overflow)?)
                .ok_or(Error::Overflow)?;
            io.write_at(absolute_offset, &data[..byte_len])?;
            Ok(())
        }
    }
}

#[cfg(feature = "alloc")]
fn write_zero_padding<IO: WriteAt>(io: &mut IO, span: FitsDataSpan) -> Result<()> {
    let padding_len = span.padding_len();
    if padding_len == 0 {
        return Ok(());
    }

    let padding_offset = span
        .offset()
        .checked_add(u64::try_from(span.logical_len()).map_err(|_| Error::Overflow)?)
        .ok_or(Error::Overflow)?;
    let zeros = vec![0u8; padding_len];
    io.write_at(padding_offset, &zeros)?;
    Ok(())
}
