//! FITS file wrapper with HDU scanning and `consus-core` trait integration.
//!
//! ## Scope
//!
//! This module is the authoritative file-domain boundary for `consus-fits`.
//! It defines:
//! - FITS file scanning over positioned I/O
//! - ordered HDU indexing and traversal
//! - synthetic path mapping from FITS HDUs to `consus-core` file traits
//! - raw dataset reads and writes for image and table HDUs
//!
//! ## Architectural mapping
//!
//! `consus-core` models hierarchical containers with groups and datasets,
//! while FITS is an ordered sequence of HDUs. This module uses the minimal
//! deterministic mapping:
//! - `/` => synthetic root group
//! - `/PRIMARY` => primary HDU dataset
//! - `/HDU/{n}` => HDU dataset at zero-based ordinal `n`
//!
//! Header cards remain available through the concrete `FitsFile` API and are
//! not projected into the `consus-core` node model.
//!
//! ## Invariants
//!
//! - HDU scan order matches on-disk order.
//! - The first HDU is primary.
//! - Header and data extents remain 2880-byte aligned.
//! - Dataset paths resolve only to HDU payloads.
//! - Image HDUs expose canonical numeric datatypes.
//! - Table HDUs expose row-wise opaque records.
//!
//! ## Current write semantics
//!
//! - Existing HDU payloads may be overwritten in-place.
//! - Structural mutation of the HDU sequence is not implemented.
//! - `create_group` is unsupported except for `/`.
//! - Partial image selections are not implemented.
//! - Table selections support `All`, `None`, and contiguous 1-D row hyperslabs.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

use consus_core::{
    Datatype, Error, FileRead, FileWrite, HasAttributes, NodeType, Result, Selection, Shape,
};
use consus_io::{Length, ReadAt, WriteAt};

use crate::datastructure::{
    FITS_LOGICAL_RECORD_LEN, FitsDataSpan, FitsHeaderBlock, FitsHeaderCardCount,
};
use crate::hdu::{FitsHdu, FitsHduIndex, FitsHduPayload, FitsHduSequence};
use crate::header::{FitsCard, FitsHeader};
use crate::image::FitsImageDescriptor;
use crate::table::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor};

/// Canonical FITS format identifier returned through `consus-core`.
pub const FITS_FORMAT_NAME: &str = "fits";

/// FITS file wrapper over positioned I/O.
///
/// The wrapper owns the underlying I/O object and an indexed HDU sequence
/// derived from a deterministic scan of the FITS container.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsFile<IO> {
    io: IO,
    hdus: FitsHduSequence,
}

#[cfg(feature = "alloc")]
impl<IO> FitsFile<IO>
where
    IO: ReadAt + Length,
{
    /// Open and scan a FITS file from positioned I/O.
    ///
    /// ## Errors
    ///
    /// Returns:
    /// - `Error::InvalidFormat` if the FITS structure is invalid
    /// - `Error::BufferTooSmall` or `Error::Io` on underlying I/O failure
    /// - `Error::Overflow` on offset/size overflow
    pub fn open(io: IO) -> Result<Self> {
        let hdus = scan_hdus(&io)?;
        Ok(Self { io, hdus })
    }

    /// Construct a FITS file from pre-scanned HDUs.
    pub fn new(io: IO, hdus: FitsHduSequence) -> Self {
        Self { io, hdus }
    }

    /// Borrow the underlying I/O object.
    pub const fn io(&self) -> &IO {
        &self.io
    }

    /// Borrow the ordered HDU sequence.
    pub const fn hdus(&self) -> &FitsHduSequence {
        &self.hdus
    }

    /// Return the number of HDUs.
    pub fn hdu_count(&self) -> usize {
        self.hdus.len()
    }

    /// Return the primary HDU.
    pub fn primary_hdu(&self) -> Option<&FitsHdu> {
        self.hdus.primary()
    }

    /// Return the HDU at zero-based ordinal `index`.
    pub fn hdu(&self, index: usize) -> Option<&FitsHdu> {
        self.hdus.get_usize(index)
    }

    /// Resolve a synthetic FITS path to an HDU.
    pub fn hdu_at_path(&self, path: &str) -> Result<&FitsHdu> {
        match parse_dataset_path(path)? {
            DatasetPath::Primary => self
                .hdus
                .primary()
                .ok_or_else(|| invalid_format_error("FITS file is missing primary HDU")),
            DatasetPath::Hdu(index) => {
                self.hdus
                    .get(FitsHduIndex::new(index))
                    .ok_or_else(|| Error::NotFound {
                        #[cfg(feature = "alloc")]
                        path: path.into(),
                    })
            }
        }
    }

    /// Read the full logical payload of the HDU at `path`.
    pub fn read_hdu_all(&self, path: &str, buf: &mut [u8]) -> Result<usize> {
        let hdu = self.hdu_at_path(path)?;
        read_hdu_payload(&self.io, hdu, &Selection::All, buf)
    }

    /// Read a raw selection from the HDU at `path`.
    pub fn read_hdu_selection(
        &self,
        path: &str,
        selection: &Selection,
        buf: &mut [u8],
    ) -> Result<usize> {
        let hdu = self.hdu_at_path(path)?;
        read_hdu_payload(&self.io, hdu, selection, buf)
    }
}

#[cfg(feature = "alloc")]
impl<IO> FitsFile<IO>
where
    IO: ReadAt + WriteAt + Length,
{
    /// Open and scan a writable FITS file from positioned I/O.
    pub fn open_mut(io: IO) -> Result<Self> {
        Self::open(io)
    }

    /// Mutably borrow the underlying I/O object.
    pub fn io_mut(&mut self) -> &mut IO {
        &mut self.io
    }

    /// Write a raw selection to the HDU at `path`.
    ///
    /// Current support:
    /// - image HDUs: `Selection::All` and `Selection::None`
    /// - table HDUs: `Selection::All`, `Selection::None`, contiguous 1-D row hyperslabs
    pub fn write_hdu_selection(
        &mut self,
        path: &str,
        selection: &Selection,
        data: &[u8],
    ) -> Result<()> {
        let hdu = self.hdu_at_path(path)?.clone();
        write_hdu_payload(&mut self.io, &hdu, selection, data)
    }
}

#[cfg(feature = "alloc")]
impl<IO> HasAttributes for FitsFile<IO> {
    fn num_attributes(&self) -> Result<usize> {
        Ok(0)
    }

    fn has_attribute(&self, _name: &str) -> Result<bool> {
        Ok(false)
    }

    fn attribute_datatype(&self, _name: &str) -> Result<Datatype> {
        Err(Error::NotFound {
            #[cfg(feature = "alloc")]
            path: "/".into(),
        })
    }

    fn read_attribute_raw(&self, _name: &str, _buf: &mut [u8]) -> Result<usize> {
        Err(Error::NotFound {
            #[cfg(feature = "alloc")]
            path: "/".into(),
        })
    }

    fn for_each_attribute(&self, _visitor: &mut dyn FnMut(&str) -> bool) -> Result<()> {
        Ok(())
    }
}

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
            Ok(DatasetPath::Primary) => Ok(self.hdus.primary().is_some()),
            Ok(DatasetPath::Hdu(index)) => Ok(self.hdus.get(FitsHduIndex::new(index)).is_some()),
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
            return Ok(self.hdus.len());
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
        read_hdu_payload(&self.io, hdu, selection, buf)
    }
}

#[cfg(feature = "alloc")]
impl<IO> FileWrite for FitsFile<IO>
where
    IO: ReadAt + WriteAt + Length,
{
    fn flush(&mut self) -> Result<()> {
        self.io.flush()
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
        write_hdu_payload(&mut self.io, &hdu, selection, data)
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetPath {
    Primary,
    Hdu(usize),
}

#[cfg(feature = "alloc")]
fn parse_dataset_path(path: &str) -> Result<DatasetPath> {
    if path == "/PRIMARY" {
        return Ok(DatasetPath::Primary);
    }

    let Some(rest) = path.strip_prefix("/HDU/") else {
        return Err(Error::NotFound {
            #[cfg(feature = "alloc")]
            path: path.into(),
        });
    };

    if rest.is_empty() || rest.contains('/') {
        return Err(Error::NotFound {
            #[cfg(feature = "alloc")]
            path: path.into(),
        });
    }

    let index = rest.parse::<usize>().map_err(|_| Error::NotFound {
        #[cfg(feature = "alloc")]
        path: path.into(),
    })?;

    Ok(DatasetPath::Hdu(index))
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
fn read_hdu_payload<IO: ReadAt>(
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
fn write_hdu_payload<IO: WriteAt>(
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

#[cfg(feature = "alloc")]
fn scan_hdus<IO>(io: &IO) -> Result<FitsHduSequence>
where
    IO: ReadAt + Length,
{
    let file_len = io.len()?;
    if file_len == 0 {
        return invalid_format("empty FITS file");
    }

    let mut hdus = Vec::new();
    let mut offset = 0u64;
    let mut index = 0usize;

    while offset < file_len {
        let (header, header_block) = read_header_at(io, offset, index == 0)?;
        let data_offset = offset
            .checked_add(
                u64::try_from(header_block.padded_byte_len()).map_err(|_| Error::Overflow)?,
            )
            .ok_or(Error::Overflow)?;

        let logical_data_len = logical_data_len_from_header(index, &header)?;
        let data_span = FitsDataSpan::new(data_offset, logical_data_len)?;
        let hdu = FitsHdu::from_header(FitsHduIndex::new(index), header, header_block, data_span)?;
        offset = data_span.end_offset()?;
        hdus.push(hdu);
        index += 1;

        if offset == file_len {
            break;
        }
    }

    FitsHduSequence::new(hdus)
}

#[cfg(feature = "alloc")]
fn read_header_at<IO: ReadAt>(
    io: &IO,
    start_offset: u64,
    primary: bool,
) -> Result<(FitsHeader, FitsHeaderBlock)> {
    let mut raw_cards = Vec::new();
    let mut block_offset = start_offset;
    let mut saw_end = false;

    loop {
        let mut block = [0u8; FITS_LOGICAL_RECORD_LEN];
        io.read_at(block_offset, &mut block)?;

        for chunk in block.chunks_exact(80) {
            let card = FitsCard::parse(chunk)?;
            raw_cards.extend_from_slice(chunk);
            if card.is_end() {
                saw_end = true;
                break;
            }
        }

        if saw_end {
            break;
        }

        block_offset = block_offset
            .checked_add(u64::try_from(FITS_LOGICAL_RECORD_LEN).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
    }

    if !saw_end {
        return invalid_format("FITS header is missing END card");
    }

    let header = if primary {
        crate::header::parse_header_bytes(&raw_cards)?
    } else {
        parse_extension_header_bytes(&raw_cards)?
    };

    let card_count = raw_cards.len() / 80;
    let header_block = FitsHeaderBlock::new(FitsHeaderCardCount::new(card_count));
    Ok((header, header_block))
}

#[cfg(feature = "alloc")]
pub(crate) fn parse_extension_header_bytes(bytes: &[u8]) -> Result<FitsHeader> {
    if bytes.len() % 80 != 0 {
        return invalid_format("FITS header byte length is not a multiple of 80");
    }

    let mut parsed_cards: Vec<FitsCard> = Vec::new();
    let mut saw_end = false;

    for chunk in bytes.chunks_exact(80) {
        let card = FitsCard::parse(chunk)?;

        if card.is_end() {
            saw_end = true;
            break;
        }

        if card.is_continue() {
            let previous = parsed_cards.last_mut().ok_or_else(|| {
                invalid_format_error("CONTINUE card cannot appear before a string-valued card")
            })?;
            let fragment = card.continue_fragment().ok_or_else(|| {
                invalid_format_error(
                    "CONTINUE card does not contain a valid string continuation fragment",
                )
            })?;
            previous.append_string_fragment(fragment)?;
        } else {
            parsed_cards.push(card);
        }
    }

    if !saw_end {
        return invalid_format("FITS header is missing END card");
    }

    Ok(FitsHeader::new(parsed_cards))
}

#[cfg(feature = "alloc")]
fn logical_data_len_from_header(index: usize, header: &FitsHeader) -> Result<usize> {
    if index == 0 {
        return image_logical_len_from_header(header);
    }

    match parse_xtension(header)? {
        Some("IMAGE") => image_logical_len_from_header(header),
        Some("TABLE") => ascii_table_logical_len_from_header(header),
        Some("BINTABLE") => binary_table_logical_len_from_header(header),
        Some(_) => invalid_format("unsupported FITS XTENSION value"),
        None => invalid_format("extension HDU is missing XTENSION"),
    }
}

#[cfg(feature = "alloc")]
fn image_logical_len_from_header(header: &FitsHeader) -> Result<usize> {
    crate::image::FitsImageDescriptor::from_header(header)?.logical_data_len()
}

#[cfg(feature = "alloc")]
fn ascii_table_logical_len_from_header(header: &FitsHeader) -> Result<usize> {
    crate::table::FitsAsciiTableDescriptor::from_header(header)?.logical_data_len()
}

#[cfg(feature = "alloc")]
fn binary_table_logical_len_from_header(header: &FitsHeader) -> Result<usize> {
    crate::table::FitsBinaryTableDescriptor::from_header(header)?.logical_data_len()
}

#[cfg(feature = "alloc")]
fn parse_xtension<'a>(header: &'a FitsHeader) -> Result<Option<&'a str>> {
    let Some(card) = header.get_standard("XTENSION") else {
        return Ok(None);
    };

    match card.value() {
        Some(crate::header::HeaderValue::String(value)) => Ok(Some(value.as_str())),
        Some(_) => invalid_format("XTENSION must contain a string value"),
        None => invalid_format("XTENSION is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn invalid_format<T>(message: &str) -> Result<T> {
    Err(invalid_format_error(message))
}

#[cfg(feature = "alloc")]
fn invalid_format_error(message: &str) -> Error {
    Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: String::from(message),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    use consus_core::{Hyperslab, HyperslabDim};
    use consus_io::MemCursor;

    use crate::datastructure::FitsBlockAlignment;

    fn card(text: &str) -> [u8; 80] {
        assert!(text.len() <= 80);
        let mut raw = [b' '; 80];
        raw[..text.len()].copy_from_slice(text.as_bytes());
        raw
    }

    fn append_header(bytes: &mut Vec<u8>, cards: &[&str]) {
        let start = bytes.len();
        for text in cards {
            bytes.extend_from_slice(&card(text));
        }
        let padded = FitsBlockAlignment::padded_len(bytes.len() - start);
        bytes.resize(start + padded, b' ');
    }

    fn append_data(bytes: &mut Vec<u8>, data: &[u8]) {
        bytes.extend_from_slice(data);
        let padded = FitsBlockAlignment::padded_len(data.len());
        bytes.resize(bytes.len() + (padded - data.len()), 0);
    }

    fn primary_image_file_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        append_header(
            &mut bytes,
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                    8",
                "NAXIS   =                    1",
                "NAXIS1  =                    4",
                "END",
            ],
        );
        append_data(&mut bytes, &[1, 2, 3, 4]);
        bytes
    }

    fn image_and_table_file_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();

        append_header(
            &mut bytes,
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                    8",
                "NAXIS   =                    1",
                "NAXIS1  =                    4",
                "END",
            ],
        );
        append_data(&mut bytes, &[1, 2, 3, 4]);

        append_header(
            &mut bytes,
            &[
                "XTENSION= 'BINTABLE'",
                "BITPIX  =                    8",
                "NAXIS   =                    2",
                "NAXIS1  =                    2",
                "NAXIS2  =                    3",
                "PCOUNT  =                    0",
                "GCOUNT  =                    1",
                "TFIELDS =                    1",
                "TFORM1  = '1I      '",
                "END",
            ],
        );
        append_data(&mut bytes, &[10, 11, 20, 21, 30, 31]);

        bytes
    }

    #[test]
    fn scans_primary_image_file() {
        let cursor = MemCursor::from_bytes(primary_image_file_bytes());
        let file = FitsFile::open(cursor).unwrap();

        assert_eq!(file.format(), FITS_FORMAT_NAME);
        assert_eq!(file.hdu_count(), 1);
        assert!(file.primary_hdu().unwrap().is_primary());
        assert_eq!(file.primary_hdu().unwrap().data_span().logical_len(), 4);
    }

    #[test]
    fn scans_multiple_hdus() {
        let cursor = MemCursor::from_bytes(image_and_table_file_bytes());
        let file = FitsFile::open(cursor).unwrap();

        assert_eq!(file.hdu_count(), 2);
        assert!(file.hdu(0).unwrap().is_primary());
        assert!(file.hdu(1).unwrap().is_binary_table());
    }

    #[test]
    fn file_read_maps_paths_to_nodes() {
        let cursor = MemCursor::from_bytes(image_and_table_file_bytes());
        let file = FitsFile::open(cursor).unwrap();

        assert!(file.exists("/").unwrap());
        assert!(file.exists("/PRIMARY").unwrap());
        assert!(file.exists("/HDU/1").unwrap());
        assert!(!file.exists("/HDU/2").unwrap());

        assert_eq!(file.node_type_at("/").unwrap(), NodeType::Group);
        assert_eq!(file.node_type_at("/PRIMARY").unwrap(), NodeType::Dataset);
        assert_eq!(file.num_children_at("/").unwrap(), 2);
    }

    #[test]
    fn file_read_reports_dataset_metadata() {
        let cursor = MemCursor::from_bytes(image_and_table_file_bytes());
        let file = FitsFile::open(cursor).unwrap();

        assert_eq!(file.dataset_shape("/PRIMARY").unwrap(), Shape::fixed(&[4]));
        assert_eq!(
            file.dataset_datatype("/PRIMARY").unwrap(),
            crate::types::Bitpix::U8.to_datatype()
        );

        assert_eq!(file.dataset_shape("/HDU/1").unwrap(), Shape::fixed(&[3]));
        assert_eq!(
            file.dataset_datatype("/HDU/1").unwrap(),
            Datatype::Opaque {
                size: 2,
                #[cfg(feature = "alloc")]
                tag: Some("fits-row".into()),
            }
        );
    }

    #[test]
    fn file_read_reads_primary_and_table_payloads() {
        let cursor = MemCursor::from_bytes(image_and_table_file_bytes());
        let file = FitsFile::open(cursor).unwrap();

        let mut primary = [0u8; 4];
        let read = file
            .read_dataset_raw("/PRIMARY", &Selection::All, &mut primary)
            .unwrap();
        assert_eq!(read, 4);
        assert_eq!(primary, [1, 2, 3, 4]);

        let selection = Selection::Hyperslab(Hyperslab::new(&[HyperslabDim {
            start: 1,
            stride: 1,
            count: 2,
            block: 1,
        }]));
        let mut rows = [0u8; 4];
        let read = file
            .read_dataset_raw("/HDU/1", &selection, &mut rows)
            .unwrap();
        assert_eq!(read, 4);
        assert_eq!(rows, [20, 21, 30, 31]);
    }

    #[test]
    fn file_write_overwrites_primary_payload_and_preserves_padding() {
        let cursor = MemCursor::from_bytes(primary_image_file_bytes());
        let mut file = FitsFile::open_mut(cursor).unwrap();

        file.write_dataset_raw("/PRIMARY", &Selection::All, &[9, 8, 7, 6])
            .unwrap();

        let mut buf = [0u8; 4];
        let read = file
            .read_dataset_raw("/PRIMARY", &Selection::All, &mut buf)
            .unwrap();
        assert_eq!(read, 4);
        assert_eq!(buf, [9, 8, 7, 6]);
    }

    #[test]
    fn file_write_overwrites_contiguous_table_rows() {
        let cursor = MemCursor::from_bytes(image_and_table_file_bytes());
        let mut file = FitsFile::open_mut(cursor).unwrap();

        let selection = Selection::Hyperslab(Hyperslab::new(&[HyperslabDim {
            start: 1,
            stride: 1,
            count: 2,
            block: 1,
        }]));
        file.write_dataset_raw("/HDU/1", &selection, &[99, 98, 77, 76])
            .unwrap();

        let mut rows = [0u8; 6];
        let read = file
            .read_dataset_raw("/HDU/1", &Selection::All, &mut rows)
            .unwrap();
        assert_eq!(read, 6);
        assert_eq!(rows, [10, 11, 99, 98, 77, 76]);
    }

    #[test]
    fn create_group_is_unsupported_except_root() {
        let cursor = MemCursor::from_bytes(primary_image_file_bytes());
        let mut file = FitsFile::open_mut(cursor).unwrap();

        assert!(file.create_group("/").is_ok());
        assert!(matches!(
            file.create_group("/new"),
            Err(Error::UnsupportedFeature { .. })
        ));
    }

    #[test]
    fn invalid_paths_are_rejected() {
        let cursor = MemCursor::from_bytes(primary_image_file_bytes());
        let file = FitsFile::open(cursor).unwrap();

        assert!(file.node_type_at("/bad").is_err());
        assert!(file.dataset_shape("/HDU/x").is_err());
        assert!(file.dataset_datatype("/HDU/1/extra").is_err());
    }
}
