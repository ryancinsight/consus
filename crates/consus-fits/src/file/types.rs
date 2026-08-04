#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::{Datatype, Error, HasAttributes, Result, Selection};
use consus_io::{Length, ReadAt, WriteAt};

use crate::datastructure::{
    FITS_LOGICAL_RECORD_LEN, FitsDataSpan, FitsHeaderBlock, FitsHeaderCardCount,
};
use crate::hdu::{FitsHdu, FitsHduIndex, FitsHduSequence};
use crate::header::{FitsCard, FitsHeader, HeaderValue};

use super::read::read_hdu_payload;
use super::write::write_hdu_payload;

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
        read_hdu_payload(self.io(), hdu, &Selection::All, buf)
    }

    /// Read a raw selection from the HDU at `path`.
    pub fn read_hdu_selection(
        &self,
        path: &str,
        selection: &Selection,
        buf: &mut [u8],
    ) -> Result<usize> {
        let hdu = self.hdu_at_path(path)?;
        read_hdu_payload(self.io(), hdu, selection, buf)
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
        write_hdu_payload(self.io_mut(), &hdu, selection, data)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DatasetPath {
    Primary,
    Hdu(usize),
}

#[cfg(feature = "alloc")]
pub(super) fn parse_dataset_path(path: &str) -> Result<DatasetPath> {
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
fn parse_xtension(header: &FitsHeader) -> Result<Option<&str>> {
    let Some(card) = header.get_standard("XTENSION") else {
        return Ok(None);
    };

    match card.value() {
        Some(HeaderValue::String(value)) => Ok(Some(value.as_str())),
        Some(_) => invalid_format("XTENSION must contain a string value"),
        None => invalid_format("XTENSION is missing a value"),
    }
}

#[cfg(feature = "alloc")]
pub(super) fn invalid_format<T>(message: &str) -> Result<T> {
    Err(invalid_format_error(message))
}

#[cfg(feature = "alloc")]
fn invalid_format_error(message: &str) -> Error {
    Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.into(),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use consus_core::{
        Datatype, Error, FileRead, FileWrite, Hyperslab, HyperslabDim, NodeType, Selection, Shape,
    };
    use consus_io::MemCursor;

    use super::{FITS_FORMAT_NAME, FitsFile};
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
