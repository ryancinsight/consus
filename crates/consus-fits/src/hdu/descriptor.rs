use consus_core::Result;

use crate::datastructure::{FitsDataSpan, FitsHeaderBlock};
use crate::header::FitsHeader;
use crate::image::FitsImageDescriptor;
use crate::table::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor};
use crate::types::HduType;

use super::index::FitsHduIndex;
use super::kind::FitsHduKind;
use super::payload::FitsHduPayload;
use super::support::parse_xtension;

/// Canonical FITS HDU descriptor.
///
/// This type couples the parsed header, header block extent, data-unit span,
/// and semantic payload descriptor for one HDU.
#[derive(Debug, Clone, PartialEq)]
pub struct FitsHdu {
    index: FitsHduIndex,
    kind: FitsHduKind,
    header: FitsHeader,
    header_block: FitsHeaderBlock,
    data_span: FitsDataSpan,
    payload: FitsHduPayload,
}

impl FitsHdu {
    /// Construct an HDU from canonical fields.
    pub const fn new(
        index: FitsHduIndex,
        kind: FitsHduKind,
        header: FitsHeader,
        header_block: FitsHeaderBlock,
        data_span: FitsDataSpan,
        payload: FitsHduPayload,
    ) -> Self {
        Self {
            index,
            kind,
            header,
            header_block,
            data_span,
            payload,
        }
    }

    /// Parse an HDU descriptor from a header and structural extents.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if the header does not match the HDU kind
    /// implied by file position and `XTENSION`.
    pub fn from_header(
        index: FitsHduIndex,
        header: FitsHeader,
        header_block: FitsHeaderBlock,
        data_span: FitsDataSpan,
    ) -> Result<Self> {
        let xtension = parse_xtension(&header)?;
        let kind = FitsHduKind::from_position_and_xtension(index, xtension)?;

        let payload = match kind {
            FitsHduKind::Primary | FitsHduKind::ImageExtension => {
                FitsHduPayload::Image(FitsImageDescriptor::from_header(&header)?)
            }
            FitsHduKind::AsciiTableExtension => {
                FitsHduPayload::AsciiTable(FitsAsciiTableDescriptor::from_header(&header)?)
            }
            FitsHduKind::BinaryTableExtension => {
                FitsHduPayload::BinaryTable(FitsBinaryTableDescriptor::from_header(&header)?)
            }
        };

        Ok(Self::new(
            index,
            kind,
            header,
            header_block,
            data_span,
            payload,
        ))
    }

    /// Return the HDU index.
    pub const fn index(&self) -> FitsHduIndex {
        self.index
    }

    /// Return the HDU kind.
    pub const fn kind(&self) -> FitsHduKind {
        self.kind
    }

    /// Return the canonical `HduType`.
    pub const fn hdu_type(&self) -> HduType {
        self.kind.hdu_type()
    }

    /// Return the parsed header.
    pub const fn header(&self) -> &FitsHeader {
        &self.header
    }

    /// Return the header block extent.
    pub const fn header_block(&self) -> FitsHeaderBlock {
        self.header_block
    }

    /// Return the data-unit span.
    pub const fn data_span(&self) -> FitsDataSpan {
        self.data_span
    }

    /// Return the semantic payload descriptor.
    pub const fn payload(&self) -> &FitsHduPayload {
        &self.payload
    }

    /// Return whether this is the primary HDU.
    pub const fn is_primary(&self) -> bool {
        self.kind.is_primary()
    }

    /// Return whether this is an extension HDU.
    pub const fn is_extension(&self) -> bool {
        self.kind.is_extension()
    }

    /// Return whether this HDU is image-like.
    pub const fn is_image(&self) -> bool {
        self.kind.is_image()
    }

    /// Return whether this HDU is an ASCII table extension.
    pub const fn is_ascii_table(&self) -> bool {
        self.kind.is_ascii_table()
    }

    /// Return whether this HDU is a binary table extension.
    pub const fn is_binary_table(&self) -> bool {
        self.kind.is_binary_table()
    }

    /// Return the image descriptor, if present.
    pub const fn image(&self) -> Option<&FitsImageDescriptor> {
        self.payload.as_image()
    }

    /// Return the ASCII table descriptor, if present.
    pub const fn ascii_table(&self) -> Option<&FitsAsciiTableDescriptor> {
        self.payload.as_ascii_table()
    }

    /// Return the binary table descriptor, if present.
    pub const fn binary_table(&self) -> Option<&FitsBinaryTableDescriptor> {
        self.payload.as_binary_table()
    }
}
