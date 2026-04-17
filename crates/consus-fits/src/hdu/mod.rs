//! FITS Header/Data Unit descriptors and sequencing.
//!
//! ## Scope
//!
//! This module is the authoritative HDU-domain boundary for `consus-fits`.
//! It defines:
//! - HDU indexing and ordered sequencing
//! - primary and extension HDU classification
//! - header/data-unit coupling
//! - image/table payload descriptor attachment
//!
//! ## FITS invariants
//!
//! A FITS file is an ordered sequence of HDUs.
//! - The first HDU is the primary HDU.
//! - Subsequent HDUs are extension HDUs.
//! - Each HDU owns exactly one header block and one data-unit span.
//! - HDU kind is derived from authoritative header semantics.
//!
//! ## Architectural role
//!
//! This module depends on the authoritative `header`, `types`,
//! `datastructure`, `image`, and `table` modules. It does not duplicate
//! FITS keyword parsing, image metadata extraction, table metadata extraction,
//! or 2880-byte blocking math.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::{Error, Result};

use crate::datastructure::{FitsDataSpan, FitsHeaderBlock};
use crate::header::{FitsHeader, HeaderValue};
use crate::image::FitsImageDescriptor;
use crate::table::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor};
use crate::types::HduType;

/// FITS HDU ordinal index.
///
/// The primary HDU always has index `0`. Extension HDUs follow in file order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FitsHduIndex(usize);

impl FitsHduIndex {
    /// Construct an HDU index from a zero-based ordinal.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based ordinal.
    pub const fn get(self) -> usize {
        self.0
    }

    /// Return whether this is the primary HDU index.
    pub const fn is_primary(self) -> bool {
        self.0 == 0
    }
}

impl From<usize> for FitsHduIndex {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

/// FITS HDU semantic classification.
///
/// This refines `crate::types::HduType` with primary/extension role semantics
/// while preserving the canonical FITS HDU taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitsHduKind {
    /// Primary HDU.
    Primary,
    /// IMAGE extension HDU.
    ImageExtension,
    /// ASCII table extension HDU.
    AsciiTableExtension,
    /// Binary table extension HDU.
    BinaryTableExtension,
}

impl FitsHduKind {
    /// Derive an HDU kind from file position and optional `XTENSION` value.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if:
    /// - the primary HDU carries `XTENSION`
    /// - an extension HDU omits `XTENSION`
    /// - the extension type is unsupported
    pub fn from_position_and_xtension(index: FitsHduIndex, xtension: Option<&str>) -> Result<Self> {
        if index.is_primary() {
            if xtension.is_some() {
                return invalid_format("primary HDU must not define XTENSION");
            }
            return Ok(Self::Primary);
        }

        match xtension.map(str::trim_end) {
            Some("IMAGE") => Ok(Self::ImageExtension),
            Some("TABLE") => Ok(Self::AsciiTableExtension),
            Some("BINTABLE") => Ok(Self::BinaryTableExtension),
            Some(_) => invalid_format("unsupported FITS XTENSION value"),
            None => invalid_format("extension HDU is missing XTENSION"),
        }
    }

    /// Return the canonical `HduType`.
    pub const fn hdu_type(self) -> HduType {
        match self {
            Self::Primary => HduType::Primary,
            Self::ImageExtension => HduType::Image,
            Self::AsciiTableExtension => HduType::Table,
            Self::BinaryTableExtension => HduType::BinTable,
        }
    }

    /// Return whether this is the primary HDU kind.
    pub const fn is_primary(self) -> bool {
        matches!(self, Self::Primary)
    }

    /// Return whether this is an extension HDU kind.
    pub const fn is_extension(self) -> bool {
        !self.is_primary()
    }

    /// Return whether this HDU carries image payload semantics.
    pub const fn is_image(self) -> bool {
        matches!(self, Self::Primary | Self::ImageExtension)
    }

    /// Return whether this HDU carries ASCII table payload semantics.
    pub const fn is_ascii_table(self) -> bool {
        matches!(self, Self::AsciiTableExtension)
    }

    /// Return whether this HDU carries binary table payload semantics.
    pub const fn is_binary_table(self) -> bool {
        matches!(self, Self::BinaryTableExtension)
    }
}

/// FITS HDU payload descriptor.
///
/// This enum attaches the parsed semantic payload descriptor corresponding to
/// the HDU kind.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum FitsHduPayload {
    /// Image payload descriptor for primary or IMAGE extension HDUs.
    Image(FitsImageDescriptor),
    /// ASCII table payload descriptor.
    AsciiTable(FitsAsciiTableDescriptor),
    /// Binary table payload descriptor.
    BinaryTable(FitsBinaryTableDescriptor),
}

#[cfg(feature = "alloc")]
impl FitsHduPayload {
    /// Return whether this payload is image-like.
    pub const fn is_image(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    /// Return whether this payload is an ASCII table.
    pub const fn is_ascii_table(&self) -> bool {
        matches!(self, Self::AsciiTable(_))
    }

    /// Return whether this payload is a binary table.
    pub const fn is_binary_table(&self) -> bool {
        matches!(self, Self::BinaryTable(_))
    }

    /// Return the image descriptor, if present.
    pub const fn as_image(&self) -> Option<&FitsImageDescriptor> {
        match self {
            Self::Image(value) => Some(value),
            _ => None,
        }
    }

    /// Return the ASCII table descriptor, if present.
    pub const fn as_ascii_table(&self) -> Option<&FitsAsciiTableDescriptor> {
        match self {
            Self::AsciiTable(value) => Some(value),
            _ => None,
        }
    }

    /// Return the binary table descriptor, if present.
    pub const fn as_binary_table(&self) -> Option<&FitsBinaryTableDescriptor> {
        match self {
            Self::BinaryTable(value) => Some(value),
            _ => None,
        }
    }
}

/// Canonical FITS HDU descriptor.
///
/// This type couples the parsed header, header block extent, data-unit span,
/// and semantic payload descriptor for one HDU.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsHdu {
    index: FitsHduIndex,
    kind: FitsHduKind,
    header: FitsHeader,
    header_block: FitsHeaderBlock,
    data_span: FitsDataSpan,
    payload: FitsHduPayload,
}

#[cfg(feature = "alloc")]
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

/// Ordered FITS HDU sequence.
///
/// This type preserves file order and provides deterministic indexed access.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitsHduSequence {
    hdus: Vec<FitsHdu>,
}

#[cfg(feature = "alloc")]
impl FitsHduSequence {
    /// Construct an HDU sequence from ordered HDUs.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if:
    /// - the sequence is empty
    /// - the first HDU is not primary
    /// - any later HDU is primary
    /// - HDU indices are not contiguous and order-preserving
    pub fn new(hdus: Vec<FitsHdu>) -> Result<Self> {
        validate_sequence(&hdus)?;
        Ok(Self { hdus })
    }

    /// Construct an empty HDU sequence.
    pub const fn empty() -> Self {
        Self { hdus: Vec::new() }
    }

    /// Return the ordered HDUs.
    pub fn hdus(&self) -> &[FitsHdu] {
        &self.hdus
    }

    /// Return the number of HDUs.
    pub fn len(&self) -> usize {
        self.hdus.len()
    }

    /// Return whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.hdus.is_empty()
    }

    /// Return the primary HDU.
    pub fn primary(&self) -> Option<&FitsHdu> {
        self.hdus.first()
    }

    /// Return the HDU at `index`.
    pub fn get(&self, index: FitsHduIndex) -> Option<&FitsHdu> {
        self.hdus.get(index.get())
    }

    /// Return the HDU at zero-based ordinal `index`.
    pub fn get_usize(&self, index: usize) -> Option<&FitsHdu> {
        self.hdus.get(index)
    }

    /// Return an iterator over the HDUs.
    pub fn iter(&self) -> core::slice::Iter<'_, FitsHdu> {
        self.hdus.iter()
    }

    /// Append an HDU while preserving FITS sequence invariants.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if the appended HDU violates ordering,
    /// primary placement, or contiguous indexing.
    pub fn push(&mut self, hdu: FitsHdu) -> Result<()> {
        let expected_index = self.hdus.len();
        if hdu.index().get() != expected_index {
            return invalid_format("FITS HDU indices must be contiguous and ordered");
        }

        if expected_index == 0 {
            if !hdu.is_primary() {
                return invalid_format("first FITS HDU must be primary");
            }
        } else if hdu.is_primary() {
            return invalid_format("only the first FITS HDU may be primary");
        }

        self.hdus.push(hdu);
        Ok(())
    }
}

#[cfg(feature = "alloc")]
impl IntoIterator for FitsHduSequence {
    type Item = FitsHdu;
    type IntoIter = alloc::vec::IntoIter<FitsHdu>;

    fn into_iter(self) -> Self::IntoIter {
        self.hdus.into_iter()
    }
}

#[cfg(feature = "alloc")]
impl<'a> IntoIterator for &'a FitsHduSequence {
    type Item = &'a FitsHdu;
    type IntoIter = core::slice::Iter<'a, FitsHdu>;

    fn into_iter(self) -> Self::IntoIter {
        self.hdus.iter()
    }
}

#[cfg(feature = "alloc")]
fn validate_sequence(hdus: &[FitsHdu]) -> Result<()> {
    if hdus.is_empty() {
        return invalid_format("FITS file must contain at least one HDU");
    }

    for (expected_index, hdu) in hdus.iter().enumerate() {
        if hdu.index().get() != expected_index {
            return invalid_format("FITS HDU indices must be contiguous and ordered");
        }

        if expected_index == 0 {
            if !hdu.is_primary() {
                return invalid_format("first FITS HDU must be primary");
            }
        } else if hdu.is_primary() {
            return invalid_format("only the first FITS HDU may be primary");
        }
    }

    Ok(())
}

#[cfg(feature = "alloc")]
fn parse_xtension<'a>(header: &'a FitsHeader) -> Result<Option<&'a str>> {
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
fn invalid_format<T>(message: &str) -> Result<T> {
    Err(Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.into(),
    })
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    use crate::datastructure::{FitsBlockAlignment, FitsHeaderCardCount};
    use crate::file::parse_extension_header_bytes;
    use crate::header::parse_header_bytes;

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

    fn primary_hdu() -> FitsHdu {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    1",
            "NAXIS1  =                    4",
            "END",
        ]);
        let header = parse_header_bytes(&bytes).unwrap();
        FitsHdu::from_header(
            FitsHduIndex::new(0),
            header,
            FitsHeaderBlock::new(FitsHeaderCardCount::new(5)),
            FitsDataSpan::new(2880, 4).unwrap(),
        )
        .unwrap()
    }

    fn image_extension_hdu() -> FitsHdu {
        let bytes = header_bytes(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    2",
            "NAXIS2  =                    3",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        FitsHdu::from_header(
            FitsHduIndex::new(1),
            header,
            FitsHeaderBlock::new(FitsHeaderCardCount::new(7)),
            FitsDataSpan::new(5760, 12).unwrap(),
        )
        .unwrap()
    }

    fn ascii_table_hdu() -> FitsHdu {
        let bytes = header_bytes(&[
            "XTENSION= 'TABLE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    8",
            "NAXIS2  =                    2",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "TFIELDS =                    1",
            "TFORM1  = 'A8      '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        FitsHdu::from_header(
            FitsHduIndex::new(1),
            header,
            FitsHeaderBlock::new(FitsHeaderCardCount::new(9)),
            FitsDataSpan::new(5760, 16).unwrap(),
        )
        .unwrap()
    }

    fn binary_table_hdu() -> FitsHdu {
        let bytes = header_bytes(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    3",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "TFIELDS =                    1",
            "TFORM1  = '1J      '",
            "END",
        ]);
        let header = parse_extension_header_bytes(&bytes).unwrap();
        FitsHdu::from_header(
            FitsHduIndex::new(1),
            header,
            FitsHeaderBlock::new(FitsHeaderCardCount::new(9)),
            FitsDataSpan::new(5760, 12).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn hdu_kind_derives_from_position_and_xtension() {
        assert_eq!(
            FitsHduKind::from_position_and_xtension(FitsHduIndex::new(0), None).unwrap(),
            FitsHduKind::Primary
        );
        assert_eq!(
            FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("IMAGE")).unwrap(),
            FitsHduKind::ImageExtension
        );
        assert_eq!(
            FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("TABLE")).unwrap(),
            FitsHduKind::AsciiTableExtension
        );
        assert_eq!(
            FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("BINTABLE"))
                .unwrap(),
            FitsHduKind::BinaryTableExtension
        );
    }

    #[test]
    fn hdu_kind_rejects_invalid_primary_and_extension_forms() {
        assert!(
            FitsHduKind::from_position_and_xtension(FitsHduIndex::new(0), Some("IMAGE")).is_err()
        );
        assert!(FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), None).is_err());
        assert!(
            FitsHduKind::from_position_and_xtension(FitsHduIndex::new(1), Some("A3DTABLE"))
                .is_err()
        );
    }

    #[test]
    fn parses_primary_hdu_descriptor() {
        let hdu = primary_hdu();
        assert!(hdu.is_primary());
        assert!(hdu.is_image());
        assert_eq!(hdu.index().get(), 0);
        assert_eq!(hdu.hdu_type(), HduType::Primary);
        assert_eq!(hdu.image().unwrap().axis_lengths(), &[4]);
        assert_eq!(hdu.data_span().logical_len(), 4);
    }

    #[test]
    fn parses_image_extension_descriptor() {
        let hdu = image_extension_hdu();
        assert!(hdu.is_extension());
        assert!(hdu.is_image());
        assert_eq!(hdu.kind(), FitsHduKind::ImageExtension);
        assert_eq!(hdu.hdu_type(), HduType::Image);
        assert_eq!(hdu.image().unwrap().axis_lengths(), &[2, 3]);
    }

    #[test]
    fn parses_ascii_table_extension_descriptor() {
        let hdu = ascii_table_hdu();
        assert!(hdu.is_ascii_table());
        assert_eq!(hdu.kind(), FitsHduKind::AsciiTableExtension);
        assert_eq!(hdu.hdu_type(), HduType::Table);
        assert_eq!(hdu.ascii_table().unwrap().rows(), 2);
        assert_eq!(hdu.ascii_table().unwrap().row_len(), 8);
    }

    #[test]
    fn parses_binary_table_extension_descriptor() {
        let hdu = binary_table_hdu();
        assert!(hdu.is_binary_table());
        assert_eq!(hdu.kind(), FitsHduKind::BinaryTableExtension);
        assert_eq!(hdu.hdu_type(), HduType::BinTable);
        assert_eq!(hdu.binary_table().unwrap().rows(), 3);
        assert_eq!(hdu.binary_table().unwrap().row_len(), 4);
    }

    #[test]
    fn sequence_requires_primary_first_and_contiguous_indices() {
        let primary = primary_hdu();
        let image = image_extension_hdu();

        let sequence = FitsHduSequence::new(vec![primary.clone(), image.clone()]).unwrap();
        assert_eq!(sequence.len(), 2);
        assert!(sequence.primary().unwrap().is_primary());
        assert_eq!(
            sequence.get(FitsHduIndex::new(1)).unwrap().kind(),
            image.kind()
        );

        let invalid_first = FitsHduSequence::new(vec![image.clone()]);
        assert!(invalid_first.is_err());

        let invalid_gap = FitsHduSequence::new(vec![
            primary,
            FitsHdu::new(
                FitsHduIndex::new(2),
                image.kind(),
                image.header().clone(),
                image.header_block(),
                image.data_span(),
                image.payload().clone(),
            ),
        ]);
        assert!(invalid_gap.is_err());
    }

    #[test]
    fn sequence_push_preserves_invariants() {
        let mut sequence = FitsHduSequence::empty();
        sequence.push(primary_hdu()).unwrap();
        sequence.push(image_extension_hdu()).unwrap();

        assert_eq!(sequence.len(), 2);
        assert!(sequence.primary().unwrap().is_primary());

        let invalid_primary_again = sequence.push(primary_hdu());
        assert!(invalid_primary_again.is_err());
    }
}
