//! FITS image metadata extraction and raw image access.
//!
//! ## Scope
//!
//! This module is the authoritative image-domain boundary for `consus-fits`.
//! It defines:
//! - multi-dimensional FITS image descriptors derived from header keywords
//! - optional physical-value scaling metadata (`BSCALE`, `BZERO`, `BLANK`)
//! - random-groups detection metadata
//! - raw image byte access over FITS data-unit spans
//!
//! ## FITS invariants
//!
//! For a standard image HDU:
//! - `BITPIX` defines the stored element representation
//! - `NAXIS` defines the rank
//! - `NAXISn` for `n ∈ [1, NAXIS]` define axis extents
//! - the logical payload size is `product(NAXISn) * element_size(BITPIX)`
//!
//! For random groups:
//! - `GROUPS = T`
//! - `NAXIS1 = 0`
//! - `GCOUNT >= 1`
//! - `PCOUNT >= 0`
//! - the logical payload size is
//!   `(PCOUNT + product(NAXIS2..NAXISn)) * GCOUNT * element_size(BITPIX)`
//!
//! FITS stores multi-byte numeric values in big-endian order.
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
use alloc::vec::Vec;

use consus_core::{Error, Result, Selection, Shape};
use consus_io::ReadAt;

use crate::datastructure::FitsDataSpan;
use crate::header::{FitsHeader, HeaderValue};
use crate::types::Bitpix;

/// Parsed FITS image scaling metadata.
///
/// Physical values are derived from stored array values by:
/// `physical = BSCALE * stored + BZERO`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitsImageScaling {
    /// Multiplicative scale factor.
    pub bscale: f64,
    /// Additive zero-point offset.
    pub bzero: f64,
    /// Optional integer blank sentinel for undefined pixels.
    pub blank: Option<i64>,
}

impl FitsImageScaling {
    /// Canonical identity scaling.
    pub const fn identity() -> Self {
        Self {
            bscale: 1.0,
            bzero: 0.0,
            blank: None,
        }
    }

    /// Return whether the scaling is the FITS identity transform.
    pub fn is_identity(self) -> bool {
        self.bscale == 1.0 && self.bzero == 0.0 && self.blank.is_none()
    }
}

impl Default for FitsImageScaling {
    fn default() -> Self {
        Self::identity()
    }
}

/// Random-groups metadata for legacy FITS image payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitsRandomGroups {
    /// Number of group parameters per group.
    pub parameter_count: usize,
    /// Number of groups.
    pub group_count: usize,
}

impl FitsRandomGroups {
    /// Return whether the random-groups descriptor is semantically empty.
    pub const fn is_empty(self) -> bool {
        self.parameter_count == 0 && self.group_count == 0
    }
}

/// Canonical FITS image descriptor.
///
/// This type is the single source of truth for image-domain metadata derived
/// from a FITS header.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsImageDescriptor {
    bitpix: Bitpix,
    shape: Shape,
    axis_lengths: Vec<usize>,
    scaling: FitsImageScaling,
    random_groups: Option<FitsRandomGroups>,
}

#[cfg(feature = "alloc")]
impl FitsImageDescriptor {
    /// Construct an image descriptor from canonical fields.
    pub fn new(
        bitpix: Bitpix,
        axis_lengths: Vec<usize>,
        scaling: FitsImageScaling,
        random_groups: Option<FitsRandomGroups>,
    ) -> Self {
        let shape = Shape::fixed(&axis_lengths);
        Self {
            bitpix,
            shape,
            axis_lengths,
            scaling,
            random_groups,
        }
    }

    /// Parse an image descriptor from a FITS header.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` when required image keywords are missing
    /// or semantically invalid.
    pub fn from_header(header: &FitsHeader) -> Result<Self> {
        let bitpix = parse_required_integer(header, "BITPIX").and_then(Bitpix::from_i64)?;

        let naxis = parse_required_integer(header, "NAXIS")?;
        if naxis < 0 {
            return invalid_format("NAXIS must be a non-negative integer");
        }
        let rank = usize::try_from(naxis).map_err(|_| Error::Overflow)?;

        let mut axis_lengths = Vec::with_capacity(rank);
        for axis_index in 1..=rank {
            let keyword = axis_keyword(axis_index)?;
            let axis_len = parse_required_integer(header, keyword)?;
            if axis_len < 0 {
                return invalid_format("NAXISn must be a non-negative integer");
            }
            axis_lengths.push(usize::try_from(axis_len).map_err(|_| Error::Overflow)?);
        }

        let scaling = FitsImageScaling {
            bscale: parse_optional_real(header, "BSCALE")?.unwrap_or(1.0),
            bzero: parse_optional_real(header, "BZERO")?.unwrap_or(0.0),
            blank: parse_optional_integer(header, "BLANK")?,
        };

        let random_groups = parse_random_groups(header, rank, &axis_lengths)?;

        Ok(Self::new(bitpix, axis_lengths, scaling, random_groups))
    }

    /// Return the stored FITS element representation.
    pub const fn bitpix(&self) -> Bitpix {
        self.bitpix
    }

    /// Return the canonical dataset shape.
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Return the axis lengths in FITS header order.
    pub fn axis_lengths(&self) -> &[usize] {
        &self.axis_lengths
    }

    /// Return the image rank.
    pub fn rank(&self) -> usize {
        self.axis_lengths.len()
    }

    /// Return the scaling metadata.
    pub const fn scaling(&self) -> FitsImageScaling {
        self.scaling
    }

    /// Return the random-groups metadata, if present.
    pub const fn random_groups(&self) -> Option<FitsRandomGroups> {
        self.random_groups
    }

    /// Return whether this descriptor represents a random-groups payload.
    pub fn is_random_groups(&self) -> bool {
        self.random_groups.is_some()
    }

    /// Return the number of stored array elements for a standard image.
    ///
    /// For random groups, this returns the product of the image axes only,
    /// excluding group parameters and group multiplicity.
    pub fn num_image_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Return the logical payload size in bytes.
    ///
    /// ## Errors
    ///
    /// Returns `Error::Overflow` if the computed byte count exceeds `usize`.
    pub fn logical_data_len(&self) -> Result<usize> {
        let element_size = self.bitpix.element_size();

        let element_count = match self.random_groups {
            None => self.num_image_elements(),
            Some(groups) => {
                let group_image_elements = if self.axis_lengths.len() <= 1 {
                    0
                } else {
                    self.axis_lengths[1..]
                        .iter()
                        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                        .ok_or(Error::Overflow)?
                };

                let per_group = groups
                    .parameter_count
                    .checked_add(group_image_elements)
                    .ok_or(Error::Overflow)?;

                per_group
                    .checked_mul(groups.group_count)
                    .ok_or(Error::Overflow)?
            }
        };

        element_count
            .checked_mul(element_size)
            .ok_or(Error::Overflow)
    }

    /// Return whether the descriptor represents an empty logical payload.
    pub fn is_empty(&self) -> Result<bool> {
        self.logical_data_len().map(|len| len == 0)
    }
}

/// Raw FITS image view over a data-unit span.
///
/// This type provides zero-copy metadata plus bounded raw-byte access through
/// `consus-io::ReadAt`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct FitsImageData {
    descriptor: FitsImageDescriptor,
    span: FitsDataSpan,
}

#[cfg(feature = "alloc")]
impl FitsImageData {
    /// Construct a raw image view from a descriptor and data span.
    pub const fn new(descriptor: FitsImageDescriptor, span: FitsDataSpan) -> Self {
        Self { descriptor, span }
    }

    /// Return the image descriptor.
    pub const fn descriptor(&self) -> &FitsImageDescriptor {
        &self.descriptor
    }

    /// Return the data-unit span.
    pub const fn span(&self) -> FitsDataSpan {
        self.span
    }

    /// Read the entire logical image payload into `buf`.
    ///
    /// ## Errors
    ///
    /// Returns:
    /// - `Error::BufferTooSmall` if `buf` is smaller than the logical payload
    /// - any underlying I/O error from `ReadAt`
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

    /// Read a raw selection from the image payload.
    ///
    /// Current support is intentionally strict:
    /// - `Selection::All` reads the full logical payload
    /// - `Selection::None` reads zero bytes
    ///
    /// Other selection variants are rejected because FITS image subarray
    /// extraction requires element-stride aware byte mapping that is outside
    /// this module's current raw-byte boundary.
    pub fn read_selection<R: ReadAt>(
        &self,
        reader: &R,
        selection: &Selection,
        buf: &mut [u8],
    ) -> Result<usize> {
        match selection {
            Selection::All => self.read_all(reader, buf),
            Selection::None => Ok(0),
            Selection::Hyperslab(_) | Selection::Points(_) => Err(Error::UnsupportedFeature {
                #[cfg(feature = "alloc")]
                feature: "FITS raw image partial selection is not implemented".into(),
            }),
        }
    }
}

#[cfg(feature = "alloc")]
fn parse_required_integer(header: &FitsHeader, keyword: &str) -> Result<i64> {
    parse_optional_integer(header, keyword)?.ok_or_else(|| Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: alloc::format!("missing required FITS image keyword: {keyword}"),
    })
}

#[cfg(feature = "alloc")]
fn parse_optional_integer(header: &FitsHeader, keyword: &str) -> Result<Option<i64>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };

    match card.value() {
        Some(HeaderValue::Integer(value)) => value.to_i64().map(Some),
        Some(_) => invalid_format("FITS image keyword must contain an integer value"),
        None => invalid_format("FITS image keyword is missing a value"),
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
        Some(_) => invalid_format("FITS image keyword must contain a numeric value"),
        None => invalid_format("FITS image keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn parse_optional_logical(header: &FitsHeader, keyword: &str) -> Result<Option<bool>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };

    match card.value() {
        Some(HeaderValue::Logical(value)) => Ok(Some(*value)),
        Some(_) => invalid_format("FITS image keyword must contain a logical value"),
        None => invalid_format("FITS image keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn parse_random_groups(
    header: &FitsHeader,
    rank: usize,
    axis_lengths: &[usize],
) -> Result<Option<FitsRandomGroups>> {
    let groups = parse_optional_logical(header, "GROUPS")?.unwrap_or(false);
    if !groups {
        return Ok(None);
    }

    if rank == 0 {
        return invalid_format("GROUPS requires NAXIS >= 1");
    }

    if axis_lengths.first().copied().unwrap_or(1) != 0 {
        return invalid_format("random groups require NAXIS1 = 0");
    }

    let pcount = parse_required_integer(header, "PCOUNT")?;
    let gcount = parse_required_integer(header, "GCOUNT")?;

    if pcount < 0 {
        return invalid_format("PCOUNT must be a non-negative integer");
    }
    if gcount < 1 {
        return invalid_format("GCOUNT must be a positive integer");
    }

    Ok(Some(FitsRandomGroups {
        parameter_count: usize::try_from(pcount).map_err(|_| Error::Overflow)?,
        group_count: usize::try_from(gcount).map_err(|_| Error::Overflow)?,
    }))
}

#[cfg(feature = "alloc")]
fn axis_keyword(axis_index: usize) -> Result<&'static str> {
    match axis_index {
        1 => Ok("NAXIS1"),
        2 => Ok("NAXIS2"),
        3 => Ok("NAXIS3"),
        4 => Ok("NAXIS4"),
        5 => Ok("NAXIS5"),
        6 => Ok("NAXIS6"),
        7 => Ok("NAXIS7"),
        8 => Ok("NAXIS8"),
        9 => Ok("NAXIS9"),
        _ => Err(Error::UnsupportedFeature {
            #[cfg(feature = "alloc")]
            feature: "FITS image ranks above 9 are not implemented".into(),
        }),
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
    use consus_io::MemCursor;

    use crate::datastructure::FitsBlockAlignment;
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

    #[test]
    fn parses_standard_image_descriptor() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    3",
            "NAXIS2  =                    2",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).unwrap();
        let descriptor = FitsImageDescriptor::from_header(&header).unwrap();

        assert_eq!(descriptor.bitpix(), Bitpix::I16);
        assert_eq!(descriptor.axis_lengths(), &[3, 2]);
        assert_eq!(descriptor.rank(), 2);
        assert_eq!(descriptor.num_image_elements(), 6);
        assert_eq!(descriptor.logical_data_len().unwrap(), 12);
        assert_eq!(descriptor.scaling(), FitsImageScaling::identity());
        assert!(!descriptor.is_random_groups());
    }

    #[test]
    fn parses_scaling_keywords() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    1",
            "NAXIS1  =                    4",
            "BSCALE  =                  2.5",
            "BZERO   =                 -1.0",
            "BLANK   =                 -999",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).unwrap();
        let descriptor = FitsImageDescriptor::from_header(&header).unwrap();

        assert_eq!(descriptor.bitpix(), Bitpix::F32);
        assert_eq!(descriptor.logical_data_len().unwrap(), 16);
        assert_eq!(
            descriptor.scaling(),
            FitsImageScaling {
                bscale: 2.5,
                bzero: -1.0,
                blank: Some(-999),
            }
        );
    }

    #[test]
    fn parses_random_groups_descriptor() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    3",
            "NAXIS1  =                    0",
            "NAXIS2  =                    5",
            "NAXIS3  =                    7",
            "GROUPS  =                    T",
            "PCOUNT  =                    2",
            "GCOUNT  =                    3",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).unwrap();
        let descriptor = FitsImageDescriptor::from_header(&header).unwrap();

        assert!(descriptor.is_random_groups());
        assert_eq!(
            descriptor.random_groups(),
            Some(FitsRandomGroups {
                parameter_count: 2,
                group_count: 3,
            })
        );
        assert_eq!(descriptor.logical_data_len().unwrap(), (2 + 35) * 3 * 2);
    }

    #[test]
    fn rejects_random_groups_without_zero_naxis1() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    5",
            "GROUPS  =                    T",
            "PCOUNT  =                    1",
            "GCOUNT  =                    2",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).unwrap();
        let error = FitsImageDescriptor::from_header(&header).unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }

    #[test]
    fn reads_full_raw_image_payload() {
        let descriptor =
            FitsImageDescriptor::new(Bitpix::U8, vec![4], FitsImageScaling::identity(), None);
        let span = FitsDataSpan::new(0, 4).unwrap();
        let image = FitsImageData::new(descriptor, span);

        let reader = MemCursor::from_bytes(vec![10, 20, 30, 40]);
        let mut buf = [0u8; 4];
        let read = image.read_all(&reader, &mut buf).unwrap();

        assert_eq!(read, 4);
        assert_eq!(buf, [10, 20, 30, 40]);
    }

    #[test]
    fn read_selection_supports_all_and_none_only() {
        let descriptor =
            FitsImageDescriptor::new(Bitpix::U8, vec![2, 2], FitsImageScaling::identity(), None);
        let span = FitsDataSpan::new(0, 4).unwrap();
        let image = FitsImageData::new(descriptor, span);

        let reader = MemCursor::from_bytes(vec![1, 2, 3, 4]);
        let mut buf = [0u8; 4];

        let read = image
            .read_selection(&reader, &Selection::All, &mut buf)
            .unwrap();
        assert_eq!(read, 4);
        assert_eq!(buf, [1, 2, 3, 4]);

        let read_none = image
            .read_selection(&reader, &Selection::None, &mut buf)
            .unwrap();
        assert_eq!(read_none, 0);
    }
}
