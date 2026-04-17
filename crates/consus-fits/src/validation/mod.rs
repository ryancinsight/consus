//! FITS file-, HDU-, and path-level invariant validation.
//!
//! ## Scope
//!
//! This module is the authoritative validation boundary for `consus-fits`.
//! It defines reusable invariant checks for:
//! - FITS file structure
//! - HDU ordering and semantic consistency
//! - synthetic `consus-core` path mapping
//!
//! ## Architectural role
//!
//! Validation is isolated from parsing, I/O, and payload decoding. Higher-level
//! modules call these helpers to enforce FITS structural contracts without
//! duplicating error construction or invariant logic.
//!
//! ## Invariants
//!
//! - A FITS file contains at least one HDU.
//! - The first HDU is the primary HDU.
//! - Later HDUs are extensions only.
//! - HDU indices are contiguous and order-preserving.
//! - Header and data extents are 2880-byte aligned on disk.
//! - Synthetic dataset paths are limited to `/PRIMARY` and `/HDU/{n}`.
//! - `/` is the only synthetic group path.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

use consus_core::{Error, Result};

use crate::datastructure::{FitsBlockAlignment, FitsDataSpan, FitsHeaderBlock};
use crate::hdu::{FitsHdu, FitsHduIndex, FitsHduSequence};

/// Canonical synthetic FITS root path.
pub const FITS_ROOT_PATH: &str = "/";

/// Canonical synthetic FITS primary HDU dataset path.
pub const FITS_PRIMARY_PATH: &str = "/PRIMARY";

/// Parsed synthetic FITS path classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitsPathKind {
    /// Synthetic root group path `/`.
    Root,
    /// Primary HDU dataset path `/PRIMARY`.
    PrimaryDataset,
    /// Extension or primary HDU dataset path `/HDU/{n}`.
    HduDataset(FitsHduIndex),
}

impl FitsPathKind {
    /// Return whether this path denotes the synthetic root group.
    pub const fn is_root(self) -> bool {
        matches!(self, Self::Root)
    }

    /// Return whether this path denotes a dataset.
    pub const fn is_dataset(self) -> bool {
        matches!(self, Self::PrimaryDataset | Self::HduDataset(_))
    }

    /// Return the HDU index addressed by this path, if any.
    pub const fn hdu_index(self) -> Option<FitsHduIndex> {
        match self {
            Self::PrimaryDataset => Some(FitsHduIndex::new(0)),
            Self::HduDataset(index) => Some(index),
            Self::Root => None,
        }
    }
}

/// Validate a synthetic FITS path and classify it.
///
/// Accepted paths:
/// - `/`
/// - `/PRIMARY`
/// - `/HDU/{n}` where `{n}` is a non-negative decimal integer
///
/// ## Errors
///
/// Returns `Error::NotFound` if the path is not a valid synthetic FITS path.
#[cfg(feature = "alloc")]
pub fn validate_path(path: &str) -> Result<FitsPathKind> {
    if path == FITS_ROOT_PATH {
        return Ok(FitsPathKind::Root);
    }

    if path == FITS_PRIMARY_PATH {
        return Ok(FitsPathKind::PrimaryDataset);
    }

    let Some(rest) = path.strip_prefix("/HDU/") else {
        return Err(not_found(path));
    };

    if rest.is_empty() || rest.contains('/') {
        return Err(not_found(path));
    }

    let index = rest.parse::<usize>().map_err(|_| not_found(path))?;
    Ok(FitsPathKind::HduDataset(FitsHduIndex::new(index)))
}

/// Validate that a path denotes the synthetic FITS root group.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if the path is not `/`.
#[cfg(feature = "alloc")]
pub fn validate_root_path(path: &str) -> Result<()> {
    match validate_path(path)? {
        FitsPathKind::Root => Ok(()),
        _ => invalid_format("FITS root operation requires path '/'"),
    }
}

/// Validate that a path denotes a synthetic FITS dataset path.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if the path is not `/PRIMARY` or `/HDU/{n}`.
#[cfg(feature = "alloc")]
pub fn validate_dataset_path(path: &str) -> Result<FitsHduIndex> {
    match validate_path(path)? {
        FitsPathKind::PrimaryDataset => Ok(FitsHduIndex::new(0)),
        FitsPathKind::HduDataset(index) => Ok(index),
        FitsPathKind::Root => invalid_format("FITS dataset operation requires an HDU dataset path"),
    }
}

/// Validate a header block extent.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if:
/// - the padded header length is not 2880-byte aligned
/// - the unpadded header length exceeds the padded length
#[cfg(feature = "alloc")]
pub fn validate_header_block(header_block: FitsHeaderBlock) -> Result<()> {
    if header_block.byte_len() > header_block.padded_byte_len() {
        return invalid_format("FITS header block byte length exceeds padded byte length");
    }

    if !FitsBlockAlignment::is_aligned(header_block.padded_byte_len()) {
        return invalid_format("FITS header block padded length is not 2880-byte aligned");
    }

    Ok(())
}

/// Validate a data-unit span.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if:
/// - the padded data length is not 2880-byte aligned when non-zero
/// - the logical data length exceeds the padded length
#[cfg(feature = "alloc")]
pub fn validate_data_span(span: FitsDataSpan) -> Result<()> {
    if span.logical_len() > span.padded_len() {
        return invalid_format("FITS data span logical length exceeds padded length");
    }

    if span.padded_len() != 0 && !FitsBlockAlignment::is_aligned(span.padded_len()) {
        return invalid_format("FITS data span padded length is not 2880-byte aligned");
    }

    Ok(())
}

/// Validate one HDU's structural invariants.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if:
/// - the HDU index does not match its primary/extension role
/// - the header block is invalid
/// - the data span is invalid
#[cfg(feature = "alloc")]
pub fn validate_hdu(hdu: &FitsHdu) -> Result<()> {
    if hdu.index().is_primary() != hdu.is_primary() {
        return invalid_format("FITS HDU primary role does not match HDU index");
    }

    validate_header_block(hdu.header_block())?;
    validate_data_span(hdu.data_span())?;
    Ok(())
}

/// Validate an ordered FITS HDU sequence.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if:
/// - the sequence is empty
/// - the first HDU is not primary
/// - any later HDU is primary
/// - HDU indices are not contiguous
/// - any HDU has invalid structural extents
#[cfg(feature = "alloc")]
pub fn validate_hdu_sequence(sequence: &FitsHduSequence) -> Result<()> {
    if sequence.is_empty() {
        return invalid_format("FITS file must contain at least one HDU");
    }

    for (expected_index, hdu) in sequence.iter().enumerate() {
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

        validate_hdu(hdu)?;
    }

    Ok(())
}

/// Validate that a dataset path resolves within an HDU sequence.
///
/// ## Errors
///
/// Returns:
/// - `Error::NotFound` if the path is syntactically valid but does not resolve
/// - `Error::InvalidFormat` if the path is not a dataset path
#[cfg(feature = "alloc")]
pub fn validate_dataset_path_exists(
    path: &str,
    sequence: &FitsHduSequence,
) -> Result<FitsHduIndex> {
    let index = validate_dataset_path(path)?;
    if sequence.get(index).is_some() {
        Ok(index)
    } else {
        Err(not_found(path))
    }
}

/// Validate that a root path resolves within an HDU sequence.
///
/// The synthetic root always exists for any valid FITS file.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if the path is not `/`.
#[cfg(feature = "alloc")]
pub fn validate_root_path_exists(path: &str, _sequence: &FitsHduSequence) -> Result<()> {
    validate_root_path(path)
}

#[cfg(feature = "alloc")]
fn invalid_format<T>(message: &str) -> Result<T> {
    Err(Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.into(),
    })
}

#[cfg(feature = "alloc")]
fn not_found(path: &str) -> Error {
    Error::NotFound {
        #[cfg(feature = "alloc")]
        path: path.into(),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    use alloc::vec::Vec;

    use crate::datastructure::{FitsHeaderCardCount, FitsLogicalRecord};
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
            FitsDataSpan::new(FitsLogicalRecord::byte_len() as u64, 4).unwrap(),
        )
        .unwrap()
    }

    fn image_extension_hdu() -> FitsHdu {
        let bytes = header_bytes(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    1",
            "NAXIS1  =                    2",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "END",
        ]);
        let header = crate::file::parse_extension_header_bytes(&bytes).unwrap();
        FitsHdu::from_header(
            FitsHduIndex::new(1),
            header,
            FitsHeaderBlock::new(FitsHeaderCardCount::new(7)),
            FitsDataSpan::new((FitsLogicalRecord::byte_len() * 2) as u64, 2).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn validate_path_accepts_root_primary_and_hdu_paths() {
        assert_eq!(validate_path("/").unwrap(), FitsPathKind::Root);
        assert_eq!(
            validate_path("/PRIMARY").unwrap(),
            FitsPathKind::PrimaryDataset
        );
        assert_eq!(
            validate_path("/HDU/3").unwrap(),
            FitsPathKind::HduDataset(FitsHduIndex::new(3))
        );
    }

    #[test]
    fn validate_path_rejects_invalid_forms() {
        assert!(validate_path("").is_err());
        assert!(validate_path("/HDU/").is_err());
        assert!(validate_path("/HDU/x").is_err());
        assert!(validate_path("/HDU/1/extra").is_err());
        assert!(validate_path("/BAD").is_err());
    }

    #[test]
    fn validate_dataset_path_requires_dataset_paths() {
        assert_eq!(
            validate_dataset_path("/PRIMARY").unwrap(),
            FitsHduIndex::new(0)
        );
        assert_eq!(
            validate_dataset_path("/HDU/2").unwrap(),
            FitsHduIndex::new(2)
        );
        assert!(validate_dataset_path("/").is_err());
    }

    #[test]
    fn validate_root_path_requires_root() {
        assert!(validate_root_path("/").is_ok());
        assert!(validate_root_path("/PRIMARY").is_err());
    }

    #[test]
    fn validate_header_block_accepts_aligned_extent() {
        let block = FitsHeaderBlock::new(FitsHeaderCardCount::new(5));
        assert!(validate_header_block(block).is_ok());
    }

    #[test]
    fn validate_data_span_accepts_aligned_extent() {
        let span = FitsDataSpan::new(2880, 100).unwrap();
        assert!(validate_data_span(span).is_ok());
    }

    #[test]
    fn validate_hdu_accepts_primary_and_extension() {
        assert!(validate_hdu(&primary_hdu()).is_ok());
        assert!(validate_hdu(&image_extension_hdu()).is_ok());
    }

    #[test]
    fn validate_hdu_sequence_accepts_primary_then_extension() {
        let sequence = FitsHduSequence::new(vec![primary_hdu(), image_extension_hdu()]).unwrap();
        assert!(validate_hdu_sequence(&sequence).is_ok());
    }

    #[test]
    fn validate_dataset_path_exists_checks_resolution() {
        let sequence = FitsHduSequence::new(vec![primary_hdu(), image_extension_hdu()]).unwrap();

        assert_eq!(
            validate_dataset_path_exists("/PRIMARY", &sequence).unwrap(),
            FitsHduIndex::new(0)
        );
        assert_eq!(
            validate_dataset_path_exists("/HDU/1", &sequence).unwrap(),
            FitsHduIndex::new(1)
        );
        assert!(validate_dataset_path_exists("/HDU/2", &sequence).is_err());
    }

    #[test]
    fn validate_root_path_exists_accepts_root() {
        let sequence = FitsHduSequence::new(vec![primary_hdu()]).unwrap();
        assert!(validate_root_path_exists("/", &sequence).is_ok());
        assert!(validate_root_path_exists("/PRIMARY", &sequence).is_err());
    }
}
