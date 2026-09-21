//! Bounded EXIF orientation parsing and coordinate mapping.
//!
//! [`parse`](crate::exif::parse) accepts the TIFF payload following an EXIF `Exif\0\0`
//! signature. It traverses a bounded image-file-directory graph, validates
//! presentation metadata that affects display interpretation, and returns the
//! primary image's orientation. Embedded thumbnail metadata is not presented:
//! its declared extent and JPEG boundary markers are validated, but the nested
//! image is not decoded.

use crate::{DecodeError, DecodeErrorKind};

mod tiff;

#[cfg(test)]
mod tests;

/// The eight display orientations encoded by EXIF tag `0x0112`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Orientation {
    /// Preserve the encoded row and column directions.
    #[default]
    Normal,
    /// Reflect the image across its vertical axis.
    MirrorHorizontal,
    /// Rotate the image by 180 degrees.
    RotateHalf,
    /// Reflect the image across its horizontal axis.
    MirrorVertical,
    /// Reflect the image across its top-left to bottom-right diagonal.
    Transpose,
    /// Rotate the image 90 degrees clockwise.
    RotateClockwise,
    /// Reflect the image across its top-right to bottom-left diagonal.
    Transverse,
    /// Rotate the image 90 degrees counter-clockwise.
    RotateCounterClockwise,
}

impl TryFrom<u16> for Orientation {
    type Error = DecodeError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::MirrorHorizontal),
            3 => Ok(Self::RotateHalf),
            4 => Ok(Self::MirrorVertical),
            5 => Ok(Self::Transpose),
            6 => Ok(Self::RotateClockwise),
            7 => Ok(Self::Transverse),
            8 => Ok(Self::RotateCounterClockwise),
            _ => Err(DecodeError::new(DecodeErrorKind::Malformed)),
        }
    }
}

#[deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
impl Orientation {
    /// Returns whether display normalization exchanges width and height.
    #[must_use]
    pub const fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Transpose
                | Self::RotateClockwise
                | Self::Transverse
                | Self::RotateCounterClockwise
        )
    }

    /// Maps one normalized output coordinate to its encoded source coordinate.
    ///
    /// `width` and `height` describe the encoded source grid. For orientations
    /// that swap axes, the valid output range is `x < height, y < width`;
    /// otherwise it is `x < width, y < height`.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeErrorKind::Malformed`] when either source dimension is
    /// zero or the requested output coordinate lies outside the normalized
    /// output grid.
    pub fn source_coordinate(
        self,
        width: u32,
        height: u32,
        x: u32,
        y: u32,
    ) -> Result<(u32, u32), DecodeError> {
        if width == 0 || height == 0 {
            return Err(DecodeError::new(DecodeErrorKind::Malformed));
        }
        let (output_width, output_height) = if self.swaps_axes() {
            (height, width)
        } else {
            (width, height)
        };
        if x >= output_width || y >= output_height {
            return Err(DecodeError::new(DecodeErrorKind::Malformed));
        }

        Ok(match self {
            Self::Normal => (x, y),
            Self::MirrorHorizontal => (reverse_coordinate(width, x)?, y),
            Self::RotateHalf => (
                reverse_coordinate(width, x)?,
                reverse_coordinate(height, y)?,
            ),
            Self::MirrorVertical => (x, reverse_coordinate(height, y)?),
            Self::Transpose => (y, x),
            Self::RotateClockwise => (y, reverse_coordinate(height, x)?),
            Self::Transverse => (
                reverse_coordinate(width, y)?,
                reverse_coordinate(height, x)?,
            ),
            Self::RotateCounterClockwise => (reverse_coordinate(width, y)?, x),
        })
    }
}

#[deny(clippy::arithmetic_side_effects)]
fn reverse_coordinate(extent: u32, coordinate: u32) -> Result<u32, DecodeError> {
    extent
        .checked_sub(1)
        .and_then(|last| last.checked_sub(coordinate))
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::Malformed))
}

/// Parses a bounded TIFF payload and returns the primary EXIF orientation.
///
/// Missing orientation metadata resolves to [`Orientation::Normal`]. Unknown
/// metadata is ignored unless it changes presentation semantics; unsupported
/// color-space, density, interoperability, profile, or thumbnail encodings are
/// rejected rather than silently reinterpreted.
///
/// # Errors
///
/// Returns [`DecodeErrorKind::Malformed`] for invalid TIFF structure or field
/// values, [`DecodeErrorKind::Unsupported`] for recognized display metadata
/// outside this contract, and [`DecodeErrorKind::TooLarge`] when the bounded
/// directory graph or an extent calculation exceeds its limit.
pub fn parse(tiff: &[u8]) -> Result<Orientation, DecodeError> {
    tiff::parse(tiff)
}
