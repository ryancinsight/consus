//! Bounded JPEG encoding and decoding.
//!
//! Decoding supports 8-bit sequential, extended sequential, and progressive
//! Huffman DCT streams with grayscale, RGB/YCbCr, CMYK, or YCCK components. It
//! also supports 8 through 16-bit single-component lossless Huffman streams.
//! Entropy data must end exactly at its following marker; truncated scans are
//! rejected instead of being padded with synthetic coefficients.

use crate::{DecodeError, DecodeErrorKind, DecodeLimits, DecodedImage};
use jpeg_encoder::{ColorType, Encoder};

const WORKING_METADATA_BOUND: usize = 4096;
const MAX_DCT_BYTES_PER_TILE: usize = 3200;

mod bitstream;
mod decoder;
mod transform;

#[cfg(test)]
mod fixture_data;

/// Decodes one JPEG image without applying EXIF orientation.
///
/// The returned orientation describes how a presentation layer may transform
/// the encoded grid. A zero value in any [`DecodeLimits`] field rejects every
/// input.
///
/// # Errors
///
/// Returns a classified [`DecodeError`] for malformed or unsupported input,
/// exceeded limits, or a refused bounded allocation.
pub fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<DecodedImage, DecodeError> {
    decoder::decode(bytes, limits)
}

/// Returns a working-storage bound for every supported JPEG at `width` by `height`.
///
/// The result covers coefficients or lossless samples, component planes,
/// allocation descriptors, and the final output while those allocations
/// coexist. It is suitable for [`DecodeLimits::max_working_bytes`] when the
/// caller knows the encoded-grid dimensions before decoding.
///
/// # Errors
///
/// Returns [`DecodeErrorKind::Malformed`] for a zero dimension and
/// [`DecodeErrorKind::TooLarge`] if the bound does not fit in [`usize`].
pub fn working_storage_bound(width: u32, height: u32) -> Result<usize, DecodeError> {
    if width == 0 || height == 0 {
        return Err(DecodeError::new(DecodeErrorKind::Malformed));
    }
    let width = usize::try_from(width).map_err(|_| DecodeError::new(DecodeErrorKind::TooLarge))?;
    let height =
        usize::try_from(height).map_err(|_| DecodeError::new(DecodeErrorKind::TooLarge))?;
    let tiles = width
        .div_ceil(transform::BLOCK_SIDE)
        .checked_mul(height.div_ceil(transform::BLOCK_SIDE))
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
    WORKING_METADATA_BOUND
        .checked_add(
            tiles
                .checked_mul(MAX_DCT_BYTES_PER_TILE)
                .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?,
        )
        .and_then(|bytes| bytes.checked_add(pixels.checked_mul(3)?))
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))
}

/// Encodes one tightly packed eight-bit grayscale image as JPEG.
///
/// `quality` must be in `1..=100`, dimensions must be nonzero and fit the JPEG
/// sixteen-bit dimension fields, and `pixels.len()` must equal `width * height`.
///
/// # Errors
///
/// Returns [`DecodeErrorKind::Malformed`] for invalid arguments and
/// [`DecodeErrorKind::Unsupported`] if the pure-Rust encoder rejects a valid
/// request.
pub fn encode_gray(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, DecodeError> {
    if quality == 0 || quality > 100 || width == 0 || height == 0 {
        return Err(DecodeError::new(DecodeErrorKind::Malformed));
    }
    let width = u16::try_from(width).map_err(|_| DecodeError::new(DecodeErrorKind::Malformed))?;
    let height = u16::try_from(height).map_err(|_| DecodeError::new(DecodeErrorKind::Malformed))?;
    let expected = usize::from(width)
        .checked_mul(usize::from(height))
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
    if pixels.len() != expected {
        return Err(DecodeError::new(DecodeErrorKind::Malformed));
    }
    let mut encoded = Vec::new();
    Encoder::new(&mut encoded, quality)
        .encode(pixels, width, height, ColorType::Luma)
        .map_err(|_| DecodeError::new(DecodeErrorKind::Unsupported))?;
    Ok(encoded)
}

#[cfg(test)]
mod tests;
