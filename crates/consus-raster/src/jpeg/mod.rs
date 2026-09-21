//! Bounded JPEG encoding and decoding.
//!
//! Decoding supports 8-bit sequential, extended sequential, and progressive
//! Huffman DCT streams with grayscale, RGB/YCbCr, CMYK, or YCCK components. It
//! also supports 8 through 16-bit single-component lossless Huffman streams.
//! Entropy data must end exactly at its following marker; truncated scans are
//! rejected instead of being padded with synthetic coefficients.

use crate::{DecodeError, DecodeErrorKind, DecodeLimits, DecodedImage};
use jpeg_encoder::{ColorType, Encoder};

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
