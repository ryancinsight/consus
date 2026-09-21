use crate::{DecodeError, DecodeErrorKind};
use jpeg_encoder::{ColorType, Encoder};

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
