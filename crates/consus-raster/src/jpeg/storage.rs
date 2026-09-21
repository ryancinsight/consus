use crate::{DecodeError, DecodeErrorKind};

use super::transform;

pub(super) const WORKING_METADATA_BOUND: usize = 4096;
// Four frame components each admit sampling factors up to 4×4, including
// noninterleaved scans. Each padded block retains 64 i32 coefficients and
// 64 u16 plane samples; frame MCU count cannot exceed the 8×8 image tile count.
// Three output channels occupy at most two bytes each.
const MAX_DCT_BYTES_PER_TILE: usize = 4 * 4 * 4 * 64 * 6;
const MAX_OUTPUT_BYTES_PER_PIXEL: usize = 6;

/// Returns a working-storage bound for every supported JPEG at `width` by `height`.
///
/// The result covers coefficients or lossless samples, component planes,
/// allocation descriptors, and the final output while those allocations
/// coexist. It is suitable for [`crate::DecodeLimits::max_working_bytes`] when
/// the caller knows the encoded-grid dimensions before decoding.
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
        .and_then(|bytes| bytes.checked_add(pixels.checked_mul(MAX_OUTPUT_BYTES_PER_PIXEL)?))
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))
}
