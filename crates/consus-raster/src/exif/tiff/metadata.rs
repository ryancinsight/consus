use crate::{DecodeError, DecodeErrorKind};

use super::traversal::{DirectoryKind, PixelDensity, PresentationMetadata};

pub(super) fn density_for(
    metadata: &mut PresentationMetadata,
    kind: DirectoryKind,
) -> Result<&mut PixelDensity, DecodeError> {
    match kind {
        DirectoryKind::Root => Ok(&mut metadata.primary_density),
        DirectoryKind::Thumbnail => Ok(&mut metadata.thumbnail_density),
        DirectoryKind::Exif | DirectoryKind::Gps | DirectoryKind::Interoperability => {
            Err(DecodeError::new(DecodeErrorKind::Unsupported))
        }
    }
}

pub(super) fn validate(tiff: &[u8], metadata: &PresentationMetadata) -> Result<(), DecodeError> {
    validate_density(&metadata.primary_density)?;
    validate_density(&metadata.thumbnail_density)?;
    match (metadata.thumbnail_offset, metadata.thumbnail_length) {
        (Some(offset), Some(length)) => {
            let end = offset
                .checked_add(length)
                .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
            let thumbnail = tiff
                .get(offset..end)
                .ok_or_else(|| DecodeError::new(DecodeErrorKind::Malformed))?;
            if !thumbnail.starts_with(&[0xff, 0xd8]) || !thumbnail.ends_with(&[0xff, 0xd9]) {
                return Err(DecodeError::new(DecodeErrorKind::Malformed));
            }
        }
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => {
            return Err(DecodeError::new(DecodeErrorKind::Malformed));
        }
    }
    Ok(())
}

fn validate_density(density: &PixelDensity) -> Result<(), DecodeError> {
    match (density.horizontal_resolution, density.vertical_resolution) {
        (Some((horizontal, horizontal_denominator)), Some((vertical, vertical_denominator))) => {
            let horizontal_scaled = u64::from(horizontal)
                .checked_mul(u64::from(vertical_denominator))
                .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
            let vertical_scaled = u64::from(vertical)
                .checked_mul(u64::from(horizontal_denominator))
                .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
            if horizontal_scaled != vertical_scaled {
                return Err(DecodeError::new(DecodeErrorKind::Unsupported));
            }
        }
        (None, None) if density.resolution_unit.is_none() => {}
        (None | Some(_), None) | (None, Some(_)) => {
            return Err(DecodeError::new(DecodeErrorKind::Malformed));
        }
    }
    Ok(())
}
