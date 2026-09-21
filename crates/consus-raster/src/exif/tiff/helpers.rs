use crate::{DecodeError, DecodeErrorKind};

use super::{DirectoryKind, PixelDensity, PresentationMetadata};

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

pub(super) fn push_directory(
    pending: &mut [usize; 16],
    pending_kinds: &mut [DirectoryKind; 16],
    pending_len: &mut usize,
    offset: usize,
    kind: DirectoryKind,
) -> Result<(), DecodeError> {
    let slot = pending
        .get_mut(*pending_len)
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
    *slot = offset;
    let kind_slot = pending_kinds
        .get_mut(*pending_len)
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
    *kind_slot = kind;
    *pending_len = pending_len
        .checked_add(1)
        .ok_or_else(|| DecodeError::new(DecodeErrorKind::TooLarge))?;
    Ok(())
}

pub(super) fn field_width(field_type: u16) -> Result<usize, DecodeError> {
    match field_type {
        1 | 2 | 6 | 7 => Ok(1),
        3 | 8 => Ok(2),
        4 | 9 | 11 => Ok(4),
        5 | 10 | 12 => Ok(8),
        _ => Err(DecodeError::new(DecodeErrorKind::Malformed)),
    }
}
