use crate::{DecodeError, DecodeErrorKind};

pub(super) fn encoded_width(field_type: u16) -> Result<usize, DecodeError> {
    match field_type {
        1 | 2 | 6 | 7 => Ok(1),
        3 | 8 => Ok(2),
        4 | 9 | 11 => Ok(4),
        5 | 10 | 12 => Ok(8),
        _ => Err(DecodeError::new(DecodeErrorKind::Malformed)),
    }
}
