use crate::{DecodeError, DecodeErrorKind};

pub(super) const fn malformed() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Malformed)
}

pub(super) const fn unsupported() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Unsupported)
}

pub(super) const fn too_large() -> DecodeError {
    DecodeError::new(DecodeErrorKind::TooLarge)
}

pub(super) const fn allocation() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Allocation)
}
