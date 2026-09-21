use crate::DecodeError;

use super::{Orientation, tiff};

/// Parses a bounded TIFF payload and returns the primary EXIF orientation.
///
/// Missing orientation metadata resolves to [`Orientation::Normal`]. Unknown
/// metadata is ignored unless it changes presentation semantics; unsupported
/// color-space, density, interoperability, profile, or thumbnail encodings are
/// rejected rather than silently reinterpreted.
///
/// # Errors
///
/// Returns [`crate::DecodeErrorKind::Malformed`] for invalid TIFF structure or
/// field values, [`crate::DecodeErrorKind::Unsupported`] for recognized display
/// metadata outside this contract, and [`crate::DecodeErrorKind::TooLarge`]
/// when the bounded directory graph or an extent calculation exceeds its limit.
pub fn parse(tiff: &[u8]) -> Result<Orientation, DecodeError> {
    tiff::parse(tiff)
}
