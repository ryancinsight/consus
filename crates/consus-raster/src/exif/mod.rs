//! Bounded EXIF orientation parsing and coordinate mapping.
//!
//! [`parse`](crate::exif::parse) accepts the TIFF payload following an EXIF `Exif\0\0`
//! signature. It traverses a bounded image-file-directory graph, validates
//! presentation metadata that affects display interpretation, and returns the
//! primary image's orientation. Embedded thumbnail metadata is not presented:
//! its declared extent and JPEG boundary markers are validated, but the nested
//! image is not decoded.

mod orientation;
mod parser;
mod tiff;

pub use orientation::Orientation;
pub use parser::parse;

#[cfg(test)]
mod tests;
