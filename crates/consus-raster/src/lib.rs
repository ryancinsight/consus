#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod error;
mod image;
mod limits;

pub use error::{DecodeError, DecodeErrorKind};
pub use image::{DecodedImage, PixelFormat};
pub use limits::DecodeLimits;

/// EXIF metadata parsing.
pub mod exif;
/// JPEG encoding and decoding.
pub mod jpeg;
