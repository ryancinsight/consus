#![deny(missing_docs)]
#![deny(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
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
