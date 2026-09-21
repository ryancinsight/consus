//! Bounded JPEG encoding and decoding.
//!
//! Decoding supports 8/12-bit sequential and progressive DCT streams with
//! grayscale, RGB/YCbCr, CMYK, or YCCK components, and 2 through 16-bit
//! single-component lossless streams. Huffman and arithmetic entropy coding
//! share sample reconstruction. Every scan requires a physical terminating
//! marker; arithmetic termination follows the implicit-zero rule of T.81 D.2.6.

mod arithmetic;
mod bitstream;
mod decoder;
mod encoder;
mod storage;
mod transform;

pub use decoder::decode;
pub use encoder::encode_gray;
pub use storage::working_storage_bound;

use storage::WORKING_METADATA_BOUND;

#[cfg(test)]
mod fixture_data;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod precision_tests;
