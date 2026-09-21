//! Bounded JPEG encoding and decoding.
//!
//! Decoding supports 8-bit sequential, extended sequential, and progressive
//! Huffman DCT streams with grayscale, RGB/YCbCr, CMYK, or YCCK components. It
//! also supports 8 through 16-bit single-component lossless Huffman streams.
//! Entropy data must end exactly at its following marker; truncated scans are
//! rejected instead of being padded with synthetic coefficients.

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
