use crate::exif;

mod display;

/// The encoded compression process family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Compression {
    /// Quantized transform coding.
    Lossy,
    /// Predictive coding, including any encoded point transform.
    Lossless,
}

/// Pixel representation returned by a decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelFormat {
    /// One luminance sample per pixel in one byte.
    Gray,
    /// One native-endian `u16` luminance sample per pixel.
    GrayWide,
    /// Three red, green, and blue samples per pixel in three bytes.
    Rgb,
    /// Three native-endian `u16` red, green, and blue samples per pixel.
    RgbWide,
}

/// A validated encoded-grid raster image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    format: PixelFormat,
    sample_precision: u8,
    compression: Compression,
    orientation: exif::Orientation,
}

impl DecodedImage {
    pub(crate) fn new(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        format: PixelFormat,
        sample_precision: u8,
        compression: Compression,
        orientation: exif::Orientation,
    ) -> Self {
        Self {
            width,
            height,
            pixels,
            format,
            sample_precision,
            compression,
            orientation,
        }
    }

    /// Returns the encoded-grid width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the encoded-grid height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the packed pixel bytes.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Returns the pixel representation.
    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// Returns the number of meaningful bits in each decoded sample.
    #[must_use]
    pub const fn sample_precision(&self) -> u8 {
        self.sample_precision
    }

    /// Returns the encoded compression process family.
    #[must_use]
    pub const fn compression(&self) -> Compression {
        self.compression
    }

    /// Returns the EXIF presentation orientation for the encoded pixel grid.
    #[must_use]
    pub const fn orientation(&self) -> exif::Orientation {
        self.orientation
    }

    /// Consumes the image and returns its packed pixel bytes.
    #[must_use]
    pub fn into_pixels(self) -> Vec<u8> {
        self.pixels
    }
}
