use crate::exif;

/// Pixel representation returned by a decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelFormat {
    /// One eight-bit luminance sample per pixel.
    Gray,
    /// One native-endian `u16` luminance sample per pixel.
    GrayWide,
    /// Three eight-bit red, green, and blue samples per pixel.
    Rgb,
}

/// A validated encoded-grid raster image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    format: PixelFormat,
    orientation: exif::Orientation,
}

impl DecodedImage {
    pub(crate) fn new(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        format: PixelFormat,
        orientation: exif::Orientation,
    ) -> Self {
        Self {
            width,
            height,
            pixels,
            format,
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
