use super::{DecodedImage, PixelFormat};
use std::iter::FusedIterator;
use std::slice::{ChunksExact, Iter};

impl DecodedImage {
    /// Returns encoded samples mapped to packed eight-bit display values.
    ///
    /// For a sample `s` with precision maximum `M = 2^precision - 1`, each
    /// result is `floor((s * 255 + floor(M / 2)) / M)`. The iterator preserves
    /// encoded channel order and does not apply orientation, alpha, luminance,
    /// gamma, or clinical presentation policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use consus_raster::{DecodeLimits, jpeg};
    ///
    /// let encoded = jpeg::encode_gray(&[0; 64], 8, 8, 100)?;
    /// let image = jpeg::decode(
    ///     &encoded,
    ///     DecodeLimits {
    ///         max_encoded_bytes: 1 << 20,
    ///         max_dimension: 4096,
    ///         max_pixels: 16 << 20,
    ///         max_working_bytes: 64 << 20,
    ///     },
    /// )?;
    /// assert!(image.display_samples().eq([0; 64]));
    /// # Ok::<(), consus_raster::DecodeError>(())
    /// ```
    #[must_use]
    pub fn display_samples(&self) -> impl ExactSizeIterator<Item = u8> + '_ {
        DisplaySamples {
            samples: match self.format {
                PixelFormat::Gray | PixelFormat::Rgb => PackedSamples::Bytes(self.pixels.iter()),
                PixelFormat::GrayWide | PixelFormat::RgbWide => {
                    PackedSamples::Wide(self.pixels.chunks_exact(2))
                }
            },
            maximum: (1_u32 << self.sample_precision) - 1,
        }
    }
}

enum PackedSamples<'image> {
    Bytes(Iter<'image, u8>),
    Wide(ChunksExact<'image, u8>),
}

struct DisplaySamples<'image> {
    samples: PackedSamples<'image>,
    maximum: u32,
}

impl DisplaySamples<'_> {
    fn remaining(&self) -> usize {
        match &self.samples {
            PackedSamples::Bytes(samples) => samples.len(),
            PackedSamples::Wide(samples) => samples.len(),
        }
    }
}

impl Iterator for DisplaySamples<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = match &mut self.samples {
            PackedSamples::Bytes(samples) => u32::from(*samples.next()?),
            PackedSamples::Wide(samples) => {
                let bytes: &[u8; 2] = samples
                    .next()?
                    .try_into()
                    .expect("invariant: chunks_exact(2) yields two-byte chunks");
                u32::from(u16::from_ne_bytes(*bytes))
            }
        };
        let scaled = (sample * 255 + self.maximum / 2) / self.maximum;
        Some(u8::try_from(scaled).expect("invariant: a normalized display sample is at most 255"))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DisplaySamples<'_> {
    fn len(&self) -> usize {
        self.remaining()
    }
}

impl FusedIterator for DisplaySamples<'_> {}

#[cfg(test)]
mod tests {
    use super::super::Compression;
    use super::*;
    use crate::exif::Orientation;

    fn image(precision: u8, format: PixelFormat, pixels: Vec<u8>) -> DecodedImage {
        let bytes_per_pixel = match format {
            PixelFormat::Gray => 1,
            PixelFormat::GrayWide => 2,
            PixelFormat::Rgb => 3,
            PixelFormat::RgbWide => 6,
        };
        DecodedImage::new(
            u32::try_from(pixels.len() / bytes_per_pixel).expect("invariant: test width fits u32"),
            1,
            pixels,
            format,
            precision,
            Compression::Lossless,
            Orientation::RotateClockwise,
        )
    }

    fn exhaustive_image(precision: u8) -> DecodedImage {
        let maximum = (1_u32 << precision) - 1;
        if precision <= 8 {
            image(
                precision,
                PixelFormat::Gray,
                (0..=maximum)
                    .map(|sample| {
                        u8::try_from(sample)
                            .expect("invariant: eight-bit sample precision fits one byte")
                    })
                    .collect(),
            )
        } else {
            image(
                precision,
                PixelFormat::GrayWide,
                (0..=maximum)
                    .flat_map(|sample| {
                        u16::try_from(sample)
                            .expect("invariant: sample precision fits two bytes")
                            .to_ne_bytes()
                    })
                    .collect(),
            )
        }
    }

    #[test]
    fn exhaustive_precisions_satisfy_nearest_integer_mapping() {
        for precision in 2..=16 {
            let image = exhaustive_image(precision);
            let maximum = (1_u32 << precision) - 1;
            let mut display = image.display_samples();
            assert_eq!(
                display.len(),
                usize::try_from(maximum + 1)
                    .expect("invariant: sixteen-bit sample count fits usize")
            );

            let mut previous = 0;
            for sample in 0..=maximum {
                let output = u32::from(display.next().expect("one result per encoded sample"));
                let error = (i64::from(maximum) * i64::from(output) - 255_i64 * i64::from(sample))
                    .unsigned_abs();
                assert!(
                    error <= u64::from(maximum / 2),
                    "precision {precision}, sample {sample}"
                );
                assert!(output >= previous, "precision {precision}, sample {sample}");
                if sample == 0 {
                    assert_eq!(output, 0, "precision {precision} lower endpoint");
                }
                previous = output;
            }
            assert_eq!(display.next(), None);
            assert_eq!(previous, 255);
        }
    }

    #[test]
    fn eight_bit_samples_are_identity_mapped() {
        let image = exhaustive_image(8);
        assert!(image.display_samples().eq(0..=255));
    }

    #[test]
    fn packed_formats_preserve_channel_order_and_metadata() {
        let gray = image(8, PixelFormat::Gray, vec![0, 64, 255]);
        let rgb = image(8, PixelFormat::Rgb, vec![255, 0, 64, 1, 2, 3]);
        let wide_values = [0_u16, 2048, 4095];
        let wide: Vec<u8> = wide_values.into_iter().flat_map(u16::to_ne_bytes).collect();
        let gray_wide = image(12, PixelFormat::GrayWide, wide.clone());
        let rgb_wide = image(
            12,
            PixelFormat::RgbWide,
            [4095_u16, 0, 2048, 1, 4094, 2047]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect(),
        );

        assert_eq!(gray.display_samples().collect::<Vec<_>>(), [0, 64, 255]);
        assert_eq!(
            rgb.display_samples().collect::<Vec<_>>(),
            [255, 0, 64, 1, 2, 3]
        );
        assert_eq!(
            gray_wide.display_samples().collect::<Vec<_>>(),
            [0, 128, 255]
        );
        assert_eq!(
            rgb_wide.display_samples().collect::<Vec<_>>(),
            [255, 0, 128, 0, 255, 127]
        );
        assert_eq!(gray.display_samples().len(), 3);
        assert_eq!(rgb.display_samples().len(), 6);
        assert_eq!(gray_wide.display_samples().len(), 3);
        assert_eq!(rgb_wide.display_samples().len(), 6);
        assert_eq!(gray_wide.pixels(), wide.as_slice());
        assert_eq!((gray_wide.width(), gray_wide.height()), (3, 1));
        assert_eq!(gray_wide.format(), PixelFormat::GrayWide);
        assert_eq!(gray_wide.sample_precision(), 12);
        assert_eq!(gray_wide.compression(), Compression::Lossless);
        assert_eq!(gray_wide.orientation(), Orientation::RotateClockwise);
    }
}
