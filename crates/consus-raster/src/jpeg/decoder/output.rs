use super::super::transform::{BLOCK_CELLS, BLOCK_SIDE, reconstruct, ycbcr_to_rgb};
use super::{
    Coding, ColorModel, Frame, Scan, UNSEEN, allocation, malformed, too_large, unsupported,
};
use crate::exif::Orientation;
use crate::{DecodeError, DecodedImage, PixelFormat};

pub(super) fn record_progression(frame: &mut Frame, scan: &Scan) -> Result<(), DecodeError> {
    if frame.coding == Coding::Lossless {
        return Ok(());
    }
    for scan_component in &scan.components {
        let component = frame
            .components
            .get_mut(scan_component.frame_index)
            .ok_or_else(malformed)?;
        for coefficient in scan.start..=scan.end {
            component.approximation[usize::from(coefficient)] = scan.low;
        }
    }
    Ok(())
}

pub(super) fn finish(
    frame: Frame,
    color_model: ColorModel,
    orientation: Orientation,
) -> Result<DecodedImage, DecodeError> {
    if frame.coding == Coding::Lossless {
        return finish_lossless(frame, orientation);
    }
    ensure_complete(&frame)?;
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(frame.components.len())
        .map_err(|_| allocation())?;
    for component in &frame.components {
        let table = component.quantization_values.ok_or_else(malformed)?;
        let plane_len = component
            .stored_across
            .checked_mul(component.stored_down)
            .and_then(|blocks| blocks.checked_mul(BLOCK_CELLS))
            .ok_or_else(too_large)?;
        let mut plane = Vec::new();
        plane
            .try_reserve_exact(plane_len)
            .map_err(|_| allocation())?;
        plane.resize(plane_len, 0);
        let block_count = component
            .stored_across
            .checked_mul(component.stored_down)
            .ok_or_else(too_large)?;
        for block in 0..block_count {
            let coefficient_start = component
                .coefficient_offset
                .checked_add(block.checked_mul(BLOCK_CELLS).ok_or_else(too_large)?)
                .ok_or_else(too_large)?;
            let coefficient_end = coefficient_start
                .checked_add(BLOCK_CELLS)
                .ok_or_else(too_large)?;
            let coefficients = frame
                .coefficients
                .get(coefficient_start..coefficient_end)
                .ok_or_else(malformed)?;
            let samples = reconstruct(coefficients, &table);
            let block_x = block % component.stored_across;
            let block_y = block / component.stored_across;
            let stride = component
                .stored_across
                .checked_mul(BLOCK_SIDE)
                .ok_or_else(too_large)?;
            for row in 0..BLOCK_SIDE {
                let destination = block_y
                    .checked_mul(BLOCK_SIDE)
                    .and_then(|value| value.checked_add(row))
                    .and_then(|value| value.checked_mul(stride))
                    .and_then(|value| value.checked_add(block_x * BLOCK_SIDE))
                    .ok_or_else(too_large)?;
                let end = destination.checked_add(BLOCK_SIDE).ok_or_else(too_large)?;
                plane
                    .get_mut(destination..end)
                    .ok_or_else(malformed)?
                    .copy_from_slice(&samples[row * BLOCK_SIDE..(row + 1) * BLOCK_SIDE]);
            }
        }
        planes.push(plane);
    }
    let pixel_count = frame
        .width
        .checked_mul(frame.height)
        .ok_or_else(too_large)?;
    let channels = if frame.components.len() == 1 { 1 } else { 3 };
    let output_len = pixel_count.checked_mul(channels).ok_or_else(too_large)?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(output_len)
        .map_err(|_| allocation())?;
    for y in 0..frame.height {
        for x in 0..frame.width {
            let sample = |component_index: usize| -> Result<u8, DecodeError> {
                let component = frame
                    .components
                    .get(component_index)
                    .ok_or_else(malformed)?;
                let sample_x = x
                    .checked_mul(usize::from(component.horizontal))
                    .ok_or_else(too_large)?
                    / usize::from(frame.max_horizontal);
                let sample_y = y
                    .checked_mul(usize::from(component.vertical))
                    .ok_or_else(too_large)?
                    / usize::from(frame.max_vertical);
                let stride = component
                    .stored_across
                    .checked_mul(BLOCK_SIDE)
                    .ok_or_else(too_large)?;
                let index = sample_y
                    .checked_mul(stride)
                    .and_then(|value| value.checked_add(sample_x))
                    .ok_or_else(too_large)?;
                planes
                    .get(component_index)
                    .and_then(|plane| plane.get(index))
                    .copied()
                    .ok_or_else(malformed)
            };
            match (color_model, frame.components.len()) {
                (ColorModel::Gray, 1) => pixels.push(sample(0)?),
                (ColorModel::Rgb, 3) => {
                    pixels.extend_from_slice(&[sample(0)?, sample(1)?, sample(2)?]);
                }
                (ColorModel::Ycbcr, 3) => {
                    pixels.extend_from_slice(&ycbcr_to_rgb(sample(0)?, sample(1)?, sample(2)?));
                }
                (ColorModel::Cmyk | ColorModel::Ycck, 4) => {
                    let key = sample(3)?;
                    let colors = if color_model == ColorModel::Ycck {
                        ycbcr_to_rgb(sample(0)?, sample(1)?, sample(2)?)
                            .map(|channel| 255 - channel)
                    } else {
                        [sample(0)?, sample(1)?, sample(2)?]
                    };
                    pixels.extend(colors.map(|channel| {
                        let color = u16::from(channel) * u16::from(key) / 255;
                        u8::try_from(color).expect("invariant: CMYK product divided by 255 fits u8")
                    }));
                }
                _ => return Err(unsupported()),
            }
        }
    }
    if pixels.len() != output_len {
        return Err(malformed());
    }
    Ok(DecodedImage::new(
        u32::try_from(frame.width).map_err(|_| too_large())?,
        u32::try_from(frame.height).map_err(|_| too_large())?,
        pixels,
        if channels == 1 {
            PixelFormat::Gray
        } else {
            PixelFormat::Rgb
        },
        orientation,
    ))
}

fn ensure_complete(frame: &Frame) -> Result<(), DecodeError> {
    if frame
        .components
        .iter()
        .any(|component| component.approximation[0] == UNSEEN)
    {
        return Err(malformed());
    }
    Ok(())
}

fn finish_lossless(frame: Frame, orientation: Orientation) -> Result<DecodedImage, DecodeError> {
    let pixel_count = frame
        .width
        .checked_mul(frame.height)
        .ok_or_else(too_large)?;
    if frame.coefficients.len() != pixel_count {
        return Err(malformed());
    }
    let mut pixels = Vec::new();
    let bytes_per_sample = if frame.precision <= 8 { 1 } else { 2 };
    pixels
        .try_reserve_exact(
            pixel_count
                .checked_mul(bytes_per_sample)
                .ok_or_else(too_large)?,
        )
        .map_err(|_| allocation())?;
    if frame.precision <= 8 {
        for sample in frame.coefficients {
            pixels.push(u8::try_from(sample).map_err(|_| malformed())?);
        }
    } else {
        for sample in frame.coefficients {
            pixels.extend_from_slice(
                &u16::try_from(sample)
                    .map_err(|_| malformed())?
                    .to_ne_bytes(),
            );
        }
    }
    Ok(DecodedImage::new(
        u32::try_from(frame.width).map_err(|_| too_large())?,
        u32::try_from(frame.height).map_err(|_| too_large())?,
        pixels,
        if frame.precision <= 8 {
            PixelFormat::Gray
        } else {
            PixelFormat::GrayWide
        },
        orientation,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecodeErrorKind, DecodeLimits};

    #[test]
    fn incomplete_component_is_rejected() {
        let frame = super::super::frame::parse_frame(
            &[8, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0],
            0xc0,
            DecodeLimits {
                max_encoded_bytes: 1024,
                max_dimension: 8,
                max_pixels: 64,
                max_working_bytes: 8192,
            },
        )
        .expect("valid frame");
        assert_eq!(
            ensure_complete(&frame)
                .expect_err("unscanned components")
                .kind(),
            DecodeErrorKind::Malformed
        );
    }

    #[test]
    fn rgb_component_identifiers_select_direct_samples_without_app14() {
        let mut encoded = vec![
            0xff, 0xd8, 0xff, 0xee, 0, 14, b'A', b'd', b'o', b'b', b'e', 0, 0, 0, 0, 0, 0, 0, 0xff,
            0xdb, 0, 67, 0,
        ];
        encoded.extend_from_slice(&[1; 64]);
        encoded.extend_from_slice(&[
            0xff, 0xc0, 0, 17, 8, 0, 8, 0, 8, 3, b'R', 0x11, 0, b'G', 0x11, 0, b'B', 0x11, 0, 0xff,
            0xc4, 0, 21, 0, 1, 1,
        ]);
        encoded.extend_from_slice(&[0; 14]);
        encoded.extend_from_slice(&[4, 0, 0xff, 0xc4, 0, 20, 0x10, 1]);
        encoded.extend_from_slice(&[0; 15]);
        encoded.extend_from_slice(&[
            0, 0xff, 0xda, 0, 12, 3, b'R', 0, b'G', 0, b'B', 0, 0, 63, 0, 0x42, 0x1d, 0xff, 0xd9,
        ]);
        encoded.drain(2..18);
        let decoded = super::super::decode(
            &encoded,
            DecodeLimits {
                max_encoded_bytes: 1024,
                max_dimension: 8,
                max_pixels: 64,
                max_working_bytes: 8192,
            },
        )
        .expect("direct RGB fixture after removing APP14");
        assert_eq!(decoded.format(), PixelFormat::Rgb);
        assert_eq!((decoded.width(), decoded.height()), (8, 8));
        assert_eq!(decoded.pixels(), &[129, 128, 127].repeat(64));
    }
}
