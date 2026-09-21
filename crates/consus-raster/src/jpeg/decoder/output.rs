use super::super::transform::{BLOCK_CELLS, BLOCK_SIDE, reconstruct, ycbcr_to_rgb};
use super::{Coding, Frame, Scan, UNSEEN, allocation, malformed, too_large, unsupported};
use crate::exif::Orientation;
use crate::{DecodeError, DecodeLimits, DecodedImage, PixelFormat};

pub(super) fn record_progression(frame: &mut Frame, scan: &Scan) -> Result<(), DecodeError> {
    if frame.coding != Coding::Progressive {
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
    quantization: &[Option<[u16; 64]>; 4],
    adobe_transform: Option<u8>,
    orientation: Orientation,
    limits: DecodeLimits,
) -> Result<DecodedImage, DecodeError> {
    if frame.coding == Coding::Lossless {
        return finish_lossless(frame, orientation);
    }
    if frame.coding == Coding::Progressive
        && frame
            .components
            .iter()
            .any(|component| component.approximation[0] == UNSEEN)
    {
        return Err(malformed());
    }
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(frame.components.len())
        .map_err(|_| allocation())?;
    for component in &frame.components {
        let table = quantization[component.quantization].ok_or_else(malformed)?;
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
            match frame.components.len() {
                1 => pixels.push(sample(0)?),
                3 if adobe_transform == Some(0) => {
                    pixels.extend_from_slice(&[sample(0)?, sample(1)?, sample(2)?]);
                }
                3 => pixels.extend_from_slice(&ycbcr_to_rgb(sample(0)?, sample(1)?, sample(2)?)),
                4 => {
                    let key = sample(3)?;
                    let cmy = if adobe_transform == Some(2) {
                        ycbcr_to_rgb(sample(0)?, sample(1)?, sample(2)?)
                    } else {
                        [sample(0)?, sample(1)?, sample(2)?]
                    };
                    pixels.extend(cmy.map(|channel| {
                        let color = u16::from(255 - channel) * u16::from(255 - key) / 255;
                        u8::try_from(color).expect("invariant: CMYK product divided by 255 fits u8")
                    }));
                }
                _ => return Err(unsupported()),
            }
        }
    }
    if pixels.len() != output_len || output_len > limits.max_working_bytes {
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
\n