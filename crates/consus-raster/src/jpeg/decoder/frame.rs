use super::super::bitstream::Tables;
use super::super::transform::{BLOCK_CELLS, BLOCK_SIDE, ZIGZAG};
use super::{
    Coding, Component, Frame, Scan, ScanComponent, UNSEEN, allocation, malformed, read_word,
    too_large, unsupported,
};
use crate::{DecodeError, DecodeLimits};

const _: () = assert!(
    4 * size_of::<Component>() + 4 * size_of::<ScanComponent>() + 4 * size_of::<Vec<u8>>()
        <= super::super::WORKING_METADATA_BOUND
);

pub(super) fn parse_frame(
    payload: &[u8],
    marker: u8,
    limits: DecodeLimits,
) -> Result<Frame, DecodeError> {
    let precision = *payload.first().ok_or_else(malformed)?;
    let height = usize::from(read_word(payload.get(1..).ok_or_else(malformed)?)?);
    let width = usize::from(read_word(payload.get(3..).ok_or_else(malformed)?)?);
    let count = usize::from(*payload.get(5).ok_or_else(malformed)?);
    if width == 0 || height == 0 || !matches!(count, 1 | 3 | 4) || payload.len() != 6 + count * 3 {
        return Err(malformed());
    }
    let coding = match marker {
        0xc0 | 0xc1 => Coding::Sequential,
        0xc2 => Coding::Progressive,
        0xc3 => Coding::Lossless,
        _ => return Err(unsupported()),
    };
    match coding {
        Coding::Sequential | Coding::Progressive if precision != 8 => return Err(unsupported()),
        Coding::Lossless if !(8..=16).contains(&precision) || count != 1 => {
            return Err(unsupported());
        }
        _ => {}
    }
    let width_u32 = u32::try_from(width).map_err(|_| too_large())?;
    let height_u32 = u32::try_from(height).map_err(|_| too_large())?;
    let pixels = width.checked_mul(height).ok_or_else(too_large)?;
    if width_u32 > limits.max_dimension
        || height_u32 > limits.max_dimension
        || pixels > limits.max_pixels
    {
        return Err(too_large());
    }

    let specifications = payload.get(6..).ok_or_else(malformed)?;
    let max_horizontal = specifications
        .chunks_exact(3)
        .map(|component| component[1] >> 4)
        .max()
        .ok_or_else(malformed)?;
    let max_vertical = specifications
        .chunks_exact(3)
        .map(|component| component[1] & 0x0f)
        .max()
        .ok_or_else(malformed)?;
    if max_horizontal == 0 || max_vertical == 0 || max_horizontal > 4 || max_vertical > 4 {
        return Err(malformed());
    }
    let block_extent = if coding == Coding::Lossless {
        1
    } else {
        BLOCK_SIDE
    };
    let mcu_width = usize::from(max_horizontal) * block_extent;
    let mcu_height = usize::from(max_vertical) * block_extent;
    let mcu_across = width.div_ceil(mcu_width);
    let mcu_down = height.div_ceil(mcu_height);
    let mut components = Vec::new();
    components
        .try_reserve_exact(count)
        .map_err(|_| allocation())?;
    let mut coefficient_count = 0_usize;
    let mut blocks_per_mcu = 0_usize;
    for specification in specifications.chunks_exact(3) {
        let horizontal = specification[1] >> 4;
        let vertical = specification[1] & 0x0f;
        let quantization = usize::from(specification[2]);
        if horizontal == 0
            || vertical == 0
            || horizontal > 4
            || vertical > 4
            || quantization >= 4
            || components
                .iter()
                .any(|component: &Component| component.id == specification[0])
        {
            return Err(malformed());
        }
        if coding == Coding::Lossless && (horizontal != 1 || vertical != 1 || quantization != 0) {
            return Err(unsupported());
        }
        blocks_per_mcu = blocks_per_mcu
            .checked_add(usize::from(horizontal) * usize::from(vertical))
            .ok_or_else(too_large)?;
        let (blocks_across, blocks_down, stored_across, stored_down, stored_coefficients) =
            if coding == Coding::Lossless {
                (width, height, width, height, pixels)
            } else {
                let blocks_across = width
                    .checked_mul(usize::from(horizontal))
                    .ok_or_else(too_large)?
                    .div_ceil(usize::from(max_horizontal) * BLOCK_SIDE);
                let blocks_down = height
                    .checked_mul(usize::from(vertical))
                    .ok_or_else(too_large)?
                    .div_ceil(usize::from(max_vertical) * BLOCK_SIDE);
                let stored_across = mcu_across
                    .checked_mul(usize::from(horizontal))
                    .ok_or_else(too_large)?;
                let stored_down = mcu_down
                    .checked_mul(usize::from(vertical))
                    .ok_or_else(too_large)?;
                let stored_coefficients = stored_across
                    .checked_mul(stored_down)
                    .and_then(|blocks| blocks.checked_mul(BLOCK_CELLS))
                    .ok_or_else(too_large)?;
                (
                    blocks_across,
                    blocks_down,
                    stored_across,
                    stored_down,
                    stored_coefficients,
                )
            };
        let coefficient_offset = coefficient_count;
        coefficient_count = coefficient_count
            .checked_add(stored_coefficients)
            .ok_or_else(too_large)?;
        components.push(Component {
            id: specification[0],
            horizontal,
            vertical,
            quantization,
            quantization_values: None,
            blocks_across,
            blocks_down,
            stored_across,
            stored_down,
            coefficient_offset,
            approximation: [UNSEEN; 64],
        });
    }
    if blocks_per_mcu > 10 {
        return Err(malformed());
    }

    let output_bytes = pixels
        .checked_mul(if coding == Coding::Lossless && precision > 8 {
            2
        } else if count == 1 {
            1
        } else {
            3
        })
        .ok_or_else(too_large)?;
    let coefficient_bytes = coefficient_count
        .checked_mul(size_of::<i32>())
        .ok_or_else(too_large)?;
    let component_bytes = count
        .checked_mul(size_of::<Component>())
        .ok_or_else(too_large)?;
    let scan_bytes = count
        .checked_mul(size_of::<ScanComponent>())
        .ok_or_else(too_large)?;
    let plane_bytes = if coding == Coding::Lossless {
        0
    } else {
        coefficient_count
    };
    let plane_descriptors = if coding == Coding::Lossless {
        0
    } else {
        count
            .checked_mul(size_of::<Vec<u8>>())
            .ok_or_else(too_large)?
    };
    let persistent = coefficient_bytes
        .checked_add(component_bytes)
        .ok_or_else(too_large)?;
    let scan_working = persistent.checked_add(scan_bytes).ok_or_else(too_large)?;
    let output_working = persistent
        .checked_add(plane_bytes)
        .and_then(|bytes| bytes.checked_add(plane_descriptors))
        .and_then(|bytes| bytes.checked_add(output_bytes))
        .ok_or_else(too_large)?;
    let working = scan_working.max(output_working);
    if working > limits.max_working_bytes {
        return Err(too_large());
    }
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(coefficient_count)
        .map_err(|_| allocation())?;
    coefficients.resize(coefficient_count, 0);
    Ok(Frame {
        coding,
        width,
        height,
        precision,
        max_horizontal,
        max_vertical,
        mcu_across,
        mcu_down,
        components,
        coefficients,
    })
}

pub(super) fn parse_quantization(
    payload: &[u8],
    tables: &mut [Option<[u16; 64]>; 4],
) -> Result<(), DecodeError> {
    let mut cursor = 0_usize;
    while cursor < payload.len() {
        let selector = *payload.get(cursor).ok_or_else(malformed)?;
        cursor = cursor.checked_add(1).ok_or_else(too_large)?;
        let precision = selector >> 4;
        let index = usize::from(selector & 0x0f);
        if index >= 4 {
            return Err(malformed());
        }
        if precision != 0 {
            return Err(unsupported());
        }
        let end = cursor.checked_add(64).ok_or_else(too_large)?;
        let values = payload.get(cursor..end).ok_or_else(malformed)?;
        if values.contains(&0) {
            return Err(malformed());
        }
        let mut natural = [0_u16; 64];
        for (zigzag, value) in values.iter().copied().enumerate() {
            natural[ZIGZAG[zigzag]] = u16::from(value);
        }
        tables[index] = Some(natural);
        cursor = end;
    }
    if cursor == 0 {
        return Err(malformed());
    }
    Ok(())
}

pub(super) fn parse_scan(
    payload: &[u8],
    frame: &Frame,
    tables: &Tables,
) -> Result<Scan, DecodeError> {
    let count = usize::from(*payload.first().ok_or_else(malformed)?);
    if count == 0 || count > frame.components.len() || payload.len() != 1 + count * 2 + 3 {
        return Err(malformed());
    }
    let tail = payload.get(1 + count * 2..).ok_or_else(malformed)?;
    let start = tail[0];
    let end = tail[1];
    let high = tail[2] >> 4;
    let low = tail[2] & 0x0f;
    if end > 63 {
        return Err(malformed());
    }
    if frame.coding != Coding::Lossless && (high > 13 || low > 13) {
        return Err(malformed());
    }
    if frame.coding != Coding::Lossless && start > end {
        return Err(malformed());
    }
    match frame.coding {
        Coding::Sequential if start != 0 || end != 63 || high != 0 || low != 0 => {
            return Err(malformed());
        }
        Coding::Progressive if start == 0 && end != 0 => return Err(malformed()),
        Coding::Progressive if start != 0 && count != 1 => return Err(malformed()),
        Coding::Progressive if high != 0 && high != low + 1 => return Err(malformed()),
        Coding::Lossless
            if !(1..=7).contains(&start) || end != 0 || high != 0 || low >= frame.precision =>
        {
            return Err(malformed());
        }
        _ => {}
    }
    let mut components = Vec::new();
    components
        .try_reserve_exact(count)
        .map_err(|_| allocation())?;
    for selector in payload
        .get(1..=count * 2)
        .ok_or_else(malformed)?
        .chunks_exact(2)
    {
        let frame_index = frame
            .components
            .iter()
            .position(|component| component.id == selector[0])
            .ok_or_else(malformed)?;
        if components
            .iter()
            .any(|component: &ScanComponent| component.frame_index == frame_index)
        {
            return Err(malformed());
        }
        let dc_table = usize::from(selector[1] >> 4);
        let ac_table = usize::from(selector[1] & 0x0f);
        if dc_table >= 4 || ac_table >= 4 {
            return Err(malformed());
        }
        if (start == 0 || frame.coding == Coding::Lossless)
            && (frame.coding != Coding::Progressive || high == 0)
            && tables.dc[dc_table].is_none()
            || end != 0 && tables.ac[ac_table].is_none()
        {
            return Err(malformed());
        }
        if frame.coding == Coding::Progressive {
            let approximation = &frame.components[frame_index].approximation;
            if start != 0 && approximation[0] == UNSEEN {
                return Err(malformed());
            }
            for coefficient in start..=end {
                let previous = approximation[usize::from(coefficient)];
                if high == 0 && previous != UNSEEN || high != 0 && previous != high {
                    return Err(malformed());
                }
            }
        } else if frame.coding == Coding::Sequential
            && frame.components[frame_index].approximation[0] != UNSEEN
        {
            return Err(malformed());
        }
        components.push(ScanComponent {
            frame_index,
            dc_table,
            ac_table,
        });
    }
    Ok(Scan {
        components,
        start,
        end,
        high,
        low,
    })
}

pub(super) fn capture_quantization(
    frame: &mut Frame,
    scan: &Scan,
    tables: &[Option<[u16; 64]>; 4],
) -> Result<(), DecodeError> {
    if frame.coding == Coding::Lossless {
        return Ok(());
    }
    for scan_component in &scan.components {
        let component = frame
            .components
            .get_mut(scan_component.frame_index)
            .ok_or_else(malformed)?;
        if component.quantization_values.is_none() {
            component.quantization_values = Some(
                tables
                    .get(component.quantization)
                    .copied()
                    .flatten()
                    .ok_or_else(malformed)?,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::bitstream::parse_tables;
    use super::*;
    use crate::DecodeErrorKind;

    fn frame() -> Frame {
        parse_frame(
            &[8, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0],
            0xc0,
            DecodeLimits {
                max_encoded_bytes: 1024,
                max_dimension: 8,
                max_pixels: 64,
                max_working_bytes: 8192,
            },
        )
        .expect("valid frame")
    }

    #[test]
    fn quantization_is_bound_when_component_scan_begins() {
        let mut frame = frame();
        let mut tables = [None; 4];
        tables[0] = Some([1; 64]);
        capture_quantization(
            &mut frame,
            &Scan {
                components: vec![ScanComponent {
                    frame_index: 0,
                    dc_table: 0,
                    ac_table: 0,
                }],
                start: 0,
                end: 63,
                high: 0,
                low: 0,
            },
            &tables,
        )
        .expect("first component table");
        tables[0] = Some([2; 64]);
        capture_quantization(
            &mut frame,
            &Scan {
                components: vec![ScanComponent {
                    frame_index: 1,
                    dc_table: 0,
                    ac_table: 0,
                }],
                start: 0,
                end: 63,
                high: 0,
                low: 0,
            },
            &tables,
        )
        .expect("second component table");
        assert_eq!(frame.components[0].quantization_values, Some([1; 64]));
        assert_eq!(frame.components[1].quantization_values, Some([2; 64]));
    }

    #[test]
    fn sequential_component_cannot_be_scanned_twice() {
        let mut frame = frame();
        frame.components[0].approximation[0] = 0;
        let mut tables = Tables::new();
        let mut definition = vec![0, 1];
        definition.extend_from_slice(&[0; 15]);
        definition.push(0);
        definition.extend_from_slice(&[0x10, 1]);
        definition.extend_from_slice(&[0; 15]);
        definition.push(0);
        parse_tables(&definition, &mut tables).expect("minimal Huffman tables");
        assert_eq!(
            parse_scan(&[1, 1, 0, 0, 63, 0], &frame, &tables)
                .map(|_| ())
                .expect_err("duplicate sequential scan")
                .kind(),
            DecodeErrorKind::Malformed
        );
    }
}
