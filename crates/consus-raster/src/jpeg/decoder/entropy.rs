use super::super::bitstream::{BitReader, Tables, receive_extend};
use super::super::transform::{BLOCK_CELLS, ZIGZAG};
use super::{Coding, Frame, Scan, ScanComponent, malformed, too_large};
use crate::DecodeError;

pub(super) fn decode_scan(
    bytes: &[u8],
    cursor: usize,
    frame: &mut Frame,
    tables: &Tables,
    scan: &Scan,
    restart_interval: usize,
) -> Result<usize, DecodeError> {
    if frame.coding == Coding::Lossless {
        return super::lossless::decode_lossless_scan(
            bytes,
            cursor,
            frame,
            tables,
            scan,
            restart_interval,
        );
    }
    let interleaved = scan.components.len() > 1;
    let (mcu_across, mcu_down) = if interleaved {
        (frame.mcu_across, frame.mcu_down)
    } else {
        let component = frame
            .components
            .get(scan.components[0].frame_index)
            .ok_or_else(malformed)?;
        (component.blocks_across, component.blocks_down)
    };
    let mcu_count = mcu_across.checked_mul(mcu_down).ok_or_else(too_large)?;
    let mut reader = BitReader::new(bytes, cursor);
    let mut eob_run = 0_usize;
    let mut expected_restart = 0_u8;
    let mut dc_predictors = [0_i32; 4];
    for mcu in 0..mcu_count {
        let mcu_x = mcu % mcu_across;
        let mcu_y = mcu / mcu_across;
        for scan_component in &scan.components {
            let (horizontal, vertical) = if interleaved {
                let component = frame
                    .components
                    .get(scan_component.frame_index)
                    .ok_or_else(malformed)?;
                (
                    usize::from(component.horizontal),
                    usize::from(component.vertical),
                )
            } else {
                (1, 1)
            };
            for block_y in 0..vertical {
                for block_x in 0..horizontal {
                    let block_index = {
                        let component = frame
                            .components
                            .get(scan_component.frame_index)
                            .ok_or_else(malformed)?;
                        if interleaved {
                            let row = mcu_y
                                .checked_mul(vertical)
                                .and_then(|value| value.checked_add(block_y))
                                .ok_or_else(too_large)?;
                            let column = mcu_x
                                .checked_mul(horizontal)
                                .and_then(|value| value.checked_add(block_x))
                                .ok_or_else(too_large)?;
                            row.checked_mul(component.stored_across)
                                .and_then(|value| value.checked_add(column))
                                .ok_or_else(too_large)?
                        } else {
                            mcu_y
                                .checked_mul(component.stored_across)
                                .and_then(|value| value.checked_add(mcu_x))
                                .ok_or_else(too_large)?
                        }
                    };
                    decode_dct_block(
                        &mut reader,
                        frame,
                        tables,
                        *scan_component,
                        scan,
                        block_index,
                        &mut eob_run,
                        &mut dc_predictors,
                    )?;
                }
            }
        }
        let completed = mcu.checked_add(1).ok_or_else(too_large)?;
        if restart_interval != 0 && completed % restart_interval == 0 && completed < mcu_count {
            if eob_run != 0 {
                return Err(malformed());
            }
            reader.finish_byte()?;
            let (_, marker, after_marker) = reader.marker()?;
            if marker != 0xd0 + expected_restart {
                return Err(malformed());
            }
            reader.cursor = after_marker;
            expected_restart = (expected_restart + 1) & 7;
            dc_predictors.fill(0);
        }
    }
    if eob_run != 0 {
        return Err(malformed());
    }
    reader.finish_byte()?;
    let (marker_start, marker, _) = reader.marker()?;
    if matches!(marker, 0xd0..=0xd7) {
        return Err(malformed());
    }
    Ok(marker_start)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one entropy block needs scan and predictor state"
)]
fn decode_dct_block(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    tables: &Tables,
    component: ScanComponent,
    scan: &Scan,
    block_index: usize,
    eob_run: &mut usize,
    dc_predictors: &mut [i32; 4],
) -> Result<(), DecodeError> {
    match frame.coding {
        Coding::Sequential => {
            decode_sequential(reader, frame, tables, component, block_index, dc_predictors)
        }
        Coding::Progressive if scan.start == 0 && scan.high == 0 => decode_dc_initial(
            reader,
            frame,
            tables,
            component,
            block_index,
            scan.low,
            dc_predictors,
        ),
        Coding::Progressive if scan.start == 0 => {
            let bit = reader.bit()?;
            if bit != 0 {
                let delta = 1_i32
                    .checked_shl(u32::from(scan.low))
                    .ok_or_else(malformed)?;
                let value = coefficient_mut(frame, component.frame_index, block_index, 0)?;
                // T.81 G.1.2.1 refines DC by setting the transmitted bit in
                // the two's-complement coefficient representation.
                *value |= delta;
            }
            Ok(())
        }
        Coding::Progressive if scan.high == 0 => {
            decode_ac_initial(reader, frame, tables, component, scan, block_index, eob_run)
        }
        Coding::Progressive => {
            decode_ac_refinement(reader, frame, tables, component, scan, block_index, eob_run)
        }
        Coding::Lossless => Err(malformed()),
    }
}

fn decode_sequential(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    tables: &Tables,
    component: ScanComponent,
    block_index: usize,
    predictors: &mut [i32; 4],
) -> Result<(), DecodeError> {
    let table = tables.dc[component.dc_table].ok_or_else(malformed)?;
    let category = table.decode(reader)?;
    if category > 11 {
        return Err(malformed());
    }
    let difference = receive_extend(reader, category)?;
    let predictor = predictors
        .get_mut(component.frame_index)
        .ok_or_else(malformed)?;
    *predictor = predictor.checked_add(difference).ok_or_else(malformed)?;
    if !(-2048..=2047).contains(predictor) {
        return Err(malformed());
    }
    *coefficient_mut(frame, component.frame_index, block_index, 0)? = *predictor;

    let table = tables.ac[component.ac_table].ok_or_else(malformed)?;
    let mut coefficient = 1_u8;
    while coefficient <= 63 {
        let symbol = table.decode(reader)?;
        let run = symbol >> 4;
        let size = symbol & 0x0f;
        if size == 0 {
            if run == 0 {
                break;
            }
            if run != 15 || coefficient > 48 {
                return Err(malformed());
            }
            coefficient += 16;
            continue;
        }
        if size > 10 {
            return Err(malformed());
        }
        coefficient = coefficient.checked_add(run).ok_or_else(malformed)?;
        if coefficient > 63 {
            return Err(malformed());
        }
        let value = receive_extend(reader, size)?;
        *coefficient_mut(frame, component.frame_index, block_index, coefficient)? = value;
        coefficient += 1;
    }
    Ok(())
}

fn decode_dc_initial(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    tables: &Tables,
    component: ScanComponent,
    block_index: usize,
    low: u8,
    predictors: &mut [i32; 4],
) -> Result<(), DecodeError> {
    let table = tables.dc[component.dc_table].ok_or_else(malformed)?;
    let category = table.decode(reader)?;
    if category > 11 {
        return Err(malformed());
    }
    let difference = receive_extend(reader, category)?;
    let predictor = predictors
        .get_mut(component.frame_index)
        .ok_or_else(malformed)?;
    *predictor = predictor.checked_add(difference).ok_or_else(malformed)?;
    if !(-2048..=2047).contains(predictor) {
        return Err(malformed());
    }
    *coefficient_mut(frame, component.frame_index, block_index, 0)? = predictor
        .checked_shl(u32::from(low))
        .ok_or_else(malformed)?;
    Ok(())
}

fn decode_ac_initial(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    tables: &Tables,
    component: ScanComponent,
    scan: &Scan,
    block_index: usize,
    eob_run: &mut usize,
) -> Result<(), DecodeError> {
    if *eob_run != 0 {
        *eob_run -= 1;
        return Ok(());
    }
    let table = tables.ac[component.ac_table].ok_or_else(malformed)?;
    let mut coefficient = scan.start;
    while coefficient <= scan.end {
        let symbol = table.decode(reader)?;
        let run = symbol >> 4;
        let size = symbol & 0x0f;
        if size == 0 {
            if run == 15 {
                if coefficient > scan.end.saturating_sub(15) {
                    return Err(malformed());
                }
                coefficient += 16;
                continue;
            }
            *eob_run = (1_usize << run)
                .checked_add(usize::from(reader.bits(run)?))
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(malformed)?;
            break;
        }
        if size > 10 {
            return Err(malformed());
        }
        coefficient = coefficient.checked_add(run).ok_or_else(malformed)?;
        if coefficient > scan.end {
            return Err(malformed());
        }
        let value = receive_extend(reader, size)?
            .checked_shl(u32::from(scan.low))
            .ok_or_else(malformed)?;
        *coefficient_mut(frame, component.frame_index, block_index, coefficient)? = value;
        coefficient += 1;
    }
    Ok(())
}

fn decode_ac_refinement(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    tables: &Tables,
    component: ScanComponent,
    scan: &Scan,
    block_index: usize,
    eob_run: &mut usize,
) -> Result<(), DecodeError> {
    let delta = 1_i32
        .checked_shl(u32::from(scan.low))
        .ok_or_else(malformed)?;
    if *eob_run != 0 {
        refine_existing(
            reader,
            frame,
            component.frame_index,
            block_index,
            scan.start,
            scan.end,
            delta,
        )?;
        *eob_run -= 1;
        return Ok(());
    }
    let table = tables.ac[component.ac_table].ok_or_else(malformed)?;
    let mut coefficient = scan.start;
    while coefficient <= scan.end {
        let symbol = table.decode(reader)?;
        let mut run = symbol >> 4;
        let size = symbol & 0x0f;
        let new_value = match size {
            0 if run < 15 => {
                let following = usize::from(reader.bits(run)?);
                refine_existing(
                    reader,
                    frame,
                    component.frame_index,
                    block_index,
                    coefficient,
                    scan.end,
                    delta,
                )?;
                *eob_run = (1_usize << run)
                    .checked_add(following)
                    .and_then(|value| value.checked_sub(1))
                    .ok_or_else(malformed)?;
                return Ok(());
            }
            0 => {
                run = 16;
                None
            }
            1 => Some(if reader.bit()? == 0 { -delta } else { delta }),
            _ => return Err(malformed()),
        };
        loop {
            if coefficient > scan.end {
                return Err(malformed());
            }
            let value = coefficient_mut(frame, component.frame_index, block_index, coefficient)?;
            if *value != 0 {
                refine_value(reader, value, delta)?;
            } else if run == 0 {
                *value = new_value.ok_or_else(malformed)?;
                coefficient += 1;
                break;
            } else {
                run -= 1;
            }
            coefficient += 1;
            if run == 0 && new_value.is_none() {
                break;
            }
        }
    }
    Ok(())
}

fn refine_existing(
    reader: &mut BitReader<'_>,
    frame: &mut Frame,
    component: usize,
    block: usize,
    start: u8,
    end: u8,
    delta: i32,
) -> Result<(), DecodeError> {
    for coefficient in start..=end {
        let value = coefficient_mut(frame, component, block, coefficient)?;
        if *value != 0 {
            refine_value(reader, value, delta)?;
        }
    }
    Ok(())
}

fn refine_value(
    reader: &mut BitReader<'_>,
    value: &mut i32,
    delta: i32,
) -> Result<(), DecodeError> {
    if reader.bit()? != 0 && value.abs() & delta == 0 {
        *value = if *value > 0 {
            value.checked_add(delta)
        } else {
            value.checked_sub(delta)
        }
        .ok_or_else(malformed)?;
    }
    Ok(())
}

fn coefficient_mut(
    frame: &mut Frame,
    component: usize,
    block: usize,
    zigzag: u8,
) -> Result<&mut i32, DecodeError> {
    let component = frame.components.get(component).ok_or_else(malformed)?;
    let block_count = component
        .stored_across
        .checked_mul(component.stored_down)
        .ok_or_else(too_large)?;
    if block >= block_count {
        return Err(malformed());
    }
    let natural = *ZIGZAG.get(usize::from(zigzag)).ok_or_else(malformed)?;
    let index = component
        .coefficient_offset
        .checked_add(block.checked_mul(BLOCK_CELLS).ok_or_else(too_large)?)
        .and_then(|value| value.checked_add(natural))
        .ok_or_else(too_large)?;
    frame.coefficients.get_mut(index).ok_or_else(malformed)
}
