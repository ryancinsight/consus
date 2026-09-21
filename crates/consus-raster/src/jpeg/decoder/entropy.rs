use super::super::bitstream::{BitReader, Tables, receive_extend};
use super::scan::{DctScanState, coefficient_mut, decode_dct_scan};
use super::{Coding, Frame, Scan, ScanComponent, malformed};
use crate::DecodeError;

const DC_CATEGORY_PRECISION_OFFSET: u8 = 3;
const AC_CATEGORY_PRECISION_OFFSET: u8 = 2;

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
    let mut state = HuffmanScanState {
        reader: BitReader::new(bytes, cursor),
        tables,
        eob_run: 0,
        dc_predictors: [0; 4],
    };
    decode_dct_scan(frame, scan, restart_interval, &mut state)
}

struct HuffmanScanState<'bytes, 'tables> {
    reader: BitReader<'bytes>,
    tables: &'tables Tables,
    eob_run: usize,
    dc_predictors: [i32; 4],
}

impl DctScanState for HuffmanScanState<'_, '_> {
    fn decode_block(
        &mut self,
        frame: &mut Frame,
        component: ScanComponent,
        scan: &Scan,
        block_index: usize,
    ) -> Result<(), DecodeError> {
        decode_dct_block(
            &mut self.reader,
            frame,
            self.tables,
            component,
            scan,
            block_index,
            &mut self.eob_run,
            &mut self.dc_predictors,
        )
    }

    fn restart(&mut self, expected_restart: u8) -> Result<(), DecodeError> {
        if self.eob_run != 0 {
            return Err(malformed());
        }
        self.reader.finish_byte()?;
        let (_, marker, after_marker) = self.reader.marker()?;
        if marker != 0xd0 + expected_restart {
            return Err(malformed());
        }
        self.reader.cursor = after_marker;
        self.dc_predictors.fill(0);
        Ok(())
    }

    fn finish(&mut self) -> Result<usize, DecodeError> {
        if self.eob_run != 0 {
            return Err(malformed());
        }
        self.reader.finish_byte()?;
        let (marker_start, marker, _) = self.reader.marker()?;
        if matches!(marker, 0xd0..=0xd7) {
            return Err(malformed());
        }
        Ok(marker_start)
    }
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
        Coding::Baseline | Coding::Sequential => {
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
    let dc_category_limit =
        coefficient_category_limit(frame.precision, DC_CATEGORY_PRECISION_OFFSET)?;
    if category > dc_category_limit {
        return Err(malformed());
    }
    let difference = receive_extend(reader, category)?;
    let predictor = update_dc_predictor(
        predictors,
        component.frame_index,
        difference,
        dc_category_limit,
    )?;
    *coefficient_mut(frame, component.frame_index, block_index, 0)? = predictor;

    let table = tables.ac[component.ac_table].ok_or_else(malformed)?;
    let ac_category_limit =
        coefficient_category_limit(frame.precision, AC_CATEGORY_PRECISION_OFFSET)?;
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
        if size > ac_category_limit {
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
    let dc_category_limit =
        coefficient_category_limit(frame.precision, DC_CATEGORY_PRECISION_OFFSET)?;
    if category > dc_category_limit {
        return Err(malformed());
    }
    let difference = receive_extend(reader, category)?;
    let predictor = update_dc_predictor(
        predictors,
        component.frame_index,
        difference,
        dc_category_limit,
    )?;
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
    let ac_category_limit =
        coefficient_category_limit(frame.precision, AC_CATEGORY_PRECISION_OFFSET)?;
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
        if size > ac_category_limit {
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

fn coefficient_category_limit(precision: u8, precision_offset: u8) -> Result<u8, DecodeError> {
    precision
        .checked_add(precision_offset)
        .ok_or_else(malformed)
}

fn update_dc_predictor(
    predictors: &mut [i32; 4],
    component: usize,
    difference: i32,
    category_limit: u8,
) -> Result<i32, DecodeError> {
    let upper_bound = 1_i32
        .checked_shl(u32::from(category_limit))
        .ok_or_else(malformed)?;
    let lower_bound = upper_bound.checked_neg().ok_or_else(malformed)?;
    let predictor = predictors.get_mut(component).ok_or_else(malformed)?;
    *predictor = predictor.checked_add(difference).ok_or_else(malformed)?;
    if !(lower_bound..upper_bound).contains(predictor) {
        return Err(malformed());
    }
    Ok(*predictor)
}
