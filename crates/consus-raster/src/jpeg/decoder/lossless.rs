use super::super::bitstream::{BitReader, Tables, receive_extend};
use super::{Frame, Scan, malformed, too_large, unsupported};
use crate::DecodeError;

pub(super) fn decode_lossless_scan(
    bytes: &[u8],
    cursor: usize,
    frame: &mut Frame,
    tables: &Tables,
    scan: &Scan,
    restart_interval: usize,
) -> Result<usize, DecodeError> {
    if scan.components.len() != 1 || frame.components.len() != 1 {
        return Err(unsupported());
    }
    let component = scan.components[0];
    let table = tables.dc[component.dc_table].ok_or_else(malformed)?;
    let reduced_precision = frame
        .precision
        .checked_sub(scan.low)
        .ok_or_else(malformed)?;
    let initial = 1_i32
        .checked_shl(u32::from(reduced_precision - 1))
        .ok_or_else(malformed)?;
    let maximum = 1_i32
        .checked_shl(u32::from(reduced_precision))
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(malformed)?;
    let sample_count = frame
        .width
        .checked_mul(frame.height)
        .ok_or_else(too_large)?;
    let mut reader = BitReader::new(bytes, cursor);
    let mut expected_restart = 0_u8;
    let mut restart_origin = true;
    for sample in 0..sample_count {
        let category = table.decode(&mut reader)?;
        if category > 16 {
            return Err(malformed());
        }
        // T.81 Table H.2 reserves category 16 for -32768 and encodes no
        // additional magnitude bits.
        let difference = if category == 16 {
            -32_768
        } else {
            receive_extend(&mut reader, category)?
        };
        let x = sample % frame.width;
        let y = sample / frame.width;
        let predictor = if restart_origin || sample == 0 {
            initial
        } else if y == 0 {
            *frame.coefficients.get(sample - 1).ok_or_else(malformed)?
        } else if x == 0 {
            *frame
                .coefficients
                .get(sample.checked_sub(frame.width).ok_or_else(malformed)?)
                .ok_or_else(malformed)?
        } else {
            let left = *frame.coefficients.get(sample - 1).ok_or_else(malformed)?;
            let above_index = sample.checked_sub(frame.width).ok_or_else(malformed)?;
            let above = *frame.coefficients.get(above_index).ok_or_else(malformed)?;
            let upper_left = *frame
                .coefficients
                .get(above_index.checked_sub(1).ok_or_else(malformed)?)
                .ok_or_else(malformed)?;
            lossless_predictor(scan.start, left, above, upper_left)?
        };
        // T.81 A.4 lossless reconstruction is modulo 2^16. The reduced
        // precision bound below then rejects values unavailable at P-Pt.
        let value = predictor
            .checked_add(difference)
            .ok_or_else(malformed)?
            .rem_euclid(65_536);
        if !(0..=maximum).contains(&value) {
            return Err(malformed());
        }
        *frame.coefficients.get_mut(sample).ok_or_else(malformed)? = value;
        restart_origin = false;

        let completed = sample.checked_add(1).ok_or_else(too_large)?;
        if restart_interval != 0 && completed % restart_interval == 0 && completed < sample_count {
            reader.finish_byte()?;
            let (_, marker, after_marker) = reader.marker()?;
            if marker != 0xd0 + expected_restart {
                return Err(malformed());
            }
            reader.cursor = after_marker;
            expected_restart = (expected_restart + 1) & 7;
            restart_origin = true;
        }
    }
    reader.finish_byte()?;
    let (marker_start, marker, _) = reader.marker()?;
    if matches!(marker, 0xd0..=0xd7) {
        return Err(malformed());
    }
    for sample in &mut frame.coefficients {
        *sample = sample
            .checked_shl(u32::from(scan.low))
            .ok_or_else(malformed)?;
    }
    Ok(marker_start)
}

fn lossless_predictor(
    selection: u8,
    left: i32,
    above: i32,
    upper_left: i32,
) -> Result<i32, DecodeError> {
    match selection {
        1 => Ok(left),
        2 => Ok(above),
        3 => Ok(upper_left),
        4 => left
            .checked_add(above)
            .and_then(|v| v.checked_sub(upper_left))
            .ok_or_else(malformed),
        5 => above
            .checked_sub(upper_left)
            .map(|difference| left + difference.div_euclid(2))
            .ok_or_else(malformed),
        6 => left
            .checked_sub(upper_left)
            .map(|difference| above + difference.div_euclid(2))
            .ok_or_else(malformed),
        7 => left
            .checked_add(above)
            .map(|sum| sum.div_euclid(2))
            .ok_or_else(malformed),
        _ => Err(malformed()),
    }
}
\n