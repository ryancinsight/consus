//! Arithmetic DCT scan decoding for sequential and progressive JPEG.
//!
//! The decision trees and context indices follow ITU-T T.81 sections F.1.4
//! and G.1.3. The implementation was cross-checked against `jdarith.c` from
//! libjpeg-turbo 3.1.2, which derives from Independent JPEG Group software
//! developed by Guido Vollbeding (1997-2015). The shared scan walker owns MCU
//! ordering and restart placement; this module owns only arithmetic models.

use super::super::arithmetic::{Reader, State};
use super::conditioning::{Conditioning, difference_class};
use super::scan::{DctScanState, coefficient_mut, decode_dct_scan};
use super::{Coding, Frame, Scan, ScanComponent, malformed};
use crate::DecodeError;

const DC_BINS: usize = 49;
const AC_BINS: usize = 245;
const REFINEMENT_BINS: usize = 189;
const TABLES: usize = 4;
const COMPONENTS: usize = 4;
const DC_X1: usize = 20;
const AC_X2_LOW: usize = 189;
const AC_X2_HIGH: usize = 217;
const DIFFERENCE_CATEGORY_BITS: u8 = 15;

/// Decodes one arithmetic-coded DCT scan.
pub(super) fn decode_scan(
    bytes: &[u8],
    cursor: usize,
    frame: &mut Frame,
    conditioning: &Conditioning,
    scan: &Scan,
    restart_interval: usize,
) -> Result<usize, DecodeError> {
    let mut state = ArithmeticScanState {
        bytes,
        reader: Reader::new(bytes, cursor)?,
        conditioning,
        dc_stats: [[State::new(); DC_BINS]; TABLES],
        ac_stats: [[State::new(); AC_BINS]; TABLES],
        refinement_stats: [[State::new(); REFINEMENT_BINS]; TABLES],
        fixed: State::fixed(),
        dc_contexts: [0; COMPONENTS],
        dc_predictors: [0; COMPONENTS],
    };
    decode_dct_scan(frame, scan, restart_interval, &mut state)
}

struct ArithmeticScanState<'input, 'tables> {
    bytes: &'input [u8],
    reader: Reader<'input>,
    conditioning: &'tables Conditioning,
    dc_stats: [[State; DC_BINS]; TABLES],
    ac_stats: [[State; AC_BINS]; TABLES],
    refinement_stats: [[State; REFINEMENT_BINS]; TABLES],
    fixed: State,
    dc_contexts: [usize; COMPONENTS],
    dc_predictors: [i32; COMPONENTS],
}

impl DctScanState for ArithmeticScanState<'_, '_> {
    fn decode_block(
        &mut self,
        frame: &mut Frame,
        component: ScanComponent,
        scan: &Scan,
        block_index: usize,
    ) -> Result<(), DecodeError> {
        match frame.coding {
            Coding::Baseline | Coding::Sequential => {
                self.decode_dc_initial(frame, component, scan.low, block_index)?;
                self.decode_ac_initial(frame, component, 1, 63, scan.low, block_index)
            }
            Coding::Progressive if scan.start == 0 && scan.high == 0 => {
                self.decode_dc_initial(frame, component, scan.low, block_index)
            }
            Coding::Progressive if scan.start == 0 => {
                self.decode_dc_refinement(frame, component, scan.low, block_index)
            }
            Coding::Progressive if scan.high == 0 => self.decode_ac_initial(
                frame,
                component,
                scan.start,
                scan.end,
                scan.low,
                block_index,
            ),
            Coding::Progressive => self.decode_ac_refinement(
                frame,
                component,
                scan.start,
                scan.end,
                scan.low,
                block_index,
            ),
            Coding::Lossless => Err(malformed()),
        }
    }

    fn restart(&mut self, expected_restart: u8) -> Result<(), DecodeError> {
        let (_, marker, after_marker) = self.reader.finish()?;
        if marker != 0xd0 + expected_restart {
            return Err(malformed());
        }
        self.reader = Reader::new(self.bytes, after_marker)?;
        self.dc_stats = [[State::new(); DC_BINS]; TABLES];
        self.ac_stats = [[State::new(); AC_BINS]; TABLES];
        self.refinement_stats = [[State::new(); REFINEMENT_BINS]; TABLES];
        self.fixed = State::fixed();
        self.dc_contexts.fill(0);
        self.dc_predictors.fill(0);
        Ok(())
    }

    fn finish(&mut self) -> Result<usize, DecodeError> {
        let (marker_start, marker, _) = self.reader.finish()?;
        if matches!(marker, 0xd0..=0xd7) {
            return Err(malformed());
        }
        Ok(marker_start)
    }
}

impl ArithmeticScanState<'_, '_> {
    fn decode_dc_initial(
        &mut self,
        frame: &mut Frame,
        component: ScanComponent,
        low: u8,
        block_index: usize,
    ) -> Result<(), DecodeError> {
        let table = component.dc_table;
        let context = *self
            .dc_contexts
            .get(component.frame_index)
            .ok_or_else(malformed)?;
        let maximum_category = frame.precision.checked_add(3).ok_or_else(malformed)?;
        let bins = self.dc_stats.get_mut(table).ok_or_else(malformed)?;
        let (difference, _) = decode_difference(&mut self.reader, bins, context, DC_X1)?;
        enforce_dct_magnitude(difference, maximum_category)?;

        let conditioning = *self.conditioning.dc.get(table).ok_or_else(malformed)?;
        let next_context = usize::from(difference_class(difference, conditioning)) * 4;
        *self
            .dc_contexts
            .get_mut(component.frame_index)
            .ok_or_else(malformed)? = next_context;

        let predictor = self
            .dc_predictors
            .get_mut(component.frame_index)
            .ok_or_else(malformed)?;
        *predictor = predictor.checked_add(difference).ok_or_else(malformed)?;
        enforce_signed_category(*predictor, maximum_category)?;
        *coefficient_mut(frame, component.frame_index, block_index, 0)? = predictor
            .checked_shl(u32::from(low))
            .ok_or_else(malformed)?;
        Ok(())
    }

    fn decode_ac_initial(
        &mut self,
        frame: &mut Frame,
        component: ScanComponent,
        start: u8,
        end: u8,
        low: u8,
        block_index: usize,
    ) -> Result<(), DecodeError> {
        let table = component.ac_table;
        let conditioning = *self.conditioning.ac.get(table).ok_or_else(malformed)?;
        let maximum_category = frame.precision.checked_add(2).ok_or_else(malformed)?;
        let mut coefficient = start;
        while coefficient <= end {
            let mut context = 3 * (usize::from(coefficient) - 1);
            if self.ac_decision(table, context)? {
                break;
            }
            while !self.ac_decision(table, context + 1)? {
                coefficient = coefficient.checked_add(1).ok_or_else(malformed)?;
                if coefficient > end {
                    return Err(malformed());
                }
                context += 3;
            }

            let negative = self.fixed_decision()?;
            context += 2;
            let mut magnitude_category = u32::from(self.ac_decision(table, context)?);
            if magnitude_category != 0 && self.ac_decision(table, context)? {
                magnitude_category = 2;
                context = if coefficient <= conditioning {
                    AC_X2_LOW
                } else {
                    AC_X2_HIGH
                };
                loop {
                    if !self.ac_decision(table, context)? {
                        break;
                    }
                    magnitude_category = magnitude_category.checked_shl(1).ok_or_else(malformed)?;
                    context = context.checked_add(1).ok_or_else(malformed)?;
                    enforce_magnitude_category(magnitude_category, maximum_category)?;
                }
            }
            let magnitude = self.decode_magnitude_bits(
                table,
                context.checked_add(14).ok_or_else(malformed)?,
                magnitude_category,
            )?;
            let value = signed_dct_magnitude(magnitude, negative, maximum_category)?
                .checked_shl(u32::from(low))
                .ok_or_else(malformed)?;
            *coefficient_mut(frame, component.frame_index, block_index, coefficient)? = value;
            coefficient = coefficient.checked_add(1).ok_or_else(malformed)?;
        }
        Ok(())
    }

    fn decode_dc_refinement(
        &mut self,
        frame: &mut Frame,
        component: ScanComponent,
        low: u8,
        block_index: usize,
    ) -> Result<(), DecodeError> {
        if self.fixed_decision()? {
            let delta = 1_i32.checked_shl(u32::from(low)).ok_or_else(malformed)?;
            *coefficient_mut(frame, component.frame_index, block_index, 0)? |= delta;
        }
        Ok(())
    }

    fn decode_ac_refinement(
        &mut self,
        frame: &mut Frame,
        component: ScanComponent,
        start: u8,
        end: u8,
        low: u8,
        block_index: usize,
    ) -> Result<(), DecodeError> {
        let table = component.ac_table;
        let delta = 1_i32.checked_shl(u32::from(low)).ok_or_else(malformed)?;
        let negative_delta = delta.checked_neg().ok_or_else(malformed)?;
        let mut previous_end = 0;
        for coefficient in (1..=end).rev() {
            if *coefficient_mut(frame, component.frame_index, block_index, coefficient)? != 0 {
                previous_end = coefficient;
                break;
            }
        }

        let mut coefficient = start;
        while coefficient <= end {
            let mut context = 3 * (usize::from(coefficient) - 1);
            if coefficient > previous_end && self.refinement_decision(table, context)? {
                break;
            }
            loop {
                let value =
                    *coefficient_mut(frame, component.frame_index, block_index, coefficient)?;
                if value != 0 {
                    if self.refinement_decision(table, context + 2)? {
                        *coefficient_mut(frame, component.frame_index, block_index, coefficient)? =
                            if value < 0 {
                                value.checked_add(negative_delta)
                            } else {
                                value.checked_add(delta)
                            }
                            .ok_or_else(malformed)?;
                    }
                    break;
                }
                if self.refinement_decision(table, context + 1)? {
                    *coefficient_mut(frame, component.frame_index, block_index, coefficient)? =
                        if self.fixed_decision()? {
                            negative_delta
                        } else {
                            delta
                        };
                    break;
                }
                coefficient = coefficient.checked_add(1).ok_or_else(malformed)?;
                if coefficient > end {
                    return Err(malformed());
                }
                context += 3;
            }
            coefficient = coefficient.checked_add(1).ok_or_else(malformed)?;
        }
        Ok(())
    }

    fn ac_decision(&mut self, table: usize, context: usize) -> Result<bool, DecodeError> {
        let state = self
            .ac_stats
            .get_mut(table)
            .and_then(|bins| bins.get_mut(context))
            .ok_or_else(malformed)?;
        self.reader.decision(state)
    }

    fn refinement_decision(&mut self, table: usize, context: usize) -> Result<bool, DecodeError> {
        let state = self
            .refinement_stats
            .get_mut(table)
            .and_then(|bins| bins.get_mut(context))
            .ok_or_else(malformed)?;
        self.reader.decision(state)
    }

    fn fixed_decision(&mut self) -> Result<bool, DecodeError> {
        self.reader.decision(&mut self.fixed)
    }

    fn decode_magnitude_bits(
        &mut self,
        table: usize,
        context: usize,
        magnitude_category: u32,
    ) -> Result<u32, DecodeError> {
        let mut magnitude = magnitude_category;
        let mut mask = magnitude_category >> 1;
        while mask != 0 {
            if self.ac_decision(table, context)? {
                magnitude |= mask;
            }
            mask >>= 1;
        }
        magnitude.checked_add(1).ok_or_else(malformed)
    }
}

/// Decodes the shared arithmetic sign and magnitude tree.
///
/// `zero_context` selects S0, `x1_context` selects the first magnitude-category
/// bin. The signed 16-bit category bound rejects malformed input before it can
/// consume an unbounded sequence of decisions. The returned category is the
/// power-of-two bound used by the JPEG conditioning models.
pub(super) fn decode_difference(
    reader: &mut Reader<'_>,
    bins: &mut [State],
    zero_context: usize,
    x1_context: usize,
) -> Result<(i32, u32), DecodeError> {
    if !decision(reader, bins, zero_context)? {
        return Ok((0, 0));
    }
    let negative = decision(reader, bins, zero_context + 1)?;
    let mut context = zero_context + 2 + usize::from(negative);
    let mut magnitude_category = u32::from(decision(reader, bins, context)?);
    if magnitude_category != 0 {
        context = x1_context;
        loop {
            if !decision(reader, bins, context)? {
                break;
            }
            magnitude_category = magnitude_category.checked_shl(1).ok_or_else(malformed)?;
            context = context.checked_add(1).ok_or_else(malformed)?;
            enforce_magnitude_category(magnitude_category, DIFFERENCE_CATEGORY_BITS)?;
        }
    }
    let bit_context = context.checked_add(14).ok_or_else(malformed)?;
    let mut magnitude = magnitude_category;
    let mut mask = magnitude_category >> 1;
    while mask != 0 {
        if decision(reader, bins, bit_context)? {
            magnitude |= mask;
        }
        mask >>= 1;
    }
    let magnitude = magnitude.checked_add(1).ok_or_else(malformed)?;
    Ok((
        signed_magnitude(magnitude, negative, DIFFERENCE_CATEGORY_BITS)?,
        magnitude_category,
    ))
}

fn decision(
    reader: &mut Reader<'_>,
    bins: &mut [State],
    context: usize,
) -> Result<bool, DecodeError> {
    reader.decision(bins.get_mut(context).ok_or_else(malformed)?)
}

fn enforce_magnitude_category(value: u32, maximum_category: u8) -> Result<(), DecodeError> {
    let maximum = 1_u32
        .checked_shl(u32::from(maximum_category.saturating_sub(1)))
        .ok_or_else(malformed)?;
    if value > maximum {
        return Err(malformed());
    }
    Ok(())
}

fn signed_magnitude(
    magnitude: u32,
    negative: bool,
    maximum_category: u8,
) -> Result<i32, DecodeError> {
    let exclusive_upper = 1_u32
        .checked_shl(u32::from(maximum_category))
        .ok_or_else(malformed)?;
    if magnitude > exclusive_upper || (!negative && magnitude == exclusive_upper) {
        return Err(malformed());
    }
    let magnitude = i32::try_from(magnitude).map_err(|_| malformed())?;
    if negative {
        magnitude.checked_neg().ok_or_else(malformed)
    } else {
        Ok(magnitude)
    }
}

fn signed_dct_magnitude(
    magnitude: u32,
    negative: bool,
    maximum_category: u8,
) -> Result<i32, DecodeError> {
    let exclusive_upper = 1_u32
        .checked_shl(u32::from(maximum_category))
        .ok_or_else(malformed)?;
    if magnitude >= exclusive_upper {
        return Err(malformed());
    }
    let magnitude = i32::try_from(magnitude).map_err(|_| malformed())?;
    if negative {
        magnitude.checked_neg().ok_or_else(malformed)
    } else {
        Ok(magnitude)
    }
}

fn enforce_dct_magnitude(value: i32, maximum_category: u8) -> Result<(), DecodeError> {
    let exclusive_upper = 1_u32
        .checked_shl(u32::from(maximum_category))
        .ok_or_else(malformed)?;
    if value.unsigned_abs() >= exclusive_upper {
        return Err(malformed());
    }
    Ok(())
}

fn enforce_signed_category(value: i32, maximum_category: u8) -> Result<(), DecodeError> {
    let exclusive_upper = 1_i32
        .checked_shl(u32::from(maximum_category))
        .ok_or_else(malformed)?;
    let inclusive_lower = exclusive_upper.checked_neg().ok_or_else(malformed)?;
    if !(inclusive_lower..exclusive_upper).contains(&value) {
        return Err(malformed());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
