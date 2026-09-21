//! Shared traversal for block-based JPEG entropy scans.

use super::super::transform::{BLOCK_CELLS, ZIGZAG};
use super::{Frame, Scan, ScanComponent, malformed, too_large};
use crate::DecodeError;

/// Entropy-specific state advanced by the shared MCU traversal.
///
/// Implementations decode one addressed block, consume and reset at restart
/// boundaries, and finish at the marker following the scan. The generic walk
/// monomorphizes these calls for each entropy coding process.
pub(super) trait DctScanState {
    /// Decodes one scan component block at its validated frame index.
    fn decode_block(
        &mut self,
        frame: &mut Frame,
        component: ScanComponent,
        scan: &Scan,
        block_index: usize,
    ) -> Result<(), DecodeError>;

    /// Consumes the next restart marker and resets entropy-specific state.
    fn restart(&mut self, expected_restart: u8) -> Result<(), DecodeError>;

    /// Finishes the entropy segment and returns the following marker offset.
    fn finish(&mut self) -> Result<usize, DecodeError>;
}

/// Walks every MCU and component block in one DCT scan.
pub(super) fn decode_dct_scan<S: DctScanState>(
    frame: &mut Frame,
    scan: &Scan,
    restart_interval: usize,
    state: &mut S,
) -> Result<usize, DecodeError> {
    let interleaved = scan.components.len() > 1;
    let (mcu_across, mcu_down) = if interleaved {
        (frame.mcu_across, frame.mcu_down)
    } else {
        let scan_component = scan.components.first().ok_or_else(malformed)?;
        let component = frame
            .components
            .get(scan_component.frame_index)
            .ok_or_else(malformed)?;
        (component.blocks_across, component.blocks_down)
    };
    let mcu_count = mcu_across.checked_mul(mcu_down).ok_or_else(too_large)?;
    let mut expected_restart = 0_u8;

    for mcu in 0..mcu_count {
        let mcu_x = mcu % mcu_across;
        let mcu_y = mcu / mcu_across;
        for scan_component in &scan.components {
            let component = frame
                .components
                .get(scan_component.frame_index)
                .ok_or_else(malformed)?;
            let (horizontal, vertical) = if interleaved {
                (
                    usize::from(component.horizontal),
                    usize::from(component.vertical),
                )
            } else {
                (1, 1)
            };
            let stored_across = component.stored_across;

            for block_y in 0..vertical {
                for block_x in 0..horizontal {
                    let (row, column) = if interleaved {
                        (
                            mcu_y
                                .checked_mul(vertical)
                                .and_then(|value| value.checked_add(block_y))
                                .ok_or_else(too_large)?,
                            mcu_x
                                .checked_mul(horizontal)
                                .and_then(|value| value.checked_add(block_x))
                                .ok_or_else(too_large)?,
                        )
                    } else {
                        (mcu_y, mcu_x)
                    };
                    let block_index = row
                        .checked_mul(stored_across)
                        .and_then(|value| value.checked_add(column))
                        .ok_or_else(too_large)?;
                    state.decode_block(frame, *scan_component, scan, block_index)?;
                }
            }
        }

        let completed = mcu.checked_add(1).ok_or_else(too_large)?;
        if restart_interval != 0
            && completed.is_multiple_of(restart_interval)
            && completed < mcu_count
        {
            state.restart(expected_restart)?;
            expected_restart = (expected_restart + 1) & 7;
        }
    }
    state.finish()
}

pub(super) fn coefficient_mut(
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
