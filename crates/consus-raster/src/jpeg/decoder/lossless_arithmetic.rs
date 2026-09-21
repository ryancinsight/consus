use super::arithmetic::decode_difference;
use super::conditioning::{Conditioning, difference_class};
use super::lossless::{Differences, decode_samples};
use super::{Frame, Scan, allocation, malformed, unsupported};
use crate::DecodeError;
use crate::jpeg::arithmetic::{Reader, State};

/// T.81 H.1.2.3 assigns 100 joint-neighbor bins and two 29-bin magnitude groups.
const STATISTICS_BINS: usize = 158;

#[cfg(test)]
mod tests;

pub(super) fn decode_scan(
    bytes: &[u8],
    cursor: usize,
    frame: &mut Frame,
    conditioning: &Conditioning,
    scan: &Scan,
    restart_interval: usize,
) -> Result<usize, DecodeError> {
    if frame.components.len() != 1 || scan.components.len() != 1 {
        return Err(unsupported());
    }
    let component = scan.components.first().ok_or_else(malformed)?;
    let bounds = *conditioning
        .dc
        .get(component.dc_table)
        .ok_or_else(malformed)?;
    let mut above = Vec::new();
    above
        .try_reserve_exact(frame.width)
        .map_err(|_| allocation())?;
    above.resize(frame.width, 0);
    let mut state = ArithmeticDifferences {
        bytes,
        reader: Reader::new(bytes, cursor)?,
        bins: [State::new(); STATISTICS_BINS],
        above,
        left: 0,
        bounds,
    };
    decode_samples(frame, scan, restart_interval, &mut state)
}

struct ArithmeticDifferences<'input> {
    bytes: &'input [u8],
    reader: Reader<'input>,
    bins: [State; STATISTICS_BINS],
    above: Vec<u8>,
    left: u8,
    bounds: (u8, u8),
}

impl Differences for ArithmeticDifferences<'_> {
    fn decode(&mut self, column: usize) -> Result<i32, DecodeError> {
        if column == 0 {
            self.left = 0;
        }
        let above = self.above.get_mut(column).ok_or_else(malformed)?;
        // H.2 indexes columns by the left difference and rows by the above difference.
        let context = 20 * usize::from(*above) + 4 * usize::from(self.left);
        let magnitude_context = if *above >= 3 { 129 } else { 100 };
        let (difference, _) =
            decode_difference(&mut self.reader, &mut self.bins, context, magnitude_context)?;
        let class = difference_class(difference, self.bounds);
        *above = class;
        self.left = class;
        Ok(difference)
    }

    fn restart(&mut self, expected: u8) -> Result<(), DecodeError> {
        let (_, marker, after) = self.reader.finish()?;
        if marker != 0xd0 + expected {
            return Err(malformed());
        }
        self.reader = Reader::new(self.bytes, after)?;
        self.bins.fill(State::new());
        self.above.fill(0);
        self.left = 0;
        Ok(())
    }

    fn finish(&mut self) -> Result<usize, DecodeError> {
        let (start, marker, _) = self.reader.finish()?;
        if matches!(marker, 0xd0..=0xd7) {
            return Err(malformed());
        }
        Ok(start)
    }
}
