//! Bounded JPEG arithmetic entropy reading and marker retention.

use super::state::State;
use crate::{DecodeError, DecodeErrorKind};

const NORMALIZED_INTERVAL: u32 = 0x8000;
const CODE_REGISTER_SHIFT: u32 = 16;
const BYTE_INSERT_SHIFT: u32 = 8;
const MAX_RENORMALIZATION_SHIFTS: usize = u16::BITS as usize;

/// Allocation-free arithmetic entropy reader retaining the physical marker.
#[derive(Debug)]
pub(in crate::jpeg) struct Reader<'input> {
    bytes: &'input [u8],
    cursor: usize,
    marker: Option<Marker>,
    code: u32,
    interval: u32,
    bits_until_byte: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Marker {
    start: usize,
    code: u8,
    after: usize,
}

impl<'input> Reader<'input> {
    /// Initializes the decoder from two entropy bytes at `cursor`.
    ///
    /// A marker encountered during initialization is retained and supplies the
    /// zero input required by T.81 D.2.6 for all subsequent byte requests.
    pub(in crate::jpeg) fn new(bytes: &'input [u8], cursor: usize) -> Result<Self, DecodeError> {
        if cursor > bytes.len() {
            return Err(malformed());
        }
        let mut reader = Self {
            bytes,
            cursor,
            marker: None,
            code: 0,
            interval: 0x1_0000,
            bits_until_byte: 0,
        };
        let first = reader.entropy_byte()?;
        reader.code |= u32::from(first) << BYTE_INSERT_SHIFT;
        reader.code <<= 8;
        let second = reader.entropy_byte()?;
        reader.code |= u32::from(second) << BYTE_INSERT_SHIFT;
        reader.code <<= 8;
        Ok(reader)
    }

    /// Decodes one binary decision and updates its context state.
    pub(in crate::jpeg) fn decision(&mut self, state: &mut State) -> Result<bool, DecodeError> {
        self.renormalize()?;
        let mps = state.mps();
        let probability = state.probability().ok_or_else(malformed)?;
        self.interval = self
            .interval
            .checked_sub(probability.qe)
            .ok_or_else(malformed)?;
        let code_value = self.code >> CODE_REGISTER_SHIFT;

        let decision = if code_value < self.interval {
            if self.interval >= NORMALIZED_INTERVAL {
                return Ok(mps);
            }
            if self.interval < probability.qe {
                let lps = !mps;
                state.observe_lps(probability);
                lps
            } else {
                state.observe_mps(probability);
                mps
            }
        } else {
            self.code -= self.interval << CODE_REGISTER_SHIFT;
            let decision = if self.interval < probability.qe {
                state.observe_mps(probability);
                mps
            } else {
                state.observe_lps(probability);
                !mps
            };
            self.interval = probability.qe;
            decision
        };
        Ok(decision)
    }

    /// Locates and returns the marker start, code, and first following byte.
    ///
    /// Remaining entropy bytes are scanned with the same stuffing grammar as
    /// decoder input. End-of-input, a truncated `FF` run, or absence of a
    /// physical marker is malformed input.
    pub(in crate::jpeg) fn finish(&mut self) -> Result<(usize, u8, usize), DecodeError> {
        while self.marker.is_none() {
            self.entropy_byte()?;
        }
        let marker = self.marker.ok_or_else(malformed)?;
        Ok((marker.start, marker.code, marker.after))
    }

    fn renormalize(&mut self) -> Result<(), DecodeError> {
        for _ in 0..MAX_RENORMALIZATION_SHIFTS {
            if self.interval >= NORMALIZED_INTERVAL {
                return Ok(());
            }
            if self.bits_until_byte == 0 {
                let byte = self.entropy_byte()?;
                self.code |= u32::from(byte) << BYTE_INSERT_SHIFT;
                self.bits_until_byte = 8;
            }
            self.interval <<= 1;
            self.code <<= 1;
            self.bits_until_byte -= 1;
        }
        Err(malformed())
    }

    fn entropy_byte(&mut self) -> Result<u8, DecodeError> {
        if self.marker.is_some() {
            return Ok(0);
        }
        let marker_start = self.cursor;
        let byte = self.take_byte()?;
        if byte != 0xff {
            return Ok(byte);
        }

        let mut following = self.take_byte()?;
        if following == 0 {
            return Ok(0xff);
        }
        while following == 0xff {
            following = self.take_byte()?;
        }
        if following == 0 {
            return Err(malformed());
        }
        self.marker = Some(Marker {
            start: marker_start,
            code: following,
            after: self.cursor,
        });
        Ok(0)
    }

    fn take_byte(&mut self) -> Result<u8, DecodeError> {
        let byte = self.bytes.get(self.cursor).copied().ok_or_else(malformed)?;
        self.cursor = self.cursor.checked_add(1).ok_or_else(too_large)?;
        Ok(byte)
    }
}

const fn malformed() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Malformed)
}

const fn too_large() -> DecodeError {
    DecodeError::new(DecodeErrorKind::TooLarge)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decisions(reader: &mut Reader<'_>, state: &mut State, count: usize) -> String {
        (0..count)
            .map(|_| {
                if reader.decision(state).expect("valid arithmetic decision") {
                    '1'
                } else {
                    '0'
                }
            })
            .collect()
    }

    #[test]
    fn adaptive_decisions_match_independent_annex_d_vector() {
        // Generated from the T.81 D.2 flow charts and Table D.3 independently of
        // this implementation, then checked against libjpeg-turbo 3.1.2 arith_decode.
        let bytes = [0x94, 0x67, 0xb3, 0x4f, 0x74, 0xe5, 0x6b, 0x6b, 0xff, 0xd9];
        let mut reader = Reader::new(&bytes, 0).expect("valid arithmetic segment");
        let mut state = State::new();
        assert_eq!(
            decisions(&mut reader, &mut state, 64),
            "0011111110101111010011001101111110111111001011011111000111101111"
        );
        assert_eq!(reader.finish().expect("terminating marker"), (8, 0xd9, 10));
    }

    #[test]
    fn fixed_probability_decisions_do_not_adapt() {
        let bytes = [0x6d, 0x27, 0xb4, 0x11, 0xff, 0xd9];
        let mut reader = Reader::new(&bytes, 0).expect("valid arithmetic segment");
        let mut state = State::fixed();
        assert_eq!(
            decisions(&mut reader, &mut state, 32),
            "00010111101101010100111000101000"
        );
        assert_eq!(state, State::fixed());
        assert_eq!(reader.finish().expect("terminating marker"), (4, 0xd9, 6));
    }

    #[test]
    fn conditional_exchange_decisions_follow_reference_sequence() {
        let bytes = [0x00, 0x00, 0xff, 0xd9];
        let mut reader = Reader::new(&bytes, 0).expect("valid arithmetic segment");
        let mut state = State::new();

        assert!(!reader.decision(&mut state).expect("first decision"));
        assert!(reader.decision(&mut state).expect("exchange decision"));
        assert!(reader.decision(&mut state).expect("post-exchange decision"));

        assert_eq!(reader.finish().expect("terminating marker"), (2, 0xd9, 4));
    }

    #[test]
    fn immediate_zero_stuffs_exactly_one_entropy_ff() {
        let bytes = [0xff, 0x00, 0x21, 0x43, 0xff, 0xd0];
        let mut reader = Reader::new(&bytes, 0).expect("stuffed entropy");

        assert_eq!(reader.code, 0xff21_0000);
        assert_eq!(reader.cursor, 3);
        assert_eq!(reader.finish().expect("stuffed marker"), (4, 0xd0, 6));
    }

    #[test]
    fn fill_ff_is_retained_only_before_physical_markers() {
        for marker in [0xd0, 0xd9] {
            let bytes = [0xff, 0xff, marker, 0xa5];
            let mut reader = Reader::new(&bytes, 0).expect("filled marker");
            let mut state = State::new();

            assert_eq!(
                decisions(&mut reader, &mut state, 40),
                "0111111111111111111111111111111111111111"
            );
            assert_eq!(reader.finish().expect("retained marker"), (0, marker, 3));
        }
    }

    #[test]
    fn fill_ff_run_ending_in_zero_is_malformed() {
        let initialization = Reader::new(&[0xff, 0xff, 0x00, 0x21], 0)
            .expect_err("fill cannot precede a stuffing zero");
        assert_eq!(initialization.kind(), DecodeErrorKind::Malformed);

        let bytes = [0x12, 0x34, 0xff, 0xff, 0x00, 0x56];
        let mut reader = Reader::new(&bytes, 0).expect("two initialization bytes");
        let finish = reader
            .finish()
            .expect_err("flush fill cannot precede a stuffing zero");
        assert_eq!(finish.kind(), DecodeErrorKind::Malformed);
    }

    #[test]
    fn marker_retention_supplies_normative_zero_input() {
        let bytes = [0xff, 0xd0, 0xa5];
        let mut reader = Reader::new(&bytes, 0).expect("marker initializes with zero input");
        let mut state = State::new();

        assert_eq!(
            decisions(&mut reader, &mut state, 40),
            "0111111111111111111111111111111111111111"
        );
        assert_eq!(reader.finish().expect("retained marker"), (0, 0xd0, 2));
    }

    #[test]
    fn finish_scans_unread_flush_bytes_without_consuming_following_data() {
        let bytes = [0x94, 0x67, 0x12, 0xff, 0x00, 0x44, 0xff, 0xff, 0xd9, 0xa5];
        let mut reader = Reader::new(&bytes, 0).expect("valid arithmetic segment");
        let mut state = State::new();
        assert!(!reader.decision(&mut state).expect("first decision"));
        assert_eq!(reader.finish().expect("scanned marker"), (6, 0xd9, 9));
        assert_eq!(reader.finish().expect("retained marker"), (6, 0xd9, 9));
    }

    #[test]
    fn cursor_and_physical_termination_are_required() {
        for (bytes, cursor) in [
            (&[][..], 0),
            (&[0x12][..], 0),
            (&[0xff][..], 0),
            (&[0x12, 0x34][..], 3),
        ] {
            let error = Reader::new(bytes, cursor).expect_err("invalid initialization");
            assert_eq!(error.kind(), DecodeErrorKind::Malformed);
        }

        for bytes in [
            &[0x12, 0x34][..],
            &[0x12, 0x34, 0xff][..],
            &[0x12, 0x34, 0xff, 0xff][..],
            &[0x12, 0x34, 0xff, 0x00, 0x56][..],
        ] {
            let mut reader = Reader::new(bytes, 0).expect("two initialization bytes");
            let error = reader.finish().expect_err("physical marker is required");
            assert_eq!(error.kind(), DecodeErrorKind::Malformed);
        }
    }
}
