use super::malformed;
use crate::DecodeError;

/// T.81 F.1.4.4.1.2: zero, small positive/negative, large positive/negative.
pub(super) fn difference_class(value: i32, (lower, upper): (u8, u8)) -> u8 {
    let magnitude = value.unsigned_abs();
    let zero_bound = (1_u32 << lower) >> 1;
    if magnitude <= zero_bound {
        0
    } else {
        let sign = u8::from(value < 0);
        if magnitude > 1_u32 << upper {
            3 + sign
        } else {
            1 + sign
        }
    }
}

/// T.81 B.2.4.3 conditioning values, indexed by entropy-table destination.
pub(super) struct Conditioning {
    pub(super) dc: [(u8, u8); 4],
    pub(super) ac: [u8; 4],
}

impl Conditioning {
    pub(super) const fn new() -> Self {
        Self {
            dc: [(0, 1); 4],
            ac: [5; 4],
        }
    }

    pub(super) fn parse(&mut self, payload: &[u8]) -> Result<(), DecodeError> {
        let mut entries = payload.chunks_exact(2);
        if payload.is_empty() || !entries.remainder().is_empty() {
            return Err(malformed());
        }
        for entry in &mut entries {
            let [selector, value] = *entry else {
                return Err(malformed());
            };
            let destination = usize::from(selector & 15);
            match selector >> 4 {
                0 => {
                    let lower = value & 15;
                    let upper = value >> 4;
                    if lower > upper {
                        return Err(malformed());
                    }
                    *self.dc.get_mut(destination).ok_or_else(malformed)? = (lower, upper);
                }
                1 if (1..=63).contains(&value) => {
                    *self.ac.get_mut(destination).ok_or_else(malformed)? = value;
                }
                _ => return Err(malformed()),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecodeErrorKind;

    #[test]
    fn conditioning_values_cover_the_admitted_domain() {
        for lower in 0..=15 {
            for upper in lower..=15 {
                let mut tables = Conditioning::new();
                tables.parse(&[3, upper << 4 | lower]).unwrap();
                assert_eq!(tables.dc[3], (lower, upper));
                assert_eq!(tables.dc[0], (0, 1));
            }
        }
        for split in 1..=63 {
            let mut tables = Conditioning::new();
            tables.parse(&[0x12, split]).unwrap();
            assert_eq!(tables.ac, [5, 5, split, 5]);
        }
    }

    #[test]
    fn malformed_conditioning_is_rejected() {
        for payload in [
            &[][..],
            &[0],
            &[0, 0x01],
            &[4, 0x10],
            &[0x14, 5],
            &[0x10, 0],
            &[0x10, 64],
            &[0x20, 5],
        ] {
            assert_eq!(
                Conditioning::new().parse(payload).unwrap_err().kind(),
                DecodeErrorKind::Malformed
            );
        }
    }

    #[test]
    fn difference_classes_pin_exclusive_and_inclusive_bounds() {
        for lower in 0..=15 {
            for upper in lower..=15 {
                let bounds = (lower, upper);
                let zero_bound = (1_i32 << lower) >> 1;
                let small_bound = 1_i32 << upper;
                assert_eq!(difference_class(zero_bound, bounds), 0);
                assert_eq!(difference_class(-zero_bound, bounds), 0);
                assert_eq!(difference_class(zero_bound + 1, bounds), 1);
                assert_eq!(difference_class(-zero_bound - 1, bounds), 2);
                assert_eq!(difference_class(small_bound, bounds), 1);
                assert_eq!(difference_class(-small_bound, bounds), 2);
                assert_eq!(difference_class(small_bound + 1, bounds), 3);
                assert_eq!(difference_class(-small_bound - 1, bounds), 4);
            }
        }
    }
}
