//! JPEG QM probability states and T.81 Table D.3.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Probability {
    pub(super) qe: u32,
    pub(super) next_lps: u8,
    pub(super) next_mps: u8,
    pub(super) switch_mps: bool,
}

const fn probability(qe: u32, lps_index: u8, mps_index: u8, changes_mps: bool) -> Probability {
    Probability {
        qe,
        next_lps: lps_index,
        next_mps: mps_index,
        switch_mps: changes_mps,
    }
}

// T.81 Table D.3. Entry 113 is the non-adapting equiprobable state used by
// the JPEG arithmetic models and encoded by libjpeg-turbo's fixed bin.
const PROBABILITIES: [Probability; 114] = [
    probability(0x5a1d, 1, 1, true),
    probability(0x2586, 14, 2, false),
    probability(0x1114, 16, 3, false),
    probability(0x080b, 18, 4, false),
    probability(0x03d8, 20, 5, false),
    probability(0x01da, 23, 6, false),
    probability(0x00e5, 25, 7, false),
    probability(0x006f, 28, 8, false),
    probability(0x0036, 30, 9, false),
    probability(0x001a, 33, 10, false),
    probability(0x000d, 35, 11, false),
    probability(0x0006, 9, 12, false),
    probability(0x0003, 10, 13, false),
    probability(0x0001, 12, 13, false),
    probability(0x5a7f, 15, 15, true),
    probability(0x3f25, 36, 16, false),
    probability(0x2cf2, 38, 17, false),
    probability(0x207c, 39, 18, false),
    probability(0x17b9, 40, 19, false),
    probability(0x1182, 42, 20, false),
    probability(0x0cef, 43, 21, false),
    probability(0x09a1, 45, 22, false),
    probability(0x072f, 46, 23, false),
    probability(0x055c, 48, 24, false),
    probability(0x0406, 49, 25, false),
    probability(0x0303, 51, 26, false),
    probability(0x0240, 52, 27, false),
    probability(0x01b1, 54, 28, false),
    probability(0x0144, 56, 29, false),
    probability(0x00f5, 57, 30, false),
    probability(0x00b7, 59, 31, false),
    probability(0x008a, 60, 32, false),
    probability(0x0068, 62, 33, false),
    probability(0x004e, 63, 34, false),
    probability(0x003b, 32, 35, false),
    probability(0x002c, 33, 9, false),
    probability(0x5ae1, 37, 37, true),
    probability(0x484c, 64, 38, false),
    probability(0x3a0d, 65, 39, false),
    probability(0x2ef1, 67, 40, false),
    probability(0x261f, 68, 41, false),
    probability(0x1f33, 69, 42, false),
    probability(0x19a8, 70, 43, false),
    probability(0x1518, 72, 44, false),
    probability(0x1177, 73, 45, false),
    probability(0x0e74, 74, 46, false),
    probability(0x0bfb, 75, 47, false),
    probability(0x09f8, 77, 48, false),
    probability(0x0861, 78, 49, false),
    probability(0x0706, 79, 50, false),
    probability(0x05cd, 48, 51, false),
    probability(0x04de, 50, 52, false),
    probability(0x040f, 50, 53, false),
    probability(0x0363, 51, 54, false),
    probability(0x02d4, 52, 55, false),
    probability(0x025c, 53, 56, false),
    probability(0x01f8, 54, 57, false),
    probability(0x01a4, 55, 58, false),
    probability(0x0160, 56, 59, false),
    probability(0x0125, 57, 60, false),
    probability(0x00f6, 58, 61, false),
    probability(0x00cb, 59, 62, false),
    probability(0x00ab, 61, 63, false),
    probability(0x008f, 61, 32, false),
    probability(0x5b12, 65, 65, true),
    probability(0x4d04, 80, 66, false),
    probability(0x412c, 81, 67, false),
    probability(0x37d8, 82, 68, false),
    probability(0x2fe8, 83, 69, false),
    probability(0x293c, 84, 70, false),
    probability(0x2379, 86, 71, false),
    probability(0x1edf, 87, 72, false),
    probability(0x1aa9, 87, 73, false),
    probability(0x174e, 72, 74, false),
    probability(0x1424, 72, 75, false),
    probability(0x119c, 74, 76, false),
    probability(0x0f6b, 74, 77, false),
    probability(0x0d51, 75, 78, false),
    probability(0x0bb6, 77, 79, false),
    probability(0x0a40, 77, 48, false),
    probability(0x5832, 80, 81, true),
    probability(0x4d1c, 88, 82, false),
    probability(0x438e, 89, 83, false),
    probability(0x3bdd, 90, 84, false),
    probability(0x34ee, 91, 85, false),
    probability(0x2eae, 92, 86, false),
    probability(0x299a, 93, 87, false),
    probability(0x2516, 86, 71, false),
    probability(0x5570, 88, 89, true),
    probability(0x4ca9, 95, 90, false),
    probability(0x44d9, 96, 91, false),
    probability(0x3e22, 97, 92, false),
    probability(0x3824, 99, 93, false),
    probability(0x32b4, 99, 94, false),
    probability(0x2e17, 93, 86, false),
    probability(0x56a8, 95, 96, true),
    probability(0x4f46, 101, 97, false),
    probability(0x47e5, 102, 98, false),
    probability(0x41cf, 103, 99, false),
    probability(0x3c3d, 104, 100, false),
    probability(0x375e, 99, 93, false),
    probability(0x5231, 105, 102, false),
    probability(0x4c0f, 106, 103, false),
    probability(0x4639, 107, 104, false),
    probability(0x415e, 103, 99, false),
    probability(0x5627, 105, 106, true),
    probability(0x50e7, 108, 107, false),
    probability(0x4b85, 109, 103, false),
    probability(0x5597, 110, 109, false),
    probability(0x504f, 111, 107, false),
    probability(0x5a10, 110, 111, true),
    probability(0x5522, 112, 109, false),
    probability(0x59eb, 112, 111, true),
    probability(0x5a1d, 113, 113, false),
];

/// One adaptive binary probability state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::jpeg) struct State(u8);

impl State {
    /// Returns the T.81 initial state: index zero with MPS sense zero.
    pub(in crate::jpeg) const fn new() -> Self {
        Self(0)
    }

    /// Returns the non-adapting equiprobable state used for fixed decisions.
    pub(in crate::jpeg) const fn fixed() -> Self {
        Self(113)
    }

    const fn from_parts(index: u8, mps: bool) -> Self {
        Self(index | if mps { 0x80 } else { 0 })
    }

    const fn index(self) -> u8 {
        self.0 & 0x7f
    }

    pub(super) const fn mps(self) -> bool {
        self.0 & 0x80 != 0
    }

    pub(super) fn probability(self) -> Option<Probability> {
        PROBABILITIES.get(usize::from(self.index())).copied()
    }

    pub(super) const fn observe_mps(&mut self, probability: Probability) {
        *self = Self::from_parts(probability.next_mps, self.mps());
    }

    pub(super) const fn observe_lps(&mut self, probability: Probability) {
        let mps = if probability.switch_mps {
            !self.mps()
        } else {
            self.mps()
        };
        *self = Self::from_parts(probability.next_lps, mps);
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lps_exchange_and_mps_observation_follow_table_transitions() {
        let mut state = State::new();
        let probability = state.probability().expect("initial table state");
        state.observe_lps(probability);
        assert_eq!(state, State::from_parts(1, true));

        let probability = state.probability().expect("post-exchange table state");
        state.observe_mps(probability);
        assert_eq!(state, State::from_parts(2, true));
    }

    #[test]
    fn table_entries_match_t81_boundaries_and_fixed_state() {
        assert_eq!(std::mem::size_of::<State>(), 1);
        assert_eq!(PROBABILITIES[0], probability(0x5a1d, 1, 1, true));
        assert_eq!(PROBABILITIES[13], probability(0x0001, 12, 13, false));
        assert_eq!(PROBABILITIES[64], probability(0x5b12, 65, 65, true));
        assert_eq!(PROBABILITIES[112], probability(0x59eb, 112, 111, true));
        assert_eq!(PROBABILITIES[113], probability(0x5a1d, 113, 113, false));
    }
}
