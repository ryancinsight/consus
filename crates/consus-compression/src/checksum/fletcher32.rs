//! Fletcher-32 checksum as used by HDF5.
//!
//! ## Algorithm
//!
//! Processes input as a sequence of 16-bit little-endian words.
//! Maintains two running sums, `s1` and `s2`, modulo 65535:
//!
//! ```text
//!   s1 = (s1 + word_i) mod 65535
//!   s2 = (s2 + s1)     mod 65535
//! ```
//!
//! Checksum = `(s2 << 16) | s1`
//!
//! If the input length is odd, the last byte is padded with a zero byte
//! to form the final 16-bit word.
//!
//! ## Initial State
//!
//! HDF5 convention: `s1 = 0, s2 = 0`.
//! This differs from the textbook Fletcher-32 initialisation of `s1 = 0xFFFF`.
//!
//! ## Reference
//!
//! HDF5 specification, Section III.H: Fletcher32 Checksum.

use super::traits::Checksum;

/// Fletcher-32 checksum (HDF5 convention).
///
/// ## Invariants
///
/// - `s1` and `s2` are always in `[0, 0xFFFE]` after each modular reduction.
/// - The modulus 65535 (`0xFFFF`) guarantees that intermediate sums in `u32`
///   cannot overflow: the maximum value of `s1` before reduction is
///   `0xFFFE + 0xFFFF = 0x1FFFD`, which fits in `u32`.
#[derive(Debug, Clone)]
pub struct Fletcher32 {
    s1: u32,
    s2: u32,
}

impl Fletcher32 {
    /// Fletcher-32 modulus.
    const MOD: u32 = 65535;

    /// Create a new Fletcher-32 instance with HDF5-convention zero
    /// initialisation.
    #[must_use]
    pub const fn new() -> Self {
        Self { s1: 0, s2: 0 }
    }

    /// Process a single 16-bit word into the running sums.
    #[inline]
    fn process_word(&mut self, word: u32) {
        self.s1 = (self.s1 + word) % Self::MOD;
        self.s2 = (self.s2 + self.s1) % Self::MOD;
    }
}

impl Default for Fletcher32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Checksum for Fletcher32 {
    type Output = u32;

    /// Feed data into the Fletcher-32 computation.
    ///
    /// Input bytes are consumed as 16-bit little-endian words. If the
    /// total accumulated length across all `update` calls is odd at the
    /// time of [`finalize`](Checksum::finalize), the last byte is
    /// implicitly padded with `0x00`.
    ///
    /// This implementation processes complete pairs within the current
    /// slice. An odd trailing byte is paired with `0x00` immediately,
    /// matching HDF5 behaviour where each `update` call supplies a
    /// self-contained segment of the data stream.
    fn update(&mut self, data: &[u8]) {
        let mut i = 0;
        let len = data.len();

        // Process complete 16-bit LE word pairs.
        while i + 1 < len {
            let word = u32::from(data[i]) | (u32::from(data[i + 1]) << 8);
            self.process_word(word);
            i += 2;
        }

        // If there is a trailing odd byte, pad with 0x00 to form a word.
        if i < len {
            let word = u32::from(data[i]);
            self.process_word(word);
        }
    }

    /// Return the 32-bit Fletcher checksum.
    ///
    /// ```text
    /// result = (s2 << 16) | s1
    /// ```
    fn finalize(&self) -> u32 {
        (self.s2 << 16) | self.s1
    }

    /// Reset to the HDF5 zero-initialised state.
    fn reset(&mut self) {
        self.s1 = 0;
        self.s2 = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty input: both sums remain zero.
    #[test]
    fn empty_input() {
        let result = Fletcher32::compute(&[]);
        assert_eq!(result, 0x0000_0000);
    }

    /// Single 16-bit LE word [0x01, 0x02] → word = 0x0201.
    ///
    /// s1 = 0x0201, s2 = 0x0201.
    /// Result = 0x0201_0201.
    #[test]
    fn single_word() {
        let result = Fletcher32::compute(&[0x01, 0x02]);
        assert_eq!(result, 0x0201_0201);
    }

    /// Two 16-bit LE words [0x01, 0x02, 0x03, 0x04].
    ///
    /// Word 0: 0x0201 → s1 = 0x0201, s2 = 0x0201.
    /// Word 1: 0x0403 → s1 = 0x0201 + 0x0403 = 0x0604,
    ///                   s2 = 0x0201 + 0x0604 = 0x0805.
    /// Result = 0x0805_0604.
    #[test]
    fn two_words() {
        let result = Fletcher32::compute(&[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(result, 0x0805_0604);
    }

    /// Odd-length input: [0x01, 0x02, 0x03].
    ///
    /// Word 0: LE(0x01, 0x02) = 0x0201 → s1 = 0x0201, s2 = 0x0201.
    /// Word 1: pad(0x03, 0x00) = 0x0003 → s1 = 0x0201 + 0x0003 = 0x0204,
    ///                                     s2 = 0x0201 + 0x0204 = 0x0405.
    /// Result = 0x0405_0204.
    #[test]
    fn odd_length_input() {
        let result = Fletcher32::compute(&[0x01, 0x02, 0x03]);
        assert_eq!(result, 0x0405_0204);
    }

    /// Incremental update must match single-shot compute.
    #[test]
    fn incremental_matches_single_shot() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];

        let single = Fletcher32::compute(&data);

        let mut incremental = Fletcher32::new();
        // Feed two bytes at a time (complete words).
        incremental.update(&data[..2]);
        incremental.update(&data[2..4]);
        incremental.update(&data[4..6]);
        incremental.update(&data[6..8]);
        let inc_result = incremental.finalize();

        assert_eq!(inc_result, single);
    }

    /// Reset produces a fresh state that recomputes identically.
    #[test]
    fn reset_and_recompute() {
        let data = [0x10, 0x20, 0x30, 0x40];

        let mut hasher = Fletcher32::new();
        hasher.update(&data);
        let first = hasher.finalize();

        hasher.reset();
        hasher.update(&data);
        let second = hasher.finalize();

        assert_eq!(first, second);
    }

    /// Verify modular reduction: construct input that forces sums past 65535.
    ///
    /// Using 0xFFFF words: word value = 0xFFFF.
    /// After first word:  s1 = 0xFFFF % 65535 = 0, s2 = 0.
    /// Note: 0xFFFF = 65535 and 65535 mod 65535 = 0.
    ///
    /// More useful: word value = 0xFFFE (= 65534).
    /// After word 0: s1 = 65534, s2 = 65534.
    /// After word 1: s1 = (65534 + 65534) % 65535 = 65533,
    ///               s2 = (65534 + 65533) % 65535 = 65532.
    #[test]
    fn modular_reduction() {
        // Two words of value 0xFFFE = [0xFE, 0xFF, 0xFE, 0xFF].
        let data = [0xFE, 0xFF, 0xFE, 0xFF];
        let result = Fletcher32::compute(&data);

        let expected_s1: u32 = (65534 + 65534) % 65535; // 65533
        let expected_s2: u32 = (65534 + expected_s1) % 65535; // 65532
        let expected = (expected_s2 << 16) | expected_s1;

        assert_eq!(expected_s1, 65533);
        assert_eq!(expected_s2, 65532);
        assert_eq!(result, expected);
    }

    /// Single byte input: [0xFF] → padded to word 0x00FF.
    ///
    /// s1 = 0xFF = 255, s2 = 255.
    /// Result = (255 << 16) | 255 = 0x00FF_00FF.
    #[test]
    fn single_byte_padded() {
        let result = Fletcher32::compute(&[0xFF]);
        assert_eq!(result, 0x00FF_00FF);
    }
}
