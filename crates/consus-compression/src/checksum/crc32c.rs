//! CRC-32C checksum (Castagnoli polynomial).
//!
//! ## Polynomial
//!
//! Reflected form: `0x82F63B78`.
//!
//! ## Why this exists alongside [`super::crc32::Crc32`]
//!
//! They are not interchangeable. Zarr v3 specifies **crc32c** for its
//! checksum codec, while HDF5's filter and this crate's other users want
//! IEEE. Reaching for the IEEE implementation where the format says
//! Castagnoli produces checksums no conformant reader accepts — a silent
//! write defect — so both live here under names that say which is which.
//!
//! ## Correctness
//!
//! The CRC-32C check value for the ASCII string `"123456789"` is
//! `0xE3069283` (the standard vector, distinct from IEEE's `0xCBF43926`).
//! Verified in the unit tests below.

use super::reflected::{fold, reflected_table};
use super::traits::Checksum;

/// Castagnoli CRC-32C polynomial in reflected (bit-reversed) form.
const POLYNOMIAL: u32 = 0x82F6_3B78;

/// Initial CRC register value (all bits set).
const INIT: u32 = 0xFFFF_FFFF;

/// XOR mask applied at finalization (all bits set, producing the complement).
const XOROUT: u32 = 0xFFFF_FFFF;

/// Compile-time CRC-32C lookup table (256 entries).
const CRC_TABLE: [u32; 256] = reflected_table(POLYNOMIAL);

/// CRC-32C checksum (Castagnoli polynomial).
///
/// ## State machine
///
/// - **Initial state**: `0xFFFF_FFFF`
/// - **Update rule**: `state = CRC_TABLE[((state ^ byte) & 0xFF)] ^ (state >> 8)`
/// - **Finalization**: `state ^ 0xFFFF_FFFF`
///
/// The pre- and post-conditioning ensures leading and trailing zero bytes
/// affect the value, which is required for correct error detection.
#[derive(Clone)]
pub struct Crc32c {
    /// Running CRC register. Holds the pre-conditioned intermediate value.
    state: u32,
}

impl Crc32c {
    /// Create a new CRC-32C instance with the standard initial value.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { state: INIT }
    }

    /// Compute the CRC-32C of a complete byte slice in one call.
    #[inline]
    #[must_use]
    pub fn compute_slice(data: &[u8]) -> u32 {
        fold(INIT, &CRC_TABLE, data) ^ XOROUT
    }
}

impl Default for Crc32c {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Checksum for Crc32c {
    type Output = u32;

    #[inline]
    fn update(&mut self, data: &[u8]) {
        self.state = fold(self.state, &CRC_TABLE, data);
    }

    #[inline]
    fn finalize(&self) -> u32 {
        self.state ^ XOROUT
    }

    #[inline]
    fn reset(&mut self) {
        self.state = INIT;
    }
}

#[cfg(test)]
mod tests {
    use super::super::crc32::Crc32;
    use super::*;

    /// CRC-32C of the empty byte sequence: `INIT ^ XOROUT == 0`.
    #[test]
    fn empty_data() {
        assert_eq!(Crc32c::new().finalize(), 0x0000_0000);
    }

    /// The canonical CRC-32C check value, from RFC 3720 appendix B.
    #[test]
    fn check_value_123456789() {
        assert_eq!(Crc32c::compute_slice(b"123456789"), 0xE306_9283);
    }

    /// CRC-32C must not silently equal CRC-32.
    ///
    /// This is the assertion that would have caught wiring the IEEE
    /// implementation into a format that specifies Castagnoli — the two
    /// agree on no realistic input, and a test that only checked "some
    /// checksum was produced" would not notice the substitution.
    #[test]
    fn differs_from_ieee_crc32() {
        let input = b"123456789";
        assert_ne!(Crc32c::compute_slice(input), Crc32::compute_slice(input));
        assert_eq!(Crc32::compute_slice(input), 0xCBF4_3926);
    }

    /// Incremental update matches single-shot computation.
    #[test]
    fn incremental_matches_single_shot() {
        let single = Crc32c::compute_slice(b"123456789");
        let mut incremental = Crc32c::new();
        incremental.update(b"1234");
        incremental.update(b"56789");
        assert_eq!(incremental.finalize(), single);
    }

    /// A single flipped bit changes the checksum — the property the codec
    /// relies on to detect corruption.
    #[test]
    fn detects_a_single_bit_flip() {
        let clean = Crc32c::compute_slice(&[0x00, 0x11, 0x22, 0x33]);
        let flipped = Crc32c::compute_slice(&[0x00, 0x11, 0x22, 0x32]);
        assert_ne!(clean, flipped);
    }

    /// `reset` restores the default-constructed state.
    #[test]
    fn reset_restores_initial_state() {
        let mut crc = Crc32c::new();
        crc.update(b"noise");
        crc.reset();
        crc.update(b"123456789");
        assert_eq!(crc.finalize(), 0xE306_9283);
    }
}
