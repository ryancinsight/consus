//! CRC-32 checksum (IEEE 802.3 polynomial).
//!
//! ## Polynomial
//!
//! Generator polynomial: x³² + x²⁶ + x²³ + x²² + x¹⁶ + x¹² + x¹¹ + x¹⁰ + x⁸ + x⁷ + x⁵ + x⁴ + x² + x + 1
//! Reflected form: `0xEDB88320`
//!
//! ## Algorithm
//!
//! Standard table-driven CRC-32 with byte-at-a-time processing.
//! The 256-entry lookup table is computed at compile time via `const` evaluation.
//!
//! ## Correctness
//!
//! The CRC-32/ISO-HDLC check value for the ASCII string `"123456789"` is
//! `0xCBF43926`. This is verified in the unit tests below.

use super::traits::Checksum;

/// IEEE 802.3 CRC-32 polynomial in reflected (bit-reversed) form.
const POLYNOMIAL: u32 = 0xEDB8_8320;

/// Initial CRC register value (all bits set).
const INIT: u32 = 0xFFFF_FFFF;

/// XOR mask applied at finalization (all bits set, producing the complement).
const XOROUT: u32 = 0xFFFF_FFFF;

/// Compute the 256-entry CRC-32 lookup table at compile time.
///
/// For each byte value `i` in `0..256`, the entry is computed by iterating
/// 8 bit positions. At each position, if the least-significant bit is set,
/// the value is shifted right by 1 and XORed with the polynomial; otherwise
/// it is simply shifted right by 1.
///
/// ## Proof sketch
///
/// Each table entry `T[i]` equals `CRC(i)` where `i` is treated as a
/// degree-7 polynomial over GF(2), divided by the generator polynomial
/// in reflected bit order. The 8-iteration loop is the standard
/// bit-at-a-time division algorithm.
const fn make_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i: usize = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compile-time CRC-32 lookup table (256 entries).
const CRC_TABLE: [u32; 256] = make_crc_table();

/// CRC-32 checksum (IEEE 802.3 polynomial).
///
/// ## Polynomial
///
/// Generator polynomial: x³² + x²⁶ + x²³ + x²² + x¹⁶ + x¹² + x¹¹ + x¹⁰ + x⁸ + x⁷ + x⁵ + x⁴ + x² + x + 1
/// Reflected form: `0xEDB88320`
///
/// ## Algorithm
///
/// Standard table-driven CRC-32 with byte-at-a-time processing.
/// The lookup table is computed at compile time using const evaluation.
///
/// ## State machine
///
/// - **Initial state**: `0xFFFF_FFFF`
/// - **Update rule**: `state = CRC_TABLE[((state ^ byte) & 0xFF) as usize] ^ (state >> 8)`
/// - **Finalization**: `state ^ 0xFFFF_FFFF`
///
/// The pre- and post-conditioning with `0xFFFF_FFFF` ensures that leading
/// and trailing zero bytes affect the checksum value, which is required
/// for correct error detection.
#[derive(Clone)]
pub struct Crc32 {
    /// Running CRC register. Holds the pre-conditioned intermediate value.
    state: u32,
}

impl Crc32 {
    /// Create a new CRC-32 instance with the standard initial value.
    #[inline]
    pub fn new() -> Self {
        Self { state: INIT }
    }

    /// Compute the CRC-32 of a complete byte slice in one call.
    ///
    /// Equivalent to creating a new instance, calling `update`, and
    /// calling `finalize`.
    #[inline]
    pub fn compute_slice(data: &[u8]) -> u32 {
        let mut crc = Self::new();
        crc.update(data);
        crc.finalize()
    }
}

impl Default for Crc32 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Checksum for Crc32 {
    type Output = u32;

    /// Feed `data` into the running CRC-32 computation.
    ///
    /// For each byte `b` in `data`:
    ///   `state = CRC_TABLE[((state ^ b) & 0xFF) as usize] ^ (state >> 8)`
    ///
    /// This is incremental: `update(a); update(b)` produces the same
    /// result as `update(a ++ b)`.
    #[inline]
    fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &byte in data {
            let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
            crc = CRC_TABLE[index] ^ (crc >> 8);
        }
        self.state = crc;
    }

    /// Return the finalized CRC-32 value.
    ///
    /// Applies the output XOR mask (`0xFFFF_FFFF`) to the running state.
    /// Does not mutate the instance; repeated calls return the same value
    /// until `update` or `reset` is called.
    #[inline]
    fn finalize(&self) -> u32 {
        self.state ^ XOROUT
    }

    /// Reset the CRC-32 instance to its initial state.
    #[inline]
    fn reset(&mut self) {
        self.state = INIT;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CRC-32 of the empty byte sequence.
    ///
    /// With `init = 0xFFFF_FFFF` and no data, finalize returns
    /// `0xFFFF_FFFF ^ 0xFFFF_FFFF = 0x0000_0000`.
    #[test]
    fn empty_data() {
        let crc = Crc32::new();
        assert_eq!(crc.finalize(), 0x0000_0000);
    }

    /// The canonical CRC-32/ISO-HDLC check value.
    ///
    /// CRC-32 of the ASCII string `"123456789"` (9 bytes, no NUL terminator)
    /// must equal `0xCBF43926`. This is the universally cited test vector for
    /// CRC-32 with the IEEE 802.3 polynomial.
    #[test]
    fn check_value_123456789() {
        let result = Crc32::compute_slice(b"123456789");
        assert_eq!(result, 0xCBF4_3926);
    }

    /// Incremental update produces the same result as single-shot compute.
    ///
    /// Splitting `"123456789"` into `"1234"` and `"56789"` and calling
    /// `update` twice must yield the same checksum as a single `update`
    /// over the entire string.
    #[test]
    fn incremental_matches_single_shot() {
        let single = Crc32::compute_slice(b"123456789");

        let mut incremental = Crc32::new();
        incremental.update(b"1234");
        incremental.update(b"56789");
        assert_eq!(incremental.finalize(), single);
    }

    /// After `reset`, recomputing produces the same result as a fresh instance.
    #[test]
    fn reset_and_recompute() {
        let mut crc = Crc32::new();
        crc.update(b"some data");
        let _ = crc.finalize();

        crc.reset();
        crc.update(b"123456789");
        assert_eq!(crc.finalize(), 0xCBF4_3926);
    }

    /// The `Checksum::compute` convenience method works correctly.
    #[test]
    fn trait_compute_convenience() {
        let result = Crc32::compute(b"123456789");
        assert_eq!(result, 0xCBF4_3926);
    }

    /// Verify CRC table entries against the bit-at-a-time algorithm.
    ///
    /// `CRC_TABLE[0]` must be 0 (zero input, 8 zero-bit shifts, no XOR).
    /// `CRC_TABLE[i]` is the result of running 8 iterations of the
    /// reflected-polynomial division on byte value `i`.
    ///
    /// For i=1: the first iteration XORs with the polynomial, then 7
    /// more shifts yield `0x77073096` (well-known first non-zero entry).
    #[test]
    fn table_spot_check() {
        /// Compute a single CRC table entry by running the 8-iteration
        /// bit-at-a-time division on `byte_val`.
        fn crc_entry(byte_val: u8) -> u32 {
            let mut v = byte_val as u32;
            for _ in 0..8 {
                if v & 1 != 0 {
                    v = (v >> 1) ^ POLYNOMIAL;
                } else {
                    v >>= 1;
                }
            }
            v
        }

        assert_eq!(CRC_TABLE[0], 0x0000_0000);
        assert_eq!(CRC_TABLE[0], crc_entry(0x00));
        // CRC_TABLE[1] = 0x77073096 (not the raw polynomial; the byte
        // value 0x01 undergoes all 8 shift-and-XOR iterations).
        assert_eq!(CRC_TABLE[1], 0x7707_3096);
        assert_eq!(CRC_TABLE[1], crc_entry(0x01));
        assert_eq!(CRC_TABLE[255], crc_entry(0xFF));
    }

    /// Single-byte CRC values for boundary bytes 0x00 and 0xFF.
    #[test]
    fn single_byte_boundary() {
        // CRC of a single 0x00 byte:
        // state = INIT, index = (0xFFFFFFFF ^ 0) & 0xFF = 0xFF
        // state = CRC_TABLE[0xFF] ^ (0xFFFFFFFF >> 8) = CRC_TABLE[255] ^ 0x00FFFFFF
        // finalize = state ^ 0xFFFFFFFF
        let crc_zero = Crc32::compute_slice(&[0x00]);
        let expected_zero = (CRC_TABLE[255] ^ 0x00FF_FFFF) ^ XOROUT;
        assert_eq!(crc_zero, expected_zero);

        let crc_ff = Crc32::compute_slice(&[0xFF]);
        // index = (0xFFFFFFFF ^ 0xFF) & 0xFF = 0x00
        // state = CRC_TABLE[0] ^ (0xFFFFFFFF >> 8) = 0 ^ 0x00FFFFFF = 0x00FFFFFF
        // finalize = 0x00FFFFFF ^ 0xFFFFFFFF = 0xFF000000
        assert_eq!(crc_ff, 0xFF00_0000);
    }

    /// Determinism: computing the same input twice yields identical results.
    #[test]
    fn deterministic() {
        let data = b"determinism check payload";
        let a = Crc32::compute_slice(data);
        let b = Crc32::compute_slice(data);
        assert_eq!(a, b);
    }

    /// Three-way split produces the same result as single-shot.
    #[test]
    fn three_way_incremental() {
        let mut crc = Crc32::new();
        crc.update(b"123");
        crc.update(b"456");
        crc.update(b"789");
        assert_eq!(crc.finalize(), 0xCBF4_3926);
    }
}
