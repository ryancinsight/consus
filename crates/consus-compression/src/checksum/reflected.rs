//! Shared table generation for reflected (bit-reversed) CRC algorithms.
//!
//! CRC-32 and CRC-32C differ only in their generator polynomial; the table
//! construction and the byte-at-a-time update rule are identical. This module
//! is the single owner of both, so a second reflected CRC costs a polynomial
//! constant rather than a second copy of the algorithm.

/// Compute the 256-entry lookup table for a reflected CRC at compile time.
///
/// For each byte value `i`, the entry is computed by iterating 8 bit
/// positions: if the least-significant bit is set, shift right and XOR with
/// the polynomial, otherwise shift right.
///
/// ## Proof sketch
///
/// Each entry `T[i]` equals `CRC(i)` where `i` is treated as a degree-7
/// polynomial over GF(2), divided by the generator polynomial in reflected
/// bit order. The 8-iteration loop is the standard bit-at-a-time division.
pub(super) const fn reflected_table(polynomial: u32) -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i: usize = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ polynomial;
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

/// Fold `data` into a running reflected-CRC register.
///
/// `state = TABLE[((state ^ byte) & 0xFF)] ^ (state >> 8)` per byte, which is
/// associative over concatenation — the incremental-equivalence half of the
/// [`super::traits::Checksum`] contract.
#[inline]
pub(super) fn fold(state: u32, table: &[u32; 256], data: &[u8]) -> u32 {
    let mut crc = state;
    for &byte in data {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = table[index] ^ (crc >> 8);
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two polynomials must produce genuinely different tables.
    ///
    /// A table generator that ignored its argument would satisfy every
    /// check-value test for whichever polynomial happened to be baked in,
    /// so the distinguishing assertion belongs here.
    #[test]
    fn distinct_polynomials_yield_distinct_tables() {
        let ieee = reflected_table(0xEDB8_8320);
        let castagnoli = reflected_table(0x82F6_3B78);
        assert_ne!(ieee, castagnoli);
        // Entry 0 is zero for any polynomial: no bits are ever set.
        assert_eq!(ieee[0], 0);
        assert_eq!(castagnoli[0], 0);
    }

    /// Folding in two pieces equals folding the concatenation.
    #[test]
    fn fold_is_associative_over_concatenation() {
        let table = reflected_table(0x82F6_3B78);
        let whole = fold(0xFFFF_FFFF, &table, b"123456789");
        let split = fold(fold(0xFFFF_FFFF, &table, b"1234"), &table, b"56789");
        assert_eq!(whole, split);
    }
}
