//! Unit tests for checksum algorithms.
//!
//! Verifies correctness against known test vectors and validates
//! the incremental equivalence invariant.

use consus_compression::checksum::{Checksum, Crc32, Fletcher32, Lookup3};

// =============================================================================
// CRC-32 Tests (IEEE 802.3 polynomial)
// =============================================================================

mod crc32 {
    use super::*;

    /// CRC-32/ISO-HDLC check value for "123456789".
    ///
    /// Reference: IEEE 802.3 standard, CRC-32 check value.
    /// The ASCII string "123456789" (9 bytes, no NUL terminator) must produce
    /// `0xCBF43926`. This is the universally cited test vector.
    #[test]
    fn check_value_123456789() {
        let result = Crc32::compute(b"123456789");
        assert_eq!(
            result, 0xCBF4_3926,
            "CRC-32 check value must match IEEE 802.3 reference"
        );
    }

    /// Empty input produces `0x00000000`.
    ///
    /// With `init = 0xFFFFFFFF` and no data, finalize returns
    /// `0xFFFFFFFF ^ 0xFFFFFFFF = 0x00000000`.
    #[test]
    fn empty_input() {
        let result = Crc32::compute(b"");
        assert_eq!(result, 0x0000_0000, "CRC-32 of empty input must be zero");
    }

    /// Single byte `0x00`.
    ///
    /// Verifies correct handling of leading zeros affecting the checksum.
    #[test]
    fn single_zero_byte() {
        let mut crc = Crc32::new();
        crc.update(&[0x00]);
        let result = crc.finalize();
        // Compute expected value analytically
        let expected = {
            // index = (0xFFFFFFFF ^ 0x00) & 0xFF = 0xFF
            // state = CRC_TABLE[255] ^ (0xFFFFFFFF >> 8)
            // finalize = state ^ 0xFFFFFFFF
            let table_255 = compute_crc_table_entry(255);
            let state = table_255 ^ 0x00FF_FFFF;
            state ^ 0xFFFF_FFFF
        };
        assert_eq!(
            result, expected,
            "CRC-32 of single 0x00 byte must match analytical value"
        );
    }

    /// Single byte `0xFF`.
    ///
    /// index = (0xFFFFFFFF ^ 0xFF) & 0xFF = 0x00
    /// state = CRC_TABLE[0] ^ (0xFFFFFFFF >> 8) = 0 ^ 0x00FFFFFF = 0x00FFFFFF
    /// finalize = 0x00FFFFFF ^ 0xFFFFFFFF = 0xFF000000
    #[test]
    fn single_ff_byte() {
        let result = Crc32::compute(&[0xFF]);
        assert_eq!(
            result, 0xFF00_0000,
            "CRC-32 of single 0xFF byte must be 0xFF000000"
        );
    }

    /// Incremental update produces the same result as single-shot compute.
    ///
    /// Invariant: `update(a); update(b)` equals `update(a ++ b)`.
    #[test]
    fn incremental_matches_single_shot() {
        let single = Crc32::compute(b"123456789");

        let mut incremental = Crc32::new();
        incremental.update(b"1234");
        incremental.update(b"56789");

        assert_eq!(
            incremental.finalize(),
            single,
            "incremental CRC-32 must match single-shot"
        );
    }

    /// Three-way split produces the same result.
    #[test]
    fn three_way_incremental() {
        let single = Crc32::compute(b"123456789");

        let mut crc = Crc32::new();
        crc.update(b"123");
        crc.update(b"456");
        crc.update(b"789");

        assert_eq!(
            crc.finalize(),
            single,
            "three-way incremental must match single-shot"
        );
    }

    /// Reset clears state for recomputation.
    #[test]
    fn reset_and_recompute() {
        let mut crc = Crc32::new();
        crc.update(b"some data");
        let _ = crc.finalize();

        crc.reset();
        crc.update(b"123456789");

        assert_eq!(
            crc.finalize(),
            0xCBF4_3926,
            "reset must restore initial state"
        );
    }

    /// Determinism: identical inputs produce identical outputs.
    #[test]
    fn deterministic() {
        let data = b"determinism check payload";
        let a = Crc32::compute(data);
        let b = Crc32::compute(data);
        assert_eq!(a, b, "CRC-32 must be deterministic");
    }

    /// Multiple updates followed by finalize produce correct result.
    #[test]
    fn multiple_updates_then_finalize() {
        let data = b"The quick brown fox jumps over the lazy dog";

        // Single-shot
        let single = Crc32::compute(data);

        // Byte-by-byte incremental
        let mut crc = Crc32::new();
        for byte in data {
            crc.update(&[*byte]);
        }

        assert_eq!(
            crc.finalize(),
            single,
            "byte-by-byte incremental must match single-shot"
        );
    }

    /// Large data block.
    #[test]
    fn large_data_block() {
        let data: Vec<u8> = (0u8..=255).cycle().take(100_000).collect();
        let result = Crc32::compute(&data);

        // Verify determinism and non-triviality
        let result2 = Crc32::compute(&data);
        assert_eq!(result, result2, "large data CRC-32 must be deterministic");
        assert_ne!(result, 0x0000_0000, "large data CRC-32 must not be zero");
    }

    /// Helper: compute CRC table entry at index `i` using bit-at-a-time algorithm.
    fn compute_crc_table_entry(i: u8) -> u32 {
        const POLYNOMIAL: u32 = 0xEDB8_8320;
        let mut v = i as u32;
        for _ in 0..8 {
            if v & 1 != 0 {
                v = (v >> 1) ^ POLYNOMIAL;
            } else {
                v >>= 1;
            }
        }
        v
    }
}

// =============================================================================
// Fletcher-32 Tests (HDF5 filter ID 3)
// =============================================================================

mod fletcher32 {
    use super::*;

    /// Empty input: both sums remain zero.
    ///
    /// Result = (0 << 16) | 0 = 0x00000000.
    #[test]
    fn empty_input() {
        let result = Fletcher32::compute(&[]);
        assert_eq!(
            result, 0x0000_0000,
            "Fletcher-32 of empty input must be zero"
        );
    }

    /// Single 16-bit LE word [0x01, 0x02] → word = 0x0201.
    ///
    /// s1 = 0x0201, s2 = 0x0201.
    /// Result = 0x0201_0201.
    #[test]
    fn single_word() {
        let result = Fletcher32::compute(&[0x01, 0x02]);
        assert_eq!(
            result, 0x0201_0201,
            "Fletcher-32 of [0x01, 0x02] must be 0x02010201"
        );
    }

    /// Two 16-bit LE words.
    ///
    /// Word 0: 0x0201 → s1 = 0x0201, s2 = 0x0201.
    /// Word 1: 0x0403 → s1 = 0x0201 + 0x0403 = 0x0604,
    ///          s2 = 0x0201 + 0x0604 = 0x0805.
    /// Result = 0x0805_0604.
    #[test]
    fn two_words() {
        let result = Fletcher32::compute(&[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(
            result, 0x0805_0604,
            "Fletcher-32 of two words must be 0x08050604"
        );
    }

    /// Odd-length input: trailing byte padded with 0x00.
    ///
    /// [0x01, 0x02, 0x03]:
    /// Word 0: LE(0x01, 0x02) = 0x0201 → s1 = 0x0201, s2 = 0x0201.
    /// Word 1: pad(0x03, 0x00) = 0x0003 → s1 = 0x0201 + 0x0003 = 0x0204,
    ///          s2 = 0x0201 + 0x0204 = 0x0405.
    /// Result = 0x0405_0204.
    #[test]
    fn odd_length_input() {
        let result = Fletcher32::compute(&[0x01, 0x02, 0x03]);
        assert_eq!(
            result, 0x0405_0204,
            "Fletcher-32 with odd-length input must handle padding"
        );
    }

    /// Single byte padded to 16-bit word.
    ///
    /// [0xFF] → word = 0x00FF.
    /// s1 = 0xFF = 255, s2 = 255.
    /// Result = 0x00FF_00FF.
    #[test]
    fn single_byte_padded() {
        let result = Fletcher32::compute(&[0xFF]);
        assert_eq!(
            result, 0x00FF_00FF,
            "Fletcher-32 of [0xFF] must be 0x00FF00FF"
        );
    }

    /// Modular reduction: sums wrap modulo 65535.
    ///
    /// Two words of value 0xFFFE (= 65534).
    /// After word 0: s1 = 65534, s2 = 65534.
    /// After word 1: s1 = (65534 + 65534) % 65535 = 65533,
    ///               s2 = (65534 + 65533) % 65535 = 65532.
    /// Result = (65532 << 16) | 65533.
    #[test]
    fn modular_reduction() {
        let data = [0xFE, 0xFF, 0xFE, 0xFF]; // Two 0xFFFE words
        let result = Fletcher32::compute(&data);

        let expected_s1: u32 = (65534 + 65534) % 65535; // 65533
        let expected_s2: u32 = (65534 + expected_s1) % 65535; // 65532
        let expected = (expected_s2 << 16) | expected_s1;

        assert_eq!(
            result, expected,
            "Fletcher-32 must correctly reduce modulo 65535"
        );
    }

    /// Incremental update matches single-shot.
    #[test]
    fn incremental_matches_single_shot() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let single = Fletcher32::compute(&data);

        let mut incremental = Fletcher32::new();
        incremental.update(&data[..2]);
        incremental.update(&data[2..4]);
        incremental.update(&data[4..6]);
        incremental.update(&data[6..8]);

        assert_eq!(
            incremental.finalize(),
            single,
            "incremental Fletcher-32 must match single-shot"
        );
    }

    /// Reset clears state.
    #[test]
    fn reset_and_recompute() {
        let data = [0x10, 0x20, 0x30, 0x40];

        let mut hasher = Fletcher32::new();
        hasher.update(&data);
        let first = hasher.finalize();

        hasher.reset();
        hasher.update(&data);
        let second = hasher.finalize();

        assert_eq!(first, second, "reset must allow recomputation");
    }

    /// Determinism.
    #[test]
    fn deterministic() {
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let a = Fletcher32::compute(&data);
        let b = Fletcher32::compute(&data);
        assert_eq!(a, b, "Fletcher-32 must be deterministic");
    }

    /// Large data block.
    #[test]
    fn large_data_block() {
        let data: Vec<u8> = (0u8..=255).cycle().take(100_000).collect();
        let result = Fletcher32::compute(&data);

        // Verify non-triviality
        assert_ne!(
            result, 0x0000_0000,
            "large data Fletcher-32 must not be zero"
        );
    }

    /// All zeros.
    #[test]
    fn all_zeros() {
        let data = vec![0u8; 1024];
        let result = Fletcher32::compute(&data);
        // s1 = sum of 512 zero words = 0
        // s2 = cumulative sum of zeros = 0
        assert_eq!(result, 0x0000_0000, "Fletcher-32 of all zeros must be zero");
    }

    /// All ones (0xFF).
    #[test]
    fn all_ones() {
        let data = vec![0xFFu8; 1024];
        let result = Fletcher32::compute(&data);
        // Each word = 0xFFFF = 65535.
        // s1 = (0 + 65535) % 65535 = 0 after the first word, and remains 0
        // for every subsequent word. s2 = (0 + 0) % 65535 = 0 likewise.
        // Therefore the checksum is zero.
        assert_eq!(
            result, 0x0000_0000,
            "Fletcher-32 of all 0xFF must be zero (65535 mod 65535 = 0)"
        );
    }
}

// =============================================================================
// Lookup3 Tests (Jenkins hash, HDF5 v2 metadata checksums)
// =============================================================================

mod lookup3 {
    use super::*;

    /// Empty input with initval=0 returns `c` without final mix.
    ///
    /// a = b = c = 0xdeadbeef + 0 + 0 = 0xdeadbeef.
    /// Remaining = 0 → return c immediately.
    /// Therefore `hash(b"") == 0xdeadbeef`.
    #[test]
    #[cfg(feature = "alloc")]
    fn empty_input() {
        let result = Lookup3::compute(b"");
        assert_eq!(
            result, 0xdeadbeef,
            "Lookup3 of empty input must be 0xdeadbeef"
        );
    }

    /// Single zero byte.
    #[test]
    #[cfg(feature = "alloc")]
    fn single_zero_byte() {
        // len = 1, initval = 0
        // base = 0xdeadbeef + 1 = 0xdeadbef0
        // a = b = c = 0xdeadbef0
        // remaining = 1 → a += d[0] = 0, so a unchanged
        // final_mix(0xdeadbef0, 0xdeadbef0, 0xdeadbef0)
        let result = Lookup3::compute(b"\x00");
        assert_ne!(
            result, 0xdeadbeef,
            "Lookup3 of single zero must differ from empty"
        );
    }

    /// Determinism.
    #[test]
    #[cfg(feature = "alloc")]
    fn deterministic() {
        let inputs: &[&[u8]] = &[
            b"",
            b"\x00",
            b"a",
            b"ab",
            b"abc",
            b"abcd",
            b"abcde",
            b"abcdefghijkl",
            b"abcdefghijklm",
            b"message digest",
            b"Four score and seven years ago",
        ];

        for input in inputs {
            let a = Lookup3::compute(input);
            let b = Lookup3::compute(input);
            assert_eq!(a, b, "Lookup3 must be deterministic for {:?}", input);
        }
    }

    /// Distinct inputs produce distinct hashes (collision resistance sanity check).
    #[test]
    #[cfg(feature = "alloc")]
    fn distinct_inputs_differ() {
        assert_ne!(Lookup3::compute(b""), Lookup3::compute(b"\x00"));
        assert_ne!(Lookup3::compute(b"a"), Lookup3::compute(b"b"));
        assert_ne!(Lookup3::compute(b"abc"), Lookup3::compute(b"abd"));
        assert_ne!(Lookup3::compute(b"hello"), Lookup3::compute(b"world"));
    }

    /// All remainder lengths (0..=12) produce distinct results.
    #[test]
    #[cfg(feature = "alloc")]
    fn all_remainder_lengths() {
        let mut seen = [0u32; 13];
        for len in 0..=12 {
            let input: Vec<u8> = vec![0xAA; len];
            seen[len] = Lookup3::compute(&input);
        }

        // No two distinct-length inputs should collide
        for i in 0..13 {
            for j in (i + 1)..13 {
                assert_ne!(
                    seen[i], seen[j],
                    "Lookup3 collision between lengths {} and {}",
                    i, j
                );
            }
        }
    }

    /// Input longer than 12 bytes exercises the main mixing loop.
    #[test]
    #[cfg(feature = "alloc")]
    fn longer_than_twelve_bytes() {
        let data = b"abcdefghijklmnopqrstuvwxyz";
        let result = Lookup3::compute(data);
        assert_ne!(
            result, 0xdeadbeef,
            "Lookup3 of long input must differ from empty"
        );
    }

    /// Exactly 12 bytes: one remainder block, no chunk-loop iterations.
    #[test]
    #[cfg(feature = "alloc")]
    fn exactly_twelve_bytes() {
        let data = b"abcdefghijkl";
        assert_eq!(data.len(), 12);
        let result = Lookup3::compute(data);
        assert_ne!(
            result, 0xdeadbeef,
            "Lookup3 of 12 bytes must be non-trivial"
        );
    }

    /// Exactly 24 bytes: two chunks processed by the main loop.
    #[test]
    #[cfg(feature = "alloc")]
    fn exactly_twentyfour_bytes() {
        let data = b"abcdefghijklmnopqrstuvwx";
        assert_eq!(data.len(), 24);
        let result = Lookup3::compute(data);
        let result2 = Lookup3::compute(data);
        assert_eq!(result, result2, "Lookup3 of 24 bytes must be deterministic");
    }

    /// Incremental update through Checksum trait matches direct hash.
    #[test]
    #[cfg(feature = "alloc")]
    fn checksum_trait_matches_direct() {
        use consus_compression::checksum::lookup3::hash;

        let data = b"Four score and seven years ago";
        let direct = hash(data);

        let mut hasher = Lookup3::default();
        hasher.update(data);
        assert_eq!(
            hasher.finalize(),
            direct,
            "Checksum trait must match direct hash"
        );
    }

    /// Incremental update across multiple calls.
    #[test]
    #[cfg(feature = "alloc")]
    fn checksum_trait_incremental() {
        use consus_compression::checksum::lookup3::hash;

        let data = b"Four score and seven years ago";
        let direct = hash(data);

        let mut hasher = Lookup3::default();
        hasher.update(&data[..10]);
        hasher.update(&data[10..20]);
        hasher.update(&data[20..]);

        assert_eq!(
            hasher.finalize(),
            direct,
            "incremental Lookup3 must match direct"
        );
    }

    /// Reset clears accumulated state.
    #[test]
    #[cfg(feature = "alloc")]
    fn checksum_trait_reset() {
        let mut hasher = Lookup3::default();
        hasher.update(b"some data");
        hasher.reset();
        hasher.update(b"");
        assert_eq!(hasher.finalize(), 0xdeadbeef, "reset must clear state");
    }

    /// Large data block.
    #[test]
    #[cfg(feature = "alloc")]
    fn large_data_block() {
        let data: Vec<u8> = (0u8..=255).cycle().take(100_000).collect();
        let result = Lookup3::compute(&data);
        let result2 = Lookup3::compute(&data);
        assert_eq!(
            result, result2,
            "Lookup3 of large data must be deterministic"
        );
        assert_ne!(
            result, 0xdeadbeef,
            "Lookup3 of large data must be non-trivial"
        );
    }
}

// =============================================================================
// Checksum Composition Invariants
// =============================================================================

mod composition_invariants {
    use super::*;

    /// CRC-32 incremental equivalence: two updates equals one update of concatenation.
    #[test]
    fn crc32_incremental_equivalence() {
        let a = b"part one";
        let b = b"part two";
        let concatenated: Vec<u8> = a.iter().chain(b.iter()).copied().collect();

        let single = Crc32::compute(&concatenated);

        let mut incremental = Crc32::new();
        incremental.update(a);
        incremental.update(b);

        assert_eq!(
            incremental.finalize(),
            single,
            "CRC-32 incremental equivalence invariant violated"
        );
    }

    /// Fletcher-32 incremental equivalence.
    #[test]
    fn fletcher32_incremental_equivalence() {
        let a = b"alpha";
        let b = b"beta";
        let concatenated: Vec<u8> = a.iter().chain(b.iter()).copied().collect();

        let single = Fletcher32::compute(&concatenated);

        let mut incremental = Fletcher32::new();
        incremental.update(a);
        incremental.update(b);

        assert_eq!(
            incremental.finalize(),
            single,
            "Fletcher-32 incremental equivalence invariant violated"
        );
    }

    /// Lookup3 incremental equivalence (requires alloc).
    #[test]
    #[cfg(feature = "alloc")]
    fn lookup3_incremental_equivalence() {
        use consus_compression::checksum::lookup3::hash;

        let a = b"first segment";
        let b = b"second segment";
        let concatenated: Vec<u8> = a.iter().chain(b.iter()).copied().collect();

        let single = hash(&concatenated);

        let mut hasher = Lookup3::default();
        hasher.update(a);
        hasher.update(b);

        assert_eq!(
            hasher.finalize(),
            single,
            "Lookup3 incremental equivalence invariant violated"
        );
    }

    /// CRC-32 reset produces identical result to fresh instance.
    #[test]
    fn crc32_reset_invariant() {
        let data = b"test data for reset";

        let mut crc1 = Crc32::new();
        crc1.update(data);
        let result1 = crc1.finalize();

        crc1.reset();
        crc1.update(data);
        let result2 = crc1.finalize();

        let fresh_result = Crc32::compute(data);

        assert_eq!(
            result1, fresh_result,
            "initial computation must match fresh"
        );
        assert_eq!(result2, fresh_result, "reset computation must match fresh");
    }

    /// Fletcher-32 reset produces identical result to fresh instance.
    #[test]
    fn fletcher32_reset_invariant() {
        let data = b"test data for reset";

        let mut f1 = Fletcher32::new();
        f1.update(data);
        let result1 = f1.finalize();

        f1.reset();
        f1.update(data);
        let result2 = f1.finalize();

        let fresh_result = Fletcher32::compute(data);

        assert_eq!(
            result1, fresh_result,
            "initial computation must match fresh"
        );
        assert_eq!(result2, fresh_result, "reset computation must match fresh");
    }

    /// Different checksums produce different values for non-trivial input.
    #[test]
    fn different_algorithms_differ() {
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();

        let crc = Crc32::compute(&data);
        let fletcher = Fletcher32::compute(&data);

        // These should almost certainly differ for non-trivial input
        assert_ne!(
            crc, fletcher,
            "CRC-32 and Fletcher-32 should produce different values"
        );
    }

    /// All-zero input.
    #[test]
    fn all_zeros_input() {
        let zeros = vec![0u8; 1024];

        let crc = Crc32::compute(&zeros);
        let fletcher = Fletcher32::compute(&zeros);

        // CRC of zeros is non-trivial due to pre/post-conditioning
        assert_ne!(crc, 0x0000_0000, "CRC-32 of zeros must be non-zero");

        // Fletcher of zeros is zero (sum of zero words)
        assert_eq!(fletcher, 0x0000_0000, "Fletcher-32 of zeros must be zero");
    }

    /// All-ones input.
    #[test]
    fn all_ones_input() {
        let ones = vec![0xFFu8; 1024];

        let crc = Crc32::compute(&ones);
        let fletcher = Fletcher32::compute(&ones);

        assert_ne!(crc, 0x0000_0000, "CRC-32 of ones must be non-zero");
        // 0xFFFF mod 65535 = 0 for every word, so both s1 and s2 stay zero.
        assert_eq!(
            fletcher, 0x0000_0000,
            "Fletcher-32 of all 0xFF must be zero (65535 mod 65535 = 0)"
        );
    }
}
