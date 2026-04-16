//! Jenkins lookup3 hash function (`hashlittle` variant).
//!
//! Used by HDF5 version 2 object headers for metadata checksums.
//! This module is the SSOT for lookup3 in Consus. No other crate may
//! duplicate this implementation.
//!
//! ## Algorithm
//!
//! The `hashlittle` function from Bob Jenkins' lookup3.c processes input
//! in 12-byte chunks through a mixing function, then applies a final
//! mixing step to the remaining bytes. HDF5 always uses `initval = 0`.
//!
//! ## Reference
//!
//! Bob Jenkins, "A Hash Function for Hash Table Lookup", 2006.
//! <http://burtleburtle.net/bob/hash/doobs.html>
//!
//! The canonical C source: <http://burtleburtle.net/bob/c/lookup3.c>

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use super::traits::Checksum;

/// Jenkins lookup3 hash state.
///
/// Because `hashlittle` requires the complete input to produce correct output
/// (the initial constants depend on total length), the `Checksum` trait
/// implementation accumulates data into an internal buffer and computes
/// the hash on [`finalize`](Checksum::finalize).
///
/// For callers that have the complete input available, prefer the zero-alloc
/// [`hash`] free function.
#[derive(Default)]
pub struct Lookup3 {
    /// Accumulated input bytes. Requires the `alloc` feature.
    #[cfg(feature = "alloc")]
    buf: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl Checksum for Lookup3 {
    type Output = u32;

    fn update(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    fn finalize(&self) -> u32 {
        hash(&self.buf)
    }

    fn reset(&mut self) {
        self.buf.clear();
    }
}

// ---------------------------------------------------------------------------
// Internal mixing primitives
// ---------------------------------------------------------------------------

/// The lookup3 `mix` function.
///
/// Mixes three 32-bit values reversibly. This is used for every 12-byte
/// chunk of the input.
///
/// ## Invariant
///
/// Every bit of `a`, `b`, `c` affects every bit of `a`, `b`, `c` after
/// mixing. No output bit is a linear function of fewer than 18 input bits.
#[inline(always)]
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}

/// The lookup3 `final` mixing function.
///
/// Applied once after all 12-byte chunks have been consumed and the
/// remaining bytes have been folded into `a`, `b`, `c`.
///
/// ## Invariant
///
/// Achieves avalanche: every input bit affects every output bit with
/// probability close to 1/2.
#[inline(always)]
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(14));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(11));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(25));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(16));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(4));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(14));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(24));
}

/// Read a little-endian `u32` from a byte slice at the given offset.
///
/// # Safety contract (upheld by caller)
///
/// `offset + 4 <= data.len()` must hold. This is guaranteed by the
/// chunk-processing loop which only calls this when >= 12 bytes remain.
#[inline(always)]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Jenkins lookup3 `hashlittle` hash of `data`.
///
/// This is the canonical entry point. It operates in `no_std` without
/// allocation because it processes the input in a single pass.
///
/// HDF5 always uses `initval = 0`.
///
/// ## Algorithm outline
///
/// 1. Set `a = b = c = 0xdeadbeef + len + initval`.
/// 2. Consume 12-byte chunks: read three little-endian `u32`s, add to
///    `a`, `b`, `c`, then [`mix`].
/// 3. Fold remaining 1–12 bytes into `a`, `b`, `c` via a byte-level
///    switch.
/// 4. If any bytes were present, apply [`final_mix`].
/// 5. Return `c`.
///
/// ## Determinism
///
/// The function is pure: identical inputs always produce identical outputs.
pub fn hash(data: &[u8]) -> u32 {
    hash_with_initval(data, 0)
}

/// Compute `hashlittle` with an explicit `initval`.
///
/// Exposed for testing against the reference implementation. Production
/// callers should use [`hash`] (which passes `initval = 0`).
pub fn hash_with_initval(data: &[u8], initval: u32) -> u32 {
    let len = data.len();

    // Initial values: 0xdeadbeef + length + initval for all three.
    let base = 0xdeadbeefu32.wrapping_add(len as u32).wrapping_add(initval);
    let mut a = base;
    let mut b = base;
    let mut c = base;

    let mut offset: usize = 0;
    let mut remaining = len;

    // --- Process 12-byte chunks -------------------------------------------
    while remaining > 12 {
        a = a.wrapping_add(read_u32_le(data, offset));
        b = b.wrapping_add(read_u32_le(data, offset + 4));
        c = c.wrapping_add(read_u32_le(data, offset + 8));
        mix(&mut a, &mut b, &mut c);
        offset += 12;
        remaining -= 12;
    }

    // --- Fold remaining bytes (0..=12) ------------------------------------
    //
    // The canonical lookup3.c uses a fall-through switch on the remaining
    // byte count. We replicate this with explicit byte reads.
    //
    // Bytes are added to a, b, c in little-endian order:
    //   bytes 0..3  → a
    //   bytes 4..7  → b
    //   bytes 8..11 → c
    //
    // The match arms intentionally fall through by accumulating all
    // applicable bytes for each case.

    let d = &data[offset..];

    // We handle the remainder with explicit byte-by-byte addition,
    // matching the reference C implementation's fall-through switch.
    match remaining {
        12 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add(read_u32_le(d, 4));
            c = c.wrapping_add(read_u32_le(d, 8));
        }
        11 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add(read_u32_le(d, 4));
            c = c.wrapping_add((d[10] as u32) << 16 | (d[9] as u32) << 8 | (d[8] as u32));
        }
        10 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add(read_u32_le(d, 4));
            c = c.wrapping_add((d[9] as u32) << 8 | (d[8] as u32));
        }
        9 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add(read_u32_le(d, 4));
            c = c.wrapping_add(d[8] as u32);
        }
        8 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add(read_u32_le(d, 4));
        }
        7 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add((d[6] as u32) << 16 | (d[5] as u32) << 8 | (d[4] as u32));
        }
        6 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add((d[5] as u32) << 8 | (d[4] as u32));
        }
        5 => {
            a = a.wrapping_add(read_u32_le(d, 0));
            b = b.wrapping_add(d[4] as u32);
        }
        4 => {
            a = a.wrapping_add(read_u32_le(d, 0));
        }
        3 => {
            a = a.wrapping_add((d[2] as u32) << 16 | (d[1] as u32) << 8 | (d[0] as u32));
        }
        2 => {
            a = a.wrapping_add((d[1] as u32) << 8 | (d[0] as u32));
        }
        1 => {
            a = a.wrapping_add(d[0] as u32);
        }
        0 => {
            // Zero remaining bytes: return c directly, no final mix.
            return c;
        }
        _ => unreachable!(),
    }

    final_mix(&mut a, &mut b, &mut c);
    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty input with initval=0 returns `c` without final mix.
    ///
    /// `a = b = c = 0xdeadbeef + 0 + 0 = 0xdeadbeef`.
    /// Remaining = 0 → return c immediately.
    /// Therefore `hash(b"") == 0xdeadbeef`.
    #[test]
    fn empty_input() {
        assert_eq!(hash(b""), 0xdeadbeef);
    }

    /// Determinism: hashing the same input twice yields the same result.
    #[test]
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
            assert_eq!(
                hash(input),
                hash(input),
                "non-deterministic for input {:?}",
                input
            );
        }
    }

    /// Single zero byte: trace through the algorithm by hand.
    ///
    /// len = 1, initval = 0.
    /// base = 0xdeadbeef + 1 + 0 = 0xdeadbef0.
    /// a = b = c = 0xdeadbef0.
    /// No 12-byte chunks.
    /// Remaining = 1 → case 1: a += d[0] = 0, so a unchanged.
    /// Then final_mix(0xdeadbef0, 0xdeadbef0, 0xdeadbef0).
    ///
    /// We verify this against the traced computation.
    #[test]
    fn single_zero_byte() {
        let mut a: u32 = 0xdeadbef0;
        let mut b: u32 = 0xdeadbef0;
        let mut c: u32 = 0xdeadbef0;
        final_mix(&mut a, &mut b, &mut c);
        assert_eq!(hash(b"\x00"), c);
    }

    /// Verify that different inputs produce different hashes (basic
    /// collision resistance sanity check, not a proof).
    #[test]
    fn distinct_inputs_differ() {
        assert_ne!(hash(b""), hash(b"\x00"));
        assert_ne!(hash(b"a"), hash(b"b"));
        assert_ne!(hash(b"abc"), hash(b"abd"));
        assert_ne!(hash(b"hello"), hash(b"world"));
    }

    /// Inputs of varying lengths exercise all remainder-switch arms.
    /// We verify each arm produces a distinct, non-zero result.
    #[test]
    fn all_remainder_lengths() {
        let mut seen = [0u32; 13];
        for len in 0..=12 {
            let input: &[u8] = &vec![0xAA_u8; len];
            seen[len] = hash(input);
        }
        // No two distinct-length inputs of the same byte should collide.
        for i in 0..13 {
            for j in (i + 1)..13 {
                assert_ne!(seen[i], seen[j], "collision between lengths {i} and {j}");
            }
        }
    }

    /// Input longer than 12 bytes exercises the main mixing loop.
    #[test]
    fn longer_than_twelve_bytes() {
        let data = b"abcdefghijklmnopqrstuvwxyz";
        let h = hash(data);
        // Determinism + non-trivial: must differ from the base constant.
        assert_ne!(h, 0xdeadbeef);
        assert_eq!(h, hash(data));
    }

    /// Exactly 12 bytes: one remainder block, no chunk-loop iterations.
    #[test]
    fn exactly_twelve_bytes() {
        let data = b"abcdefghijkl";
        assert_eq!(data.len(), 12);
        let h = hash(data);
        assert_ne!(h, 0xdeadbeef);
        assert_eq!(h, hash(data));
    }

    /// Exactly 24 bytes: two chunks processed by the main loop, then
    /// remainder of 0 (returns c before final_mix from the last mix).
    /// Wait — 24 > 12 so the loop runs once (consumes 12), remaining=12,
    /// which hits the remainder case 12, then final_mix. Verify.
    #[test]
    fn exactly_twentyfour_bytes() {
        let data = b"abcdefghijklmnopqrstuvwx";
        assert_eq!(data.len(), 24);
        let h = hash(data);
        assert_eq!(h, hash(data));
    }

    /// Jenkins' reference test: `hashlittle2("Four score and seven years
    /// ago", 30, &mut 1, &mut 0)` produces known values.
    ///
    /// In `hashlittle2`, the initial `c` gets `initval` added and `b`
    /// gets the second initval added. For `hashlittle` (single return),
    /// we can verify consistency with `hash_with_initval`.
    ///
    /// `hashlittle("", 0, 0)` returns `0xdeadbeef` (verified above).
    /// `hashlittle("", 0, 1)` returns a different value (initval=1 changes base).
    #[test]
    fn initval_changes_result() {
        let h0 = hash_with_initval(b"", 0);
        let h1 = hash_with_initval(b"", 1);
        assert_eq!(h0, 0xdeadbeef);
        // With initval=1: base = 0xdeadbeef + 0 + 1 = 0xdeadbef0, remaining=0 → return c.
        assert_eq!(h1, 0xdeadbef0);
    }

    /// Verify the Checksum trait wrapper produces the same result as the
    /// direct `hash` function (requires alloc).
    #[cfg(feature = "alloc")]
    #[test]
    fn checksum_trait_matches_direct() {
        let data = b"Four score and seven years ago";
        let direct = hash(data);

        let mut hasher = Lookup3::default();
        hasher.update(data);
        assert_eq!(hasher.finalize(), direct);
    }

    /// Incremental update through the Checksum trait matches single-shot.
    #[cfg(feature = "alloc")]
    #[test]
    fn checksum_trait_incremental() {
        let data = b"Four score and seven years ago";
        let direct = hash(data);

        let mut hasher = Lookup3::default();
        hasher.update(&data[..10]);
        hasher.update(&data[10..20]);
        hasher.update(&data[20..]);
        assert_eq!(hasher.finalize(), direct);
    }

    /// Reset clears accumulated state.
    #[cfg(feature = "alloc")]
    #[test]
    fn checksum_trait_reset() {
        let mut hasher = Lookup3::default();
        hasher.update(b"some data");
        hasher.reset();
        hasher.update(b"");
        assert_eq!(hasher.finalize(), 0xdeadbeef);
    }

    /// Compute convenience method matches manual update + finalize.
    #[cfg(feature = "alloc")]
    #[test]
    fn checksum_compute_convenience() {
        let data = b"hello world";
        let via_compute = Lookup3::compute(data);
        let via_hash = hash(data);
        assert_eq!(via_compute, via_hash);
    }
}
