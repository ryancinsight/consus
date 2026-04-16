//! Byte shuffle filter (HDF5 filter ID 2).
//!
//! Rearranges bytes to group corresponding byte positions of each element
//! together, improving compression ratio for typed data by exploiting
//! byte-level correlation within fixed-width elements.
//!
//! ## Algorithm (Forward — Shuffle)
//!
//! For `N` elements of `typesize` `T` bytes each, the shuffle permutation
//! σ : [0, N·T) → [0, N·T) is defined by:
//!
//! ```text
//! output[j · N + i] = input[i · T + j]
//! ```
//!
//! where `i ∈ [0, N)` indexes the element and `j ∈ [0, T)` indexes the
//! byte position within an element.
//!
//! ## Algorithm (Reverse — Unshuffle)
//!
//! The inverse permutation σ⁻¹ reconstructs the original interleaved layout:
//!
//! ```text
//! output[i · T + j] = input[j · N + i]
//! ```
//!
//! ## Theorem
//!
//! The shuffle/unshuffle pair forms an involution:
//!
//! ```text
//! unshuffle(shuffle(data, T), T) = data
//! ```
//!
//! for any `data` whose length is divisible by `T`.
//!
//! ## Proof
//!
//! Let `σ(k) = (k mod T) · N + (k / T)` be the shuffle permutation on
//! linear index `k`, where `k = i · T + j` gives `i = k / T`, `j = k mod T`.
//!
//! Let `σ⁻¹(k) = (k mod N) · T + (k / N)` be the unshuffle permutation on
//! linear index `k`, where `k = j · N + i` gives `j = k / N`, `i = k mod N`.
//!
//! Compose:
//!
//! ```text
//! σ⁻¹(σ(k)) = σ⁻¹((k mod T) · N + (k / T))
//!            = ((k / T) mod N) · T + ((k mod T) · N + (k / T)) / N
//! ```
//!
//! Since `i = k / T ∈ [0, N)`, we have `(k / T) mod N = i` and
//! `((k mod T) · N + i) / N = k mod T` (because `i < N`).
//!
//! Therefore `σ⁻¹(σ(k)) = i · T + j = k`. □

use alloc::vec;
use alloc::vec::Vec;

use consus_core::{Error, Result};

use super::traits::{Filter, FilterDirection};

/// Byte shuffle filter.
///
/// Groups corresponding byte positions of fixed-width elements together,
/// transforming `[e0_b0, e0_b1, ..., e1_b0, e1_b1, ...]` into
/// `[e0_b0, e1_b0, ..., e0_b1, e1_b1, ...]`.
///
/// This byte transposition concentrates similar values (e.g., all most-
/// significant bytes) into contiguous runs, which downstream entropy
/// coders and LZ-family compressors exploit for higher compression ratios.
///
/// ## HDF5
///
/// Corresponds to HDF5 filter ID 2 (`H5Z_FILTER_SHUFFLE`).
#[derive(Debug, Clone)]
pub struct ShuffleFilter {
    /// Element size in bytes. Must be ≥ 1.
    typesize: usize,
}

impl ShuffleFilter {
    /// Create a new shuffle filter for elements of `typesize` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `typesize == 0`. A zero-byte element has no bytes to
    /// shuffle, and would cause division by zero in the permutation.
    #[must_use]
    pub fn new(typesize: usize) -> Self {
        assert!(typesize > 0, "ShuffleFilter: typesize must be > 0");
        Self { typesize }
    }

    /// Return the element size this filter was configured with.
    #[must_use]
    pub fn typesize(&self) -> usize {
        self.typesize
    }

    /// Forward shuffle: group bytes by byte-position across elements.
    ///
    /// For `N` elements of `T` bytes each:
    ///   `output[j * N + i] = input[i * T + j]`
    fn shuffle(&self, data: &[u8]) -> Vec<u8> {
        let t = self.typesize;
        let n = data.len() / t;
        let mut output = vec![0u8; data.len()];
        for i in 0..n {
            for j in 0..t {
                output[j * n + i] = data[i * t + j];
            }
        }
        output
    }

    /// Reverse shuffle: reconstruct original interleaved byte layout.
    ///
    /// For `N` elements of `T` bytes each:
    ///   `output[i * T + j] = input[j * N + i]`
    fn unshuffle(&self, data: &[u8]) -> Vec<u8> {
        let t = self.typesize;
        let n = data.len() / t;
        let mut output = vec![0u8; data.len()];
        for i in 0..n {
            for j in 0..t {
                output[i * t + j] = data[j * n + i];
            }
        }
        output
    }
}

impl Filter for ShuffleFilter {
    fn name(&self) -> &str {
        "shuffle"
    }

    /// Apply the shuffle filter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFormat`] if `data.len()` is not divisible
    /// by the configured `typesize`.
    fn apply(&self, direction: FilterDirection, data: &[u8]) -> Result<Vec<u8>> {
        let t = self.typesize;

        if data.len() % t != 0 {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "shuffle: data length {} is not divisible by typesize {}",
                    data.len(),
                    t,
                ),
            });
        }

        // typesize == 1: every element is a single byte; shuffle is identity.
        if t == 1 {
            return Ok(data.to_vec());
        }

        // Empty data is trivially valid.
        if data.is_empty() {
            return Ok(Vec::new());
        }

        match direction {
            FilterDirection::Forward => Ok(self.shuffle(data)),
            FilterDirection::Reverse => Ok(self.unshuffle(data)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip with typesize=4 on 16 bytes (4 elements of 4 bytes).
    ///
    /// Verifies `unshuffle(shuffle(data)) == data`.
    #[test]
    fn round_trip_typesize_4() {
        let filter = ShuffleFilter::new(4);
        // 4 elements × 4 bytes = 16 bytes, representing f32-like data.
        let input: Vec<u8> = (0x01..=0x10).collect();
        assert_eq!(input.len(), 16);

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");
        let restored = filter
            .apply(FilterDirection::Reverse, &shuffled)
            .expect("reverse must succeed");

        assert_eq!(restored, input, "round-trip must recover original data");
    }

    /// Round-trip with typesize=8 on 32 bytes (4 elements of 8 bytes).
    ///
    /// Verifies `unshuffle(shuffle(data)) == data` for f64-like data.
    #[test]
    fn round_trip_typesize_8() {
        let filter = ShuffleFilter::new(8);
        // 4 elements × 8 bytes = 32 bytes.
        let input: Vec<u8> = (0x20..=0x3F).collect();
        assert_eq!(input.len(), 32);

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");
        let restored = filter
            .apply(FilterDirection::Reverse, &shuffled)
            .expect("reverse must succeed");

        assert_eq!(restored, input, "round-trip must recover original data");
    }

    /// Verify exact byte positions after forward shuffle.
    ///
    /// Input: 4 elements of typesize 4:
    ///   e0=[0x01,0x02,0x03,0x04], e1=[0x05,0x06,0x07,0x08],
    ///   e2=[0x09,0x0A,0x0B,0x0C], e3=[0x0D,0x0E,0x0F,0x10]
    ///
    /// N=4, T=4. Permutation: output[j*4+i] = input[i*4+j].
    ///
    /// Expected output groups by byte position:
    ///   byte-0 of each element: [0x01, 0x05, 0x09, 0x0D]
    ///   byte-1 of each element: [0x02, 0x06, 0x0A, 0x0E]
    ///   byte-2 of each element: [0x03, 0x07, 0x0B, 0x0F]
    ///   byte-3 of each element: [0x04, 0x08, 0x0C, 0x10]
    #[test]
    fn shuffle_byte_positions_typesize_4() {
        let filter = ShuffleFilter::new(4);
        let input: Vec<u8> = (0x01..=0x10).collect();

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let expected: Vec<u8> = vec![
            0x01, 0x05, 0x09, 0x0D, // byte-0 of elements 0..3
            0x02, 0x06, 0x0A, 0x0E, // byte-1 of elements 0..3
            0x03, 0x07, 0x0B, 0x0F, // byte-2 of elements 0..3
            0x04, 0x08, 0x0C, 0x10, // byte-3 of elements 0..3
        ];

        assert_eq!(
            shuffled, expected,
            "shuffled bytes must match analytical permutation"
        );
    }

    /// typesize=1 is identity: single-byte elements have nothing to transpose.
    #[test]
    fn typesize_1_is_identity() {
        let filter = ShuffleFilter::new(1);
        let input: Vec<u8> = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE];

        let forward = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");
        assert_eq!(forward, input, "typesize=1 forward must be identity");

        let reverse = filter
            .apply(FilterDirection::Reverse, &input)
            .expect("reverse must succeed");
        assert_eq!(reverse, input, "typesize=1 reverse must be identity");
    }

    /// Error when data length is not divisible by typesize.
    #[test]
    fn error_on_misaligned_length() {
        let filter = ShuffleFilter::new(4);
        // 7 bytes is not divisible by 4.
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

        let result = filter.apply(FilterDirection::Forward, &input);
        assert!(result.is_err(), "misaligned input must produce an error");

        // Verify the error is InvalidFormat with a descriptive message.
        match result.unwrap_err() {
            Error::InvalidFormat { message } => {
                assert!(
                    message.contains("7"),
                    "error message must contain the actual data length '7', got: {message}"
                );
                assert!(
                    message.contains("4"),
                    "error message must contain the typesize '4', got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Constructor panics on typesize=0.
    #[test]
    #[should_panic(expected = "typesize must be > 0")]
    fn panics_on_typesize_zero() {
        let _filter = ShuffleFilter::new(0);
    }

    /// Empty input is valid for any typesize and returns empty output.
    #[test]
    fn empty_input_returns_empty() {
        let filter = ShuffleFilter::new(4);
        let input: Vec<u8> = Vec::new();

        let forward = filter
            .apply(FilterDirection::Forward, &input)
            .expect("empty forward must succeed");
        assert!(forward.is_empty(), "forward of empty must be empty");

        let reverse = filter
            .apply(FilterDirection::Reverse, &input)
            .expect("empty reverse must succeed");
        assert!(reverse.is_empty(), "reverse of empty must be empty");
    }

    /// Single element: shuffle of one element is identity (no other elements
    /// to interleave with).
    #[test]
    fn single_element_is_identity() {
        let filter = ShuffleFilter::new(4);
        let input: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        // N=1, T=4: output[j*1+0] = input[0*4+j] = input[j], so output == input.
        assert_eq!(shuffled, input, "single element shuffle must be identity");
    }

    /// Verify exact byte positions after forward shuffle with typesize=2.
    ///
    /// Input: 4 elements of 2 bytes: [0x01,0x02, 0x03,0x04, 0x05,0x06, 0x07,0x08]
    /// N=4, T=2.
    /// Expected: byte-0 group [0x01,0x03,0x05,0x07], byte-1 group [0x02,0x04,0x06,0x08]
    #[test]
    fn shuffle_byte_positions_typesize_2() {
        let filter = ShuffleFilter::new(2);
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let expected: Vec<u8> = vec![
            0x01, 0x03, 0x05, 0x07, // byte-0 of elements 0..3
            0x02, 0x04, 0x06, 0x08, // byte-1 of elements 0..3
        ];

        assert_eq!(
            shuffled, expected,
            "shuffled bytes must match analytical permutation for typesize=2"
        );
    }
}
