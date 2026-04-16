//! Shared test fixtures for consus-compression tests.
//!
//! Provides reproducible data generators for testing compression
//! algorithms and filter pipelines. All generators use deterministic
//! patterns to ensure test reproducibility.
//!
//! ## Generators
//!
//! - `random_data`: Pseudo-random bytes from a fixed seed (reproducible)
//! - `gradient_data`: Linear gradient pattern 0, 1, 2, ..., 255, 0, 1, ...
//! - `zeroes`: All-zero bytes
//! - `ones`: All-0xFF bytes

use std::cell::RefCell;

thread_local! {
    /// Thread-local PRNG state for reproducible pseudo-random generation.
    /// Uses a simple Linear Congruential Generator (LCG) with constants
    /// from Numerical Recipes.
    static RNG_STATE: RefCell<u64> = RefCell::new(0x123456789ABCDEF0);
}

/// Advance the LCG and return the next random u64.
fn next_u64() -> u64 {
    RNG_STATE.with(|state| {
        let mut s = state.borrow_mut();
        // Numerical Recipes LCG constants
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    })
}

/// Reset the PRNG to the initial seed for reproducible sequences.
pub fn reset_seed() {
    RNG_STATE.with(|state| {
        *state.borrow_mut() = 0x123456789ABCDEF0;
    });
}

/// Generate reproducible pseudo-random bytes.
///
/// Uses a deterministic LCG (Linear Congruential Generator) with a fixed seed.
/// Call `reset_seed()` before generating to ensure the same sequence each time.
///
/// # Example
///
/// ```
/// use consus_compression_tests::fixtures::{random_data, reset_seed};
///
/// reset_seed();
/// let data1 = random_data(1024);
/// reset_seed();
/// let data2 = random_data(1024);
/// assert_eq!(data1, data2, "same seed produces same sequence");
/// ```
pub fn random_data(size: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(size);

    let mut remaining = size;
    while remaining > 0 {
        let rand_val = next_u64();
        let bytes_to_take = remaining.min(8);
        result.extend_from_slice(&rand_val.to_ne_bytes()[..bytes_to_take]);
        remaining -= bytes_to_take;
    }

    result.truncate(size);
    result
}

/// Generate a gradient pattern: 0, 1, 2, ..., 255, 0, 1, 2, ...
///
/// This pattern is highly compressible by all LZ-family and entropy coders,
/// making it ideal for verifying that compression actually reduces size.
///
/// # Example
///
/// ```
/// use consus_compression_tests::fixtures::gradient_data;
///
/// let data = gradient_data(10);
/// assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
pub fn gradient_data(size: usize) -> Vec<u8> {
    (0u8..=255).cycle().take(size).collect()
}

/// Generate a vector of all-zero bytes.
///
/// All-zero data is the best case for run-length encoding and most
/// compression algorithms. Useful for testing compression of sparse data.
///
/// # Example
///
/// ```
/// use consus_compression_tests::fixtures::zeroes;
///
/// let data = zeroes(1024);
/// assert_eq!(data.len(), 1024);
/// assert!(data.iter().all(|&b| b == 0));
/// ```
pub fn zeroes(size: usize) -> Vec<u8> {
    vec![0u8; size]
}

/// Generate a vector of all-0xFF bytes.
///
/// All-ones data is another compressible pattern that tests handling
/// of high-bit values and sign-extension behavior.
///
/// # Example
///
/// ```
/// use consus_compression_tests::fixtures::ones;
///
/// let data = ones(1024);
/// assert_eq!(data.len(), 1024);
/// assert!(data.iter().all(|&b| b == 0xFF));
/// ```
pub fn ones(size: usize) -> Vec<u8> {
    vec![0xFFu8; size]
}

/// Generate alternating bytes: 0x00, 0xFF, 0x00, 0xFF, ...
///
/// Tests handling of high-frequency oscillating patterns.
pub fn alternating_data(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| if i % 2 == 0 { 0x00 } else { 0xFF })
        .collect()
}

/// Generate a repeating pattern of incrementing byte pairs.
///
/// Pattern: [0, 1], [2, 3], [4, 5], ..., [254, 255], [0, 1], ...
/// Useful for testing shuffle filters with 2-byte elements.
pub fn paired_gradient_data(size: usize) -> Vec<u8> {
    (0u8..=255)
        .flat_map(|b| [b, b.wrapping_add(1)])
        .cycle()
        .take(size)
        .collect()
}

/// Generate 4-byte element data suitable for shuffle testing.
///
/// Creates elements where each element's bytes increment together,
/// producing correlated bytes within each element position after shuffle.
pub fn quad_gradient_data(size: usize) -> Vec<u8> {
    (0u8..=255)
        .flat_map(|b| [b, b.wrapping_add(1), b.wrapping_add(2), b.wrapping_add(3)])
        .cycle()
        .take(size)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_data_reproducible() {
        reset_seed();
        let data1 = random_data(1024);
        reset_seed();
        let data2 = random_data(1024);
        assert_eq!(data1, data2, "same seed must produce same sequence");
    }

    #[test]
    fn random_data_different_sizes() {
        reset_seed();
        let data0 = random_data(0);
        let data1 = random_data(1);
        let data7 = random_data(7);
        let data8 = random_data(8);
        let data100 = random_data(100);

        assert_eq!(data0.len(), 0);
        assert_eq!(data1.len(), 1);
        assert_eq!(data7.len(), 7);
        assert_eq!(data8.len(), 8);
        assert_eq!(data100.len(), 100);
    }

    #[test]
    fn gradient_data_correct_pattern() {
        let data = gradient_data(260);
        assert_eq!(data.len(), 260);

        // First 256 bytes: 0, 1, 2, ..., 255
        for i in 0..256 {
            assert_eq!(data[i], i as u8, "gradient byte {} mismatch", i);
        }

        // Next 4 bytes: 0, 1, 2, 3
        assert_eq!(data[256], 0);
        assert_eq!(data[257], 1);
        assert_eq!(data[258], 2);
        assert_eq!(data[259], 3);
    }

    #[test]
    fn zeroes_all_zero() {
        let data = zeroes(1024);
        assert!(data.iter().all(|&b| b == 0), "all bytes must be zero");
    }

    #[test]
    fn ones_all_ones() {
        let data = ones(1024);
        assert!(data.iter().all(|&b| b == 0xFF), "all bytes must be 0xFF");
    }

    #[test]
    fn alternating_pattern_correct() {
        let data = alternating_data(10);
        assert_eq!(
            data,
            vec![0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF]
        );
    }

    #[test]
    fn paired_gradient_correct() {
        let data = paired_gradient_data(8);
        // [0,1], [2,3], [4,5], [6,7]
        assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn quad_gradient_correct() {
        let data = quad_gradient_data(12);
        // [0,1,2,3], [4,5,6,7], [8,9,10,11]
        assert_eq!(data, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }
}
