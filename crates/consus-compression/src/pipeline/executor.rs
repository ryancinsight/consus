//! Ordered filter pipeline executor.
//!
//! Applies a sequence of [`Filter`] instances to chunk data in a defined
//! order. The pipeline is the composition mechanism for HDF5/Zarr filter
//! chains where chunks pass through shuffle → compression → checksum, etc.
//!
//! ## Invariant
//!
//! For pipeline `P = [F₁, F₂, ..., Fₙ]`:
//!
//! ```text
//! P.execute(Reverse, P.execute(Forward, data)?) == data
//! ```
//!
//! ## Proof
//!
//! Forward application computes:
//!
//! ```text
//! result = Fₙ(... F₂(F₁(data)))
//! ```
//!
//! Reverse application iterates in reverse order, computing:
//!
//! ```text
//! result = F₁⁻¹(F₂⁻¹(... Fₙ⁻¹(data)))
//! ```
//!
//! Composing reverse after forward:
//!
//! ```text
//! F₁⁻¹(F₂⁻¹(... Fₙ⁻¹(Fₙ(... F₂(F₁(data))))))
//! ```
//!
//! By the [`Filter`] invertibility invariant (`Fₖ⁻¹(Fₖ(x)) = x`), the
//! innermost pair cancels: `Fₙ⁻¹(Fₙ(...)) = ...`. Applying this
//! cancellation inductively from the inside out yields `data`. □

use alloc::boxed::Box;
use alloc::vec::Vec;

use consus_core::Result;

use super::traits::{Filter, FilterDirection};

/// Ordered filter pipeline executor.
///
/// Holds a sequence of [`Filter`] trait objects and applies them in the
/// correct order for the given [`FilterDirection`]:
///
/// - **Forward** (write path): filters applied first-to-last.
/// - **Reverse** (read path): filters applied last-to-first.
///
/// ## Construction
///
/// Filters are added via [`push`](Self::push) in the logical forward order.
/// The executor manages reversal internally during [`execute`](Self::execute).
pub struct FilterPipeline {
    /// Filters in forward (write-path) application order.
    filters: Vec<Box<dyn Filter>>,
}

impl FilterPipeline {
    /// Create an empty filter pipeline.
    ///
    /// An empty pipeline acts as the identity transformation: `execute`
    /// returns a copy of the input regardless of direction.
    #[must_use]
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Append a filter to the end of the pipeline.
    ///
    /// Filters are applied in insertion order during forward execution
    /// and in reverse insertion order during reverse execution.
    pub fn push(&mut self, filter: Box<dyn Filter>) {
        self.filters.push(filter);
    }

    /// Return the number of filters in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Return `true` if the pipeline contains no filters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Execute the pipeline on `data` in the given direction.
    ///
    /// - [`FilterDirection::Forward`]: applies filters `[F₁, F₂, ..., Fₙ]`
    ///   in order, threading each output as the next filter's input.
    /// - [`FilterDirection::Reverse`]: applies filters `[Fₙ, ..., F₂, F₁]`
    ///   (reverse order), each in [`FilterDirection::Reverse`] mode.
    ///
    /// An empty pipeline returns a copy of `data` unchanged.
    ///
    /// # Errors
    ///
    /// Propagates the first error from any filter in the chain.
    pub fn execute(&self, direction: FilterDirection, data: &[u8]) -> Result<Vec<u8>> {
        if self.filters.is_empty() {
            return Ok(data.to_vec());
        }

        let mut current = data.to_vec();

        match direction {
            FilterDirection::Forward => {
                for filter in &self.filters {
                    current = filter.apply(FilterDirection::Forward, &current)?;
                }
            }
            FilterDirection::Reverse => {
                for filter in self.filters.iter().rev() {
                    current = filter.apply(FilterDirection::Reverse, &current)?;
                }
            }
        }

        Ok(current)
    }
}

impl Default for FilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::nbit::NbitFilter;
    use crate::pipeline::shuffle::ShuffleFilter;

    /// Empty pipeline returns data unchanged for both directions.
    #[test]
    fn empty_pipeline_is_identity() {
        let pipeline = FilterPipeline::new();
        let input: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);

        let forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("empty forward must succeed");
        assert_eq!(forward, input, "empty forward must return input unchanged");

        let reverse = pipeline
            .execute(FilterDirection::Reverse, &input)
            .expect("empty reverse must succeed");
        assert_eq!(reverse, input, "empty reverse must return input unchanged");
    }

    /// Single shuffle filter round-trip through the pipeline.
    #[test]
    fn single_shuffle_round_trip() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(4)));

        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());

        // 4 elements × 4 bytes = 16 bytes.
        let input: Vec<u8> = (0x01..=0x10).collect();
        assert_eq!(input.len(), 16);

        let forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        // Forward output must differ from input (shuffle transposes bytes).
        assert_ne!(
            forward, input,
            "shuffled output must differ from input for multi-element data"
        );

        let restored = pipeline
            .execute(FilterDirection::Reverse, &forward)
            .expect("reverse must succeed");
        assert_eq!(restored, input, "round-trip must recover original data");
    }

    /// Multi-filter pipeline: shuffle then nbit. Round-trip verification.
    ///
    /// Pipeline: [ShuffleFilter(typesize=1), NbitFilter(4, 8)]
    ///
    /// Using typesize=1 shuffle (identity) so the nbit filter receives
    /// the raw bytes directly, allowing us to verify nbit independently
    /// within the pipeline composition.
    #[test]
    fn multi_filter_shuffle_nbit_round_trip() {
        let mut pipeline = FilterPipeline::new();
        // Shuffle with typesize=1 is identity, ensuring nbit receives original bytes.
        pipeline.push(Box::new(ShuffleFilter::new(1)));
        // Pack 8-bit elements down to 4 bits.
        pipeline.push(Box::new(NbitFilter::new(4, 8)));

        assert_eq!(pipeline.len(), 2);

        // All values fit in 4 bits (0..=15).
        let input: Vec<u8> = vec![0x03, 0x07, 0x0F, 0x01, 0x00, 0x0A, 0x05, 0x0E];

        let forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        // Forward output is smaller: 8 elements × 4 bits = 4 bytes + 4 byte header = 8 bytes.
        // Input is 8 bytes. The nbit-packed result (with header) is 8 bytes total.
        // This is expected: packing 8 bytes at 4 bits = 4 packed bytes + 4 header bytes.
        assert_eq!(
            forward.len(),
            8,
            "packed output must be 4-byte header + 4 packed bytes"
        );

        let restored = pipeline
            .execute(FilterDirection::Reverse, &forward)
            .expect("reverse must succeed");
        assert_eq!(
            restored, input,
            "multi-filter round-trip must recover original data"
        );
    }

    /// Verify reverse order: filters are applied in reverse during Reverse.
    ///
    /// Pipeline: [ShuffleFilter(typesize=2), NbitFilter(4, 8)]
    ///
    /// Forward: shuffle first, then nbit pack.
    /// Reverse: nbit unpack first, then unshuffle.
    ///
    /// We verify this by checking that the intermediate state after
    /// forward matches the expected shuffle-then-pack sequence, and that
    /// reverse correctly inverts it.
    #[test]
    fn reverse_order_is_correct() {
        let shuffle = ShuffleFilter::new(2);
        let nbit = NbitFilter::new(4, 8);

        // 4 elements of 2 bytes = 8 bytes. All byte values fit in 4 bits.
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];

        // Compute expected intermediate states manually.
        // Step 1: shuffle with typesize=2 on 8 bytes (4 elements of 2 bytes).
        // N=4, T=2. output[j*4+i] = input[i*2+j]
        // j=0: input[0],input[2],input[4],input[6] = 0x01,0x03,0x05,0x07
        // j=1: input[1],input[3],input[5],input[7] = 0x02,0x04,0x06,0x08
        let expected_after_shuffle: Vec<u8> = vec![0x01, 0x03, 0x05, 0x07, 0x02, 0x04, 0x06, 0x08];

        let after_shuffle = shuffle
            .apply(FilterDirection::Forward, &input)
            .expect("shuffle must succeed");
        assert_eq!(
            after_shuffle, expected_after_shuffle,
            "shuffle intermediate must match analytical result"
        );

        // Step 2: nbit pack the shuffled data (8 bytes, 4 bits each).
        let expected_forward = nbit
            .apply(FilterDirection::Forward, &after_shuffle)
            .expect("nbit pack must succeed");

        // Now run through the pipeline.
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(2)));
        pipeline.push(Box::new(NbitFilter::new(4, 8)));

        let pipeline_forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("pipeline forward must succeed");

        assert_eq!(
            pipeline_forward, expected_forward,
            "pipeline forward must match manual shuffle-then-pack"
        );

        // Reverse must undo in opposite order: nbit unpack first, then unshuffle.
        let pipeline_reverse = pipeline
            .execute(FilterDirection::Reverse, &pipeline_forward)
            .expect("pipeline reverse must succeed");

        assert_eq!(
            pipeline_reverse, input,
            "pipeline reverse must recover original data"
        );
    }

    /// Multi-filter pipeline with non-trivial shuffle (typesize=4) and nbit.
    ///
    /// Uses 4-byte elements packed to 8 bits (i.e., only the lowest byte
    /// is significant). Verifies full round-trip through shuffle + nbit.
    #[test]
    fn shuffle_4_then_nbit_8_of_32_round_trip() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(4)));
        pipeline.push(Box::new(NbitFilter::new(8, 8)));

        // 3 elements × 4 bytes = 12 bytes.
        // Values: each 4-byte element has all bytes in [0, 15] so they
        // fit in 8 bits and round-trip through 8-bit nbit identity packing.
        let input: Vec<u8> = vec![
            0x0A, 0x0B, 0x0C, 0x0D, // element 0
            0x01, 0x02, 0x03, 0x04, // element 1
            0x07, 0x08, 0x09, 0x0F, // element 2
        ];

        let forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let restored = pipeline
            .execute(FilterDirection::Reverse, &forward)
            .expect("reverse must succeed");

        assert_eq!(
            restored, input,
            "shuffle(4)+nbit(8,8) round-trip must recover original data"
        );

        // Verify element bytes individually.
        assert_eq!(restored[0], 0x0A, "element 0, byte 0");
        assert_eq!(restored[1], 0x0B, "element 0, byte 1");
        assert_eq!(restored[2], 0x0C, "element 0, byte 2");
        assert_eq!(restored[3], 0x0D, "element 0, byte 3");
        assert_eq!(restored[4], 0x01, "element 1, byte 0");
        assert_eq!(restored[5], 0x02, "element 1, byte 1");
        assert_eq!(restored[6], 0x03, "element 1, byte 2");
        assert_eq!(restored[7], 0x04, "element 1, byte 3");
        assert_eq!(restored[8], 0x07, "element 2, byte 0");
        assert_eq!(restored[9], 0x08, "element 2, byte 1");
        assert_eq!(restored[10], 0x09, "element 2, byte 2");
        assert_eq!(restored[11], 0x0F, "element 2, byte 3");
    }

    /// Default trait impl produces an empty pipeline.
    #[test]
    fn default_is_empty() {
        let pipeline = FilterPipeline::default();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }

    /// Pipeline propagates filter errors.
    #[test]
    fn propagates_filter_error() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(4)));

        // 5 bytes is not divisible by typesize=4.
        let bad_input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05];

        let result = pipeline.execute(FilterDirection::Forward, &bad_input);
        assert!(result.is_err(), "misaligned input must propagate error");

        match result.unwrap_err() {
            consus_core::Error::InvalidFormat { message } => {
                assert!(
                    message.contains("5"),
                    "error must mention data length, got: {message}"
                );
                assert!(
                    message.contains("4"),
                    "error must mention typesize, got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }
}
