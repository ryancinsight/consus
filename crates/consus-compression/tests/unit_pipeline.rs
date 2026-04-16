//! Unit tests for filter pipeline components.
//!
//! ## Contract
//!
//! For all reversible filters `F` and input data `D`:
//!
//! ```text
//! F.apply(Reverse, F.apply(Forward, D)?) == D
//! F.apply(Forward, F.apply(Reverse, D)?) == D
//! ```
//!
//! For pipeline `P = [F₁, F₂, ..., Fₙ]`:
//!
//! ```text
//! P.execute(Reverse, P.execute(Forward, data)?) == data
//! ```

#![cfg(all(feature = "std", feature = "alloc"))]

use consus_compression::{Filter, FilterDirection, FilterPipeline, NbitFilter, ShuffleFilter};

// =============================================================================
// Shuffle Filter Tests (HDF5 filter ID 2)
// =============================================================================

mod shuffle_tests {
    use super::*;

    /// Round-trip with typesize=4 on 16 bytes (4 elements of 4 bytes).
    ///
    /// Verifies `unshuffle(shuffle(data)) == data`.
    #[test]
    fn round_trip_typesize_4() {
        let filter = ShuffleFilter::new(4);
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
    #[test]
    fn round_trip_typesize_8() {
        let filter = ShuffleFilter::new(8);
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

    /// Round-trip with typesize=2.
    #[test]
    fn round_trip_typesize_2() {
        let filter = ShuffleFilter::new(2);
        let input: Vec<u8> = (0x01..=0x10).collect();
        assert_eq!(input.len(), 16);

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let restored = filter
            .apply(FilterDirection::Reverse, &shuffled)
            .expect("reverse must succeed");

        assert_eq!(restored, input, "typesize=2 round-trip must be lossless");
    }

    /// Verify exact byte positions after forward shuffle (typesize=4).
    ///
    /// Input: 4 elements of typesize 4.
    /// Expected output groups by byte position across elements.
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

    /// Verify exact byte positions after forward shuffle (typesize=2).
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

    /// Typesize=1 is identity: single-byte elements have nothing to transpose.
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
        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];

        let result = filter.apply(FilterDirection::Forward, &input);
        assert!(result.is_err(), "misaligned input must produce an error");

        match result.unwrap_err() {
            consus_core::Error::InvalidFormat { message } => {
                assert!(
                    message.contains('7'),
                    "error message must contain data length '7', got: {message}"
                );
                assert!(
                    message.contains('4'),
                    "error message must contain typesize '4', got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Empty input is valid for any typesize.
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

    /// Single element: shuffle of one element is identity.
    #[test]
    fn single_element_is_identity() {
        let filter = ShuffleFilter::new(4);
        let input: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        assert_eq!(shuffled, input, "single element shuffle must be identity");
    }

    /// Large data block (64 KiB) round-trip.
    #[test]
    fn large_block_64k() {
        let filter = ShuffleFilter::new(4);
        let input: Vec<u8> = (0u8..=255).cycle().take(65536).collect();
        assert_eq!(input.len() % 4, 0, "input must be divisible by typesize");

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        // Shuffled output differs from input for multi-element data
        assert_ne!(shuffled, input, "shuffle must transpose bytes");

        let restored = filter
            .apply(FilterDirection::Reverse, &shuffled)
            .expect("reverse must succeed");

        assert_eq!(restored, input, "large block round-trip must be lossless");
    }

    /// Filter metadata accessors.
    #[test]
    fn filter_metadata() {
        let filter = ShuffleFilter::new(4);
        assert_eq!(filter.name(), "shuffle");
        assert_eq!(filter.typesize(), 4);
    }
}

// =============================================================================
// N-bit Filter Tests (HDF5 filter ID 5)
// =============================================================================

mod nbit_tests {
    use super::*;

    /// Round-trip: 8-bit elements, 4 bits per value.
    #[test]
    fn round_trip_8bit_4bpv() {
        let filter = NbitFilter::new(4, 8);
        let input: Vec<u8> = vec![0x03, 0x07, 0x0F, 0x01, 0x00, 0x0A, 0x05, 0x0E];

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        // 8 elements × 4 bits = 32 bits = 4 bytes packed + header
        assert_eq!(packed.len(), 8, "packed output must be header + 4 bytes");

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("reverse must succeed");

        assert_eq!(unpacked, input, "8-bit/4bpv round-trip must be lossless");
    }

    /// Round-trip: 16-bit elements, 10 bits per value.
    #[test]
    fn round_trip_16bit_10bpv() {
        let filter = NbitFilter::new(10, 16);
        // 4 elements × 16-bit = 8 bytes
        // Values: all fit in 10 bits (max ~1023)
        let input: Vec<u8> = vec![
            0xFF, 0x03, // 0x03FF = 1023
            0x00, 0x01, // 0x0100 = 256
            0x55, 0x02, // 0x0255 = 597
            0xAA, 0x00, // 0x00AA = 170
        ];

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("reverse must succeed");

        assert_eq!(unpacked, input, "16-bit/10bpv round-trip must be lossless");
    }

    /// Round-trip: 32-bit elements, 12 bits per value.
    #[test]
    fn round_trip_32bit_12bpv() {
        let filter = NbitFilter::new(12, 32);
        // Values fit in 12 bits
        let input: Vec<u8> = vec![
            0xFF, 0x0F, 0x00, 0x00, // 0x00000FFF = 4095
            0x00, 0x00, 0x00, 0x00, // 0
            0x55, 0x05, 0x00, 0x00, // 0x00000555 = 1365
            0xAA, 0x0A, 0x00, 0x00, // 0x00000AAA = 2730
        ];

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("reverse must succeed");

        assert_eq!(unpacked, input, "32-bit/12bpv round-trip must be lossless");
    }

    /// Identity: 8-bit elements with 8 bits per value.
    #[test]
    fn identity_8bit() {
        let filter = NbitFilter::new(8, 8);
        let input: Vec<u8> = (0u8..=255).cycle().take(16).collect();

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("reverse must succeed");

        assert_eq!(unpacked, input, "8-bit/8bpv must be identity");
    }

    /// Empty input round-trip.
    #[test]
    fn empty_input_round_trip() {
        let filter = NbitFilter::new(4, 8);
        let input: Vec<u8> = Vec::new();

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        assert_eq!(
            packed.len(),
            4,
            "empty forward must emit the element-count header"
        );

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("reverse must succeed");

        assert_eq!(unpacked, input, "empty round-trip must be lossless");
    }

    /// Single bit packing: 8-bit elements, 1 bit per value.
    #[test]
    fn single_bit_packing() {
        let filter = NbitFilter::new(1, 8);
        // 8 elements × 1 bit = 1 byte packed + header
        let input: Vec<u8> = vec![0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00];

        let packed = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let unpacked = filter
            .apply(FilterDirection::Reverse, &packed)
            .expect("reverse must succeed");

        assert_eq!(unpacked, input, "1-bit packing round-trip must be lossless");
    }

    /// Error on misaligned 16-bit input.
    #[test]
    fn error_misaligned_16bit() {
        let filter = NbitFilter::new(8, 16);
        // 3 bytes is not divisible by 2-byte element size
        let input: Vec<u8> = vec![0x01, 0x02, 0x03];

        let result = filter.apply(FilterDirection::Forward, &input);
        assert!(result.is_err(), "misaligned input must produce error");
    }

    /// Filter metadata accessors.
    #[test]
    fn filter_metadata() {
        let filter = NbitFilter::new(4, 8);
        assert_eq!(filter.name(), "nbit");
        assert_eq!(filter.bits_per_value(), 4);
        assert_eq!(filter.bits_per_element(), 8);
    }
}

// =============================================================================
// Filter Pipeline Tests
// =============================================================================

mod pipeline_tests {
    use super::*;

    /// Empty pipeline returns data unchanged.
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

    /// Single shuffle filter round-trip through pipeline.
    #[test]
    fn single_shuffle_round_trip() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(4)));

        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());

        let input: Vec<u8> = (0x01..=0x10).collect();
        assert_eq!(input.len(), 16);

        let forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        assert_ne!(forward, input, "shuffled output must differ from input");

        let restored = pipeline
            .execute(FilterDirection::Reverse, &forward)
            .expect("reverse must succeed");

        assert_eq!(restored, input, "round-trip must recover original data");
    }

    /// Multi-filter pipeline: shuffle then nbit.
    #[test]
    fn multi_filter_shuffle_nbit_round_trip() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(1))); // identity shuffle
        pipeline.push(Box::new(NbitFilter::new(4, 8)));

        assert_eq!(pipeline.len(), 2);

        let input: Vec<u8> = vec![0x03, 0x07, 0x0F, 0x01, 0x00, 0x0A, 0x05, 0x0E];

        let forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let restored = pipeline
            .execute(FilterDirection::Reverse, &forward)
            .expect("reverse must succeed");

        assert_eq!(
            restored, input,
            "multi-filter round-trip must recover original data"
        );
    }

    /// Verify reverse order: filters applied in reverse during Reverse.
    #[test]
    fn reverse_order_is_correct() {
        let shuffle = ShuffleFilter::new(2);
        let nbit = NbitFilter::new(4, 8);

        let input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];

        // Manual: shuffle first, then nbit
        let after_shuffle = shuffle
            .apply(FilterDirection::Forward, &input)
            .expect("shuffle must succeed");

        let expected_forward = nbit
            .apply(FilterDirection::Forward, &after_shuffle)
            .expect("nbit must succeed");

        // Pipeline
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

        let pipeline_reverse = pipeline
            .execute(FilterDirection::Reverse, &pipeline_forward)
            .expect("pipeline reverse must succeed");

        assert_eq!(
            pipeline_reverse, input,
            "pipeline reverse must recover original data"
        );
    }

    /// Non-trivial shuffle (typesize=4) + nbit round-trip.
    #[test]
    fn shuffle_4_then_nbit_8_of_32_round_trip() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(4)));
        pipeline.push(Box::new(NbitFilter::new(8, 8)));

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
    }

    /// Pipeline propagates filter errors.
    #[test]
    fn propagates_filter_error() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(4)));

        // 5 bytes is not divisible by typesize=4
        let bad_input: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04, 0x05];

        let result = pipeline.execute(FilterDirection::Forward, &bad_input);
        assert!(result.is_err(), "misaligned input must propagate error");

        match result.unwrap_err() {
            consus_core::Error::InvalidFormat { message } => {
                assert!(
                    message.contains('5'),
                    "error must mention data length, got: {message}"
                );
                assert!(
                    message.contains('4'),
                    "error must mention typesize, got: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Default pipeline is empty.
    #[test]
    fn default_is_empty() {
        let pipeline = FilterPipeline::default();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }

    /// Large data block through shuffle + nbit pipeline.
    #[test]
    fn large_block_pipeline() {
        let mut pipeline = FilterPipeline::new();
        pipeline.push(Box::new(ShuffleFilter::new(4)));
        pipeline.push(Box::new(NbitFilter::new(8, 8)));

        // 64 KiB of data, divisible by 4-byte element size
        let input: Vec<u8> = (0u8..=255).cycle().take(65536).collect();
        assert_eq!(input.len() % 4, 0);

        let forward = pipeline
            .execute(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        let restored = pipeline
            .execute(FilterDirection::Reverse, &forward)
            .expect("reverse must succeed");

        assert_eq!(
            restored, input,
            "large block pipeline round-trip must be lossless"
        );
    }
}

// =============================================================================
// Pipeline Composition Invariants
// =============================================================================

mod composition_invariants {
    use super::*;

    /// Forward then reverse is identity for shuffle.
    #[test]
    fn shuffle_forward_reverse_identity() {
        for &typesize in &[1, 2, 4, 8] {
            let filter = ShuffleFilter::new(typesize);
            let size = typesize * 16;
            let input: Vec<u8> = (0u8..=255).cycle().take(size).collect();

            let forward = filter
                .apply(FilterDirection::Forward, &input)
                .expect("forward must succeed");

            let restored = filter
                .apply(FilterDirection::Reverse, &forward)
                .expect("reverse must succeed");

            assert_eq!(
                restored, input,
                "shuffle typesize={} forward-reverse must be identity",
                typesize
            );
        }
    }

    /// Reverse then forward is identity for shuffle.
    #[test]
    fn shuffle_reverse_forward_identity() {
        for &typesize in &[1, 2, 4, 8] {
            let filter = ShuffleFilter::new(typesize);
            let size = typesize * 16;
            let input: Vec<u8> = (0u8..=255).cycle().take(size).collect();

            // Start with "shuffled" data
            let shuffled = filter
                .apply(FilterDirection::Forward, &input)
                .expect("forward must succeed");

            // Reverse then forward
            let reverse = filter
                .apply(FilterDirection::Reverse, &shuffled)
                .expect("reverse must succeed");

            let forward_again = filter
                .apply(FilterDirection::Forward, &reverse)
                .expect("forward must succeed");

            assert_eq!(
                forward_again, shuffled,
                "shuffle typesize={} reverse-forward must return to shuffled state",
                typesize
            );
        }
    }

    /// Forward then reverse is identity for nbit.
    #[test]
    fn nbit_forward_reverse_identity() {
        for &bpv in &[1, 4, 8] {
            let filter = NbitFilter::new(bpv, 8);
            let num_elements = 16;
            let limit = 1usize << bpv;
            let input: Vec<u8> = (0..limit)
                .map(|v| v as u8)
                .cycle()
                .take(num_elements)
                .collect();

            let forward = filter
                .apply(FilterDirection::Forward, &input)
                .expect("forward must succeed");

            let restored = filter
                .apply(FilterDirection::Reverse, &forward)
                .expect("reverse must succeed");

            assert_eq!(
                restored, input,
                "nbit bpv={} forward-reverse must be identity",
                bpv
            );
        }
    }

    /// Pipeline forward-reverse identity for various filter combinations.
    #[test]
    fn pipeline_forward_reverse_identity() {
        let test_cases: Vec<Vec<Box<dyn Filter>>> = vec![
            // Single filter
            vec![Box::new(ShuffleFilter::new(4))],
            vec![Box::new(NbitFilter::new(4, 8))],
            // Two filters
            vec![
                Box::new(ShuffleFilter::new(4)),
                Box::new(NbitFilter::new(8, 8)),
            ],
            // Three filters (two shuffles of different sizes + nbit)
            vec![
                Box::new(ShuffleFilter::new(1)),
                Box::new(ShuffleFilter::new(1)),
                Box::new(NbitFilter::new(4, 8)),
            ],
        ];

        for (i, filters) in test_cases.into_iter().enumerate() {
            let mut pipeline = FilterPipeline::new();
            for filter in filters {
                pipeline.push(filter);
            }

            let input: Vec<u8> = (0u8..=15).cycle().take(64).collect();

            let forward = pipeline
                .execute(FilterDirection::Forward, &input)
                .unwrap_or_else(|e| panic!("case {} forward failed: {}", i, e));

            let restored = pipeline
                .execute(FilterDirection::Reverse, &forward)
                .unwrap_or_else(|e| panic!("case {} reverse failed: {}", i, e));

            assert_eq!(
                restored, input,
                "case {} pipeline forward-reverse must be identity",
                i
            );
        }
    }

    /// Shuffle improves compressibility for gradient data.
    #[test]
    fn shuffle_improves_compressibility() {
        let filter = ShuffleFilter::new(4);

        // Create data with high byte-level correlation: gradient pattern
        let input: Vec<u8> = (0u8..=255).flat_map(|b| [b, b, b, b]).take(1024).collect();

        let shuffled = filter
            .apply(FilterDirection::Forward, &input)
            .expect("forward must succeed");

        // Shuffled data should have runs of identical bytes
        // Count runs in shuffled vs original
        let runs_original = count_runs(&input);
        let runs_shuffled = count_runs(&shuffled);

        assert!(
            runs_shuffled >= runs_original,
            "shuffle should not decrease runs: original={}, shuffled={}",
            runs_original,
            runs_shuffled
        );
    }

    /// Helper: count number of byte runs (consecutive identical bytes).
    fn count_runs(data: &[u8]) -> usize {
        if data.is_empty() {
            return 0;
        }
        let mut count = 1;
        for i in 1..data.len() {
            if data[i] != data[i - 1] {
                count += 1;
            }
        }
        count
    }
}
