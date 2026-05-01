//! Fuzz target: Parquet footer and column-chunk decoder (heap-buffer and logic).
//!
//! ## Strategy
//!
//! Drive `ParquetReader::new` with adversarial byte sequences to exercise:
//!
//! 1. Footer trailer magic and length validation.
//! 2. Thrift compact binary `FileMetadata` decoding.
//! 3. Schema element parsing and `ParquetDatasetDescriptor` materialization.
//! 4. Column chunk page header decoding (DataPage v1/v2, DictionaryPage).
//! 5. Level decoding (definition/repetition RLE).
//! 6. Value decoding: PLAIN, RLE/bit-packing, dictionary index expansion.
//!
//! All `Result` errors are discarded; only panics cause fuzzer failures.
#![no_main]

use consus_parquet::ParquetReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Stage 1: footer trailer validation + Thrift FileMetadata decoding.
    let Ok(reader) = ParquetReader::new(data) else {
        return;
    };

    // Stage 2: drive column-chunk decoding across all row groups and columns
    // to exercise page-header dispatch, level decoding, value decoding, and
    // any enabled decompression paths.
    let rg_count = reader.metadata().row_groups.len();
    let col_count = reader.dataset().column_count();

    for rg in 0..rg_count {
        for col in 0..col_count {
            // Errors (BufferTooSmall, InvalidFormat, UnsupportedFeature) are
            // all expected for adversarial input; panics are not.
            let _ = reader.read_column_chunk(rg, col);
        }
    }
});
