//! The Parquet fuzz target's API surface, type-checked by the normal test build.
//!
//! `fuzz/fuzz_targets/fuzz_parquet_decoder.rs` cannot be compiled on every
//! developer host: `libfuzzer-sys` builds a C++ shim that requires the MSVC
//! toolchain on Windows, so `cargo fuzz build` runs only in CI. This test
//! exercises the exact calls that target makes — including the
//! `metadata()`/`dataset()`/`read_column_chunk` chain reached only after a
//! successful `new` — so a rename or signature change in the fuzzed surface
//! breaks the ordinary gate rather than the weekly fuzz job.
#![cfg(feature = "alloc")]

use consus_parquet::ParquetReader;

#[test]
fn parquet_decoder_fuzz_surface_is_callable() {
    // No footer trailer magic: rejected deterministically, exactly the early
    // return the fuzz target takes for most inputs.
    let Ok(reader) = ParquetReader::new(&[0u8; 8]) else {
        // Stage 2 of the target is unreachable for this input, but it must
        // still type-check; `unreachable_code` is avoided by returning here.
        return;
    };

    // Stage 2: the exact accessor chain the fuzz target drives. Reached only
    // when a reader constructs, which adversarial bytes rarely achieve.
    let rg_count = reader.metadata().row_groups.len();
    let col_count = reader.dataset().column_count();
    for rg in 0..rg_count {
        for col in 0..col_count {
            let _ = reader.read_column_chunk(rg, col);
        }
    }
}
