//! Fuzz target: FITS header and table-descriptor parsing.
//!
//! ## Strategy
//!
//! Drive the header card parser and both table-descriptor parsers with
//! adversarial byte sequences to exercise:
//!
//! 1. Card tokenisation and keyword/value classification.
//! 2. `TFIELDS`-driven column-vector reservation.
//! 3. `TFORMn` parsing, per-column byte widths, and the `NAXIS1` sum check.
//! 4. `TBCOL`/`PCOUNT` offset arithmetic.
//!
//! All `Result` errors are discarded; only panics and aborts cause failures.
#![no_main]

use consus_fits::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor, parse_header_bytes};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Stage 1: card tokenisation. Fails deterministically for input that is
    // not a whole number of 80-byte cards; no panic must occur.
    let Ok(header) = parse_header_bytes(data) else {
        return;
    };

    // Stage 2: both table-descriptor parsers. Each validates `XTENSION`
    // first, so at most one proceeds past that check for any given input;
    // driving both keeps the target independent of which one the fuzzer
    // happens to reach.
    let _ = FitsAsciiTableDescriptor::from_header(&header);
    let _ = FitsBinaryTableDescriptor::from_header(&header);
});
