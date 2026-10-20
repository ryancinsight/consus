//! Fuzz target: ZIP container reader.
//!
//! ## Strategy
//!
//! Open adversarial bytes as an archive and decode every member, exercising:
//!
//! 1. End-record search, including comments that embed its signature.
//! 2. ZIP64 locator and end-record resolution.
//! 3. Central directory parsing: entry-count capacity, name encoding, extra
//!    fields, ZIP64 sentinels and duplicate names.
//! 4. Local header checks, payload bounds, the DEFLATE expansion bound,
//!    bounded raw DEFLATE decoding, and size and CRC-32 verification.
//!
//! Every malformed input must surface as a typed `consus_zip::Error`. Only
//! panics, and allocations past the `-rss_limit_mb` bound, are failures.
#![no_main]

use std::io::Cursor;

use consus_zip::ZipReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(mut reader) = ZipReader::new(Cursor::new(data)) else {
        return;
    };
    for index in 0..reader.entries().len() {
        // Each member's typed error is an expected outcome; panics are not.
        let _ = reader.read_entry(index);
    }
});
