//! Fuzz target: NPY and NPZ readers.
//!
//! ## Strategy
//!
//! Decode adversarial bytes as a bare NPY stream for every supported element
//! type and as an NPZ archive whose members are each read at every element
//! type, exercising:
//!
//! 1. Magic, version and header-length framing, including unsupported
//!    versions and lengths at the header cap.
//! 2. Latin-1 and UTF-8 header decoding.
//! 3. The Python-literal dictionary lexer: quoting, escapes, comments,
//!    line joins, nesting and shape integer literals.
//! 4. Dtype comparison, shape overflow and the bounded payload reservation.
//!
//! Every malformed input must surface as a typed `consus_npy::Error`. Only
//! panics, and allocations past the `-rss_limit_mb` bound, are failures.
#![no_main]

use std::io::Cursor;

use consus_npy::{NpyElement, NpzReader, read_npy};
use consus_zip::ZipReader;
use libfuzzer_sys::fuzz_target;

/// Reads `data` as one NPY stream of `T`; a typed error is an expected outcome.
fn npy<T: NpyElement>(data: &[u8]) {
    let _ = read_npy::<T>(Cursor::new(data));
}

/// Reads every member of the archive in `data` as an array of `T`.
fn npz<T: NpyElement>(data: &[u8], names: &[String]) {
    let Ok(mut reader) = NpzReader::new(Cursor::new(data)) else {
        return;
    };
    for name in names {
        let _ = reader.by_name::<T>(name);
    }
}

fuzz_target!(|data: &[u8]| {
    npy::<i32>(data);
    npy::<i64>(data);
    npy::<f32>(data);
    npy::<f64>(data);
    let Ok(archive) = ZipReader::new(Cursor::new(data)) else {
        return;
    };
    let names: Vec<String> = archive
        .entries()
        .iter()
        .map(|entry| entry.name().to_owned())
        .collect();
    npz::<i32>(data, &names);
    npz::<i64>(data, &names);
    npz::<f32>(data, &names);
    npz::<f64>(data, &names);
});
