//! Generator for `zip_crate.zip`, the third-party interoperability fixture.
//!
//! Not compiled with this crate. To regenerate, place this file as
//! `src/main.rs` of a scratch binary whose manifest declares
//! `zip = { version = "=9.0.5", default-features = false, features = ["deflate-flate2-zlib-rs"] }`,
//! then run `cargo run -- zip_crate.zip` and copy the archive here.

use std::io::Write;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, DateTime, ZipWriter};

fn main() {
    let mut zip = ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let base = SimpleFileOptions::default().last_modified_time(DateTime::default());
    zip.start_file("stored.txt", base.compression_method(CompressionMethod::Stored)).unwrap();
    zip.write_all(b"written by the zip crate").unwrap();
    zip.start_file("dir/deflated.bin", base.compression_method(CompressionMethod::Deflated)).unwrap();
    let pattern: Vec<u8> = (0..4096u32).map(|i| u8::try_from(i % 7).unwrap()).collect();
    zip.write_all(&pattern).unwrap();
    zip.start_file("ünïcode.txt", base.compression_method(CompressionMethod::Deflated)).unwrap();
    zip.write_all("ä".as_bytes()).unwrap();
    let bytes = zip.finish().unwrap().into_inner();
    std::fs::write(std::env::args().nth(1).unwrap(), bytes).unwrap();
}
