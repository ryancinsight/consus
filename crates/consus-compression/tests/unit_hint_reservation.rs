//! A capacity hint the allocator cannot supply never refuses an honest stream.
//!
//! This binary's allocator refuses any allocation above `ALLOCATION_LIMIT`, so
//! on every host a stored stream of `STORED_LEN` bytes, which can produce 1032
//! times that, has a hint that cannot be reserved while its data needs only
//! `STORED_LEN` bytes.

#![cfg(all(feature = "deflate", feature = "gzip"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::null_mut;

use consus_compression::codec::deflate::DeflateCodec;
use consus_compression::codec::flate::MAX_DEFLATE_EXPANSION;
use consus_compression::codec::gzip::GzipCodec;
use consus_compression::codec::raw_deflate::RawDeflateCodec;
use consus_compression::codec::zlib::ZlibCodec;
use consus_compression::{Codec, CompressionLevel};

const ALLOCATION_LIMIT: usize = 256 << 20;
const STORED_LEN: usize = 1 << 20;

struct RefusesLargeAllocations;

// SAFETY: both methods forward to `System` unchanged, or `alloc` returns null,
// which `GlobalAlloc` defines as allocation failure. The provided `alloc_zeroed`
// and `realloc` allocate through `alloc`, so they honor the limit too.
unsafe impl GlobalAlloc for RefusesLargeAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > ALLOCATION_LIMIT {
            return null_mut();
        }
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract for `layout`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` come from an earlier allocation by `System`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: RefusesLargeAllocations = RefusesLargeAllocations;

/// The refusal the other test relies on.
#[test]
fn allocator_refuses_a_reservation_above_the_limit() {
    let mut buffer = Vec::<u8>::new();
    assert!(buffer.try_reserve_exact(ALLOCATION_LIMIT + 1).is_err());
    assert!(buffer.try_reserve_exact(STORED_LEN).is_ok());
}

/// Stored (level 0) streams of one payload under every framing, each with the
/// codecs that decode it: `ZlibCodec` takes both the zlib and the gzip stream.
fn stored_streams(data: &[u8]) -> Vec<(&'static str, Vec<u8>, Vec<&'static dyn Codec>)> {
    let stored = |codec: &dyn Codec| {
        codec
            .compress(data, CompressionLevel(0))
            .expect("level-0 compress must succeed")
    };
    vec![
        ("raw", stored(&RawDeflateCodec), vec![&RawDeflateCodec]),
        (
            "zlib",
            stored(&DeflateCodec),
            vec![&DeflateCodec, &ZlibCodec],
        ),
        ("gzip", stored(&GzipCodec), vec![&GzipCodec, &ZlibCodec]),
    ]
}

/// An honest stream that decodes under hint 0 or the exact size decodes under
/// every larger hint, including ones whose reservation cannot be satisfied, and
/// the result does not keep the part of a satisfied reservation it did not use.
#[test]
fn honest_stream_decodes_under_any_hint() {
    let data: Vec<u8> = (0..STORED_LEN)
        .map(|index| u8::try_from(index % 251).expect("invariant: a remainder below 251 fits u8"))
        .collect();
    let tera = usize::try_from(1u64 << 40).unwrap_or(usize::MAX);
    let half = ALLOCATION_LIMIT / 2;
    for (framing, stream, codecs) in stored_streams(&data) {
        assert!(
            stream.len() * MAX_DEFLATE_EXPANSION > ALLOCATION_LIMIT,
            "{framing}: the producible size must exceed the allocator limit"
        );
        for codec in codecs {
            let hints = [0, data.len(), data.len() + 1, half, tera, usize::MAX];
            for hint in hints {
                let decoded = codec.decompress(&stream, hint).unwrap_or_else(|error| {
                    panic!("{framing} via {}: hint {hint}: {error:?}", codec.name())
                });
                assert!(
                    decoded == data,
                    "{framing} via {}: hint {hint}: content",
                    codec.name()
                );
                assert!(
                    decoded.capacity() <= 2 * data.len(),
                    "{framing} via {}: hint {hint} kept capacity {}",
                    codec.name(),
                    decoded.capacity()
                );
            }
        }
    }
}
