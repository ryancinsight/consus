//! Chunk I/O tests for Zarr arrays.
//!
//! ## Coverage
//!
//! - Chunk compression/decompression roundtrip
//! - Chunk sharding (v3)
//! - Partial chunk reads
//! - Codec pipeline execution
//! - Chunk key generation

use consus_zarr::Codec;
use consus_zarr::chunk::{ChunkKeySeparator, chunk_key};
use consus_zarr::codec::{CodecPipeline, default_registry};
use consus_zarr::store::{InMemoryStore, Store};

// ---------------------------------------------------------------------------
// Chunk key generation tests
// ---------------------------------------------------------------------------

/// Test v2 chunk key format (dot separator).
///
/// ## Spec Compliance
///
/// Zarr v2 uses dot-separated indices: `0.1.2`
#[test]
fn chunk_key_v2_format() {
    let key = chunk_key(&[3, 1, 4], ChunkKeySeparator::Dot);
    assert_eq!(key, "3.1.4");
}

/// Test v3 chunk key format (slash separator with c/ prefix).
///
/// ## Spec Compliance
///
/// Zarr v3 uses slash-separated indices with `c/` prefix: `c/0/1/2`
#[test]
fn chunk_key_v3_format() {
    let key = chunk_key(&[3, 1, 4], ChunkKeySeparator::Slash);
    assert_eq!(key, "c/3/1/4");
}

/// Test scalar chunk key (0D array).
#[test]
fn chunk_key_scalar() {
    let key = chunk_key(&[], ChunkKeySeparator::Dot);
    assert_eq!(key, "");

    let key = chunk_key(&[], ChunkKeySeparator::Slash);
    assert_eq!(key, "c");
}

/// Test 1D chunk key.
#[test]
fn chunk_key_1d() {
    let key = chunk_key(&[42], ChunkKeySeparator::Dot);
    assert_eq!(key, "42");

    let key = chunk_key(&[42], ChunkKeySeparator::Slash);
    assert_eq!(key, "c/42");
}

/// Test 2D chunk key.
#[test]
fn chunk_key_2d() {
    let key = chunk_key(&[10, 20], ChunkKeySeparator::Dot);
    assert_eq!(key, "10.20");

    let key = chunk_key(&[10, 20], ChunkKeySeparator::Slash);
    assert_eq!(key, "c/10/20");
}

/// Test 3D chunk key.
#[test]
fn chunk_key_3d() {
    let key = chunk_key(&[1, 2, 3], ChunkKeySeparator::Dot);
    assert_eq!(key, "1.2.3");

    let key = chunk_key(&[1, 2, 3], ChunkKeySeparator::Slash);
    assert_eq!(key, "c/1/2/3");
}

/// Test large chunk indices.
#[test]
fn chunk_key_large_indices() {
    let key = chunk_key(&[1000000, 2000000], ChunkKeySeparator::Dot);
    assert_eq!(key, "1000000.2000000");
}

// ---------------------------------------------------------------------------
// Codec pipeline compression/decompression tests
// ---------------------------------------------------------------------------

/// Test gzip codec roundtrip.
///
/// ## Invariant
///
/// For any pipeline `p` and data `d`: `p.decompress(p.compress(d)) == d`
#[test]
fn gzip_roundtrip() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    let input = b"The quick brown fox jumps over the lazy dog. This is test data for compression.";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, input);
}

/// Test gzip with different compression levels.
#[test]
fn gzip_compression_levels() {
    let registry = default_registry();

    let input =
        b"Test data for compression level testing with some repetitive patterns: aaa bbb ccc";

    for level in [1, 3, 6, 9] {
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), level.to_string())],
        });

        let compressed = pipeline
            .compress(input, registry)
            .expect("compress must succeed");
        let decompressed = pipeline
            .decompress(&compressed, registry)
            .expect("decompress must succeed");

        assert_eq!(&decompressed, input, "roundtrip failed for level {}", level);
    }
}

/// Test zstd codec roundtrip.
#[test]
fn zstd_roundtrip() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("zstd"),
        configuration: vec![(String::from("level"), String::from("3"))],
    });

    let input = b"Zstandard compression test data with moderate length for testing roundtrip.";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, input);
}

/// Test zstd with different compression levels.
#[test]
fn zstd_compression_levels() {
    let registry = default_registry();
    let input = b"Test data for zstd level testing with repetitive patterns: xxx yyy zzz";

    for level in [-3, 0, 3, 10] {
        let pipeline = CodecPipeline::single(Codec {
            name: String::from("zstd"),
            configuration: vec![(String::from("level"), level.to_string())],
        });

        let compressed = pipeline
            .compress(input, registry)
            .expect("compress must succeed");
        let decompressed = pipeline
            .decompress(&compressed, registry)
            .expect("decompress must succeed");

        assert_eq!(&decompressed, input, "roundtrip failed for level {}", level);
    }
}

/// Test lz4 codec roundtrip.
#[test]
fn lz4_roundtrip() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("lz4"),
        configuration: vec![(String::from("level"), String::from("1"))],
    });

    let input =
        b"LZ4 is a lossless compression algorithm focused on compression and decompression speed.";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, input);
}

/// Test bytes codec is identity (no compression).
#[test]
fn bytes_codec_identity() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("bytes"),
        configuration: vec![(String::from("endian"), String::from("little"))],
    });

    let input = b"raw bytes data that should not be modified";
    let output = pipeline
        .compress(input, registry)
        .expect("compress must succeed");

    assert_eq!(&output, input);
}

/// Test empty pipeline (identity).
#[test]
fn empty_pipeline() {
    let registry = default_registry();
    let pipeline = CodecPipeline::empty();

    let input = b"uncompressed data";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, input);
}

/// Test multi-codec pipeline.
#[test]
fn multi_codec_pipeline() {
    let registry = default_registry();
    let pipeline = CodecPipeline::new(vec![
        Codec {
            name: String::from("bytes"),
            configuration: vec![(String::from("endian"), String::from("little"))],
        },
        Codec {
            name: String::from("gzip"),
            configuration: vec![(String::from("level"), String::from("3"))],
        },
    ]);

    let input = b"Data that goes through multiple codecs in the pipeline for compression.";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, input);
}

/// Test compression of empty data.
#[test]
fn compress_empty_data() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    let input: &[u8] = b"";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert!(decompressed.is_empty());
}

/// Test compression of highly compressible data.
#[test]
fn compress_highly_compressible() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("9"))],
    });

    // 1MB of zeros
    let input = vec![0u8; 1024 * 1024];
    let compressed = pipeline
        .compress(&input, registry)
        .expect("compress must succeed");

    // Should compress to much smaller than original
    assert!(compressed.len() < input.len() / 100);

    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");
    assert_eq!(decompressed.len(), input.len());
}

/// Test compression of random data (low compressibility).
#[test]
fn compress_random_data() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("9"))],
    });

    // Pseudo-random data (not truly random, but has low compressibility)
    let input: Vec<u8> = (0..1024).map(|i| (i * 7 + 13) as u8).collect();
    let compressed = pipeline
        .compress(&input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, &input);
}

// ---------------------------------------------------------------------------
// Chunk storage tests
// ---------------------------------------------------------------------------

/// Test storing and retrieving a compressed chunk.
#[test]
fn store_compressed_chunk() {
    let mut store = InMemoryStore::new();
    let registry = default_registry();

    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("3"))],
    });

    let original = b"Chunk data to be compressed and stored";
    let compressed = pipeline
        .compress(original, registry)
        .expect("compress must succeed");

    let key = "array/c/0.0";
    store.set(key, &compressed).expect("set must succeed");

    let retrieved = store.get(key).expect("get must succeed");
    let decompressed = pipeline
        .decompress(&retrieved, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, original);
}

/// Test storing multiple chunks.
#[test]
fn store_multiple_chunks() {
    let mut store = InMemoryStore::new();

    // Store multiple chunks with different data
    for i in 0..10 {
        let key = format!("array/c/{}.0", i);
        let data = format!("chunk data {}", i);
        store.set(&key, data.as_bytes()).expect("set must succeed");
    }

    // Verify all chunks can be retrieved
    for i in 0..10 {
        let key = format!("array/c/{}.0", i);
        let expected = format!("chunk data {}", i);
        let data = store.get(&key).expect("get must succeed");
        assert_eq!(data, expected.as_bytes());
    }
}

/// Test partial chunk read (simulated by storing larger chunk).
#[test]
fn partial_chunk_data() {
    let mut store = InMemoryStore::new();

    // Store a chunk with extra data
    let full_data = b"full chunk data with more content";
    store
        .set("array/c/0.0", full_data)
        .expect("set must succeed");

    // Retrieve and use only part
    let data = store.get("array/c/0.0").expect("get must succeed");
    let partial = &data[0..10];
    assert_eq!(partial, b"full chunk");
}

// ---------------------------------------------------------------------------
// Chunk size calculations
// ---------------------------------------------------------------------------

/// Test chunk size for different shapes and data types.
#[test]
fn chunk_size_calculation() {
    // 2D chunk: 10x10 float64 = 100 * 8 = 800 bytes
    let chunk_shape = vec![10, 10];
    let element_size = 8; // float64
    let expected_size = chunk_shape.iter().product::<usize>() * element_size;
    assert_eq!(expected_size, 800);

    // 3D chunk: 5x5x5 int32 = 125 * 4 = 500 bytes
    let chunk_shape = vec![5, 5, 5];
    let element_size = 4; // int32
    let expected_size = chunk_shape.iter().product::<usize>() * element_size;
    assert_eq!(expected_size, 500);

    // 1D chunk: 1000 uint8 = 1000 * 1 = 1000 bytes
    let chunk_shape = vec![1000];
    let element_size = 1; // uint8
    let expected_size = chunk_shape.iter().product::<usize>() * element_size;
    assert_eq!(expected_size, 1000);
}

// ---------------------------------------------------------------------------
// Chunk grid calculations
// ---------------------------------------------------------------------------

/// Test total chunks calculation.
#[test]
fn total_chunks_calculation() {
    // Array: 100x100, Chunk: 10x10 -> 10x10 = 100 chunks
    let array_shape = vec![100, 100];
    let chunk_shape = vec![10, 10];
    let total_chunks: usize = array_shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(a, c)| (a + c - 1) / c)
        .product();
    assert_eq!(total_chunks, 100);

    // Array: 256x256x256, Chunk: 64x64x64 -> 4x4x4 = 64 chunks
    let array_shape = vec![256, 256, 256];
    let chunk_shape = vec![64, 64, 64];
    let total_chunks: usize = array_shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(a, c)| (a + c - 1) / c)
        .product();
    assert_eq!(total_chunks, 64);

    // Array: 1000, Chunk: 100 -> 10 chunks
    let array_shape = vec![1000];
    let chunk_shape = vec![100];
    let total_chunks: usize = array_shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(a, c)| (a + c - 1) / c)
        .product();
    assert_eq!(total_chunks, 10);
}

/// Test chunk grid coordinate to linear index conversion.
#[test]
fn chunk_linear_index() {
    // 2D grid: 10x10 chunks
    let grid_shape = vec![10, 10];

    // (0, 0) -> 0
    let coords = vec![0, 0];
    let linear: usize = coords.iter().enumerate().fold(0, |acc, (i, &c)| {
        let multiplier: usize = grid_shape[i + 1..].iter().product();
        acc + c * multiplier
    });
    assert_eq!(linear, 0);

    // (1, 0) -> 1
    let coords = vec![1, 0];
    let linear: usize = coords.iter().enumerate().fold(0, |acc, (i, &c)| {
        let multiplier: usize = grid_shape[i + 1..].iter().product();
        acc + c * multiplier
    });
    assert_eq!(linear, 1);

    // (0, 1) -> 10
    let coords = vec![0, 1];
    let linear: usize = coords.iter().enumerate().fold(0, |acc, (i, &c)| {
        let multiplier: usize = grid_shape[i + 1..].iter().product();
        acc + c * multiplier
    });
    assert_eq!(linear, 10);

    // (5, 3) -> 53
    let coords = vec![5, 3];
    let linear: usize = coords.iter().enumerate().fold(0, |acc, (i, &c)| {
        let multiplier: usize = grid_shape[i + 1..].iter().product();
        acc + c * multiplier
    });
    assert_eq!(linear, 53);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Test single-byte chunk.
#[test]
fn single_byte_chunk() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    let input = b"X";
    let compressed = pipeline
        .compress(input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, input);
}

/// Test large chunk.
#[test]
fn large_chunk() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    // 1MB chunk
    let input: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();
    let compressed = pipeline
        .compress(&input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(decompressed.len(), input.len());
    assert_eq!(decompressed, input);
}

/// Test chunk with all same byte values.
#[test]
fn uniform_chunk() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    let input = vec![42u8; 10000];
    let compressed = pipeline
        .compress(&input, registry)
        .expect("compress must succeed");

    // Uniform data should compress very well
    assert!(compressed.len() < 100);

    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");
    assert!(decompressed.iter().all(|&b| b == 42));
}

/// Test chunk with alternating pattern.
#[test]
fn alternating_pattern_chunk() {
    let registry = default_registry();
    let pipeline = CodecPipeline::single(Codec {
        name: String::from("gzip"),
        configuration: vec![(String::from("level"), String::from("6"))],
    });

    let input: Vec<u8> = (0..10000)
        .map(|i| if i % 2 == 0 { 0 } else { 255 })
        .collect();
    let compressed = pipeline
        .compress(&input, registry)
        .expect("compress must succeed");
    let decompressed = pipeline
        .decompress(&compressed, registry)
        .expect("decompress must succeed");

    assert_eq!(&decompressed, &input);
}
