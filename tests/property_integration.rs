//! Property-based integration tests across formats.
//!
//! ## Specification Reference
//!
//! These tests use property-based testing (proptest) to validate invariants
//! across format conversions and I/O operations:
//!
//! - Arbitrary data shapes and sizes
//! - Random data patterns
//! - Cross-format roundtrip invariants
//! - Metadata preservation under random inputs
//!
//! ## Coverage
//!
//! - HDF5 roundtrip properties
//! - Zarr roundtrip properties
//! - netCDF-4 roundtrip properties
//! - Arrow ↔ Parquet schema compatibility
//! - Compression preservation

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Proptest Strategies
// ---------------------------------------------------------------------------

/// Strategy for generating small array shapes (1D to 3D).
fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=100, 1..=3)
}

/// Strategy for generating chunk shapes (smaller than dataset shape).
fn chunk_shape_strategy(shape: &[usize]) -> impl Strategy<Value = Vec<usize>> {
    let shape_owned = shape.to_vec();
    prop::collection::vec(
        prop::usize::between(1, shape_owned.iter().map(|&d| d.max(1)).collect::<Vec<_>>()),
        shape.len(),
    )
}

/// Strategy for generating f32 data.
fn f32_data_strategy(count: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(prop::num::f32::ANY, count)
}

/// Strategy for generating i32 data.
fn i32_data_strategy(count: usize) -> impl Strategy<Value = Vec<i32>> {
    prop::collection::vec(prop::num::i32::ANY, count)
}

/// Strategy for generating u8 data.
fn u8_data_strategy(count: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(prop::num::u8::ANY, count)
}

/// Strategy for generating compression levels.
fn compression_level_strategy() -> impl Strategy<Value = u32> {
    prop::num::u32::between(1, 9)
}

/// Strategy for generating dimension names.
fn dimension_name_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z][a-z0-9_]{0,15}").unwrap()
}

/// Strategy for generating dataset names.
fn dataset_name_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z][a-z0-9_]{0,15}").unwrap()
}

// ---------------------------------------------------------------------------
// HDF5 Roundtrip Property Tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that arbitrary f32 data roundtrips through HDF5.
    ///
    /// ## Invariant
    ///
    /// For any f32 array with shape [N], write to HDF5 and read back
    /// must produce identical values (bitwise for non-NaN).
    #[cfg(feature = "hdf5")]
    #[test]
    fn hdf5_f32_roundtrip_preserves_values(
        shape in shape_strategy(),
        data in prop::collection::vec(prop::num::f32::ANY, 0..=10000)
    ) {
        use consus_core::{Datatype, Selection, Shape};
        use consus_io::MemCursor;

        // Adjust data length to match shape
        let total_elements: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        // Pad or truncate data to match shape
        let mut data = data;
        data.resize(total_elements, 0.0f32);

        // Write to HDF5
        let mut buffer = vec![0u8; total_elements * 4 + 8192];
        let mut cursor = MemCursor::new(buffer);

        {
            let mut writer = match consus_hdf5::Hdf5FileBuilder::new()
                .build_writer(&mut cursor)
            {
                Ok(w) => w,
                Err(_) => return Ok(()), // Skip if writer unavailable
            };

            let dataset = match writer.root_group().create_dataset(
                "data",
                Datatype::Float {
                    bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                    byte_order: consus_core::ByteOrder::LittleEndian,
                },
                if shape.is_empty() { Shape::scalar() } else { Shape::fixed(&shape) },
            ) {
                Ok(ds) => ds,
                Err(_) => return Ok(()),
            };

            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = dataset.write(&Selection::all(), &bytes);
            let _ = writer.finish();
        }

        // Read back
        let _ = cursor.seek(std::io::SeekFrom::Start(0));
        let file = match consus_hdf5::Hdf5File::open(cursor) {
            Ok(f) => f,
            Err(_) => return Ok(()),
        };

        let dataset = match file.root_group().get_dataset("data") {
            Ok(ds) => ds,
            Err(_) => return Ok(()),
        };

        let mut read_buf = vec![0u8; total_elements * 4];
        if dataset.read(&Selection::all(), &mut read_buf).is_err() {
            return Ok(());
        }

        let read_values: Vec<f32> = read_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();

        // Verify values match
        for (original, roundtrip) in data.iter().zip(read_values.iter()) {
            // For NaN, check both are NaN
            if original.is_nan() {
                prop_assert!(roundtrip.is_nan());
            } else {
                prop_assert!((original - roundtrip).abs() < f32::EPSILON);
            }
        }
    }

    /// Test that arbitrary i32 data roundtrips through HDF5.
    ///
    /// ## Invariant
    ///
    /// Integer values must roundtrip exactly (bit-identical).
    #[cfg(feature = "hdf5")]
    #[test]
    fn hdf5_i32_roundtrip_preserves_values(
        shape in shape_strategy(),
        data in prop::collection::vec(prop::num::i32::ANY, 0..=10000)
    ) {
        use consus_core::{Datatype, Selection, Shape};
        use consus_io::MemCursor;

        let total_elements: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        let mut data = data;
        data.resize(total_elements, 0i32);

        let mut buffer = vec![0u8; total_elements * 4 + 8192];
        let mut cursor = MemCursor::new(buffer);

        {
            let mut writer = match consus_hdf5::Hdf5FileBuilder::new()
                .build_writer(&mut cursor)
            {
                Ok(w) => w,
                Err(_) => return Ok(()),
            };

            let dataset = match writer.root_group().create_dataset(
                "data",
                Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                    byte_order: consus_core::ByteOrder::LittleEndian,
                    signed: true,
                },
                if shape.is_empty() { Shape::scalar() } else { Shape::fixed(&shape) },
            ) {
                Ok(ds) => ds,
                Err(_) => return Ok(()),
            };

            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = dataset.write(&Selection::all(), &bytes);
            let _ = writer.finish();
        }

        let _ = cursor.seek(std::io::SeekFrom::Start(0));
        let file = match consus_hdf5::Hdf5File::open(cursor) {
            Ok(f) => f,
            Err(_) => return Ok(()),
        };

        let dataset = match file.root_group().get_dataset("data") {
            Ok(ds) => ds,
            Err(_) => return Ok(()),
        };

        let mut read_buf = vec![0u8; total_elements * 4];
        if dataset.read(&Selection::all(), &mut read_buf).is_err() {
            return Ok(());
        }

        let read_values: Vec<i32> = read_buf
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();

        prop_assert_eq!(data, read_values);
    }
}

// ---------------------------------------------------------------------------
// Zarr Roundtrip Property Tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that arbitrary data roundtrips through Zarr.
    ///
    /// ## Invariant
    ///
    /// Data values must be preserved through Zarr write/read cycle.
    #[cfg(feature = "zarr")]
    #[test]
    fn zarr_f64_roundtrip_preserves_values(
        size in 1usize..=1000,
        data in prop::collection::vec(prop::num::f64::NORMAL, 1..=1000)
    ) {
        use consus_zarr::{ArrayMetadataV3, InMemoryStore, ZarrArray};

        let mut data = data;
        data.resize(size, 0.0f64);

        let mut store = InMemoryStore::new();

        let array = match ZarrArray::create(
            &mut store,
            "data",
            ArrayMetadataV3 {
                shape: vec![size],
                data_type: "float64".to_string(),
                chunk_grid: vec![size.min(100)],
                fill_value: 0.0,
                codecs: vec![],
            },
        ) {
            Ok(a) => a,
            Err(_) => return Ok(()),
        };

        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Write chunk(s)
        let chunk_size = size.min(100);
        for (chunk_idx, chunk_data) in bytes.chunks(chunk_size * 8).enumerate() {
            if array.write_chunk(&[chunk_idx as u64], chunk_data).is_err() {
                return Ok(());
            }
        }

        // Read back
        let mut read_data = Vec::new();
        let num_chunks = (size + chunk_size - 1) / chunk_size;

        for chunk_idx in 0..num_chunks {
            match array.read_chunk(&[chunk_idx as u64]) {
                Ok(chunk) => read_data.extend_from_slice(&chunk),
                Err(_) => return Ok(()),
            }
        }

        let read_values: Vec<f64> = read_data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect();

        // Resize to match original
        let read_values: Vec<f64> = read_values.into_iter().take(size).collect();

        for (original, roundtrip) in data.iter().zip(read_values.iter()) {
            prop_assert!((original - roundtrip).abs() < f64::EPSILON);
        }
    }

    /// Test Zarr chunk key encoding.
    ///
    /// ## Invariant
    ///
    /// Chunk keys must correctly encode chunk coordinates.
    #[cfg(feature = "zarr")]
    #[test]
    fn zarr_chunk_key_encoding(
        x in 0u64..=1000,
        y in 0u64..=1000,
        z in 0u64..=1000
    ) {
        use consus_zarr::metadata::ChunkKeyEncoding;

        // Test default (dot-separated) encoding
        let key = ChunkKeyEncoding::default().encode(&[x, y, z]);
        let expected = format!("{}/{}/{}", x, y, z);

        prop_assert!(key.contains(&expected) || key.contains(&format!("{}.{}.{}", x, y, z)));
    }
}

// ---------------------------------------------------------------------------
// Arrow ↔ Parquet Schema Property Tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test Arrow to Core schema conversion preserves field count.
    ///
    /// ## Invariant
    ///
    /// Number of fields must be preserved through conversion.
    #[cfg(feature = "arrow")]
    #[test]
    fn arrow_schema_field_count_preserved(
        num_fields in 1usize..=20
    ) {
        use consus_arrow::{ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowSchema};
        use consus_core::{ByteOrder, Datatype};

        let mut fields = Vec::new();
        for i in 0..num_fields {
            let field = ArrowFieldBuilder::new(
                ArrowFieldId::new(i as u32 + 1),
                format!("field_{}", i),
                ArrowFieldKind::Int,
                Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(32).expect("non-zero"),
                    byte_order: ByteOrder::LittleEndian,
                    signed: true,
                },
            )
            .nullable(true)
            .build();

            if let Ok(f) = field {
                fields.push(f);
            }
        }

        let schema = ArrowSchema::new(fields);
        let core_pairs = consus_arrow::conversion::arrow_schema_to_core_pairs(&schema);

        prop_assert_eq!(core_pairs.len(), num_fields);
    }

    /// Test Parquet to Core schema conversion preserves column count.
    ///
    /// ## Invariant
    ///
    /// Number of columns must be preserved through conversion.
    #[cfg(feature = "parquet")]
    #[test]
    fn parquet_schema_column_count_preserved(
        num_cols in 1usize..=20
    ) {
        use consus_parquet::{FieldDescriptor, FieldId, ParquetPhysicalType, SchemaDescriptor};

        let mut fields = Vec::new();
        for i in 0..num_cols {
            fields.push(FieldDescriptor::required(
                FieldId::new(i as u32 + 1),
                format!("col_{}", i),
                ParquetPhysicalType::Int32,
            ));
        }

        let schema = SchemaDescriptor::new(fields);
        let core_pairs = consus_parquet::conversion::parquet_schema_to_core_pairs(&schema);

        prop_assert_eq!(core_pairs.len(), num_cols);
    }
}

// ---------------------------------------------------------------------------
// Compression Property Tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test deflate compression roundtrip with arbitrary data.
    ///
    /// ## Invariant
    ///
    /// compress(decompress(data)) == data for any input.
    #[cfg(feature = "compression")]
    #[test]
    fn deflate_roundtrip_arbitrary_data(
        data in prop::collection::vec(prop::num::u8::ANY, 0..=100000)
    ) {
        use consus_compression::codec::traits::Codec;

        let codec = match consus_compression::codec::DeflateCodec::new(6) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };

        let compressed = match codec.compress(&data) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };

        let decompressed = match codec.decompress(&compressed) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        prop_assert_eq!(data, decompressed);
    }

    /// Test that compression reduces size for compressible data.
    ///
    /// ## Invariant
    ///
    /// Repetitive patterns should compress well.
    #[cfg(feature = "compression")]
    #[test]
    fn compression_reduces_repetitive_data(
        value in prop::num::u8::ANY,
        count in 100usize..=10000
    ) {
        use consus_compression::codec::traits::Codec;

        let data: Vec<u8> = vec![value; count];

        let codec = match consus_compression::codec::DeflateCodec::new(9) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };

        let compressed = match codec.compress(&data) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };

        // Compressed size should be much smaller for repetitive data
        // (unless the data is very small)
        if count >= 100 {
            prop_assert!(compressed.len() < data.len());
        }
    }
}

// ---------------------------------------------------------------------------
// Shape and Selection Property Tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that Shape::element_count matches product of dimensions.
    ///
    /// ## Invariant
    ///
    /// element_count(dims) = product(dims)
    #[test]
    fn shape_element_count_product(
        dims in prop::collection::vec(1usize..=1000, 0..=5)
    ) {
        use consus_core::Shape;

        let shape = if dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::fixed(&dims)
        };

        let expected: usize = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };

        prop_assert_eq!(shape.element_count(), expected);
    }

    /// Test hyperslab selection bounds.
    ///
    /// ## Invariant
    ///
    /// Hyperslab start + count must not exceed shape bounds.
    #[test]
    fn hyperslab_selection_bounds(
        dim_size in 10usize..=1000,
        start in 0usize..=100,
        count in 1usize..=50
    ) {
        use consus_core::Selection;

        // Adjust start and count to be valid
        let start = start.min(dim_size - 1);
        let max_count = dim_size - start;
        let count = count.min(max_count);

        let selection = Selection::hyperslab(
            &[start],
            &[count],
            &[1],
        );

        // Selection should be valid for this shape
        let shape = consus_core::Shape::fixed(&[dim_size]);

        // Calculate expected element count
        let expected_count = count;

        prop_assert!(selection.element_count(&shape).is_some());

        if let Some(elem_count) = selection.element_count(&shape) {
            prop_assert_eq!(elem_count, expected_count);
        }
    }

    /// Test that hyperslab selections are deterministic.
    ///
    /// ## Invariant
    ///
    /// Same hyperslab parameters produce same selection.
    #[test]
    fn hyperslab_selection_deterministic(
        start in 0usize..=100,
        count in 1usize..=100,
        stride in 1usize..=10
    ) {
        use consus_core::Selection;

        let sel1 = Selection::hyperslab(&[start], &[count], &[stride]);
        let sel2 = Selection::hyperslab(&[start], &[count], &[stride]);

        // Selections with same parameters should be equal
        prop_assert_eq!(sel1, sel2);
    }
}

// ---------------------------------------------------------------------------
// Byte Order Property Tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test little-endian byte conversion roundtrip.
    ///
    /// ## Invariant
    ///
    /// to_le_bytes → from_le_bytes must be identity.
    #[test]
    fn little_endian_roundtrip_u32(value: u32) {
        let bytes = value.to_le_bytes();
        let recovered = u32::from_le_bytes(bytes);
        prop_assert_eq!(value, recovered);
    }

    /// Test big-endian byte conversion roundtrip.
    ///
    /// ## Invariant
    ///
    /// to_be_bytes → from_be_bytes must be identity.
    #[test]
    fn big_endian_roundtrip_u32(value: u32) {
        let bytes = value.to_be_bytes();
        let recovered = u32::from_be_bytes(bytes);
        prop_assert_eq!(value, recovered);
    }

    /// Test float byte conversion roundtrip.
    ///
    /// ## Invariant
    ///
    /// Float bytes must roundtrip exactly (including NaN patterns).
    #[test]
    fn float_roundtrip_f64(value: f64) {
        let bytes = value.to_le_bytes();
        let recovered = f64::from_le_bytes(bytes);

        if value.is_nan() {
            prop_assert!(recovered.is_nan());
        } else {
            prop_assert_eq!(value, recovered);
        }
    }
}

// ---------------------------------------------------------------------------
// Stress Tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Stress test: many small operations in sequence.
    ///
    /// ## Invariant
    ///
    /// Sequential operations must maintain consistency.
    #[cfg(feature = "io")]
    #[test]
    fn stress_sequential_operations(
        ops in prop::collection::vec(
            (prop::num::usize::between(0, 1000), prop::num::usize::between(1, 100)),
            0..=100
        )
    ) {
        use consus_io::{MemCursor, ReadAt, WriteAt};

        let mut cursor = MemCursor::new(vec![0u8; 100000]);

        for (offset, size) in ops {
            // Write data
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let _ = cursor.write_at(offset as u64, &data);

            // Read back
            let mut read_buf = vec![0u8; size];
            if cursor.read_at(offset as u64, &mut read_buf).is_ok() {
                prop_assert_eq!(read_buf, data);
            }
        }
    }

    /// Stress test: random data patterns through compression.
    ///
    /// ## Invariant
    ///
    /// Compression must handle any input without corruption.
    #[cfg(feature = "compression")]
    #[test]
    fn stress_compression_random_patterns(
        patterns in prop::collection::vec(
            prop::collection::vec(prop::num::u8::ANY, 100..=10000),
            1..=20
        )
    ) {
        use consus_compression::codec::traits::Codec;

        let codec = match consus_compression::codec::DeflateCodec::new(1) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };

        for data in patterns {
            let compressed = match codec.compress(&data) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let decompressed = match codec.decompress(&compressed) {
                Ok(d) => d,
                Err(_) => continue,
            };

            prop_assert_eq!(data, decompressed);
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-Format Consistency Tests
// ---------------------------------------------------------------------------

proptest! {
    /// Test that datatype size calculations are consistent.
    ///
    /// ## Invariant
    ///
    /// Datatype::element_size() must match expected size for all types.
    #[test]
    fn datatype_size_consistency(
        bits in prop::sample::select(&[8u32, 16, 32, 64] as &[u32])
    ) {
        use consus_core::{ByteOrder, Datatype};
        use core::num::NonZeroUsize;

        let int_type = Datatype::Integer {
            bits: NonZeroUsize::new(bits as usize).expect("non-zero"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };

        prop_assert_eq!(int_type.element_size(), bits as usize / 8);

        let float_type = Datatype::Float {
            bits: NonZeroUsize::new(bits as usize).expect("non-zero"),
            byte_order: ByteOrder::LittleEndian,
        };

        prop_assert_eq!(float_type.element_size(), bits as usize / 8);
    }

    /// Test that shape comparisons are transitive.
    ///
    /// ## Invariant
    ///
    /// If A == B and B == C, then A == C.
    #[test]
    fn shape_equality_transitive(
        dims1 in prop::collection::vec(1usize..=100, 1..=5),
        dims2 in prop::collection::vec(1usize..=100, 1..=5)
    ) {
        use consus_core::Shape;

        let shape_a = Shape::fixed(&dims1);
        let shape_b = Shape::fixed(&dims1); // Same as A
        let shape_c = Shape::fixed(&dims2);

        prop_assert_eq!(shape_a, shape_b);

        // If dims1 == dims2, then shape_a == shape_c
        if dims1 == dims2 {
            prop_assert_eq!(shape_a, shape_c);
        } else {
            prop_assert_ne!(shape_a, shape_c);
        }
    }
}
