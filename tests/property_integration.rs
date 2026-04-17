//! Property-based integration tests across stable Consus APIs.
//!
//! ## Specification
//!
//! These tests validate algebraic and byte-level invariants that are stable
//! across the current public API surface exposed by the workspace test crate:
//!
//! - shape element-count semantics
//! - hyperslab determinism and bounds-sensitive element counting
//! - byte-order roundtrips
//! - datatype element-size consistency
//! - in-memory random-access I/O correctness
//! - compression roundtrip correctness and size reduction on repetitive input
//! - Arrow and Parquet schema conversion cardinality preservation
//!
//! ## Scope
//!
//! This file intentionally avoids obsolete backend-specific writer APIs that no
//! longer match the current repository surface. Every assertion inspects
//! computed values.

use proptest::prelude::*;

// -----------------------------------------------------------------------------
// Shared strategies
// -----------------------------------------------------------------------------

fn non_scalar_shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=32, 1..=4)
}

fn maybe_scalar_shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=32, 0..=4)
}

#[cfg(feature = "compression")]
fn compression_payload_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..=65536)
}

// -----------------------------------------------------------------------------
// Shape and selection properties
// -----------------------------------------------------------------------------

proptest! {
    /// Invariant:
    /// `Shape::element_count()` equals the product of extents, with scalar shape
    /// contributing exactly one element.
    #[test]
    fn shape_element_count_product(
        dims in maybe_scalar_shape_strategy()
    ) {
        use consus_core::Shape;

        let shape = if dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::fixed(&dims)
        };

        let expected = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };

        prop_assert_eq!(shape.num_elements(), expected);
    }

    /// Invariant:
    /// A 1-D hyperslab with `stride = 1` and valid `start/count` has element
    /// count equal to `count`.
    #[test]
    fn hyperslab_selection_bounds_and_count(
        dim_size in 1usize..=1024,
        start_seed in 0usize..=2048,
        count_seed in 1usize..=2048
    ) {
        use consus_core::{Hyperslab, HyperslabDim, Selection, Shape};

        let start = start_seed % dim_size;
        let max_count = dim_size - start;
        let count = count_seed.min(max_count.max(1));

        let selection = Selection::Hyperslab(Hyperslab::new(&[HyperslabDim::range(start, count)]));
        let shape = Shape::fixed(&[dim_size]);

        let actual = selection.num_elements(&shape);
        prop_assert_eq!(actual, count);
    }

    /// Invariant:
    /// Equal hyperslab constructor inputs produce equal selections.
    #[test]
    fn hyperslab_selection_deterministic(
        start in 0usize..=128,
        count in 1usize..=128,
        stride in 1usize..=16
    ) {
        use consus_core::{Hyperslab, HyperslabDim, Selection};

        let dim = HyperslabDim {
            start,
            stride,
            count,
            block: 1,
        };
        let left = Selection::Hyperslab(Hyperslab::new(&[dim.clone()]));
        let right = Selection::Hyperslab(Hyperslab::new(&[dim]));

        prop_assert_eq!(left, right);
    }

    /// Invariant:
    /// For any non-scalar shape, a full-range 1-D hyperslab over the first axis
    /// counts exactly that axis extent.
    #[test]
    fn first_axis_full_hyperslab_count_matches_extent(
        dims in non_scalar_shape_strategy()
    ) {
        use consus_core::{Hyperslab, HyperslabDim, Selection, Shape};

        let shape = Shape::fixed(&dims);
        let first_extent = dims[0];
        let selection = Selection::Hyperslab(Hyperslab::new(&[HyperslabDim::range(0, first_extent)]));

        prop_assert_eq!(selection.num_elements(&Shape::fixed(&[first_extent])), first_extent);
        prop_assert!(shape.num_elements() >= first_extent);
    }
}

// -----------------------------------------------------------------------------
// Datatype and byte-order properties
// -----------------------------------------------------------------------------

proptest! {
    /// Invariant:
    /// Integer and float datatype element sizes equal `bits / 8`.
    #[test]
    fn datatype_size_consistency(
        bits in prop::sample::select(vec![8usize, 16, 32, 64])
    ) {
        use consus_core::{ByteOrder, Datatype};
        use core::num::NonZeroUsize;

        let int_type = Datatype::Integer {
            bits: NonZeroUsize::new(bits).expect("bits are non-zero"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };
        prop_assert_eq!(int_type.element_size(), Some(bits / 8));

        let float_type = Datatype::Float {
            bits: NonZeroUsize::new(bits).expect("bits are non-zero"),
            byte_order: ByteOrder::LittleEndian,
        };
        prop_assert_eq!(float_type.element_size(), Some(bits / 8));
    }

    /// Invariant:
    /// `to_le_bytes` followed by `from_le_bytes` is identity.
    #[test]
    fn little_endian_roundtrip_u32(value: u32) {
        let bytes = value.to_le_bytes();
        let recovered = u32::from_le_bytes(bytes);
        prop_assert_eq!(value, recovered);
    }

    /// Invariant:
    /// `to_be_bytes` followed by `from_be_bytes` is identity.
    #[test]
    fn big_endian_roundtrip_u32(value: u32) {
        let bytes = value.to_be_bytes();
        let recovered = u32::from_be_bytes(bytes);
        prop_assert_eq!(value, recovered);
    }

    /// Invariant:
    /// `f64` byte roundtrip preserves the exact bit pattern.
    #[test]
    fn float_roundtrip_f64_bits(value: f64) {
        let bytes = value.to_le_bytes();
        let recovered = f64::from_le_bytes(bytes);
        prop_assert_eq!(value.to_bits(), recovered.to_bits());
    }

    /// Invariant:
    /// Shape equality is reflexive and transitive for identical extents.
    #[test]
    fn shape_equality_transitive(
        dims in non_scalar_shape_strategy()
    ) {
        use consus_core::Shape;

        let a = Shape::fixed(&dims);
        let b = Shape::fixed(&dims);
        let c = Shape::fixed(&dims);

        prop_assert_eq!(a.clone(), b.clone());
        prop_assert_eq!(b.clone(), c.clone());
        prop_assert_eq!(a, c);
    }
}

// -----------------------------------------------------------------------------
// In-memory I/O properties
// -----------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Invariant:
    /// Writing bytes at an offset and reading the same range returns identical
    /// bytes.
    #[test]
    fn memcursor_write_then_read_roundtrip(
        offset in 0usize..=4096,
        data in prop::collection::vec(any::<u8>(), 0..=512)
    ) {
        use consus_io::{MemCursor, ReadAt, WriteAt};

        let capacity = offset + data.len() + 32;
        let mut cursor = MemCursor::with_capacity(capacity);

        cursor
            .write_at(offset as u64, &data)
            .expect("write_at must succeed within allocated capacity");

        let mut read_back = vec![0u8; data.len()];
        cursor
            .read_at(offset as u64, &mut read_back)
            .expect("read_at must succeed for written range");
        prop_assert_eq!(read_back, data);
    }

    /// Invariant:
    /// Sequential non-overlapping writes preserve each written segment exactly.
    #[test]
    fn memcursor_sequential_non_overlapping_writes_preserve_segments(
        first in prop::collection::vec(any::<u8>(), 0..=256),
        second in prop::collection::vec(any::<u8>(), 0..=256),
        gap in 0usize..=64
    ) {
        use consus_io::{MemCursor, ReadAt, WriteAt};

        let first_offset = 17usize;
        let second_offset = first_offset + first.len() + gap;
        let capacity = second_offset + second.len() + 17;

        let mut cursor = MemCursor::with_capacity(capacity);

        cursor.write_at(first_offset as u64, &first).expect("first write must succeed");
        cursor.write_at(second_offset as u64, &second).expect("second write must succeed");

        let mut first_read = vec![0u8; first.len()];
        let mut second_read = vec![0u8; second.len()];

        cursor.read_at(first_offset as u64, &mut first_read).expect("first read must succeed");
        cursor.read_at(second_offset as u64, &mut second_read).expect("second read must succeed");
        prop_assert_eq!(first_read, first);
        prop_assert_eq!(second_read, second);
    }
}

// -----------------------------------------------------------------------------
// Compression properties
// -----------------------------------------------------------------------------

#[cfg(feature = "compression")]
proptest! {
    /// Invariant:
    /// Deflate decompress(compress(data)) == data for arbitrary byte input.
    #[test]
    fn deflate_roundtrip_arbitrary_data(
        data in compression_payload_strategy()
    ) {
        use consus_compression::codec::deflate::DeflateCodec;
        use consus_compression::codec::traits::{Codec, CompressionLevel};

        let codec = DeflateCodec;
        let compressed = codec
            .compress(&data, CompressionLevel(6))
            .expect("deflate compression must succeed");
        let decompressed = codec
            .decompress(&compressed, data.len())
            .expect("deflate decompression must succeed");

        prop_assert_eq!(decompressed, data);
    }

    /// Invariant:
    /// Highly repetitive data compresses to fewer bytes than the original.
    #[test]
    fn deflate_reduces_repetitive_data(
        value in any::<u8>(),
        count in 128usize..=16384
    ) {
        use consus_compression::codec::deflate::DeflateCodec;
        use consus_compression::codec::traits::{Codec, CompressionLevel};

        let data = vec![value; count];
        let codec = DeflateCodec;
        let compressed = codec
            .compress(&data, CompressionLevel(9))
            .expect("deflate compression must succeed");

        prop_assert!(compressed.len() < data.len());
    }

    /// Invariant:
    /// LZ4 decompress(compress(data)) == data for arbitrary byte input.
    #[test]
    fn lz4_roundtrip_arbitrary_data(
        data in compression_payload_strategy()
    ) {
        use consus_compression::codec::lz4::Lz4Codec;
        use consus_compression::codec::traits::{Codec, CompressionLevel};

        let codec = Lz4Codec;
        let compressed = codec
            .compress(&data, CompressionLevel::default())
            .expect("lz4 compression must succeed");
        let decompressed = codec
            .decompress(&compressed, data.len())
            .expect("lz4 decompression must succeed");

        prop_assert_eq!(decompressed, data);
    }

    /// Invariant:
    /// Zstd decompress(compress(data)) == data for arbitrary byte input.
    #[test]
    fn zstd_roundtrip_arbitrary_data(
        data in compression_payload_strategy()
    ) {
        use consus_compression::codec::traits::{Codec, CompressionLevel};
        use consus_compression::codec::zstd::ZstdCodec;

        let codec = ZstdCodec;
        let compressed = codec
            .compress(&data, CompressionLevel(3))
            .expect("zstd compression must succeed");
        let decompressed = codec
            .decompress(&compressed, data.len())
            .expect("zstd decompression must succeed");

        prop_assert_eq!(decompressed, data);
    }
}

// -----------------------------------------------------------------------------
// Arrow and Parquet schema conversion properties
// -----------------------------------------------------------------------------

#[cfg(feature = "arrow")]
proptest! {
    /// Invariant:
    /// Arrow-to-core conversion preserves field cardinality.
    #[test]
    fn arrow_schema_field_count_preserved(
        num_fields in 1usize..=20
    ) {
        use consus_arrow::{ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowSchema};
        use consus_core::{ByteOrder, Datatype};
        use core::num::NonZeroUsize;

        let mut fields = Vec::with_capacity(num_fields);
        for i in 0..num_fields {
            let field = ArrowFieldBuilder::new(
                ArrowFieldId::new(i as u32 + 1),
                format!("field_{i}"),
                ArrowFieldKind::Int,
                Datatype::Integer {
                    bits: NonZeroUsize::new(32).expect("32 is non-zero"),
                    byte_order: ByteOrder::LittleEndian,
                    signed: true,
                },
            )
            .nullable(true)
            .build()
            .expect("field construction must succeed");

            fields.push(field);
        }

        let schema = ArrowSchema::new(fields);
        let core_pairs = consus_arrow::conversion::arrow_schema_to_core_pairs(&schema);

        prop_assert_eq!(core_pairs.len(), num_fields);
    }
}

#[cfg(feature = "parquet")]
proptest! {
    /// Invariant:
    /// Parquet-to-core conversion preserves column cardinality.
    #[test]
    fn parquet_schema_column_count_preserved(
        num_cols in 1usize..=20
    ) {
        use consus_parquet::{FieldDescriptor, FieldId, ParquetPhysicalType, SchemaDescriptor};

        let mut fields = Vec::with_capacity(num_cols);
        for i in 0..num_cols {
            fields.push(FieldDescriptor::required(
                FieldId::new(i as u32 + 1),
                format!("col_{i}"),
                ParquetPhysicalType::Int32,
            ));
        }

        let schema = SchemaDescriptor::new(fields);
        let core_pairs = consus_parquet::conversion::parquet_schema_to_core_pairs(&schema);

        prop_assert_eq!(core_pairs.len(), num_cols);
    }
}
