use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype, Hyperslab, HyperslabDim, Selection, Shape};
use consus_hdf5::dataset::chunk::{ChunkLocation, read_chunk_raw};
use consus_hdf5::dataset::selection::{decompose_hyperslab, map_selection_to_chunks};
use consus_io::MemCursor;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn contiguous_chunk_read_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdf5_contiguous_chunk_read");

    for bytes in [4 * 1024usize, 64 * 1024, 1024 * 1024] {
        let data: Vec<u8> = (0..bytes).map(|i| (i & 0xFF) as u8).collect();
        let cursor = MemCursor::from_bytes(data.clone());
        let location = ChunkLocation {
            address: 0,
            size: bytes as u64,
            filter_mask: 0,
        };
        let registry = consus_compression::DefaultCodecRegistry::new();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bytes), &bytes, |b, &_size| {
            b.iter(|| {
                let result = read_chunk_raw(&cursor, &location, bytes, &[], &registry, None)
                    .expect("contiguous chunk read must succeed");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn undefined_chunk_fill_value_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdf5_undefined_chunk_fill_value");
    let registry = consus_compression::DefaultCodecRegistry::new();
    let location = ChunkLocation {
        address: consus_hdf5::constants::UNDEFINED_ADDRESS,
        size: 0,
        filter_mask: 0,
    };

    for bytes in [4 * 1024usize, 64 * 1024, 1024 * 1024] {
        let fill_value = [0xAB, 0xCD, 0xEF, 0x01];
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bytes), &bytes, |b, &_size| {
            b.iter(|| {
                let result = read_chunk_raw(
                    &MemCursor::new(),
                    &location,
                    bytes,
                    &[],
                    &registry,
                    Some(&fill_value),
                )
                .expect("undefined chunk fill-value expansion must succeed");
                black_box(result);
            });
        });
    }

    group.finish();
}

fn selection_decomposition_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdf5_selection_decomposition");

    let dataset_shape = Shape::fixed(&[512, 512, 64]);
    let chunk_dims = [64usize, 64, 16];

    let simple = Hyperslab::new(&[
        HyperslabDim::range(32, 384),
        HyperslabDim::range(48, 320),
        HyperslabDim::range(4, 40),
    ]);

    let strided = Hyperslab::new(&[
        HyperslabDim {
            start: 0,
            stride: 32,
            count: 16,
            block: 8,
        },
        HyperslabDim {
            start: 8,
            stride: 24,
            count: 16,
            block: 8,
        },
        HyperslabDim {
            start: 0,
            stride: 8,
            count: 8,
            block: 4,
        },
    ]);

    group.bench_function("decompose_simple_hyperslab", |b| {
        b.iter(|| {
            let slices = decompose_hyperslab(black_box(&simple), black_box(&chunk_dims))
                .expect("simple hyperslab decomposition must succeed");
            let total_elements: usize = slices.iter().map(|s| s.num_elements()).sum();
            black_box((slices.len(), total_elements));
        });
    });

    group.bench_function("decompose_strided_hyperslab", |b| {
        b.iter(|| {
            let slices = decompose_hyperslab(black_box(&strided), black_box(&chunk_dims))
                .expect("strided hyperslab decomposition must succeed");
            let total_elements: usize = slices.iter().map(|s| s.num_elements()).sum();
            black_box((slices.len(), total_elements));
        });
    });

    group.bench_function("map_selection_all", |b| {
        b.iter(|| {
            let slices = map_selection_to_chunks(
                black_box(&Selection::All),
                black_box(&dataset_shape),
                black_box(&chunk_dims),
            )
            .expect("full selection mapping must succeed");
            let total_elements: usize = slices.iter().map(|s| s.num_elements()).sum();
            black_box((slices.len(), total_elements));
        });
    });

    group.bench_function("map_selection_hyperslab", |b| {
        let selection = Selection::Hyperslab(strided.clone());
        b.iter(|| {
            let slices = map_selection_to_chunks(
                black_box(&selection),
                black_box(&dataset_shape),
                black_box(&chunk_dims),
            )
            .expect("hyperslab selection mapping must succeed");
            let total_elements: usize = slices.iter().map(|s| s.num_elements()).sum();
            black_box((slices.len(), total_elements));
        });
    });

    group.finish();
}

fn datatype_metadata_baseline_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdf5_metadata_baseline");

    let datatype = Datatype::Integer {
        bits: NonZeroUsize::new(32).expect("32 is non-zero"),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::fixed(&[1024, 1024]);

    group.bench_function("shape_and_datatype_access", |b| {
        b.iter(|| {
            let rank = shape.rank();
            let dims = shape.current_dims();
            let element_bits = match &datatype {
                Datatype::Integer { bits, .. } => bits.get(),
                _ => 0,
            };
            black_box((rank, dims.len(), element_bits));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    contiguous_chunk_read_benchmark,
    undefined_chunk_fill_value_benchmark,
    selection_decomposition_benchmark,
    datatype_metadata_baseline_benchmark
);
criterion_main!(benches);
