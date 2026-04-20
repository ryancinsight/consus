use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype, Hyperslab, HyperslabDim, Selection, Shape};
use consus_hdf5::dataset::chunk::{ChunkLocation, read_chunk_raw};
use consus_hdf5::dataset::selection::{decompose_hyperslab, map_selection_to_chunks};
use consus_hdf5::file::Hdf5File;
use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
use consus_hdf5::property_list::{DatasetCreationProps, DatasetLayout};
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
        let selection_dims: &[u16] = &[];
        let registry = consus_compression::DefaultCodecRegistry::new();

        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bytes), &bytes, |b, &_size| {
            b.iter(|| {
                let result =
                    read_chunk_raw(&cursor, &location, bytes, selection_dims, &registry, None)
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
    let selection_dims: &[u16] = &[];
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
                    selection_dims,
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

    let dataset_shape = Shape::fixed(vec![512, 512, 64]);
    let chunk_dims = vec![64usize, 64, 16];

    let simple = Hyperslab::new(
        &[
            HyperslabDim::range(32, 384),
            HyperslabDim::range(48, 320),
            HyperslabDim::range(4, 40),
        ][..],
    );

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
    let shape = Shape::fixed(&[1024, 1024][..]);

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

fn chunked_dataset_read_throughput_benchmark(c: &mut Criterion) {
    const ROWS: usize = 128;
    const COLS: usize = 128;
    const TOTAL: usize = ROWS * COLS;

    let data: Vec<u8> = (0..TOTAL).map(|i| i as u8).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[ROWS, COLS][..]);
    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![32, 32]),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let addr = builder
        .add_dataset("chunked_u8", &dt, &shape, &data, &dcpl)
        .expect("build chunked dataset");
    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let mut group = c.benchmark_group("hdf5_chunked_dataset_read_throughput");
    group.throughput(Throughput::Bytes(TOTAL as u64));
    group.bench_function("read_all_bytes_128x128_u8", |b| {
        b.iter(|| {
            let result = file
                .read_chunked_dataset_all_bytes(black_box(addr))
                .expect("read chunked dataset must succeed");
            black_box(result);
        });
    });
    group.finish();
}

fn compressed_chunked_dataset_read_benchmark(c: &mut Criterion) {
    const ROWS: usize = 64;
    const COLS: usize = 64;
    const TOTAL: usize = ROWS * COLS;

    let data: Vec<u8> = (0..TOTAL).map(|i| i as u8).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[ROWS, COLS][..]);
    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![16, 16]),
        filters: vec![1],
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let addr = builder
        .add_dataset("compressed_u8", &dt, &shape, &data, &dcpl)
        .expect("build compressed chunked dataset");
    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let mut group = c.benchmark_group("hdf5_compressed_chunked_dataset_read");
    group.throughput(Throughput::Bytes(TOTAL as u64));
    group.bench_function("read_all_bytes_64x64_deflate", |b| {
        b.iter(|| {
            let result = file
                .read_chunked_dataset_all_bytes(black_box(addr))
                .expect("read compressed chunked dataset must succeed");
            black_box(result);
        });
    });
    group.finish();
}

fn contiguous_dataset_throughput_benchmark(c: &mut Criterion) {
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_hdf5::file::Hdf5File;
    use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
    use consus_hdf5::property_list::{DatasetCreationProps, DatasetLayout};
    use consus_io::MemCursor;
    use std::hint::black_box;

    const ROWS: usize = 1024;
    const COLS: usize = 1024;
    let total_bytes = ROWS * COLS;

    let dt = Datatype::Integer {
        bits: core::num::NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[ROWS, COLS][..]);
    let raw: Vec<u8> = (0..total_bytes).map(|i| (i % 256) as u8).collect();

    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Contiguous,
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("bench", &dt, &shape, &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).unwrap();
    let datasets = file.list_root_group().unwrap();
    let (_, addr, _) = &datasets[0];
    let dataset = file.dataset_at(*addr).unwrap();
    let data_address = dataset.data_address.unwrap();

    let mut group = c.benchmark_group("contiguous_dataset_throughput");
    group.throughput(criterion::Throughput::Bytes(total_bytes as u64));
    group.bench_function("1MB_contiguous_read", |b| {
        b.iter(|| {
            let mut buf = vec![0u8; total_bytes];
            file.read_contiguous_dataset_bytes(data_address, 0, &mut buf)
                .unwrap();
            black_box(&buf);
        });
    });
    group.finish();
}

fn v4_chunked_dataset_throughput_benchmark(c: &mut Criterion) {
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_hdf5::file::Hdf5File;
    use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
    use consus_hdf5::property_list::{DatasetCreationProps, DatasetLayout};
    use consus_io::MemCursor;
    use std::hint::black_box;

    const ROWS: usize = 128;
    const COLS: usize = 128;
    let total_bytes = ROWS * COLS;

    let dt = Datatype::Integer {
        bits: core::num::NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[ROWS, COLS][..]);
    let raw: Vec<u8> = (0..total_bytes).map(|i| (i % 256) as u8).collect();

    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![32, 32]),
        layout_version: Some(4),
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("bench", &dt, &shape, &raw, &dcpl)
        .unwrap();
    let bytes = builder.finish().unwrap();

    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).unwrap();
    let datasets = file.list_root_group().unwrap();
    let addr = datasets[0].1;

    let mut group = c.benchmark_group("v4_chunked_dataset_throughput");
    group.throughput(criterion::Throughput::Bytes(total_bytes as u64));
    group.bench_function("128x128_v4_btree_v2", |b| {
        b.iter(|| {
            let result = file.read_chunked_dataset_all_bytes(addr).unwrap();
            black_box(&result);
        });
    });
    group.finish();
}

fn zstd_compressed_chunked_dataset_read_benchmark(c: &mut Criterion) {
    use consus_core::Compression;

    const ROWS: usize = 64;
    const COLS: usize = 64;
    const TOTAL: usize = ROWS * COLS;

    let data: Vec<u8> = (0..TOTAL).map(|i| i as u8).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[ROWS, COLS][..]);
    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![16, 16]),
        compression: Compression::Zstd { level: 3 },
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let addr = builder
        .add_dataset("zstd_u8", &dt, &shape, &data, &dcpl)
        .expect("build zstd compressed dataset");
    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let mut group = c.benchmark_group("hdf5_zstd_compressed_chunked_dataset_read");
    group.throughput(Throughput::Bytes(TOTAL as u64));
    group.bench_function("read_all_bytes_64x64_zstd", |b| {
        b.iter(|| {
            let result = file
                .read_chunked_dataset_all_bytes(black_box(addr))
                .expect("read zstd compressed dataset must succeed");
            black_box(result);
        });
    });
    group.finish();
}

fn lz4_compressed_chunked_dataset_read_benchmark(c: &mut Criterion) {
    use consus_core::Compression;

    const ROWS: usize = 64;
    const COLS: usize = 64;
    const TOTAL: usize = ROWS * COLS;

    let data: Vec<u8> = (0..TOTAL).map(|i| i as u8).collect();
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[ROWS, COLS][..]);
    let dcpl = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![16, 16]),
        compression: Compression::Lz4,
        ..DatasetCreationProps::default()
    };

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let addr = builder
        .add_dataset("lz4_u8", &dt, &shape, &data, &dcpl)
        .expect("build lz4 compressed dataset");
    let bytes = builder.finish().expect("finish file");
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");

    let mut group = c.benchmark_group("hdf5_lz4_compressed_chunked_dataset_read");
    group.throughput(Throughput::Bytes(TOTAL as u64));
    group.bench_function("read_all_bytes_64x64_lz4", |b| {
        b.iter(|| {
            let result = file
                .read_chunked_dataset_all_bytes(black_box(addr))
                .expect("read lz4 compressed dataset must succeed");
            black_box(result);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    contiguous_chunk_read_benchmark,
    undefined_chunk_fill_value_benchmark,
    selection_decomposition_benchmark,
    datatype_metadata_baseline_benchmark,
    chunked_dataset_read_throughput_benchmark,
    compressed_chunked_dataset_read_benchmark,
    contiguous_dataset_throughput_benchmark,
    v4_chunked_dataset_throughput_benchmark,
    zstd_compressed_chunked_dataset_read_benchmark,
    lz4_compressed_chunked_dataset_read_benchmark
);
criterion_main!(benches);
