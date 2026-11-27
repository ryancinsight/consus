use consus_hdf5::address::ParseContext;
use consus_hdf5::heap::resolve_vl_references;
use consus_io::MemCursor;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

const REFERENCE_COUNT: usize = 64;
const PAYLOAD: [u8; 64] = [b'v'; 64];

fn global_heap_fixture() -> (MemCursor, Vec<u8>) {
    let context = ParseContext::new(8, 8);
    let reference_size = 4 + context.offset_bytes() + 4;
    let mut collection = vec![0_u8; 112];
    collection[0..4].copy_from_slice(b"GCOL");
    collection[4] = 1;
    collection[8..16].copy_from_slice(&112_u64.to_le_bytes());
    collection[16..18].copy_from_slice(&1_u16.to_le_bytes());
    collection[18..20].copy_from_slice(&1_u16.to_le_bytes());
    collection[24..32].copy_from_slice(&64_u64.to_le_bytes());
    collection[32..96].copy_from_slice(&PAYLOAD);

    let mut references = vec![0_u8; REFERENCE_COUNT * reference_size];
    for reference in references.chunks_exact_mut(reference_size) {
        reference[0..4].copy_from_slice(&1_u32.to_le_bytes());
        reference[4..12].copy_from_slice(&0_u64.to_le_bytes());
        reference[12..16].copy_from_slice(&1_u32.to_le_bytes());
    }

    (MemCursor::from_bytes(collection), references)
}

fn resolve_repeated_global_heap_references(c: &mut Criterion) {
    let context = ParseContext::new(8, 8);
    let (source, references) = global_heap_fixture();
    let mut group = c.benchmark_group("hdf5_global_heap_reference_resolution");
    group.throughput(Throughput::Elements(64));
    group.bench_function("resolve_64_references_from_one_collection", |b| {
        b.iter(|| {
            let values =
                resolve_vl_references(black_box(&source), black_box(&references), &context)
                    .expect("global heap references must resolve");
            black_box(values);
        });
    });
    group.finish();
}

criterion_group!(benches, resolve_repeated_global_heap_references);
criterion_main!(benches);
