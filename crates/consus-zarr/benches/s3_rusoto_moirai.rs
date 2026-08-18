//! Comparative S3 range-read benchmark for atlas ADR-0045 P4.
//!
//! The benchmark deliberately measures both clients against the same live
//! S3-compatible endpoint, object, range, and credentials. It is intended for
//! the MinIO CI lane; local runs fail clearly when the endpoint credentials are
//! not supplied rather than silently measuring a mock or a skipped benchmark.

use std::sync::Arc;

use consus_io::{AsyncReadAt, S3Config, S3MoiraiReader, S3Reader};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use rusoto_core::{HttpClient, Region, credential::StaticProvider};
use rusoto_s3::S3Client as RusotoS3Client;

const DEFAULT_OBJECT_LEN: usize = 1024 * 1024;
const DEFAULT_RANGE_LEN: usize = 256 * 1024;

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!("{name} is required for the ADR-0045 S3 benchmark; run it in the MinIO lane")
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn deterministic_object(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| u8::try_from((index * 31 + 17) % 251).expect("modulo 251 fits in u8"))
        .collect()
}

fn benchmark_s3_range_reads(c: &mut Criterion) {
    let endpoint = required_env("S3_TEST_ENDPOINT");
    let bucket = std::env::var("S3_TEST_BUCKET").unwrap_or_else(|_| "consus-test".to_string());
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let access_key = required_env("AWS_ACCESS_KEY_ID");
    let secret_key = required_env("AWS_SECRET_ACCESS_KEY");
    let key = std::env::var("S3_BENCH_KEY").unwrap_or_else(|_| "benchmark.bin".to_string());
    let object_len = env_usize("S3_BENCH_OBJECT_LEN", DEFAULT_OBJECT_LEN);
    let range_len = env_usize("S3_BENCH_RANGE_LEN", DEFAULT_RANGE_LEN);
    assert!(object_len > 0, "S3_BENCH_OBJECT_LEN must be positive");
    assert!(
        range_len > 0 && range_len <= object_len,
        "S3_BENCH_RANGE_LEN must be in 1..=S3_BENCH_OBJECT_LEN"
    );
    let position = (object_len - range_len) / 3;
    let object = deterministic_object(object_len);

    // Upload and validate once, outside the measured region. Both readers then
    // consume the same immutable object and byte range.
    let moirai = moirai::global();
    let native_client = consus_io::S3Client::with_endpoint(
        endpoint.clone(),
        region.clone(),
        access_key.clone(),
        secret_key.clone(),
        std::env::var("AWS_SESSION_TOKEN").ok(),
        bucket.clone(),
    );
    moirai
        .block_on(native_client.put(&key, &object))
        .expect("upload benchmark object through native backend");
    assert_eq!(
        moirai
            .block_on(native_client.head_len(&key))
            .expect("HEAD benchmark object"),
        object_len as u64,
        "benchmark object length must match the generated source"
    );

    let native_reader = S3MoiraiReader::new(S3Config {
        endpoint: endpoint.clone(),
        region: region.clone(),
        access_key: access_key.clone(),
        secret_key: secret_key.clone(),
        session_token: std::env::var("AWS_SESSION_TOKEN").ok(),
        bucket: bucket.clone(),
        key: key.clone(),
    });
    let rusoto_client = RusotoS3Client::new_with(
        HttpClient::new().expect("construct rusoto HTTP dispatcher"),
        StaticProvider::new_minimal(access_key, secret_key),
        Region::Custom {
            name: region,
            endpoint,
        },
    );
    let rusoto_reader = S3Reader::with_client(Arc::new(rusoto_client), bucket, key);
    let tokio = tokio::runtime::Runtime::new().expect("construct tokio runtime");
    let offset = position as u64;

    let mut group = c.benchmark_group("s3_range_read");
    group.throughput(Throughput::Bytes(range_len as u64));
    group.bench_function("native_moirai", |b| {
        b.iter(|| {
            let mut buf = vec![0u8; range_len];
            moirai
                .block_on(native_reader.read_at(offset, &mut buf))
                .expect("native ranged read");
            black_box(buf);
        });
    });
    group.bench_function("legacy_rusoto", |b| {
        b.iter(|| {
            let mut buf = vec![0u8; range_len];
            tokio
                .block_on(rusoto_reader.read_at(offset, &mut buf))
                .expect("legacy ranged read");
            black_box(buf);
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_s3_range_reads);
criterion_main!(benches);
