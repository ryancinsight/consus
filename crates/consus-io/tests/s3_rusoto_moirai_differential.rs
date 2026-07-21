//! ADR-015 P4 (correctness gate): the native moirai S3 reader must produce
//! byte-identical results to the legacy rusoto reader for `GetObject(Range)` and
//! `HeadObject`. Both clients hit one in-process mock S3 endpoint, so this runs
//! without Docker/MinIO (the comparative *performance* benchmark against MinIO is
//! a separate CI job; this test is the functional differential).

#![cfg(all(feature = "s3", feature = "s3-moirai"))]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

use consus_io::{AsyncLength, AsyncReadAt, S3Client, S3Config, S3MoiraiReader, S3Reader};
use rusoto_core::{HttpClient, Region, credential::StaticProvider};
use rusoto_s3::S3Client as RusotoS3Client;

/// Parse an inclusive `Range: bytes=START-END` from a request head.
fn parse_range(head: &str) -> Option<(usize, usize)> {
    for line in head.lines() {
        if let Some(spec) = line
            .to_ascii_lowercase()
            .strip_prefix("range:")
            .map(str::trim)
            .and_then(|v| v.strip_prefix("bytes="))
            .map(str::to_string)
        {
            let mut it = spec.split('-');
            let s = it.next()?.trim().parse().ok()?;
            let e = it.next()?.trim().parse().ok()?;
            return Some((s, e));
        }
    }
    None
}

/// Mock S3 endpoint: lenient on path/auth/addressing (path-style or virtual-host),
/// serving `object` for any `GET`+Range (206) and `HEAD` (200 + Content-Length).
/// Accepts connections indefinitely on a detached thread (dies at process exit).
fn spawn_mock_s3(object: Vec<u8>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let object = object.clone();
            std::thread::spawn(move || {
                let mut acc = Vec::new();
                let mut tmp = [0u8; 2048];
                loop {
                    let end = loop {
                        if let Some(p) = acc.windows(4).position(|w| w == b"\r\n\r\n") {
                            break Some(p + 4);
                        }
                        match stream.read(&mut tmp) {
                            Ok(0) => break None,
                            Ok(n) => acc.extend_from_slice(&tmp[..n]),
                            Err(_) => break None,
                        }
                    };
                    let Some(end) = end else { break };
                    let head = String::from_utf8_lossy(&acc[..end]).into_owned();
                    acc.drain(..end);
                    let method = head.split_whitespace().next().unwrap_or("").to_string();

                    let resp = if method == "HEAD" {
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                            object.len()
                        )
                        .into_bytes()
                    } else if let Some((s, e)) = parse_range(&head) {
                        let e = e.min(object.len().saturating_sub(1));
                        let slice = &object[s..=e];
                        let mut r = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\r\n",
                            slice.len()
                        )
                        .into_bytes();
                        r.extend_from_slice(slice);
                        r
                    } else {
                        // Full GET fallback (rusoto HeadObject sometimes precedes).
                        let mut r = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                            object.len()
                        )
                        .into_bytes();
                        r.extend_from_slice(&object);
                        r
                    };
                    if stream.write_all(&resp).is_err() {
                        break;
                    }
                }
            });
        }
    });
    port
}

#[test]
fn rusoto_and_moirai_read_byte_identical() {
    // Deterministic object with a non-trivial byte pattern.
    let object: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    let port = spawn_mock_s3(object.clone());
    let endpoint = format!("http://127.0.0.1:{port}");
    let (pos, len) = (1000usize, 1500usize);

    // ── moirai reader (no tokio) ─────────────────────────────────────────────
    let moirai_reader = S3MoiraiReader::new(S3Config {
        endpoint: endpoint.clone(),
        region: "us-east-1".to_string(),
        access_key: "test".to_string(),
        secret_key: "secret".to_string(),
        session_token: None,
        bucket: "bucket".to_string(),
        key: "obj.bin".to_string(),
    });
    let rt = moirai::global();
    let mut moirai_buf = vec![0u8; len];
    rt.block_on(moirai_reader.read_at(pos as u64, &mut moirai_buf))
        .expect("moirai read_at");
    let moirai_len = rt.block_on(moirai_reader.len()).expect("moirai len");

    // ── rusoto reader (legacy, tokio runtime) ────────────────────────────────
    let rusoto_client = RusotoS3Client::new_with(
        HttpClient::new().expect("construct rusoto HTTP dispatcher"),
        StaticProvider::new_minimal("test".to_string(), "secret".to_string()),
        Region::Custom {
            name: "local".to_string(),
            endpoint: endpoint.clone(),
        },
    );
    let rusoto_reader = S3Reader::with_client(Arc::new(rusoto_client), "bucket", "obj.bin");
    let trt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut rusoto_buf = vec![0u8; len];
    trt.block_on(rusoto_reader.read_at(pos as u64, &mut rusoto_buf))
        .expect("rusoto read_at");
    let rusoto_len = trt.block_on(rusoto_reader.len()).expect("rusoto len");

    // ── Differential: both must equal the source bytes and each other ────────
    assert_eq!(
        moirai_buf,
        &object[pos..pos + len],
        "moirai read must equal source bytes"
    );
    assert_eq!(
        moirai_buf, rusoto_buf,
        "moirai and rusoto must read byte-identical ranges"
    );
    assert_eq!(
        moirai_len,
        object.len() as u64,
        "moirai len must equal object size"
    );
    assert_eq!(moirai_len, rusoto_len, "moirai and rusoto len must agree");
}

/// Real-endpoint gate (CI with MinIO/S3): exercises the native moirai S3 reader
/// against a live S3-compatible server using path-style addressing + SigV4.
/// Skips when `S3_TEST_ENDPOINT` is unset (i.e. local runs without a server).
///
/// CI sets `S3_TEST_ENDPOINT`, `S3_TEST_BUCKET`, `S3_TEST_KEY`,
/// `S3_TEST_OBJECT_LEN`, and `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, having
/// authorized creation of an object with `S3_TEST_OBJECT_LEN` bytes.
#[test]
fn moirai_s3_real_endpoint() {
    let Ok(endpoint) = std::env::var("S3_TEST_ENDPOINT") else {
        eprintln!("S3_TEST_ENDPOINT unset — skipping real-endpoint moirai S3 test");
        return;
    };
    let bucket = std::env::var("S3_TEST_BUCKET").unwrap_or_else(|_| "consus-test".to_string());
    let key = std::env::var("S3_TEST_KEY").unwrap_or_else(|_| "object.bin".to_string());
    let object_len: usize = std::env::var("S3_TEST_OBJECT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let access_key = std::env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID");
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY");
    let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
    let client = S3Client::with_endpoint(
        endpoint.clone(),
        region.clone(),
        access_key.clone(),
        secret_key.clone(),
        session_token.clone(),
        bucket.clone(),
    );
    let object: Vec<u8> = (0..object_len)
        .map(|index| u8::try_from((index * 31 + 17) % 251).expect("modulo 251 fits in one byte"))
        .collect();
    let rt = moirai::global();
    rt.block_on(client.put(&key, &object)).expect("real PUT");

    let reader = S3MoiraiReader::new(S3Config {
        endpoint,
        region,
        access_key,
        secret_key,
        session_token,
        bucket,
        key,
    });

    // HEAD: length matches the uploaded object.
    let len = rt.block_on(reader.len()).expect("real HEAD");
    assert_eq!(
        len as usize, object_len,
        "HEAD length must match uploaded object"
    );

    // Ranged GET: exact requested length, and re-reading the same range is
    // deterministic (real round-trips against the live server).
    let (pos, n) = (17usize, (object_len / 2).max(1));
    let mut a = vec![0u8; n];
    let mut b = vec![0u8; n];
    let read_offset = u64::try_from(pos).expect("test offset fits in u64");
    rt.block_on(reader.read_at(read_offset, &mut a))
        .expect("real GET #1");
    rt.block_on(reader.read_at(read_offset, &mut b))
        .expect("real GET #2");
    assert_eq!(
        a,
        &object[pos..pos + n],
        "ranged read must equal the uploaded source bytes"
    );
    assert_eq!(a, b, "repeated ranged reads must be byte-identical");
}
