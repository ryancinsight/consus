//! ADR-015 P4 (correctness gate): the native moirai S3 reader must produce
//! byte-identical results to the legacy rusoto reader for `GetObject(Range)` and
//! `HeadObject`. Both clients hit one in-process mock S3 endpoint, so this runs
//! without Docker/MinIO (the comparative *performance* benchmark against MinIO is
//! a separate CI job; this test is the functional differential).

#![cfg(all(feature = "s3", feature = "s3-moirai"))]

use std::io::{Read, Write};
use std::net::TcpListener;

use consus_io::{AsyncLength, AsyncReadAt, S3Config, S3MoiraiReader, S3Reader};
use rusoto_core::Region;

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
                        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", object.len())
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
    // Static credentials so rusoto's default chain resolves (env provider); the
    // mock ignores auth, but rusoto still requires credentials to sign.
    // SAFETY: single-threaded test setup before any reader is constructed.
    unsafe {
        std::env::set_var("AWS_ACCESS_KEY_ID", "test");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "secret");
    }

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
    let rusoto_reader = S3Reader::new(
        Region::Custom {
            name: "local".to_string(),
            endpoint: endpoint.clone(),
        },
        "bucket",
        "obj.bin",
    );
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
    assert_eq!(moirai_len, object.len() as u64, "moirai len must equal object size");
    assert_eq!(moirai_len, rusoto_len, "moirai and rusoto len must agree");
}
