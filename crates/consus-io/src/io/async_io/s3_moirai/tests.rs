//! Verification for the native S3 backend: a value-semantic round-trip against a
//! local mock S3 server (no MinIO/Docker required), plus the civil-date helper.
//! The SigV4 algorithm itself is covered by the known-answer test in `sigv4`.

use super::*;
use std::io::{Read, Write};
use std::net::TcpListener as StdListener;

/// Parse `Range: bytes=START-END` (inclusive) from a request head.
fn parse_range(head: &str) -> Option<(usize, usize)> {
    for line in head.lines() {
        if let Some(v) = line
            .to_ascii_lowercase()
            .strip_prefix("range:")
            .map(str::trim)
            .and_then(|v| v.strip_prefix("bytes="))
            .map(str::to_string)
        {
            let mut it = v.split('-');
            let s = it.next()?.trim().parse().ok()?;
            let e = it.next()?.trim().parse().ok()?;
            return Some((s, e));
        }
    }
    None
}

/// Minimal mock S3 server on a std thread: HEAD → Content-Length, GET+Range → 206
/// slice. Serves one keep-alive connection until the client closes it.
fn spawn_mock_s3(object: Vec<u8>) -> u16 {
    let listener = StdListener::bind("127.0.0.1:0").expect("bind mock");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut acc = Vec::new();
        let mut tmp = [0u8; 2048];
        loop {
            // Read until a full request head is buffered.
            let head_end = loop {
                if let Some(p) = acc.windows(4).position(|w| w == b"\r\n\r\n") {
                    break Some(p + 4);
                }
                match stream.read(&mut tmp) {
                    Ok(0) => break None,
                    Ok(n) => acc.extend_from_slice(&tmp[..n]),
                    Err(_) => break None,
                }
            };
            let Some(end) = head_end else { break };
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
                b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n".to_vec()
            };
            if stream.write_all(&resp).is_err() {
                break;
            }
        }
    });
    port
}

fn config_for(port: u16) -> S3Config {
    S3Config {
        endpoint: format!("http://127.0.0.1:{port}"),
        region: "us-east-1".to_string(),
        access_key: "test".to_string(),
        secret_key: "secret".to_string(),
        session_token: None,
        bucket: "bucket".to_string(),
        key: "obj.bin".to_string(),
    }
}

#[test]
fn s3_reader_len_and_ranged_read_round_trip() {
    let object: Vec<u8> = (0..1000u32).map(|i| (i % 256) as u8).collect();
    let port = spawn_mock_s3(object.clone());
    let reader = S3MoiraiReader::new(config_for(port));

    let rt = moirai::global();
    rt.block_on(async {
        // HeadObject -> length.
        let len = reader.len().await.expect("HEAD len");
        assert_eq!(len, 1000, "len must equal object size");

        // GetObject Range -> exact bytes.
        let mut buf = [0u8; 50];
        reader.read_at(100, &mut buf).await.expect("ranged read");
        assert_eq!(
            buf.as_slice(),
            &object[100..150],
            "ranged read must return the exact object bytes"
        );
    });
}

#[test]
fn civil_from_days_matches_known_dates() {
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    // 2013-05-24 is 15849 days after the Unix epoch (matches the SigV4 KAT date).
    assert_eq!(civil_from_days(15849), (2013, 5, 24));
}
