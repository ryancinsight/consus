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

// ── Stateful mock S3 (in-memory) for the full S3Client round-trip ─────────────

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use moirai_http::HttpClient;
use std::sync::Mutex;

/// Decode `%XX` percent-escapes (the client URI-encodes query values per SigV4).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(byte) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn header_value<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim().eq_ignore_ascii_case(name).then(|| v.trim())
    })
}

/// Minimal in-memory S3 supporting PUT/GET(+Range)/HEAD/DELETE/ListObjectsV2.
/// Path-style `/bucket/key`; ignores auth. Detached thread.
fn spawn_stateful_s3() -> u16 {
    let store: Arc<Mutex<BTreeMap<String, Vec<u8>>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let listener = StdListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                let mut acc = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
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
                    let Some(hend) = head_end else { break };
                    let head = String::from_utf8_lossy(&acc[..hend]).into_owned();
                    let mut parts = head.split_whitespace();
                    let method = parts.next().unwrap_or("").to_string();
                    let target = parts.next().unwrap_or("").to_string();

                    // Read PUT body (Content-Length) if present.
                    let clen: usize = header_value(&head, "content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let mut body = acc[hend..].to_vec();
                    while body.len() < clen {
                        match stream.read(&mut tmp) {
                            Ok(0) => break,
                            Ok(n) => body.extend_from_slice(&tmp[..n]),
                            Err(_) => break,
                        }
                    }
                    acc.clear();

                    let (raw_path, query) = target.split_once('?').unwrap_or((&target, ""));
                    // Strip leading "/bucket".
                    let after_bucket = raw_path.splitn(3, '/').nth(2).unwrap_or("");
                    let resp = build_response(&store, &method, after_bucket, query, &head, &body);
                    if stream.write_all(&resp).is_err() {
                        break;
                    }
                }
            });
        }
    });
    port
}

fn build_response(
    store: &Mutex<BTreeMap<String, Vec<u8>>>,
    method: &str,
    key: &str,
    query: &str,
    head: &str,
    body: &[u8],
) -> Vec<u8> {
    let ok = |status: &str, payload: &[u8]| {
        let mut r = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n\r\n",
            payload.len()
        )
        .into_bytes();
        r.extend_from_slice(payload);
        r
    };
    let empty =
        |status: &str| format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").into_bytes();

    match method {
        "PUT" => {
            store.lock().unwrap().insert(key.to_string(), body.to_vec());
            empty("200 OK")
        }
        "DELETE" => {
            store.lock().unwrap().remove(key);
            empty("204 No Content")
        }
        "HEAD" => match store.lock().unwrap().get(key) {
            Some(v) => {
                format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", v.len()).into_bytes()
            }
            None => empty("404 Not Found"),
        },
        "GET" if query.contains("list-type=2") => {
            let prefix_raw = query
                .split('&')
                .find_map(|kv| kv.strip_prefix("prefix="))
                .unwrap_or("");
            let prefix = percent_decode(prefix_raw);
            let prefix = prefix.as_str();
            let guard = store.lock().unwrap();
            let mut xml = String::from(
                "<?xml version=\"1.0\"?><ListBucketResult><IsTruncated>false</IsTruncated>",
            );
            for k in guard.keys().filter(|k| k.starts_with(prefix)) {
                xml.push_str(&format!("<Contents><Key>{k}</Key></Contents>"));
            }
            xml.push_str("</ListBucketResult>");
            ok("200 OK", xml.as_bytes())
        }
        "GET" => {
            let guard = store.lock().unwrap();
            match guard.get(key) {
                Some(v) => {
                    if let Some((s, e)) = parse_range(head) {
                        let e = e.min(v.len().saturating_sub(1));
                        ok("206 Partial Content", &v[s..=e])
                    } else {
                        ok("200 OK", v)
                    }
                }
                None => empty("404 Not Found"),
            }
        }
        _ => empty("400 Bad Request"),
    }
}

#[test]
fn s3_client_put_get_head_list_delete_round_trip() {
    let port = spawn_stateful_s3();
    let client = S3Client::new(
        Arc::new(HttpClient::new()),
        format!("http://127.0.0.1:{port}"),
        "us-east-1".to_string(),
        "test".to_string(),
        "secret".to_string(),
        None,
        "bucket".to_string(),
    );
    let rt = moirai::global();
    rt.block_on(async {
        let data = b"hello moirai s3".to_vec();
        client.put("a/obj.bin", &data).await.expect("put");
        assert!(client.exists("a/obj.bin").await.expect("exists"));
        assert_eq!(
            client.head_len("a/obj.bin").await.expect("head"),
            data.len() as u64
        );
        assert_eq!(client.get("a/obj.bin").await.expect("get"), data);
        assert_eq!(
            client.get_range("a/obj.bin", 6, 6).await.expect("range"),
            b"moirai"
        );

        client.put("a/two.bin", b"x").await.expect("put2");
        client.put("b/three.bin", b"y").await.expect("put3");
        let mut listed = client.list("a/").await.expect("list");
        listed.sort();
        assert_eq!(
            listed,
            vec!["a/obj.bin".to_string(), "a/two.bin".to_string()],
            "list must return only the prefixed keys"
        );

        client.delete("a/obj.bin").await.expect("delete");
        assert!(
            !client
                .exists("a/obj.bin")
                .await
                .expect("exists after delete")
        );
    });
}
