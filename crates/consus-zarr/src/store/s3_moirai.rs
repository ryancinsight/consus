//! Native S3-backed [`Store`] (ADR-015): object storage over the moirai HTTP/1.1
//! client + SigV4, with **no tokio and no AWS SDK**. Mirrors [`super::s3::S3Store`]
//! behaviour (prefix handling, relative-key listing) but drives the async
//! [`S3Client`] with `moirai::global().block_on` to satisfy the synchronous
//! [`Store`] contract.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use consus_core::Result;
use consus_io::S3Client;

use crate::store::Store;

/// S3-compatible Zarr store backed by the native moirai S3 client (tokio-free).
///
/// ```ignore
/// let store = S3MoiraiStore::new(
///     "http://127.0.0.1:9000", "us-east-1", "minioadmin", "minioadmin", "my-bucket",
/// )
/// .with_prefix("experiment_001.zarr/");
/// ```
#[derive(Clone)]
pub struct S3MoiraiStore {
    client: Arc<S3Client>,
    /// Prefix applied to all keys (e.g. `"my.zarr/"`).
    prefix: String,
    /// Verify objects are readable immediately after write (HEAD).
    read_after_write: bool,
}

impl S3MoiraiStore {
    /// New store for `bucket` at `endpoint` (e.g. `https://s3.us-east-1.amazonaws.com`
    /// or `http://127.0.0.1:9000`), with its own HTTP client.
    #[must_use]
    pub fn new(
        endpoint: impl Into<String>,
        region: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        bucket: impl Into<String>,
    ) -> Self {
        let client = S3Client::with_endpoint(
            endpoint.into(),
            region.into(),
            access_key.into(),
            secret_key.into(),
            None,
            bucket.into(),
        );
        Self {
            client: Arc::new(client),
            prefix: String::new(),
            read_after_write: true,
        }
    }

    /// Build from an existing [`S3Client`] (shared HTTP connection pool).
    #[must_use]
    pub fn from_client(client: Arc<S3Client>) -> Self {
        Self {
            client,
            prefix: String::new(),
            read_after_write: true,
        }
    }

    /// Set the key prefix applied to all operations.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Toggle read-after-write verification (default `true`).
    #[must_use]
    pub fn with_read_after_write(mut self, enabled: bool) -> Self {
        self.read_after_write = enabled;
        self
    }

    fn full_key(&self, key: &str) -> String {
        alloc::format!("{}{}", self.prefix, key)
    }
}

impl Store for S3MoiraiStore {
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let full = self.full_key(key);
        moirai::global().block_on(self.client.get(&full))
    }

    fn set(&mut self, key: &str, value: &[u8]) -> Result<()> {
        let full = self.full_key(key);
        moirai::global().block_on(self.client.put(&full, value))?;
        if self.read_after_write {
            moirai::global().block_on(self.client.head_len(&full))?;
        }
        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        let full = self.full_key(key);
        moirai::global().block_on(self.client.delete(&full))
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);
        let mut keys = moirai::global().block_on(self.client.list(&full_prefix))?;
        if !self.prefix.is_empty() {
            keys = keys
                .into_iter()
                .map(|k| k.strip_prefix(&self.prefix).unwrap_or(&k).to_string())
                .collect();
        }
        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    fn contains(&self, key: &str) -> Result<bool> {
        let full = self.full_key(key);
        moirai::global().block_on(self.client.exists(&full))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;

    /// Minimal in-memory S3 (PUT/GET/HEAD/DELETE/ListObjectsV2, path-style).
    fn spawn_mock() -> u16 {
        let store: Arc<Mutex<BTreeMap<String, Vec<u8>>>> = Arc::new(Mutex::new(BTreeMap::new()));
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut stream) = conn else { break };
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    let mut acc = Vec::new();
                    let mut tmp = [0u8; 4096];
                    loop {
                        let hend = loop {
                            if let Some(p) = acc.windows(4).position(|w| w == b"\r\n\r\n") {
                                break Some(p + 4);
                            }
                            match stream.read(&mut tmp) {
                                Ok(0) => break None,
                                Ok(n) => acc.extend_from_slice(&tmp[..n]),
                                Err(_) => break None,
                            }
                        };
                        let Some(hend) = hend else { break };
                        let head = String::from_utf8_lossy(&acc[..hend]).into_owned();
                        let mut it = head.split_whitespace();
                        let method = it.next().unwrap_or("").to_string();
                        let target = it.next().unwrap_or("").to_string();
                        let clen: usize = head
                            .lines()
                            .find_map(|l| {
                                let (k, v) = l.split_once(':')?;
                                k.trim()
                                    .eq_ignore_ascii_case("content-length")
                                    .then(|| v.trim())
                            })
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

                        let (path, query) = target.split_once('?').unwrap_or((&target, ""));
                        let key = path.splitn(3, '/').nth(2).unwrap_or("");
                        let resp = respond(&store, &method, key, query, &body);
                        if stream.write_all(&resp).is_err() {
                            break;
                        }
                    }
                });
            }
        });
        port
    }

    fn respond(
        store: &Mutex<BTreeMap<String, Vec<u8>>>,
        method: &str,
        key: &str,
        query: &str,
        body: &[u8],
    ) -> Vec<u8> {
        let with_body = |status: &str, p: &[u8]| {
            let mut r =
                format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n\r\n", p.len()).into_bytes();
            r.extend_from_slice(p);
            r
        };
        let empty =
            |status: &str| format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").into_bytes();
        let decode = |s: &str| -> String {
            let b = s.as_bytes();
            let mut out = Vec::new();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'%' && i + 2 < b.len() {
                    if let Ok(x) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                        out.push(x);
                        i += 3;
                        continue;
                    }
                }
                out.push(b[i]);
                i += 1;
            }
            String::from_utf8_lossy(&out).into_owned()
        };
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
                let prefix = decode(
                    query
                        .split('&')
                        .find_map(|kv| kv.strip_prefix("prefix="))
                        .unwrap_or(""),
                );
                let g = store.lock().unwrap();
                let mut xml = String::from(
                    "<?xml version=\"1.0\"?><ListBucketResult><IsTruncated>false</IsTruncated>",
                );
                for k in g.keys().filter(|k| k.starts_with(prefix.as_str())) {
                    xml.push_str(&format!("<Contents><Key>{k}</Key></Contents>"));
                }
                xml.push_str("</ListBucketResult>");
                with_body("200 OK", xml.as_bytes())
            }
            "GET" => match store.lock().unwrap().get(key) {
                Some(v) => with_body("200 OK", v),
                None => empty("404 Not Found"),
            },
            _ => empty("400 Bad Request"),
        }
    }

    #[test]
    fn s3_moirai_store_round_trip_with_prefix() {
        let port = spawn_mock();
        let mut store = S3MoiraiStore::new(
            format!("http://127.0.0.1:{port}"),
            "us-east-1",
            "test",
            "secret",
            "bucket",
        )
        .with_prefix("exp.zarr/");

        // set / get
        store.set("arr/.zarray", b"{\"shape\":[4]}").expect("set");
        assert_eq!(store.get("arr/.zarray").expect("get"), b"{\"shape\":[4]}");
        assert!(store.contains("arr/.zarray").expect("contains"));
        assert!(!store.contains("missing").expect("contains missing"));

        // list returns RELATIVE keys (store prefix stripped)
        store.set("arr/0", b"chunk0").expect("set chunk");
        let mut listed = store.list("arr/").expect("list");
        listed.sort();
        assert_eq!(
            listed,
            vec!["arr/.zarray".to_string(), "arr/0".to_string()],
            "list must strip the store prefix and return relative keys"
        );

        // delete
        store.delete("arr/0").expect("delete");
        assert!(!store.contains("arr/0").expect("contains after delete"));
    }
}
