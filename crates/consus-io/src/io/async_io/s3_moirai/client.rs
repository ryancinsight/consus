//! Full async S3 client over `moirai-http` with SigV4 signing — `GetObject`
//! (full + ranged), `HeadObject`, `PutObject`, `DeleteObject`, `ListObjectsV2`.
//! No tokio, no AWS SDK. Path-style addressing (AWS + MinIO/Ceph compatible).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use std::time::{SystemTime, UNIX_EPOCH};

use consus_core::{Error, Result};
use moirai_http::{HttpClient, Response};

use super::sigv4::{
    CanonicalRequest, Credentials, EMPTY_PAYLOAD_SHA256, authorization_header, sha256_hex,
    uri_encode,
};

fn io_error(msg: String) -> Error {
    Error::Io(std::io::Error::other(msg))
}

/// Bucket-scoped async S3 client over moirai-http.
#[derive(Clone)]
pub struct S3Client {
    http: Arc<HttpClient>,
    endpoint: String,
    region: String,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
    bucket: String,
}

impl S3Client {
    /// Construct a client. `endpoint` is e.g. `https://s3.us-east-1.amazonaws.com`
    /// or `http://127.0.0.1:9000`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        http: Arc<HttpClient>,
        endpoint: String,
        region: String,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
        bucket: String,
    ) -> Self {
        Self {
            http,
            endpoint,
            region,
            access_key,
            secret_key,
            session_token,
            bucket,
        }
    }

    /// Construct a client with a fresh internal HTTP client (Mozilla roots for
    /// HTTPS). Convenience for callers that don't share a connection pool.
    #[must_use]
    pub fn with_endpoint(
        endpoint: String,
        region: String,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
        bucket: String,
    ) -> Self {
        Self::new(
            Arc::new(HttpClient::new()),
            endpoint,
            region,
            access_key,
            secret_key,
            session_token,
            bucket,
        )
    }

    fn host_header(&self) -> String {
        let (secure, authority) = if let Some(r) = self.endpoint.strip_prefix("https://") {
            (true, r.trim_end_matches('/'))
        } else if let Some(r) = self.endpoint.strip_prefix("http://") {
            (false, r.trim_end_matches('/'))
        } else {
            (true, self.endpoint.trim_end_matches('/'))
        };
        let default = if secure { ":443" } else { ":80" };
        authority
            .strip_suffix(default)
            .unwrap_or(authority)
            .to_string()
    }

    fn base(&self) -> String {
        self.endpoint.trim_end_matches('/').to_string()
    }

    /// `({url}, {canonical_uri})` for an object key (path-style).
    fn object_paths(&self, key: &str) -> (String, String) {
        let enc = uri_encode(key, false);
        (
            format!("{}/{}/{}", self.base(), self.bucket, enc),
            format!("/{}/{}", self.bucket, enc),
        )
    }

    /// Sign and send one request. `extra` are additional headers to sign (e.g.
    /// `range`). Returns the parsed response.
    #[allow(clippy::too_many_arguments)]
    async fn send(
        &self,
        method: &str,
        canonical_uri: &str,
        canonical_query: &str,
        url: &str,
        extra: Vec<(String, String)>,
        payload_sha256: &str,
        body: Option<&[u8]>,
    ) -> Result<Response> {
        let (amz_date, date_stamp) = now_amz();
        let host = self.host_header();

        let mut to_sign: Vec<(String, String)> = vec![
            ("host".to_string(), host),
            (
                "x-amz-content-sha256".to_string(),
                payload_sha256.to_string(),
            ),
            ("x-amz-date".to_string(), amz_date.clone()),
        ];
        to_sign.extend(extra);
        if let Some(tok) = &self.session_token {
            to_sign.push(("x-amz-security-token".to_string(), tok.clone()));
        }

        let canon = CanonicalRequest {
            method,
            canonical_uri,
            canonical_query,
            headers: to_sign.clone(),
            payload_sha256_hex: payload_sha256,
        };
        let cred = Credentials {
            access_key: &self.access_key,
            secret_key: &self.secret_key,
            region: &self.region,
            service: "s3",
        };
        let auth = authorization_header(&canon, &cred, &amz_date, &date_stamp);

        let mut headers = to_sign;
        headers.push(("authorization".to_string(), auth));
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        self.http
            .request(method, url, &header_refs, body)
            .await
            .map_err(|e| io_error(format!("S3 {method} failed: {e}")))
    }

    /// `GetObject` with a byte range `[pos, pos+len)`.
    pub async fn get_range(&self, key: &str, pos: u64, len: usize) -> Result<Vec<u8>> {
        let (url, cu) = self.object_paths(key);
        let range = format!("bytes={}-{}", pos, pos + len as u64 - 1);
        let resp = self
            .send(
                "GET",
                &cu,
                "",
                &url,
                vec![("range".to_string(), range)],
                EMPTY_PAYLOAD_SHA256,
                None,
            )
            .await?;
        self.expect_body(resp, key)
    }

    /// `GetObject` (whole object).
    pub async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let (url, cu) = self.object_paths(key);
        let resp = self
            .send("GET", &cu, "", &url, vec![], EMPTY_PAYLOAD_SHA256, None)
            .await?;
        self.expect_body(resp, key)
    }

    fn expect_body(&self, resp: Response, key: &str) -> Result<Vec<u8>> {
        match resp.status {
            200 | 206 => Ok(resp.body),
            404 => Err(Error::NotFound {
                path: key.to_string(),
            }),
            s => Err(io_error(format!("S3 GET {key} returned status {s}"))),
        }
    }

    /// `HeadObject` → object byte length.
    pub async fn head_len(&self, key: &str) -> Result<u64> {
        let (url, cu) = self.object_paths(key);
        let resp = self
            .send("HEAD", &cu, "", &url, vec![], EMPTY_PAYLOAD_SHA256, None)
            .await?;
        match resp.status {
            200 => resp
                .header("content-length")
                .and_then(|v| v.trim().parse::<u64>().ok())
                .ok_or_else(|| io_error("S3 HEAD missing content-length".to_string())),
            404 => Err(Error::NotFound {
                path: key.to_string(),
            }),
            s => Err(io_error(format!("S3 HEAD {key} returned status {s}"))),
        }
    }

    /// Whether `key` exists (`HeadObject` → 200 vs 404).
    pub async fn exists(&self, key: &str) -> Result<bool> {
        match self.head_len(key).await {
            Ok(_) => Ok(true),
            Err(Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// `PutObject` — signs the actual payload hash.
    pub async fn put(&self, key: &str, body: &[u8]) -> Result<()> {
        let (url, cu) = self.object_paths(key);
        let payload = sha256_hex(body);
        let resp = self
            .send("PUT", &cu, "", &url, vec![], &payload, Some(body))
            .await?;
        match resp.status {
            200 | 201 => Ok(()),
            s => Err(io_error(format!("S3 PUT {key} returned status {s}"))),
        }
    }

    /// `DeleteObject`.
    pub async fn delete(&self, key: &str) -> Result<()> {
        let (url, cu) = self.object_paths(key);
        let resp = self
            .send("DELETE", &cu, "", &url, vec![], EMPTY_PAYLOAD_SHA256, None)
            .await?;
        match resp.status {
            200 | 204 => Ok(()),
            404 => Ok(()), // idempotent delete
            s => Err(io_error(format!("S3 DELETE {key} returned status {s}"))),
        }
    }

    /// `ListObjectsV2` — all object keys under `prefix` (handles pagination).
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let canonical_uri = format!("/{}", self.bucket);
        let mut keys = Vec::new();
        let mut continuation: Option<String> = None;

        loop {
            // Canonical query: sorted key=value, each value URI-encoded.
            let mut params: Vec<(String, String)> =
                vec![("list-type".to_string(), "2".to_string())];
            if !prefix.is_empty() {
                params.push(("prefix".to_string(), prefix.to_string()));
            }
            if let Some(token) = &continuation {
                params.push(("continuation-token".to_string(), token.clone()));
            }
            params.sort_by(|a, b| a.0.cmp(&b.0));
            let canonical_query: String = params
                .iter()
                .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
                .collect::<Vec<_>>()
                .join("&");
            let url = format!("{}/{}?{}", self.base(), self.bucket, canonical_query);

            let resp = self
                .send(
                    "GET",
                    &canonical_uri,
                    &canonical_query,
                    &url,
                    vec![],
                    EMPTY_PAYLOAD_SHA256,
                    None,
                )
                .await?;
            if resp.status != 200 {
                return Err(io_error(format!(
                    "S3 ListObjectsV2 returned status {}",
                    resp.status
                )));
            }
            let xml = String::from_utf8_lossy(&resp.body);
            let (page_keys, next) = parse_list_v2(&xml)?;
            keys.extend(page_keys);
            match next {
                Some(token) => continuation = Some(token),
                None => break,
            }
        }
        Ok(keys)
    }
}

/// Parse a `ListObjectsV2` XML response: returns `(object keys, next continuation
/// token if truncated)`. Uses `quick-xml` for correct entity decoding.
fn parse_list_v2(xml: &str) -> Result<(Vec<String>, Option<String>)> {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    let mut reader = Reader::from_str(xml);
    let mut keys = Vec::new();
    let mut next_token: Option<String> = None;
    let mut truncated = false;
    let mut current: Vec<u8> = Vec::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => current = e.local_name().as_ref().to_vec(),
            Ok(Event::End(_)) => current.clear(),
            Ok(Event::Text(t)) => {
                let text = t
                    .unescape()
                    .map_err(|e| io_error(format!("ListObjectsV2 XML decode: {e}")))?
                    .into_owned();
                match current.as_slice() {
                    b"Key" => keys.push(text),
                    b"NextContinuationToken" => next_token = Some(text),
                    b"IsTruncated" => truncated = text.trim().eq_ignore_ascii_case("true"),
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(io_error(format!("ListObjectsV2 XML parse: {e}"))),
            _ => {}
        }
    }
    Ok((keys, if truncated { next_token } else { None }))
}

/// Current UTC time as SigV4 `(amz_date = YYYYMMDDTHHMMSSZ, date_stamp = YYYYMMDD)`.
fn now_amz() -> (String, String) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, m, d) = civil_from_days(days);
    (
        format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}Z"),
        format!("{y:04}{m:02}{d:02}"),
    )
}

/// Days since the Unix epoch → `(year, month, day)` (proleptic Gregorian, UTC).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::{civil_from_days, parse_list_v2};

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2013-05-24 is 15849 days after the Unix epoch (matches the SigV4 KAT date).
        assert_eq!(civil_from_days(15849), (2013, 5, 24));
    }

    #[test]
    fn parse_list_v2_extracts_keys_and_pagination() {
        let xml = r#"<?xml version="1.0"?><ListBucketResult>
            <Name>bucket</Name><Prefix>a/</Prefix>
            <IsTruncated>true</IsTruncated>
            <NextContinuationToken>TOK&amp;1</NextContinuationToken>
            <Contents><Key>a/one.bin</Key></Contents>
            <Contents><Key>a/two &amp; three.bin</Key></Contents>
        </ListBucketResult>"#;
        let (keys, next) = parse_list_v2(xml).expect("parse");
        assert_eq!(
            keys,
            vec!["a/one.bin".to_string(), "a/two & three.bin".to_string()]
        );
        assert_eq!(
            next.as_deref(),
            Some("TOK&1"),
            "entity-decoded continuation token"
        );
    }

    #[test]
    fn parse_list_v2_not_truncated_yields_no_token() {
        let xml = r#"<ListBucketResult><IsTruncated>false</IsTruncated>
            <NextContinuationToken>ignored</NextContinuationToken>
            <Contents><Key>k</Key></Contents></ListBucketResult>"#;
        let (keys, next) = parse_list_v2(xml).expect("parse");
        assert_eq!(keys, vec!["k".to_string()]);
        assert_eq!(next, None, "non-truncated listing has no next token");
    }
}
