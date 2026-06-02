//! Native S3 object-store reader (ADR-015): `GetObject` range reads and
//! `HeadObject` over the Moirai HTTP/1.1 client with SigV4 signing — no tokio,
//! no AWS SDK. Path-style addressing (`{endpoint}/{bucket}/{key}`), which works
//! for AWS S3 and S3-compatible stores (MinIO, Ceph).

mod sigv4;

#[cfg(test)]
mod tests;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use std::time::{SystemTime, UNIX_EPOCH};

use consus_core::{Error, Result};
use moirai_http::HttpClient;

use super::super::traits::{AsyncLength, AsyncReadAt};
use sigv4::{
    CanonicalRequest, Credentials, EMPTY_PAYLOAD_SHA256, authorization_header, uri_encode,
};

/// Connection + credential configuration for an S3 object.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Base endpoint URL, e.g. `https://s3.us-east-1.amazonaws.com` or
    /// `http://127.0.0.1:9000` (MinIO). Scheme selects TLS.
    pub endpoint: String,
    /// AWS region for the signing scope (e.g. `us-east-1`).
    pub region: String,
    /// Access key id.
    pub access_key: String,
    /// Secret access key.
    pub secret_key: String,
    /// Optional STS session token (`x-amz-security-token`).
    pub session_token: Option<String>,
    /// Bucket name.
    pub bucket: String,
    /// Object key.
    pub key: String,
}

/// Asynchronous S3 byte source over Moirai HTTP, satisfying [`AsyncReadAt`] and
/// [`AsyncLength`] by issuing signed ranged `GET` / `HEAD` requests.
#[derive(Clone)]
pub struct S3MoiraiReader {
    client: Arc<HttpClient>,
    cfg: S3Config,
}

impl S3MoiraiReader {
    /// New reader with its own HTTP client (Mozilla roots for HTTPS).
    #[must_use]
    pub fn new(cfg: S3Config) -> Self {
        Self {
            client: Arc::new(HttpClient::new()),
            cfg,
        }
    }

    /// New reader sharing an existing HTTP client (connection-pool reuse).
    #[must_use]
    pub fn with_client(client: Arc<HttpClient>, cfg: S3Config) -> Self {
        Self { client, cfg }
    }

    /// Scheme/host/port parsed from the endpoint.
    fn endpoint_parts(&self) -> (bool, &str) {
        if let Some(rest) = self.cfg.endpoint.strip_prefix("https://") {
            (true, rest.trim_end_matches('/'))
        } else if let Some(rest) = self.cfg.endpoint.strip_prefix("http://") {
            (false, rest.trim_end_matches('/'))
        } else {
            (true, self.cfg.endpoint.trim_end_matches('/'))
        }
    }

    /// `Host` header value matching moirai-http's `Origin::host_header` (authority
    /// without a default port). The endpoint authority already encodes the port.
    fn host_header(&self) -> String {
        let (secure, authority) = self.endpoint_parts();
        // Strip a default port so the signed value matches the sent header.
        let default = if secure { ":443" } else { ":80" };
        authority
            .strip_suffix(default)
            .unwrap_or(authority)
            .to_string()
    }

    fn object_url(&self) -> String {
        format!(
            "{}/{}/{}",
            self.cfg.endpoint.trim_end_matches('/'),
            self.cfg.bucket,
            uri_encode(&self.cfg.key, false)
        )
    }

    fn canonical_uri(&self) -> String {
        format!("/{}/{}", self.cfg.bucket, uri_encode(&self.cfg.key, false))
    }

    /// Build the signed header set shared by GET/HEAD; returns the request headers
    /// to send (including `Authorization`).
    fn signed_headers(&self, method: &str, extra: &[(String, String)]) -> Vec<(String, String)> {
        let (amz_date, date_stamp) = now_amz();
        let host = self.host_header();

        let mut to_sign: Vec<(String, String)> = vec![
            ("host".to_string(), host.clone()),
            (
                "x-amz-content-sha256".to_string(),
                EMPTY_PAYLOAD_SHA256.to_string(),
            ),
            ("x-amz-date".to_string(), amz_date.clone()),
        ];
        to_sign.extend(extra.iter().cloned());
        if let Some(tok) = &self.cfg.session_token {
            to_sign.push(("x-amz-security-token".to_string(), tok.clone()));
        }

        let canonical_uri = self.canonical_uri();
        let canon = CanonicalRequest {
            method,
            canonical_uri: &canonical_uri,
            canonical_query: "",
            headers: to_sign.clone(),
            payload_sha256_hex: EMPTY_PAYLOAD_SHA256,
        };
        let cred = Credentials {
            access_key: &self.cfg.access_key,
            secret_key: &self.cfg.secret_key,
            region: &self.cfg.region,
            service: "s3",
        };
        let auth = authorization_header(&canon, &cred, &amz_date, &date_stamp);

        // Headers to actually send: the signed ones plus Authorization.
        let mut out = to_sign;
        out.push(("authorization".to_string(), auth));
        out
    }
}

fn io_error(msg: String) -> Error {
    Error::Io(std::io::Error::other(msg))
}

impl AsyncReadAt for S3MoiraiReader {
    async fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let end_inclusive = pos + (buf.len() as u64) - 1;
        let range = format!("bytes={pos}-{end_inclusive}");
        let headers = self.signed_headers("GET", &[("range".to_string(), range)]);
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let url = self.object_url();

        let resp = self
            .client
            .get(&url, &header_refs)
            .await
            .map_err(|e| io_error(format!("S3 GET failed: {e}")))?;

        match resp.status {
            200 | 206 => {
                if resp.body.len() < buf.len() {
                    return Err(Error::BufferTooSmall {
                        required: buf.len(),
                        provided: resp.body.len(),
                    });
                }
                buf.copy_from_slice(&resp.body[..buf.len()]);
                Ok(())
            }
            404 => Err(Error::NotFound {
                path: self.cfg.key.clone(),
            }),
            s => Err(io_error(format!("S3 GET returned status {s}"))),
        }
    }
}

impl AsyncLength for S3MoiraiReader {
    async fn len(&self) -> Result<u64> {
        let headers = self.signed_headers("HEAD", &[]);
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let url = self.object_url();

        let resp = self
            .client
            .head(&url, &header_refs)
            .await
            .map_err(|e| io_error(format!("S3 HEAD failed: {e}")))?;

        match resp.status {
            200 => resp
                .header("content-length")
                .and_then(|v| v.trim().parse::<u64>().ok())
                .ok_or_else(|| io_error("S3 HEAD missing/invalid content-length".to_string())),
            404 => Err(Error::NotFound {
                path: self.cfg.key.clone(),
            }),
            s => Err(io_error(format!("S3 HEAD returned status {s}"))),
        }
    }
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

/// Convert a count of days since the Unix epoch to a `(year, month, day)` civil
/// date (proleptic Gregorian, UTC). Howard Hinnant's `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}
