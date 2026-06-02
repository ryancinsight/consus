//! Native S3 object-store backend (ADR-015): a full async [`S3Client`]
//! (GET/PUT/DELETE/HEAD/ListObjectsV2 with SigV4 over `moirai-http`) plus
//! [`S3MoiraiReader`], an [`AsyncReadAt`]/[`AsyncLength`] adapter for reading a
//! single object via ranged GET / HEAD. No tokio, no AWS SDK. Path-style
//! addressing (AWS S3 + S3-compatible stores: MinIO, Ceph).

mod client;
mod sigv4;

#[cfg(test)]
mod tests;

use alloc::string::String;
use alloc::sync::Arc;

use consus_core::{Error, Result};
use moirai_http::HttpClient;

pub use client::S3Client;

use super::super::traits::{AsyncLength, AsyncReadAt};

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

impl S3Config {
    /// Build an [`S3Client`] from this config (bucket-scoped; `key` is ignored,
    /// methods take the key per call).
    #[must_use]
    pub fn into_client(self, http: Arc<HttpClient>) -> S3Client {
        S3Client::new(
            http,
            self.endpoint,
            self.region,
            self.access_key,
            self.secret_key,
            self.session_token,
            self.bucket,
        )
    }
}

/// Asynchronous S3 byte source for one object, satisfying [`AsyncReadAt`] and
/// [`AsyncLength`] via signed ranged `GET` / `HEAD`.
#[derive(Clone)]
pub struct S3MoiraiReader {
    client: Arc<S3Client>,
    key: String,
}

impl S3MoiraiReader {
    /// New reader with its own HTTP client (Mozilla roots for HTTPS).
    #[must_use]
    pub fn new(cfg: S3Config) -> Self {
        Self::with_client(Arc::new(HttpClient::new()), cfg)
    }

    /// New reader sharing an existing HTTP client (connection-pool reuse).
    #[must_use]
    pub fn with_client(http: Arc<HttpClient>, cfg: S3Config) -> Self {
        let key = cfg.key.clone();
        Self {
            client: Arc::new(cfg.into_client(http)),
            key,
        }
    }
}

impl AsyncReadAt for S3MoiraiReader {
    async fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let data = self.client.get_range(&self.key, pos, buf.len()).await?;
        if data.len() < buf.len() {
            return Err(Error::BufferTooSmall {
                required: buf.len(),
                provided: data.len(),
            });
        }
        buf.copy_from_slice(&data[..buf.len()]);
        Ok(())
    }
}

impl AsyncLength for S3MoiraiReader {
    async fn len(&self) -> Result<u64> {
        self.client.head_len(&self.key).await
    }
}
