//! Async S3 Object Store Reader
//!
//! Provides an asynchronous, `ReadAt` and `Length` compliant adapter over AWS S3.
//! Utilizes `rusoto_s3` to execute `GetObjectRequest`s with HTTP `Range` headers
//! to enable zero-copy partial file reads across the network.

use super::super::traits::{AsyncLength, AsyncReadAt};
use consus_core::{Error, Result};
use rusoto_core::Region;
use rusoto_s3::{GetObjectRequest, HeadObjectRequest, S3Client, S3};
use std::sync::Arc;

/// An asynchronous byte source reading directly from AWS S3 via HTTP Range requests.
///
/// Converts positioned byte reads into `GetObject` range requests,
/// enabling remote partial reads of large datasets (like HDF5, Zarr, or Parquet)
/// without downloading the entire file.
#[derive(Clone)]
pub struct S3Reader {
    client: Arc<S3Client>,
    bucket: String,
    key: String,
}

impl S3Reader {
    /// Create a new `S3Reader` for the specified bucket and key, using the default
    /// credential provider chain and the specified region.
    pub fn new(region: Region, bucket: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            client: Arc::new(S3Client::new(region)),
            bucket: bucket.into(),
            key: key.into(),
        }
    }

    /// Create a new `S3Reader` sharing an existing `S3Client`.
    pub fn with_client(client: Arc<S3Client>, bucket: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            client,
            bucket: bucket.into(),
            key: key.into(),
        }
    }
}

impl AsyncReadAt for S3Reader {
    async fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }

        let end_inclusive = pos + (buf.len() as u64) - 1;
        let range_header = format!("bytes={}-{}", pos, end_inclusive);

        let req = GetObjectRequest {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            range: Some(range_header),
            ..Default::default()
        };

        let response = self
            .client
            .get_object(req)
            .await
            .map_err(|e| match e {
                rusoto_core::RusotoError::Service(rusoto_s3::GetObjectError::NoSuchKey(_)) => {
                    Error::NotFound { path: self.key.clone() }
                }
                _ => Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("S3 get_object failed: {}", e))),
            })?;

        let stream = response.body.ok_or_else(|| {
            Error::Io(std::io::Error::new(std::io::ErrorKind::Other, "S3 get_object response body is empty"))
        })?;

        // Read all chunks from the stream
        use tokio::io::AsyncReadExt;
        let mut reader = stream.into_async_read();
        
        let mut read_bytes = 0;
        while read_bytes < buf.len() {
            let n = reader.read(&mut buf[read_bytes..]).await.map_err(|e| {
                Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("S3 stream read failed: {}", e)))
            })?;
            
            if n == 0 {
                return Err(Error::BufferTooSmall { required: buf.len(), provided: read_bytes });
            }
            read_bytes += n;
        }

        Ok(())
    }
}

impl AsyncLength for S3Reader {
    async fn len(&self) -> Result<u64> {
        let req = HeadObjectRequest {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            ..Default::default()
        };

        let response = self
            .client
            .head_object(req)
            .await
            .map_err(|e| match e {
                rusoto_core::RusotoError::Service(rusoto_s3::HeadObjectError::NoSuchKey(_)) => {
                    Error::NotFound { path: self.key.clone() }
                }
                _ => Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("S3 head_object failed: {}", e))),
            })?;

        response.content_length.map(|l| l as u64).ok_or_else(|| {
            Error::Io(std::io::Error::new(std::io::ErrorKind::Other, "S3 head_object response missing content_length"))
        })
    }
}

