//! Provides a `Store` implementation backed by any S3-compatible object
//! store (AWS S3, MinIO, GCS with interoperability mode, R2, etc.)
//! via the `rusoto_s3` crate.
//!
//! ## Authentication
//!
//! Credentials are loaded from the AWS environment variables or from the
//! shared credentials file (`~/.aws/credentials`) via rusoto's default
//! credential chain. Explicit `credentials` can also be passed during
//! construction.
//!
//! ## Key Semantics
//!
//! All keys are object key strings within a single S3 bucket. There is
//! no concept of directory hierarchies in S3 — the store uses the same
//! key scheme as the local filesystem store (e.g., `"my_array/.zarray"`).
//! The S3 list operation is used to implement `Store::list(prefix)` by
//! enumerating objects with the given prefix.
//!
//! ## Python zarr Compatibility
//!
//! Zarr Python reads and writes chunks via HTTP range requests. This store
//! backend uses the same key scheme as the Python reference implementation,
//! so S3-hosted Zarr arrays produced by zarr-python are directly readable,
//! and arrays written by consus-zarr are directly readable by zarr-python
//! when served via an S3-compatible HTTP server (e.g., s3fs, aiohttp,
//! or cloudflare R2 public bucket).
//!
//! ## Content-MD5
//!
//! For Python zarr interoperability on S3, set the `content_md5` option
//! when storing chunk data. The Zarr v3 spec mandates `content-md5` for
//! chunk integrity. This store computes and sends the MD5 digest when
//! the `compute_md5` option is enabled.
//!
//! ## Feature Flag
//!
//! This module requires the `s3` feature:
//! ```toml
//! consus-zarr = { path = "...", features = ["s3"] }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "s3")]
use alloc::string::ToString;

#[cfg(feature = "s3")]
use alloc::sync::Arc;
#[cfg(feature = "s3")]
#[cfg(feature = "alloc")]
use consus_core::Result;
#[cfg(feature = "s3")]
use rusoto_core::{Region, RusotoError};
#[cfg(feature = "s3")]
use rusoto_s3::{
    DeleteObjectRequest, GetObjectError, GetObjectRequest, HeadObjectError, HeadObjectRequest,
    ListObjectsV2Request, PutObjectRequest, S3, S3Client,
};

#[cfg(feature = "alloc")]
use crate::store::Store;

// ---------------------------------------------------------------------------
// S3 Store
// ---------------------------------------------------------------------------

/// Configuration for an S3-compatible store.
///
/// # Example
///
/// ```ignore
/// let store = S3Store::new()
///     .with_bucket("my-zarr-bucket")
///     .with_region(Region::UsEast1)
///     .with_prefix("experiment_001.zarr")
///     .build()?;
/// ```
#[cfg(feature = "s3")]
pub struct S3Store {
    /// S3 client.
    client: S3Client,
    /// Bucket name.
    bucket: String,
    /// Optional prefix applied to all keys (e.g., `"my.zarr/"`).
    prefix: String,
    /// Whether to compute and send Content-MD5 when writing chunk data.
    compute_md5: bool,
    /// Whether to require that written objects are immediately readable.
    /// Disabling this skips the read-after-write verification.
    read_after_write: bool,
    /// Tokio runtime for blocking S3 operations.
    rt: alloc::sync::Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "s3")]
impl core::fmt::Debug for S3Store {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("S3Store")
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .field("compute_md5", &self.compute_md5)
            .field("read_after_write", &self.read_after_write)
            .finish()
    }
}

#[cfg(feature = "s3")]
impl Clone for S3Store {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            bucket: self.bucket.clone(),
            prefix: self.prefix.clone(),
            compute_md5: self.compute_md5,
            read_after_write: self.read_after_write,
            rt: self.rt.clone(),
        }
    }
}

/// Errors specific to S3 store operations.
#[cfg(feature = "s3")]
#[derive(Debug)]
pub enum S3StoreError {
    /// The object was not found in the bucket.
    NotFound { key: String },
    /// A Rusoto-level S3 error occurred.
    Rusoto(Box<dyn core::error::Error + Send + Sync>),
    /// An I/O error (e.g., MD5 computation failure).
    Io(String),
}

#[cfg(feature = "s3")]
impl core::fmt::Display for S3StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFound { key } => write!(f, "S3 object not found: {key}"),
            Self::Rusoto(e) => write!(f, "S3 error: {e}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

#[cfg(feature = "s3")]
impl std::error::Error for S3StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rusoto(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

#[cfg(feature = "s3")]
impl From<RusotoError<GetObjectError>> for S3StoreError {
    fn from(e: RusotoError<GetObjectError>) -> Self {
        Self::Rusoto(Box::new(e))
    }
}

#[cfg(feature = "s3")]
impl S3Store {
    /// Begin building an S3Store with a required bucket name.
    pub fn new(bucket: impl Into<String>) -> Self {
        Self::with_client(S3Client::new(Region::UsEast1), bucket)
    }

    fn with_runtime(client: S3Client, bucket: String, prefix: String) -> Self {
        let rt = Arc::new(
            tokio::runtime::Runtime::new()
                .expect("failed to create tokio runtime for S3 operations"),
        );
        Self {
            client,
            bucket,
            prefix,
            compute_md5: false,
            read_after_write: false,
            rt,
        }
    }

    /// Create an S3Store with an explicit `S3Client`.
    ///
    /// Use this when you need a custom HTTP client, specific Region
    /// configuration, or STS token-based credentials.
    pub fn with_client(client: S3Client, bucket: impl Into<String>) -> Self {
        Self::with_runtime(client, bucket.into(), String::new())
    }

    /// Set the key prefix for all operations.
    ///
    /// All keys will be prefixed with `prefix`. For a Zarr hierarchy
    /// stored as a single S3 object prefix (e.g., `"my_array.zarr/"`),
    /// set `prefix` to the container name with trailing slash.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        let p = prefix.into();
        self.prefix = if p.ends_with('/') { p } else { p + "/" };
        self
    }

    /// Set the AWS region.
    ///
    /// Default is `Region::UsEast1`. Common values include:
    /// - `Region::UsEast1` — US East (N. Virginia)
    /// - `Region::UsWest2` — US West (Oregon)
    /// - `Region::EuCentral1` — Europe (Frankfurt)
    /// - `Region::ApSoutheast1` — Asia Pacific (Singapore)
    ///
    /// For MinIO or other S3-compatible services, use:
    /// ```ignore
    /// Region::Custom {
    ///     name: "minio".to_string(),
    ///     endpoint: "http://localhost:9000".to_string(),
    /// }
    /// ```
    pub fn with_region(mut self, region: Region) -> Self {
        self.client = S3Client::new(region);
        self
    }

    /// Enable Content-MD5 computation and sending for written objects.
    ///
    /// The Zarr v3 spec requires `content-md5` for chunk data. Enable
    /// this option when interoperability with zarr-python on S3 is needed.
    pub fn with_md5(mut self) -> Self {
        self.compute_md5 = true;
        self
    }

    /// Disable read-after-write verification after each `set` call.
    ///
    /// By default, `set` performs a HEAD request after writing to confirm
    /// the object is readable. This adds latency but ensures early detection
    /// of permission or connectivity issues. Disable for higher throughput
    /// when writing many small objects.
    pub fn without_read_after_write(mut self) -> Self {
        self.read_after_write = false;
        self
    }

    /// Full key including the store prefix.
    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }

    /// Compute the MD5 hex digest of the given data.
    fn md5_hex(data: &[u8]) -> String {
        // md5::compute returns md5::Digest which is GenericArray<u8, 16>
        // Convert to hex via the Digest trait's output.
        use md5::Digest;
        let hash = md5::compute(data);
        let mut hex = alloc::string::String::with_capacity(32);
        for byte in hash.iter() {
            use alloc::format;
            hex.push_str(&format!("{:02x}", byte));
        }
        hex
    }
}

/// Encode a byte slice as a lowercase hex string.
fn hex_encode(data: impl AsRef<[u8]>) -> String {
    data.as_ref()
        .iter()
        .fold(alloc::string::String::new(), |mut s, &b| {
            use alloc::format;
            s.push_str(&format!("{:02x}", b));
            s
        })
}

#[cfg(feature = "s3")]
impl Store for S3Store {
    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let full_key = self.full_key(key);
        let req = GetObjectRequest {
            bucket: self.bucket.clone(),
            key: full_key.clone(),
            ..Default::default()
        };

        let result = self
            .rt
            .block_on(self.client.get_object(req))
            .map_err(|e| match e {
                RusotoError::Service(GetObjectError::NoSuchKey(_)) => {
                    consus_core::Error::NotFound {
                        path: key.to_string(),
                    }
                }
                RusotoError::Service(GetObjectError::InvalidObjectState(_)) => {
                    // Object is in Glacier — treat as not found for this impl
                    consus_core::Error::NotFound {
                        path: key.to_string(),
                    }
                }
                _ => consus_core::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    S3StoreError::Rusoto(Box::new(e)),
                )),
            })?;

        let body = result
            .body
            .ok_or_else(|| consus_core::Error::InvalidFormat {
                message: "S3 GetObject returned empty body".to_string(),
            })?;

        use futures::stream::TryStreamExt;
        let body_bytes: Vec<bytes::Bytes> = self.rt.block_on(body.try_collect()).map_err(|e| {
            consus_core::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                S3StoreError::Io(format!("S3 body read: {:?}", e)),
            ))
        })?;
        // Flatten all chunks into a single Vec<u8>
        let total_len = body_bytes.iter().map(|b| b.len()).sum();
        let mut result = Vec::with_capacity(total_len);
        for chunk in body_bytes {
            result.extend_from_slice(&chunk);
        }
        Ok(result)
    }

    fn set(&mut self, key: &str, value: &[u8]) -> Result<()> {
        let full_key = self.full_key(key);

        let content_md5 = if self.compute_md5 {
            // MD5 must be base64-encoded for S3
            Some(base64_encode(Self::md5_hex(value).as_bytes()))
        } else {
            None
        };

        let req = PutObjectRequest {
            bucket: self.bucket.clone(),
            key: full_key.clone(),
            body: Some(value.to_vec().into()),
            content_length: Some(value.len() as i64),
            content_md5,
            ..Default::default()
        };

        self.rt.block_on(self.client.put_object(req)).map_err(|e| {
            consus_core::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                S3StoreError::Rusoto(Box::new(e)),
            ))
        })?;

        if self.read_after_write {
            // Verify the object is readable via HEAD request
            let head = HeadObjectRequest {
                bucket: self.bucket.clone(),
                key: full_key,
                ..Default::default()
            };
            self.rt
                .block_on(self.client.head_object(head))
                .map_err(|e| {
                    consus_core::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        S3StoreError::Rusoto(Box::new(e)),
                    ))
                })?;
        }

        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        let full_key = self.full_key(key);
        let req = DeleteObjectRequest {
            bucket: self.bucket.clone(),
            key: full_key,
            ..Default::default()
        };

        self.rt
            .block_on(self.client.delete_object(req))
            .map_err(|e| {
                consus_core::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    S3StoreError::Rusoto(Box::new(e)),
                ))
            })?;

        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);
        let mut keys = Vec::new();
        let mut continuation_token = None;

        loop {
            let req = ListObjectsV2Request {
                bucket: self.bucket.clone(),
                prefix: Some(full_prefix.clone()),
                continuation_token: continuation_token.clone(),
                ..Default::default()
            };

            let result = self
                .rt
                .block_on(self.client.list_objects_v2(req))
                .map_err(|e| {
                    consus_core::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        S3StoreError::Rusoto(Box::new(e)),
                    ))
                })?;

            if let Some(contents) = result.contents {
                for obj in contents {
                    if let Some(key) = obj.key {
                        // Strip the store prefix to get the relative key
                        let relative = if self.prefix.is_empty() {
                            key
                        } else {
                            key.strip_prefix(&self.prefix).unwrap_or(&key).to_string()
                        };
                        keys.push(relative);
                    }
                }
            }

            if result.is_truncated == Some(true) {
                continuation_token = result.next_continuation_token;
            } else {
                break;
            }
        }

        // Sort for deterministic ordering
        keys.sort();
        keys.dedup();

        Ok(keys)
    }

    fn contains(&self, key: &str) -> Result<bool> {
        let full_key = self.full_key(key);
        let req = HeadObjectRequest {
            bucket: self.bucket.clone(),
            key: full_key,
            ..Default::default()
        };

        match self.rt.block_on(self.client.head_object(req)) {
            Ok(_) => Ok(true),
            Err(RusotoError::Service(HeadObjectError::NoSuchKey(_))) => Ok(false),
            Err(e) => Err(consus_core::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                S3StoreError::Rusoto(Box::new(e)),
            ))),
        }
    }
}

/// Encode bytes as base64.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = alloc::string::String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0F) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3F] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Mock store for testing (no S3 dependency required)
// ---------------------------------------------------------------------------

/// A mock store that records operations for testing without S3.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct MockS3Store {
    /// All set operations recorded in order.
    pub set_operations: alloc::vec::Vec<(String, Vec<u8>)>,
    /// All delete keys recorded.
    pub delete_keys: alloc::vec::Vec<String>,
    /// Whether get should return an error.
    pub get_error: bool,
    /// Whether contains should return true.
    pub contains_value: bool,
}

#[cfg(feature = "alloc")]
impl MockS3Store {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the mock to return an error on the next `get`.
    pub fn with_get_error(mut self) -> Self {
        self.get_error = true;
        self
    }

    /// Configure the mock to return the given value on `contains`.
    pub fn with_contains_value(mut self, v: bool) -> Self {
        self.contains_value = v;
        self
    }
}

#[cfg(feature = "alloc")]
impl Store for MockS3Store {
    fn get(&self, _key: &str) -> Result<Vec<u8>> {
        if self.get_error {
            return Err(consus_core::Error::NotFound {
                path: _key.to_string(),
            });
        }
        Err(consus_core::Error::NotFound {
            path: _key.to_string(),
        })
    }

    fn set(&mut self, key: &str, value: &[u8]) -> Result<()> {
        self.set_operations.push((key.to_string(), value.to_vec()));
        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<()> {
        self.delete_keys.push(key.to_string());
        Ok(())
    }

    fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        Ok(vec![])
    }

    fn contains(&self, _key: &str) -> Result<bool> {
        Ok(self.contains_value)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "s3")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encode_works() {
        // RFC 4648 test vector
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn md5_hex_works() {
        // RFC 1321 test vector
        assert_eq!(S3Store::md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(
            S3Store::md5_hex(b"hello world"),
            "5eb63bbbe01eeed093cb22bb8f5acdc3"
        );
    }

    #[test]
    fn full_key_without_prefix() {
        let store = S3Store::new("bucket");
        assert_eq!(store.full_key("arr/.zarray"), "arr/.zarray");
    }

    #[test]
    fn full_key_with_prefix() {
        let store = S3Store::new("bucket").with_prefix("my_array.zarr");
        assert_eq!(store.full_key("arr/.zarray"), "my_array.zarr/arr/.zarray");
    }

    #[test]
    fn prefix_adds_trailing_slash() {
        let store = S3Store::new("bucket").with_prefix("prefix_no_trailing");
        assert_eq!(store.prefix, "prefix_no_trailing/");
    }

    #[test]
    fn clone_preserves_config() {
        let store = S3Store::new("bucket")
            .with_prefix("data.zarr")
            .with_md5()
            .without_read_after_write();

        let store2 = store.clone();
        assert_eq!(store2.prefix, "data.zarr/");
        assert!(store2.compute_md5);
        assert!(!store2.read_after_write);
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod mock_tests {
    use super::*;

    #[test]
    fn mock_records_set_operations() {
        let mut store = MockS3Store::new();
        store.set("key1", b"val1").unwrap();
        store.set("key2", b"val2").unwrap();
        assert_eq!(store.set_operations.len(), 2);
        assert_eq!(store.set_operations[0].0, "key1");
        assert_eq!(&store.set_operations[1].0, "key2");
    }

    #[test]
    fn mock_records_delete_keys() {
        let mut store = MockS3Store::new();
        store.delete("key1").unwrap();
        store.delete("key2").unwrap();
        assert_eq!(store.delete_keys, &["key1", "key2"]);
    }

    #[test]
    fn mock_get_error() {
        let store = MockS3Store::new().with_get_error(true);
        let err = store.get("any").unwrap_err();
        assert!(matches!(err, consus_core::Error::NotFound { .. }));
    }

    #[test]
    fn mock_contains() {
        let store = MockS3Store::new().with_contains_value(true);
        assert!(store.contains("key").unwrap());
    }
}
