//! AWS Signature Version 4 request signing for S3.
//!
//! Pure RustCrypto (`hmac` + `sha2`); no AWS SDK, no tokio. Implements the
//! canonical-request → string-to-sign → signing-key → signature chain from the
//! AWS SigV4 specification. Validated by a known-answer test against AWS's own
//! published `GET Object` example vector (see `tests`).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// SHA-256 of an empty payload — the `x-amz-content-sha256` value for body-less
/// requests (`GET`/`HEAD`).
pub const EMPTY_PAYLOAD_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Hex-encode bytes (lowercase).
fn hex(bytes: &[u8]) -> String {
    use core::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Hex-encoded SHA-256 of `data`.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts keys of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// AWS credentials and the target signing scope.
pub struct Credentials<'a> {
    /// Access key id.
    pub access_key: &'a str,
    /// Secret access key.
    pub secret_key: &'a str,
    /// AWS region (e.g. `us-east-1`).
    pub region: &'a str,
    /// AWS service (`s3`).
    pub service: &'a str,
}

/// The canonical components of a request to be signed.
pub struct CanonicalRequest<'a> {
    /// HTTP method (uppercase).
    pub method: &'a str,
    /// Percent-encoded path (e.g. `/bucket/key`).
    pub canonical_uri: &'a str,
    /// Canonical (sorted, encoded) query string, or `""`.
    pub canonical_query: &'a str,
    /// Headers to sign as `(lowercased-name, value)`; must include `host`,
    /// `x-amz-date`, and `x-amz-content-sha256`.
    pub headers: Vec<(String, String)>,
    /// Hex SHA-256 of the request payload.
    pub payload_sha256_hex: &'a str,
}

/// Result of signing: the signature and the derived header fields.
pub struct Signature {
    /// Lowercase hex signature.
    pub signature: String,
    /// `;`-joined sorted signed header names.
    pub signed_headers: String,
    /// `datestamp/region/service/aws4_request`.
    pub credential_scope: String,
}

/// Compute the SigV4 signature and derived fields.
///
/// `amz_date` is `YYYYMMDDTHHMMSSZ`; `date_stamp` is `YYYYMMDD`.
#[must_use]
pub fn sign(
    req: &CanonicalRequest,
    cred: &Credentials,
    amz_date: &str,
    date_stamp: &str,
) -> Signature {
    let mut headers = req.headers.clone();
    headers.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_headers: String = headers
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k, v.trim()))
        .collect();
    let signed_headers: String = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        req.method,
        req.canonical_uri,
        req.canonical_query,
        canonical_headers,
        signed_headers,
        req.payload_sha256_hex,
    );

    let credential_scope = format!(
        "{}/{}/{}/aws4_request",
        date_stamp, cred.region, cred.service
    );
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes()),
    );

    let k_date = hmac(
        format!("AWS4{}", cred.secret_key).as_bytes(),
        date_stamp.as_bytes(),
    );
    let k_region = hmac(&k_date, cred.region.as_bytes());
    let k_service = hmac(&k_region, cred.service.as_bytes());
    let k_signing = hmac(&k_service, b"aws4_request");
    let signature = hex(&hmac(&k_signing, string_to_sign.as_bytes()));

    Signature {
        signature,
        signed_headers,
        credential_scope,
    }
}

/// Build the `Authorization` header value for `req`.
#[must_use]
pub fn authorization_header(
    req: &CanonicalRequest,
    cred: &Credentials,
    amz_date: &str,
    date_stamp: &str,
) -> String {
    let s = sign(req, cred, amz_date, date_stamp);
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        cred.access_key, s.credential_scope, s.signed_headers, s.signature
    )
}

/// Percent-encode per RFC 3986 unreserved set. When `encode_slash` is false,
/// `/` is preserved (path segments); otherwise it is encoded (query values).
#[must_use]
pub fn uri_encode(input: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if unreserved || (b == b'/' && !encode_slash) {
            out.push(b as char);
        } else {
            use core::fmt::Write;
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    /// AWS-published SigV4 `GET Object` example (docs: "Examples of the complete
    /// Version 4 signing process"). Fixed inputs → a known-correct signature.
    #[test]
    fn aws_get_object_known_answer() {
        let headers = vec![
            (
                "host".to_string(),
                "examplebucket.s3.amazonaws.com".to_string(),
            ),
            ("range".to_string(), "bytes=0-9".to_string()),
            (
                "x-amz-content-sha256".to_string(),
                EMPTY_PAYLOAD_SHA256.to_string(),
            ),
            ("x-amz-date".to_string(), "20130524T000000Z".to_string()),
        ];
        let req = CanonicalRequest {
            method: "GET",
            canonical_uri: "/test.txt",
            canonical_query: "",
            headers,
            payload_sha256_hex: EMPTY_PAYLOAD_SHA256,
        };
        let cred = Credentials {
            access_key: "AKIAIOSFODNN7EXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region: "us-east-1",
            service: "s3",
        };
        let sig = sign(&req, &cred, "20130524T000000Z", "20130524");
        assert_eq!(
            sig.signature, "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41",
            "SigV4 signature must match the AWS-published vector"
        );
        assert_eq!(
            sig.signed_headers,
            "host;range;x-amz-content-sha256;x-amz-date"
        );
    }

    #[test]
    fn uri_encode_preserves_path_slash_and_encodes_specials() {
        assert_eq!(uri_encode("/my bucket/a+b", false), "/my%20bucket/a%2Bb");
        assert_eq!(uri_encode("a/b", true), "a%2Fb");
        assert_eq!(uri_encode("Az0-_.~", false), "Az0-_.~");
    }
}
