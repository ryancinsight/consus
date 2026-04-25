//! Error types for the consus-mat MATLAB file reader.

#[cfg(feature = "alloc")]
use alloc::string::String;

/// Errors produced by MATLAB .mat file parsing.
#[derive(Debug)]
pub enum MatError {
    /// Structural or semantic format violation.
    #[cfg(feature = "alloc")]
    InvalidFormat(String),
    #[cfg(not(feature = "alloc"))]
    InvalidFormat,
    /// A specific feature is not implemented (e.g. VAX floats, sparse v4).
    #[cfg(feature = "alloc")]
    UnsupportedFeature(String),
    #[cfg(not(feature = "alloc"))]
    UnsupportedFeature,
    /// Unknown MATLAB array class code.
    InvalidClass(u8),
    /// Shape invariant violated.
    #[cfg(feature = "alloc")]
    ShapeError(String),
    #[cfg(not(feature = "alloc"))]
    ShapeError,
    /// Compression/decompression failure.
    #[cfg(feature = "alloc")]
    CompressionError(String),
    #[cfg(not(feature = "alloc"))]
    CompressionError,
    /// Underlying I/O error (std only).
    #[cfg(feature = "std")]
    Io(std::io::Error),
}

impl core::fmt::Display for MatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "alloc")]
            Self::InvalidFormat(m) => write!(f, "invalid MAT format: {m}"),
            #[cfg(not(feature = "alloc"))]
            Self::InvalidFormat => write!(f, "invalid MAT format"),
            #[cfg(feature = "alloc")]
            Self::UnsupportedFeature(ft) => write!(f, "unsupported feature: {ft}"),
            #[cfg(not(feature = "alloc"))]
            Self::UnsupportedFeature => write!(f, "unsupported feature"),
            Self::InvalidClass(c) => write!(f, "invalid MATLAB array class: {c}"),
            #[cfg(feature = "alloc")]
            Self::ShapeError(m) => write!(f, "shape error: {m}"),
            #[cfg(not(feature = "alloc"))]
            Self::ShapeError => write!(f, "shape error"),
            #[cfg(feature = "alloc")]
            Self::CompressionError(m) => write!(f, "compression error: {m}"),
            #[cfg(not(feature = "alloc"))]
            Self::CompressionError => write!(f, "compression error"),
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<consus_core::Error> for MatError {
    fn from(e: consus_core::Error) -> Self {
        #[cfg(feature = "alloc")]
        {
            Self::InvalidFormat(alloc::format!("{e}"))
        }
        #[cfg(not(feature = "alloc"))]
        {
            let _ = e;
            Self::InvalidFormat
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn invalid_format_display() {
        let e = MatError::InvalidFormat("bad header".to_string());
        assert_eq!(e.to_string(), "invalid MAT format: bad header");
    }

    #[test]
    fn unsupported_feature_display() {
        let e = MatError::UnsupportedFeature("v4 sparse".to_string());
        assert_eq!(e.to_string(), "unsupported feature: v4 sparse");
    }

    #[test]
    fn invalid_class_display() {
        let e = MatError::InvalidClass(42);
        assert_eq!(e.to_string(), "invalid MATLAB array class: 42");
    }

    #[test]
    fn shape_error_display() {
        let e = MatError::ShapeError("dimension mismatch".to_string());
        assert_eq!(e.to_string(), "shape error: dimension mismatch");
    }

    #[test]
    fn compression_error_display() {
        let e = MatError::CompressionError("zlib failure".to_string());
        assert_eq!(e.to_string(), "compression error: zlib failure");
    }
}
