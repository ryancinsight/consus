//! NPY/NPZ error model.

/// NPY/NPZ operation result.
pub type Result<T> = core::result::Result<T, Error>;

/// Failures while validating or encoding NPY/NPZ data.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Underlying I/O failed.
    #[error("NPY I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// ZIP archive access failed.
    #[error("NPZ archive failed: {0}")]
    Zip(#[from] zip::result::ZipError),
    /// The input violates the NPY format contract.
    #[error("invalid NPY input: {0}")]
    InvalidFormat(String),
    /// The stored element representation does not match the requested type.
    #[error("NPY dtype mismatch: stored {stored}, requested {requested}")]
    DtypeMismatch {
        /// Descriptor read from the header.
        stored: String,
        /// Descriptor required by the requested Rust type.
        requested: &'static str,
    },
}
