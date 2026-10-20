//! ZIP container error model.

/// ZIP container operation result.
pub type Result<T> = core::result::Result<T, Error>;

/// Failures while reading or writing a ZIP container.
///
/// Every variant names the archive structure or member it concerns, so a
/// rejected input reports which declared field broke which bound.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The underlying reader or writer failed.
    #[error("ZIP I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// A structure ends before its fixed or declared length.
    #[error("ZIP {structure} is truncated")]
    Truncated {
        /// Structure being decoded.
        structure: &'static str,
    },
    /// A structure does not start with its signature.
    #[error("ZIP {structure} has signature {found:#010x}")]
    BadSignature {
        /// Structure being decoded.
        structure: &'static str,
        /// Little-endian signature word found at the structure's offset.
        found: u32,
    },
    /// No end-of-central-directory record exists in the archive tail.
    #[error("ZIP end of central directory record not found")]
    MissingEndOfCentralDirectory,
    /// The archive spans several disks, which this container does not read.
    #[error("multi-disk ZIP archives are not supported")]
    MultiDisk,
    /// A declared offset or length points outside the region that must hold it.
    #[error("ZIP {structure} lies outside its enclosing region")]
    OutOfBounds {
        /// Structure whose extent was checked.
        structure: &'static str,
    },
    /// The declared entry count cannot fit in the declared directory size.
    #[error("ZIP directory declares {declared} entries but {capacity} fit in its size")]
    EntryCount {
        /// Entry count read from the end record.
        declared: u64,
        /// Largest count the directory size admits at the minimum header length.
        capacity: u64,
    },
    /// A field holds the ZIP64 sentinel but the ZIP64 extra field omits it.
    #[error("ZIP64 extra field omits {field}")]
    MissingZip64Field {
        /// Field whose 32-bit value is the sentinel.
        field: &'static str,
    },
    /// The directory holds bytes after its declared entries.
    #[error("ZIP central directory has bytes after its last entry")]
    TrailingDirectoryBytes,
    /// A member name is neither ASCII nor flagged UTF-8, or is not valid UTF-8.
    #[error("ZIP member name is not ASCII or flagged UTF-8")]
    NameEncoding,
    /// A member name is empty or longer than the 16-bit name length field.
    #[error("ZIP member name {name:?} is invalid: {reason}")]
    InvalidName {
        /// Rejected name.
        name: String,
        /// Violated constraint.
        reason: &'static str,
    },
    /// Two members share one name.
    #[error("ZIP member {name:?} appears more than once")]
    DuplicateName {
        /// Repeated name.
        name: String,
    },
    /// A member is encrypted.
    #[error("ZIP member {name:?} is encrypted")]
    Encrypted {
        /// Member name.
        name: String,
    },
    /// A member uses a compression method other than stored (0) or deflate (8).
    #[error("ZIP member {name:?} uses unsupported compression method {method}")]
    UnsupportedMethod {
        /// Member name.
        name: String,
        /// Method code from the header.
        method: u16,
    },
    /// A member declares more output than its compressed size can produce.
    #[error(
        "ZIP member {name:?} declares {uncompressed} bytes from {compressed} compressed bytes, \
         beyond the DEFLATE expansion bound"
    )]
    ExpansionBound {
        /// Member name.
        name: String,
        /// Declared compressed size.
        compressed: u64,
        /// Declared uncompressed size.
        uncompressed: u64,
    },
    /// A member's decoded length differs from its declared uncompressed size.
    #[error("ZIP member {name:?} declares {declared} bytes but decodes to {actual}")]
    SizeMismatch {
        /// Member name.
        name: String,
        /// Declared uncompressed size.
        declared: u64,
        /// Decoded length.
        actual: u64,
    },
    /// A member's CRC-32 differs from its declared value.
    #[error("ZIP member {name:?} CRC-32 is {actual:#010x}, declared {expected:#010x}")]
    Checksum {
        /// Member name.
        name: String,
        /// CRC-32 from the central directory.
        expected: u32,
        /// CRC-32 of the decoded bytes.
        actual: u32,
    },
    /// A local header names a different member than its directory entry.
    #[error("ZIP local header for {name:?} names a different member")]
    LocalHeaderMismatch {
        /// Name from the central directory.
        name: String,
    },
    /// No member has the requested name.
    #[error("ZIP member {name:?} not found")]
    MissingEntry {
        /// Requested name.
        name: String,
    },
    /// A size or offset exceeds the platform's address range or the ZIP64 range.
    #[error("ZIP {structure} exceeds the representable range")]
    TooLarge {
        /// Quantity that overflowed.
        structure: &'static str,
    },
    /// The DEFLATE codec rejected a member's payload.
    #[error("ZIP member {name:?} DEFLATE stream failed: {source}")]
    Compression {
        /// Member name.
        name: String,
        /// Codec failure.
        #[source]
        source: consus_core::Error,
    },
}
