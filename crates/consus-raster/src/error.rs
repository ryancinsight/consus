use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Stable classification for raster decode failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeErrorKind {
    /// The input violates its format grammar or coding invariants.
    Malformed,
    /// The input requests a format feature this crate does not implement.
    Unsupported,
    /// The input or decoded representation exceeds a caller-supplied limit.
    TooLarge,
    /// The allocator refused a bounded allocation.
    Allocation,
}

/// A raster decode failure with a stable classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeError {
    kind: DecodeErrorKind,
}

impl DecodeError {
    /// Returns the stable failure classification.
    #[must_use]
    pub const fn kind(&self) -> DecodeErrorKind {
        self.kind
    }

    pub(crate) const fn new(kind: DecodeErrorKind) -> Self {
        Self { kind }
    }
}

impl Display for DecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        let message = match self.kind {
            DecodeErrorKind::Malformed => "malformed raster data",
            DecodeErrorKind::Unsupported => "unsupported raster feature",
            DecodeErrorKind::TooLarge => "raster exceeds a decode limit",
            DecodeErrorKind::Allocation => "raster allocation failed",
        };
        formatter.write_str(message)
    }
}

impl Error for DecodeError {}
