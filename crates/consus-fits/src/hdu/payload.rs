use crate::image::FitsImageDescriptor;
use crate::table::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor};

/// FITS HDU payload descriptor.
///
/// This enum attaches the parsed semantic payload descriptor corresponding to
/// the HDU kind.
#[derive(Debug, Clone, PartialEq)]
pub enum FitsHduPayload {
    /// Image payload descriptor for primary or IMAGE extension HDUs.
    Image(FitsImageDescriptor),
    /// ASCII table payload descriptor.
    AsciiTable(FitsAsciiTableDescriptor),
    /// Binary table payload descriptor.
    BinaryTable(FitsBinaryTableDescriptor),
}

impl FitsHduPayload {
    /// Return whether this payload is image-like.
    pub const fn is_image(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    /// Return whether this payload is an ASCII table.
    pub const fn is_ascii_table(&self) -> bool {
        matches!(self, Self::AsciiTable(_))
    }

    /// Return whether this payload is a binary table.
    pub const fn is_binary_table(&self) -> bool {
        matches!(self, Self::BinaryTable(_))
    }

    /// Return the image descriptor, if present.
    pub const fn as_image(&self) -> Option<&FitsImageDescriptor> {
        match self {
            Self::Image(value) => Some(value),
            _ => None,
        }
    }

    /// Return the ASCII table descriptor, if present.
    pub const fn as_ascii_table(&self) -> Option<&FitsAsciiTableDescriptor> {
        match self {
            Self::AsciiTable(value) => Some(value),
            _ => None,
        }
    }

    /// Return the binary table descriptor, if present.
    pub const fn as_binary_table(&self) -> Option<&FitsBinaryTableDescriptor> {
        match self {
            Self::BinaryTable(value) => Some(value),
            _ => None,
        }
    }
}
