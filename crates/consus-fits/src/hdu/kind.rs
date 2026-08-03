use consus_core::Result;

use crate::types::HduType;

use super::index::FitsHduIndex;
use super::support::invalid_format;

/// FITS HDU semantic classification.
///
/// This refines `crate::types::HduType` with primary/extension role semantics
/// while preserving the canonical FITS HDU taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitsHduKind {
    /// Primary HDU.
    Primary,
    /// IMAGE extension HDU.
    ImageExtension,
    /// ASCII table extension HDU.
    AsciiTableExtension,
    /// Binary table extension HDU.
    BinaryTableExtension,
}

impl FitsHduKind {
    /// Derive an HDU kind from file position and optional `XTENSION` value.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if:
    /// - the primary HDU carries `XTENSION`
    /// - an extension HDU omits `XTENSION`
    /// - the extension type is unsupported
    pub fn from_position_and_xtension(index: FitsHduIndex, xtension: Option<&str>) -> Result<Self> {
        if index.is_primary() {
            if xtension.is_some() {
                return invalid_format("primary HDU must not define XTENSION");
            }
            return Ok(Self::Primary);
        }

        match xtension.map(str::trim_end) {
            Some("IMAGE") => Ok(Self::ImageExtension),
            Some("TABLE") => Ok(Self::AsciiTableExtension),
            Some("BINTABLE") => Ok(Self::BinaryTableExtension),
            Some(_) => invalid_format("unsupported FITS XTENSION value"),
            None => invalid_format("extension HDU is missing XTENSION"),
        }
    }

    /// Return the canonical `HduType`.
    pub const fn hdu_type(self) -> HduType {
        match self {
            Self::Primary => HduType::Primary,
            Self::ImageExtension => HduType::Image,
            Self::AsciiTableExtension => HduType::Table,
            Self::BinaryTableExtension => HduType::BinTable,
        }
    }

    /// Return whether this is the primary HDU kind.
    pub const fn is_primary(self) -> bool {
        matches!(self, Self::Primary)
    }

    /// Return whether this is an extension HDU kind.
    pub const fn is_extension(self) -> bool {
        !self.is_primary()
    }

    /// Return whether this HDU carries image payload semantics.
    pub const fn is_image(self) -> bool {
        matches!(self, Self::Primary | Self::ImageExtension)
    }

    /// Return whether this HDU carries ASCII table payload semantics.
    pub const fn is_ascii_table(self) -> bool {
        matches!(self, Self::AsciiTableExtension)
    }

    /// Return whether this HDU carries binary table payload semantics.
    pub const fn is_binary_table(self) -> bool {
        matches!(self, Self::BinaryTableExtension)
    }
}
