use alloc::vec::Vec;
use consus_core::Result;

use super::descriptor::FitsHdu;
use super::index::FitsHduIndex;
use super::support::invalid_format;

/// Ordered FITS HDU sequence.
///
/// This type preserves file order and provides deterministic indexed access.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitsHduSequence {
    hdus: Vec<FitsHdu>,
}

impl FitsHduSequence {
    /// Construct an HDU sequence from ordered HDUs.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if:
    /// - the sequence is empty
    /// - the first HDU is not primary
    /// - any later HDU is primary
    /// - HDU indices are not contiguous and order-preserving
    pub fn new(hdus: Vec<FitsHdu>) -> Result<Self> {
        validate_sequence(&hdus)?;
        Ok(Self { hdus })
    }

    /// Construct an empty HDU sequence.
    pub const fn empty() -> Self {
        Self { hdus: Vec::new() }
    }

    /// Return the ordered HDUs.
    pub fn hdus(&self) -> &[FitsHdu] {
        &self.hdus
    }

    /// Return the number of HDUs.
    pub fn len(&self) -> usize {
        self.hdus.len()
    }

    /// Return whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.hdus.is_empty()
    }

    /// Return the primary HDU.
    pub fn primary(&self) -> Option<&FitsHdu> {
        self.hdus.first()
    }

    /// Return the HDU at `index`.
    pub fn get(&self, index: FitsHduIndex) -> Option<&FitsHdu> {
        self.hdus.get(index.get())
    }

    /// Return the HDU at zero-based ordinal `index`.
    pub fn get_usize(&self, index: usize) -> Option<&FitsHdu> {
        self.hdus.get(index)
    }

    /// Return an iterator over the HDUs.
    pub fn iter(&self) -> core::slice::Iter<'_, FitsHdu> {
        self.hdus.iter()
    }

    /// Append an HDU while preserving FITS sequence invariants.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if the appended HDU violates ordering,
    /// primary placement, or contiguous indexing.
    pub fn push(&mut self, hdu: FitsHdu) -> Result<()> {
        let expected_index = self.hdus.len();
        if hdu.index().get() != expected_index {
            return invalid_format("FITS HDU indices must be contiguous and ordered");
        }

        if expected_index == 0 {
            if !hdu.is_primary() {
                return invalid_format("first FITS HDU must be primary");
            }
        } else if hdu.is_primary() {
            return invalid_format("only the first FITS HDU may be primary");
        }

        self.hdus.push(hdu);
        Ok(())
    }
}

impl IntoIterator for FitsHduSequence {
    type Item = FitsHdu;
    type IntoIter = alloc::vec::IntoIter<FitsHdu>;

    fn into_iter(self) -> Self::IntoIter {
        self.hdus.into_iter()
    }
}

impl<'a> IntoIterator for &'a FitsHduSequence {
    type Item = &'a FitsHdu;
    type IntoIter = core::slice::Iter<'a, FitsHdu>;

    fn into_iter(self) -> Self::IntoIter {
        self.hdus.iter()
    }
}

fn validate_sequence(hdus: &[FitsHdu]) -> Result<()> {
    if hdus.is_empty() {
        return invalid_format("FITS file must contain at least one HDU");
    }

    for (expected_index, hdu) in hdus.iter().enumerate() {
        if hdu.index().get() != expected_index {
            return invalid_format("FITS HDU indices must be contiguous and ordered");
        }

        if expected_index == 0 {
            if !hdu.is_primary() {
                return invalid_format("first FITS HDU must be primary");
            }
        } else if hdu.is_primary() {
            return invalid_format("only the first FITS HDU may be primary");
        }
    }

    Ok(())
}
