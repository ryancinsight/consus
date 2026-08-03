/// FITS HDU ordinal index.
///
/// The primary HDU always has index `0`. Extension HDUs follow in file order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FitsHduIndex(usize);

impl FitsHduIndex {
    /// Construct an HDU index from a zero-based ordinal.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based ordinal.
    pub const fn get(self) -> usize {
        self.0
    }

    /// Return whether this is the primary HDU index.
    pub const fn is_primary(self) -> bool {
        self.0 == 0
    }
}

impl From<usize> for FitsHduIndex {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}
