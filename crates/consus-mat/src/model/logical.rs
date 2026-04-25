//! MATLAB logical (boolean) array model.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use crate::error::MatError;

/// MATLAB logical array.
///
/// ## Invariants
///
/// - `data.len() == shape.iter().product::<usize>()`.
/// - Elements are in MATLAB column-major order.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct MatLogicalArray {
    /// MATLAB shape dimensions.
    pub shape: Vec<usize>,
    /// Boolean values in MATLAB column-major order.
    pub data: Vec<bool>,
}

#[cfg(feature = "alloc")]
impl MatLogicalArray {
    /// Construct a logical array after validating shape/data cardinality.
    pub fn new(shape: Vec<usize>, data: Vec<bool>) -> Result<Self, MatError> {
        let expected_len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if data.len() != expected_len {
            return Err(MatError::ShapeError(String::from(
                "logical array data length does not match shape product",
            )));
        }
        Ok(Self { shape, data })
    }

    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Borrow the MATLAB shape vector.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the logical element buffer in MATLAB column-major order.
    pub fn data(&self) -> &[bool] {
        &self.data
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid_logical_array() {
        let la = MatLogicalArray::new(vec![1, 4], vec![true, false, true, true]).unwrap();
        assert_eq!(la.numel(), 4);
        assert_eq!(la.shape(), &[1, 4]);
        assert_eq!(la.data(), &[true, false, true, true]);
    }

    #[test]
    fn new_element_count_mismatch_returns_error() {
        let err = MatLogicalArray::new(vec![1, 3], vec![true, false]);
        assert!(err.is_err());
    }

    #[test]
    fn numel_empty_shape_returns_one() {
        let la = MatLogicalArray::new(vec![], vec![true]).unwrap();
        assert_eq!(la.numel(), 1);
    }
}
