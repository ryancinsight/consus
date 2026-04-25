//! MATLAB character array model.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use crate::error::MatError;

/// MATLAB character array (string or character matrix).
///
/// ## Invariants
///
/// - For a row vector string of length `n`: `shape == [1, n]`.
/// - `data` contains the decoded UTF-8 string value.
/// - `shape` product == number of Unicode scalar values in `data`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct MatCharArray {
    /// MATLAB shape `[nrows, ncols, ...]`.
    pub shape: Vec<usize>,
    /// Decoded character data as UTF-8.
    pub data: String,
}

#[cfg(feature = "alloc")]
impl MatCharArray {
    /// Construct a character array after validating shape/data consistency.
    pub fn new(shape: Vec<usize>, data: String) -> Result<Self, MatError> {
        let expected = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let actual = data.chars().count();
        if expected != actual {
            return Err(MatError::ShapeError(alloc::format!(
                "char array element count {} != shape product {}",
                actual,
                expected
            )));
        }
        Ok(Self { shape, data })
    }

    /// Construct a row-vector MATLAB string with shape `[1, n]`.
    pub fn row_vector(data: String) -> Self {
        let len = data.chars().count();
        Self {
            shape: vec![1, len],
            data,
        }
    }

    /// Total number of character elements.
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Borrow the MATLAB shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the decoded UTF-8 contents.
    pub fn data(&self) -> &str {
        &self.data
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid_char_array() {
        let ca = MatCharArray::new(vec![1, 5], "hello".to_string()).unwrap();
        assert_eq!(ca.numel(), 5);
        assert_eq!(ca.shape(), &[1, 5]);
        assert_eq!(ca.data(), "hello");
    }

    #[test]
    fn new_element_count_mismatch_returns_error() {
        let err = MatCharArray::new(vec![1, 3], "hello".to_string());
        assert!(err.is_err());
    }

    #[test]
    fn row_vector_sets_shape_1_n() {
        let ca = MatCharArray::row_vector("abc".to_string());
        assert_eq!(ca.shape(), &[1, 3]);
        assert_eq!(ca.data(), "abc");
    }

    #[test]
    fn numel_empty_shape_returns_one() {
        let ca = MatCharArray::new(vec![], "x".to_string()).unwrap();
        assert_eq!(ca.numel(), 1);
    }
}
