//! MATLAB cell array model.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use super::MatArray;
use crate::error::MatError;

/// MATLAB cell array (heterogeneous container).
///
/// ## Invariants
///
/// - `cells.len() == shape.iter().product::<usize>()`.
/// - Elements are stored in MATLAB column-major order.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct MatCellArray {
    /// MATLAB shape dimensions.
    pub shape: Vec<usize>,
    /// Cell elements in MATLAB column-major order.
    pub cells: Vec<MatArray>,
}

#[cfg(feature = "alloc")]
impl MatCellArray {
    /// Constructs a cell array after enforcing the shape/cardinality invariant.
    pub fn new(shape: Vec<usize>, cells: Vec<MatArray>) -> Result<Self, MatError> {
        let expected_len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if cells.len() != expected_len {
            return Err(MatError::ShapeError(String::from(
                "cell array element count does not match shape product",
            )));
        }
        Ok(Self { shape, cells })
    }

    /// Returns the total number of logical elements.
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Returns the shape as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the cell elements in MATLAB column-major order.
    pub fn cells(&self) -> &[MatArray] {
        &self.cells
    }

    /// Consumes the array and returns the underlying shape and cells.
    pub fn into_parts(self) -> (Vec<usize>, Vec<MatArray>) {
        (self.shape, self.cells)
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MatArray;
    use crate::model::numeric::{MatNumericArray, MatNumericClass};

    fn dummy_numeric() -> MatArray {
        MatArray::Numeric(MatNumericArray {
            class: MatNumericClass::Double,
            shape: vec![1, 1],
            real_data: vec![0u8; 8],
            imag_data: None,
        })
    }

    #[test]
    fn new_valid_cell_array() {
        let ca = MatCellArray::new(vec![1, 2], vec![dummy_numeric(), dummy_numeric()]).unwrap();
        assert_eq!(ca.numel(), 2);
        assert_eq!(ca.shape(), &[1, 2]);
        assert_eq!(ca.cells().len(), 2);
    }

    #[test]
    fn new_element_count_mismatch_returns_error() {
        let err = MatCellArray::new(vec![2, 3], vec![dummy_numeric(), dummy_numeric()]);
        assert!(err.is_err());
    }

    #[test]
    fn numel_empty_shape_returns_one() {
        let ca = MatCellArray::new(vec![], vec![dummy_numeric()]).unwrap();
        assert_eq!(ca.numel(), 1);
    }

    #[test]
    fn into_parts_yields_original_shape_and_cells() {
        let cells = vec![dummy_numeric()];
        let (shape, got_cells) = MatCellArray::new(vec![1, 1], cells).unwrap().into_parts();
        assert_eq!(shape, vec![1, 1]);
        assert_eq!(got_cells.len(), 1);
    }
}
