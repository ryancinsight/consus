//! MATLAB sparse matrix model (CSC format).

#[cfg(feature = "alloc")]
use alloc::{format, vec::Vec};

#[cfg(feature = "alloc")]
use crate::error::MatError;

/// MATLAB sparse matrix in CSC (compressed sparse column) format.
///
/// ## Invariants
///
/// - `col_ptrs.len() == ncols + 1`.
/// - `row_indices.len() == col_ptrs[ncols] as usize` (number of stored non-zeros).
/// - `real_data.len() == row_indices.len() * 8` (f64, 8 bytes per non-zero).
/// - `imag_data`, when present, has the same length as `real_data`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct MatSparseArray {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row index of each stored element (0-based, CSC `ir` array).
    pub row_indices: Vec<i32>,
    /// Column start pointers (CSC `jc` array); length = `ncols + 1`.
    pub col_ptrs: Vec<i32>,
    /// Raw f64 bytes for real non-zero values (little-endian, 8 bytes each).
    pub real_data: Vec<u8>,
    /// Raw f64 bytes for imaginary non-zero values, present for complex sparse.
    pub imag_data: Option<Vec<u8>>,
}

#[cfg(feature = "alloc")]
impl MatSparseArray {
    /// Construct a sparse array after validating CSC and payload invariants.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<i32>,
        col_ptrs: Vec<i32>,
        real_data: Vec<u8>,
        imag_data: Option<Vec<u8>>,
    ) -> Result<Self, MatError> {
        let array = Self {
            nrows,
            ncols,
            row_indices,
            col_ptrs,
            real_data,
            imag_data,
        };
        array.validate()?;
        Ok(array)
    }

    /// Validate structural and byte-length invariants.
    pub fn validate(&self) -> Result<(), MatError> {
        if self.col_ptrs.len() != self.ncols + 1 {
            return Err(MatError::ShapeError(format!(
                "sparse: jc.len() {} != ncols+1 {}",
                self.col_ptrs.len(),
                self.ncols + 1
            )));
        }

        let nnz_i32 = *self.col_ptrs.last().ok_or_else(|| {
            MatError::ShapeError(format!(
                "sparse: jc.len() {} != ncols+1 {}",
                self.col_ptrs.len(),
                self.ncols + 1
            ))
        })?;

        if nnz_i32 < 0 {
            return Err(MatError::ShapeError(format!(
                "sparse: jc[ncols] {} is negative",
                nnz_i32
            )));
        }

        let nnz = nnz_i32 as usize;

        if self.row_indices.len() != nnz {
            return Err(MatError::ShapeError(format!(
                "sparse: ir.len() {} != nnz {}",
                self.row_indices.len(),
                nnz
            )));
        }

        if self.real_data.len() != nnz * 8 {
            return Err(MatError::ShapeError(format!(
                "sparse: real_data.len() {} != nnz*8 {}",
                self.real_data.len(),
                nnz * 8
            )));
        }

        if let Some(imag_data) = &self.imag_data {
            if imag_data.len() != self.real_data.len() {
                return Err(MatError::ShapeError(format!(
                    "sparse: imag_data.len() {} != real_data.len() {}",
                    imag_data.len(),
                    self.real_data.len()
                )));
            }
        }

        for (idx, &row) in self.row_indices.iter().enumerate() {
            if row < 0 {
                return Err(MatError::ShapeError(format!(
                    "sparse: ir[{idx}] {} is negative",
                    row
                )));
            }
            if row as usize >= self.nrows {
                return Err(MatError::ShapeError(format!(
                    "sparse: ir[{idx}] {} >= nrows {}",
                    row, self.nrows
                )));
            }
        }

        let mut prev = 0i32;
        for (idx, &ptr) in self.col_ptrs.iter().enumerate() {
            if ptr < 0 {
                return Err(MatError::ShapeError(format!(
                    "sparse: jc[{idx}] {} is negative",
                    ptr
                )));
            }
            if idx > 0 && ptr < prev {
                return Err(MatError::ShapeError(format!(
                    "sparse: jc[{idx}] {} < jc[{}] {}",
                    ptr,
                    idx - 1,
                    prev
                )));
            }
            prev = ptr;
        }

        Ok(())
    }

    /// Number of stored non-zero elements.
    pub fn nnz(&self) -> usize {
        self.row_indices.len()
    }

    /// Whether the sparse array has an imaginary component.
    pub fn is_complex(&self) -> bool {
        self.imag_data.is_some()
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use super::*;

    fn valid_sparse() -> MatSparseArray {
        MatSparseArray::new(
            3, 3,
            vec![0i32, 2i32],
            vec![0i32, 1i32, 1i32, 2i32],
            [5.0f64.to_le_bytes(), 7.0f64.to_le_bytes()].concat(),
            None,
        ).unwrap()
    }

    #[test]
    fn nnz_returns_stored_count() {
        let sa = valid_sparse();
        assert_eq!(sa.nnz(), 2);
    }

    #[test]
    fn is_complex_false_when_real_only() {
        assert!(!valid_sparse().is_complex());
    }

    #[test]
    fn is_complex_true_when_imag_present() {
        let imag = [0.0f64.to_le_bytes(), 1.0f64.to_le_bytes()].concat();
        let sa = MatSparseArray::new(3, 3, vec![0, 2], vec![0, 1, 1, 2],
            [5.0f64.to_le_bytes(), 7.0f64.to_le_bytes()].concat(),
            Some(imag)).unwrap();
        assert!(sa.is_complex());
    }

    #[test]
    fn validate_jc_wrong_length_returns_error() {
        let err = MatSparseArray::new(3, 3, vec![0, 2], vec![0, 1, 2],
            [5.0f64.to_le_bytes(), 7.0f64.to_le_bytes()].concat(), None);
        assert!(err.is_err());
    }

    #[test]
    fn validate_ir_wrong_length_returns_error() {
        let err = MatSparseArray::new(3, 3, vec![0, 1, 2], vec![0, 1, 1, 2],
            [5.0f64.to_le_bytes(), 7.0f64.to_le_bytes()].concat(), None);
        assert!(err.is_err());
    }

    #[test]
    fn validate_real_data_wrong_length_returns_error() {
        let err = MatSparseArray::new(3, 3, vec![0, 2], vec![0, 1, 1, 2],
            5.0f64.to_le_bytes().to_vec(), None);
        assert!(err.is_err());
    }

    #[test]
    fn validate_row_index_out_of_bounds_returns_error() {
        let err = MatSparseArray::new(3, 3, vec![0, 5], vec![0, 1, 1, 2],
            [5.0f64.to_le_bytes(), 7.0f64.to_le_bytes()].concat(), None);
        assert!(err.is_err());
    }

    #[test]
    fn validate_non_monotone_jc_returns_error() {
        let err = MatSparseArray::new(3, 3, vec![0, 2], vec![0, 2, 1, 2],
            [5.0f64.to_le_bytes(), 7.0f64.to_le_bytes()].concat(), None);
        assert!(err.is_err());
    }
}
