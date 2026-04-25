//! MATLAB dense numeric array model.
//!
//! ## Invariants
//!
//! - `real_data.len() == numel() * class.element_size()`
//! - `imag_data` when present has the same length as `real_data`.
//! - Bytes are stored in little-endian order (normalized on load).
//! - Shape dimensions follow MATLAB convention (row-major index names,
//!   column-major storage).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// MATLAB numeric array class (identifies precision and signedness).
///
/// ## Element Size Contract
///
/// `element_size()` returns the number of bytes per scalar element for
/// the class. This is a compile-time constant per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatNumericClass {
    /// 64-bit IEEE 754 float (`double` in MATLAB).
    Double,
    /// 32-bit IEEE 754 float (`single` in MATLAB).
    Single,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit unsigned integer.
    Uint8,
    /// 16-bit unsigned integer.
    Uint16,
    /// 32-bit unsigned integer.
    Uint32,
    /// 64-bit unsigned integer.
    Uint64,
}

impl MatNumericClass {
    /// Number of bytes per scalar element.
    ///
    /// ## Derivation
    ///
    /// `Double`/`Int64`/`Uint64` → 8, `Single`/`Int32`/`Uint32` → 4,
    /// `Int16`/`Uint16` → 2, `Int8`/`Uint8` → 1.
    #[inline]
    pub const fn element_size(self) -> usize {
        match self {
            Self::Double | Self::Int64 | Self::Uint64 => 8,
            Self::Single | Self::Int32 | Self::Uint32 => 4,
            Self::Int16 | Self::Uint16 => 2,
            Self::Int8 | Self::Uint8 => 1,
        }
    }

    /// Canonical string name matching MATLAB `class()` output.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Double  => "double",
            Self::Single  => "single",
            Self::Int8    => "int8",
            Self::Int16   => "int16",
            Self::Int32   => "int32",
            Self::Int64   => "int64",
            Self::Uint8   => "uint8",
            Self::Uint16  => "uint16",
            Self::Uint32  => "uint32",
            Self::Uint64  => "uint64",
        }
    }
}

/// Dense numeric array (real or complex).
///
/// Data bytes are stored in MATLAB column-major order, normalized to
/// little-endian on load. The invariant `real_data.len() ==
/// numel() * class.element_size()` holds after successful parsing.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct MatNumericArray {
    /// Numeric class (precision + signedness).
    pub class: MatNumericClass,
    /// MATLAB shape dimensions `[nrows, ncols, ...]`.
    pub shape: Vec<usize>,
    /// Raw element bytes in MATLAB column-major order (little-endian).
    pub real_data: Vec<u8>,
    /// Imaginary part bytes, present for complex arrays.
    pub imag_data: Option<Vec<u8>>,
}

#[cfg(feature = "alloc")]
impl MatNumericArray {
    /// Total number of elements: `∏ shape[i]`.
    ///
    /// Returns 1 for empty (scalar) shape.
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Whether this array is complex.
    pub fn is_complex(&self) -> bool {
        self.imag_data.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_size_matches_rust_type_widths() {
        assert_eq!(MatNumericClass::Double.element_size(), 8);
        assert_eq!(MatNumericClass::Single.element_size(), 4);
        assert_eq!(MatNumericClass::Int64.element_size(), 8);
        assert_eq!(MatNumericClass::Uint64.element_size(), 8);
        assert_eq!(MatNumericClass::Int32.element_size(), 4);
        assert_eq!(MatNumericClass::Uint32.element_size(), 4);
        assert_eq!(MatNumericClass::Int16.element_size(), 2);
        assert_eq!(MatNumericClass::Uint16.element_size(), 2);
        assert_eq!(MatNumericClass::Int8.element_size(), 1);
        assert_eq!(MatNumericClass::Uint8.element_size(), 1);
    }

    #[test]
    fn as_str_returns_matlab_class_names() {
        assert_eq!(MatNumericClass::Double.as_str(), "double");
        assert_eq!(MatNumericClass::Single.as_str(), "single");
        assert_eq!(MatNumericClass::Int8.as_str(), "int8");
        assert_eq!(MatNumericClass::Int16.as_str(), "int16");
        assert_eq!(MatNumericClass::Int32.as_str(), "int32");
        assert_eq!(MatNumericClass::Int64.as_str(), "int64");
        assert_eq!(MatNumericClass::Uint8.as_str(), "uint8");
        assert_eq!(MatNumericClass::Uint16.as_str(), "uint16");
        assert_eq!(MatNumericClass::Uint32.as_str(), "uint32");
        assert_eq!(MatNumericClass::Uint64.as_str(), "uint64");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn numel_empty_shape_returns_one() {
        let a = MatNumericArray {
            class: MatNumericClass::Double,
            shape: vec![],
            real_data: vec![0u8; 8],
            imag_data: None,
        };
        assert_eq!(a.numel(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn numel_2d_shape_is_product() {
        let a = MatNumericArray {
            class: MatNumericClass::Double,
            shape: vec![3, 4],
            real_data: vec![0u8; 3 * 4 * 8],
            imag_data: None,
        };
        assert_eq!(a.numel(), 12);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn is_complex_false_when_no_imag_data() {
        let a = MatNumericArray {
            class: MatNumericClass::Double,
            shape: vec![1, 1],
            real_data: vec![0u8; 8],
            imag_data: None,
        };
        assert!(!a.is_complex());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn is_complex_true_when_imag_data_present() {
        let a = MatNumericArray {
            class: MatNumericClass::Double,
            shape: vec![1, 1],
            real_data: vec![0u8; 8],
            imag_data: Some(vec![0u8; 8]),
        };
        assert!(a.is_complex());
    }
}
