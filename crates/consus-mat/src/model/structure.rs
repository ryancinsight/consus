//! MATLAB struct array model.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use super::MatArray;
#[cfg(feature = "alloc")]
use crate::error::MatError;

/// MATLAB struct array.
///
/// ## Invariants
///
/// - Field names are stored exactly once as the first component of each
///   `data` entry. [`MatStructArray::field_names`] derives names from `data`
///   directly - there is no separate `fields` field.
/// - `data[i].1.len() == shape.iter().product::<usize>()` for all `i`.
/// - Field names in `data` keys are unique within this struct.
/// - Elements within each field's `Vec<MatArray>` are in MATLAB
///   column-major order.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct MatStructArray {
    /// MATLAB shape dimensions.
    pub shape: Vec<usize>,
    /// Per-field element vectors: `(field_name, column_major_elements)`.
    ///
    /// Field names are the single source of truth for struct field identity.
    /// Enumerate them via [`MatStructArray::field_names`].
    pub data: Vec<(String, Vec<MatArray>)>,
}

#[cfg(feature = "alloc")]
impl MatStructArray {
    /// Construct a struct array after validating all documented invariants.
    ///
    /// Field names are derived from the keys of `data`. There is no separate
    /// `fields` parameter; the `data` vector is the single source of truth.
    pub fn new(
        shape: Vec<usize>,
        data: Vec<(String, Vec<MatArray>)>,
    ) -> Result<Self, MatError> {
        let expected_len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };

        // Validate field name uniqueness.
        for i in 0..data.len() {
            for j in (i + 1)..data.len() {
                if data[i].0 == data[j].0 {
                    return Err(MatError::InvalidFormat(String::from(
                        "struct: duplicate field name",
                    )));
                }
            }
        }

        // Validate element count per field.
        for (_, values) in &data {
            if values.len() != expected_len {
                return Err(MatError::ShapeError(String::from(
                    "struct: field element count must equal shape product",
                )));
            }
        }

        Ok(Self { shape, data })
    }

    /// Number of struct elements in MATLAB column-major order.
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Ordered field names for this struct type.
    ///
    /// Names are derived from `data` keys (single source of truth).
    pub fn field_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.data.iter().map(|(n, _)| n.as_str())
    }

    /// Field storage as `(field_name, column_major_elements)` pairs.
    pub fn field_data(&self) -> &[(String, Vec<MatArray>)] {
        &self.data
    }

    /// Return the column-major values for a named field.
    pub fn field(&self, name: &str) -> Option<&[MatArray]> {
        self.data
            .iter()
            .find(|(field_name, _)| field_name == name)
            .map(|(_, values)| values.as_slice())
    }
}

#[cfg(feature = "alloc")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MatArray;
    use crate::model::numeric::{MatNumericArray, MatNumericClass};

    fn scalar_numeric(v: f64) -> MatArray {
        MatArray::Numeric(MatNumericArray {
            class: MatNumericClass::Double,
            shape: vec![1, 1],
            real_data: v.to_le_bytes().to_vec(),
            imag_data: None,
        })
    }

    fn build_struct() -> MatStructArray {
        MatStructArray::new(
            vec![1, 1],
            vec![
                ("x".to_string(), vec![scalar_numeric(1.0)]),
                ("y".to_string(), vec![scalar_numeric(2.0)]),
            ],
        ).unwrap()
    }

    #[test]
    fn field_names_returns_correct_ordered_names() {
        let sa = build_struct();
        let names: Vec<&str> = sa.field_names().collect();
        assert_eq!(names, vec!["x", "y"]);
    }

    #[test]
    fn field_returns_correct_values_for_present_name() {
        let sa = build_struct();
        let vals = sa.field("x").expect("field x must exist");
        assert_eq!(vals.len(), 1);
        if let MatArray::Numeric(na) = &vals[0] {
            let v = f64::from_le_bytes([
                na.real_data[0], na.real_data[1], na.real_data[2], na.real_data[3],
                na.real_data[4], na.real_data[5], na.real_data[6], na.real_data[7],
            ]);
            assert_eq!(v, 1.0);
        } else {
            panic!("expected Numeric");
        }
    }

    #[test]
    fn field_returns_none_for_absent_name() {
        let sa = build_struct();
        assert!(sa.field("z").is_none());
    }

    #[test]
    fn field_data_returns_all_pairs() {
        let sa = build_struct();
        assert_eq!(sa.field_data().len(), 2);
        assert_eq!(sa.field_data()[0].0, "x");
        assert_eq!(sa.field_data()[1].0, "y");
    }

    #[test]
    fn numel_is_shape_product() {
        let sa = build_struct();
        assert_eq!(sa.numel(), 1);
    }

    #[test]
    fn new_duplicate_field_name_returns_error() {
        let err = MatStructArray::new(
            vec![1, 1],
            vec![
                ("x".to_string(), vec![scalar_numeric(1.0)]),
                ("x".to_string(), vec![scalar_numeric(2.0)]),
            ],
        );
        assert!(err.is_err());
    }

    #[test]
    fn new_field_element_count_mismatch_returns_error() {
        let err = MatStructArray::new(
            vec![1, 2],
            vec![
                ("x".to_string(), vec![scalar_numeric(1.0)]),
            ],
        );
        assert!(err.is_err());
    }
}
