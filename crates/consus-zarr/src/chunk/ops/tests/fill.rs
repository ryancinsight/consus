//! Fill value and checked size arithmetic tests.

use super::super::*;
use crate::chunk::error::ChunkError;
use crate::metadata::FillValue;

#[test]
fn expand_fill_value_test() {
    let fill = FillValue::Float("42.0".to_string());
    let dtype = "<f8";
    let expanded = expand_fill_value(&fill, dtype, 4);
    assert_eq!(expanded.len(), 32); // 4 * 8 bytes
}

#[test]
fn expand_fill_value_float32_one() {
    let fill = FillValue::Float("1.0".to_string());
    let expanded = expand_fill_value(&fill, "<f4", 1);
    assert_eq!(expanded, vec![0x00, 0x00, 0x80, 0x3f]);
}

#[test]
fn expand_fill_value_float64_one() {
    let fill = FillValue::Float("1.0".to_string());
    let expanded = expand_fill_value(&fill, "<f8", 1);
    assert_eq!(expanded, 1.0f64.to_le_bytes().to_vec());
}

/// A hostile element count must be a typed error, not an allocation abort
/// or a multiply-overflow panic: `num_elements × element_size` is bounded
/// by the parser byte ceiling.
#[test]
fn hostile_fill_value_element_count_is_a_typed_error() {
    let fill = FillValue::Float("42.0".to_string());
    let result = try_expand_fill_value(&fill, "<f8", u64::MAX);
    assert!(
        matches!(result, Err(ChunkError::StoreError(_))),
        "a hostile element count must be rejected, got {result:?}"
    );
}
