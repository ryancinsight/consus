//! Normalize a dense numeric payload from its MAT storage type and byte order.

use alloc::{format, vec::Vec};
use consus_core::{ByteOrder, EndianScalar, ParseBudget, decode_each};

use super::{element::normalize_endian, tag::MiType};
use crate::{MatError, MatNumericClass};

pub(super) fn normalize_payload(
    data: Vec<u8>,
    stored: MiType,
    class: MatNumericClass,
    count: usize,
    big_endian: bool,
    budget: &ParseBudget,
) -> Result<Vec<u8>, MatError> {
    if class != MatNumericClass::Double || stored == MiType::Double {
        let width = class.element_size();
        let expected = count.checked_mul(width).ok_or_else(|| {
            MatError::ShapeError("miMATRIX dimensions exceed the addressable byte count".into())
        })?;
        if data.len() != expected {
            return Err(MatError::ShapeError(format!(
                "numeric payload has {} bytes, expected {expected} for {count} elements",
                data.len()
            )));
        }
        return Ok(normalize_endian(data, width, big_endian));
    }

    // The matrix class names the logical type. MATLAB may encode integral
    // double values in a narrower numeric sub-element (Level 5 MAT-File Format).
    let count = u64::try_from(count).map_err(|_| consus_core::Error::Overflow)?;
    let count = budget.checked_elements(count, size_of::<f64>(), "MAT v5 double values")?;
    let order = if big_endian {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    match stored {
        MiType::Int8 => expand::<i8>(&data, count, order, f64::from),
        MiType::Uint8 => expand::<u8>(&data, count, order, f64::from),
        MiType::Int16 => expand::<i16>(&data, count, order, f64::from),
        MiType::Uint16 => expand::<u16>(&data, count, order, f64::from),
        MiType::Int32 => expand::<i32>(&data, count, order, f64::from),
        MiType::Uint32 => expand::<u32>(&data, count, order, f64::from),
        MiType::Single => expand::<f32>(&data, count, order, f64::from),
        // A double array has binary64 precision. Integer values outside its
        // exactly representable set round to the nearest binary64 value.
        MiType::Int64 => expand(&data, count, order, |value: i64| value as f64),
        MiType::Uint64 => expand(&data, count, order, |value: u64| value as f64),
        _ => Err(MatError::InvalidFormat(format!(
            "invalid storage type {stored:?} for a double array"
        ))),
    }
}

fn expand<T: EndianScalar>(
    input: &[u8],
    count: usize,
    order: ByteOrder,
    to_double: impl Fn(T) -> f64,
) -> Result<Vec<u8>, MatError> {
    if count.checked_mul(T::BYTE_WIDTH) != Some(input.len()) {
        return Err(MatError::ShapeError(format!(
            "packed double payload has {} bytes for {count} elements of width {}",
            input.len(),
            T::BYTE_WIDTH
        )));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(count * size_of::<f64>())
        .map_err(|error| MatError::ShapeError(format!("cannot allocate double values: {error}")))?;
    decode_each(input, order, |value| {
        output.extend_from_slice(&to_double(value).to_le_bytes())
    })
    .expect("validated input contains exactly the declared number of scalars");
    Ok(output)
}
