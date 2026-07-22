//! Typed owned array payloads.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::{Error, Result};

/// Scalar representations supported by the typed NPY boundary.
pub trait NpyElement: Copy + Sized {
    /// NumPy dtype descriptor for little-endian storage.
    const DTYPE: &'static str;

    /// Reads one scalar from the payload.
    fn read_from(reader: &mut impl Read) -> Result<Self>;

    /// Writes one scalar to the payload.
    fn write_to(self, writer: &mut impl Write) -> Result<()>;
}

macro_rules! impl_element {
    ($ty:ty, $dtype:literal, $read:ident, $write:ident) => {
        impl NpyElement for $ty {
            const DTYPE: &'static str = $dtype;

            fn read_from(reader: &mut impl Read) -> Result<Self> {
                reader.$read::<LittleEndian>().map_err(Error::from)
            }

            fn write_to(self, writer: &mut impl Write) -> Result<()> {
                writer.$write::<LittleEndian>(self).map_err(Error::from)
            }
        }
    };
}

impl_element!(f32, "<f4", read_f32, write_f32);
impl_element!(f64, "<f8", read_f64, write_f64);
impl_element!(i32, "<i4", read_i32, write_i32);
impl_element!(i64, "<i8", read_i64, write_i64);

/// Owned typed NPY array.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray<T> {
    shape: Box<[usize]>,
    fortran_order: bool,
    values: Box<[T]>,
}

impl<T> NpyArray<T> {
    /// Constructs an array after validating that shape and payload agree.
    pub fn new(shape: impl Into<Box<[usize]>>, values: impl Into<Box<[T]>>) -> Result<Self> {
        let shape = shape.into();
        let values = values.into();
        let expected = shape.iter().try_fold(1usize, |count, &axis| {
            count.checked_mul(axis).ok_or_else(|| {
                Error::InvalidFormat(format!("shape element count overflows usize: {shape:?}"))
            })
        })?;
        if expected != values.len() {
            return Err(Error::InvalidFormat(format!(
                "shape {shape:?} requires {expected} elements, received {}",
                values.len()
            )));
        }
        Ok(Self {
            shape,
            fortran_order: false,
            values,
        })
    }

    pub(crate) fn from_parts(shape: Box<[usize]>, fortran_order: bool, values: Box<[T]>) -> Self {
        Self {
            shape,
            fortran_order,
            values,
        }
    }

    /// Returns the array shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reports whether payload order is Fortran-contiguous.
    #[must_use]
    pub fn is_fortran_order(&self) -> bool {
        self.fortran_order
    }

    /// Borrows the contiguous stored payload.
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Consumes the array into its payload.
    #[must_use]
    pub fn into_values(self) -> Box<[T]> {
        self.values
    }
}
