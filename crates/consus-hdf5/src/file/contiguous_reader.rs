//! Streaming [`std::io::Read`] over a contiguous dataset payload.
//!
//! A contiguous HDF5 dataset stores its raw elements as one byte run at a
//! single file address. [`ContiguousDatasetReader`] exposes that run as a
//! forward-only byte stream, so consumers that parse sequentially (image
//! decoders, checksummers, `io::copy`) read the payload in caller-sized steps
//! without allocating the whole dataset.
//!
//! ## Contract
//!
//! - A reader built over `[data_address, data_address + len)` never reads a
//!   byte outside that range, whatever the buffer length.
//! - Each [`Read::read`] fills at most `min(buf.len(), remaining)` bytes and
//!   returns `Ok(0)` once all `len` bytes are consumed.
//! - A failed read consumes nothing: the position is unchanged, so the same
//!   call may be retried.
//! - A [`consus_core::Error`] becomes an [`io::Error`] that carries the
//!   original error as its inner error ([`io::Error::get_ref`],
//!   [`io::Error::into_inner`]), with an [`io::ErrorKind`] chosen from the
//!   variant. Note that [`io::Error::source`](std::error::Error::source)
//!   skips the inner error by std's definition and yields the inner error's
//!   own source.

use std::io::{self, Read};

use consus_core::Error;
use consus_io::ReadAt;

use super::Hdf5File;

impl<R: ReadAt + Sync> Hdf5File<R> {
    /// Stream `len` bytes of a contiguous dataset payload starting at
    /// `data_address`.
    ///
    /// `data_address` is the value of
    /// [`Hdf5Dataset::data_address`](crate::dataset::Hdf5Dataset::data_address)
    /// for a dataset whose layout is
    /// [`StorageLayout::Contiguous`](crate::dataset::StorageLayout::Contiguous),
    /// and `len` is the payload size in bytes. Nothing is read until the
    /// returned reader is polled, and `len` is never used to size an
    /// allocation, so an untrusted `len` costs nothing until bytes are
    /// confirmed to exist.
    ///
    /// # Errors
    ///
    /// Construction cannot fail. [`Read::read`] on the returned reader fails
    /// with an [`io::Error`] wrapping the [`consus_core::Error`] raised by the
    /// source:
    ///
    /// - [`io::ErrorKind::UnexpectedEof`] when the range extends past the end
    ///   of the source ([`Error::BufferTooSmall`]);
    /// - [`io::ErrorKind::InvalidData`] when `data_address + len` overflows
    ///   `u64` ([`Error::Overflow`]);
    /// - the kind of the underlying [`io::Error`] for [`Error::Io`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Read;
    ///
    /// use consus_core::{ByteOrder, Datatype, Shape};
    /// use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps};
    /// use consus_hdf5::{Hdf5File, Hdf5FileBuilder};
    /// use consus_io::MemCursor;
    /// use core::num::NonZeroUsize;
    ///
    /// let datatype = Datatype::Integer {
    ///     bits: NonZeroUsize::new(8).unwrap(),
    ///     byte_order: ByteOrder::LittleEndian,
    ///     signed: false,
    /// };
    /// let payload = [10u8, 20, 30, 40, 50];
    ///
    /// let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    /// builder
    ///     .add_dataset(
    ///         "values",
    ///         &datatype,
    ///         &Shape::fixed(&[payload.len()]),
    ///         &payload,
    ///         &DatasetCreationProps::default(),
    ///     )
    ///     .unwrap();
    /// let file = Hdf5File::open(MemCursor::from_bytes(builder.finish().unwrap())).unwrap();
    ///
    /// let header = file.open_path("values").unwrap();
    /// let data_address = file.dataset_at(header).unwrap().data_address.unwrap();
    ///
    /// let mut streamed = Vec::new();
    /// file.contiguous_dataset_reader(data_address, payload.len() as u64)
    ///     .read_to_end(&mut streamed)
    ///     .unwrap();
    /// assert_eq!(streamed, payload);
    /// ```
    #[must_use]
    pub fn contiguous_dataset_reader(
        &self,
        data_address: u64,
        len: u64,
    ) -> ContiguousDatasetReader<'_, R> {
        ContiguousDatasetReader {
            file: self,
            data_address,
            position: 0,
            len,
        }
    }
}

/// Forward-only byte stream over one contiguous dataset payload.
///
/// Created by [`Hdf5File::contiguous_dataset_reader`]; see the module
/// documentation for the read contract.
pub struct ContiguousDatasetReader<'file, R: ReadAt> {
    file: &'file Hdf5File<R>,
    data_address: u64,
    /// Bytes already delivered; invariant: `position <= len`.
    position: u64,
    len: u64,
}

impl<R: ReadAt> ContiguousDatasetReader<'_, R> {
    /// Payload bytes not yet delivered.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.len - self.position
    }
}

impl<R: ReadAt> core::fmt::Debug for ContiguousDatasetReader<'_, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ContiguousDatasetReader")
            .field("data_address", &self.data_address)
            .field("position", &self.position)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<R: ReadAt + Sync> Read for ContiguousDatasetReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let step = usize::try_from(self.remaining()).map_or(buf.len(), |left| left.min(buf.len()));
        if step == 0 {
            return Ok(0);
        }
        // `read_contiguous_dataset_bytes` adds its operands unchecked, so the
        // far end of this step is validated here.
        let delivered = u64::try_from(step).map_err(|_| io_error(Error::Overflow))?;
        self.position
            .checked_add(delivered)
            .and_then(|end| self.data_address.checked_add(end))
            .ok_or(Error::Overflow)
            .map_err(io_error)?;
        self.file
            .read_contiguous_dataset_bytes(self.data_address, self.position, &mut buf[..step])
            .map_err(io_error)?;
        // `step <= remaining`, so the sum is at most `len`.
        self.position += delivered;
        Ok(step)
    }
}

/// Convert a consus error to an [`io::Error`] that keeps it as the inner error.
fn io_error(error: Error) -> io::Error {
    let kind = match &error {
        Error::Io(source) => source.kind(),
        Error::BufferTooSmall { .. } => io::ErrorKind::UnexpectedEof,
        Error::NotFound { .. } | Error::LinkResolutionFailed { .. } => io::ErrorKind::NotFound,
        Error::ReadOnly => io::ErrorKind::PermissionDenied,
        Error::UnsupportedFeature { .. } => io::ErrorKind::Unsupported,
        Error::InvalidFormat { .. }
        | Error::DatatypeMismatch { .. }
        | Error::ShapeError { .. }
        | Error::SelectionOutOfBounds
        | Error::CompressionError { .. }
        | Error::Corrupted { .. }
        | Error::Overflow
        | Error::ResourceLimit { .. } => io::ErrorKind::InvalidData,
        Error::InvalidHandle | Error::CapacityExceeded { .. } | Error::InternalError { .. } => {
            io::ErrorKind::Other
        }
    };
    io::Error::new(kind, error)
}

#[cfg(test)]
mod tests;
