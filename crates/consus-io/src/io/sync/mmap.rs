//! Read-only memory-mapped file reader.
//!
//! ## Architecture
//!
//! [`MmapReader`] maps a file read-only into virtual address space using
//! `memmap2::Mmap`. All reads are page-fault driven; no explicit `read()`
//! syscalls occur after construction.
//!
//! ## Safety
//!
//! Construction uses an `unsafe` block because `memmap2::Mmap::map` is
//! unsafe: if another process truncates the file while the mapping is live
//! the read will return stale or zero bytes on Unix, or raise an access
//! violation on Windows. Callers must ensure the backing file is not
//! truncated for the lifetime of the `MmapReader`.
//!
//! `memmap2::Mmap` is `Send + Sync`, so `MmapReader` is `Send + Sync`.
//!
//! ## Platform Notes
//!
//! On Windows the mapping holds the file open. On Unix, unlinking the file
//! after mapping is safe; the mapping persists until dropped.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use consus_core::{Error, Result};

use crate::io::traits::{Length, ReadAt};

/// Read-only memory-mapped file reader.
///
/// Wraps a `memmap2::Mmap` and exposes [`ReadAt`] and [`Length`] over the
/// mapped byte slice.
///
/// # Safety
///
/// The backing file must not be truncated while this reader is alive.
/// See module documentation for details.
pub struct MmapReader {
    mmap: Mmap,
}

impl MmapReader {
    /// Open the file at `path` and map it read-only.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` if the file cannot be opened or mapped.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(Error::Io)?;
        Self::from_file(&file)
    }

    /// Map an already-open `File` read-only.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` if the OS mapping call fails.
    pub fn from_file(file: &File) -> Result<Self> {
        // SAFETY: The file is readable. The caller must ensure it is not
        // truncated for the lifetime of this reader.
        let mmap = unsafe { Mmap::map(file).map_err(Error::Io)? };
        Ok(Self { mmap })
    }

    /// Return the memory-mapped byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
}

impl ReadAt for MmapReader {
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let pos_usize = pos as usize;
        let end = pos_usize.checked_add(buf.len()).ok_or(Error::Overflow)?;
        if end > self.mmap.len() {
            return Err(Error::BufferTooSmall {
                required: end,
                provided: self.mmap.len(),
            });
        }
        buf.copy_from_slice(&self.mmap[pos_usize..end]);
        Ok(())
    }
}

impl Length for MmapReader {
    fn len(&self) -> Result<u64> {
        Ok(self.mmap.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::traits::ReadAt;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp(payload: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create unique temporary file");
        file.write_all(payload).expect("write temporary file");
        file.flush().expect("flush temporary file");
        file
    }

    #[test]
    fn mmap_reader_open_and_len() {
        let file = write_temp(b"hello world");
        let reader = MmapReader::open(file.path()).expect("open must succeed");
        assert_eq!(reader.len().unwrap(), 11);
    }

    #[test]
    fn mmap_reader_read_at_beginning() {
        let file = write_temp(b"ABCDE");
        let reader = MmapReader::open(file.path()).expect("open must succeed");
        let mut buf = [0u8; 5];
        reader.read_at(0, &mut buf).expect("read_at must succeed");
        assert_eq!(&buf, b"ABCDE");
    }

    #[test]
    fn mmap_reader_read_at_offset() {
        let file = write_temp(b"hello world");
        let reader = MmapReader::open(file.path()).expect("open must succeed");
        let mut buf = [0u8; 5];
        reader.read_at(6, &mut buf).expect("read_at must succeed");
        assert_eq!(&buf, b"world");
    }

    #[test]
    fn mmap_reader_zero_len_read_succeeds() {
        let file = write_temp(b"data");
        let reader = MmapReader::open(file.path()).expect("open must succeed");
        reader
            .read_at(100, &mut [])
            .expect("zero-length read must succeed");
    }

    #[test]
    fn mmap_reader_out_of_bounds_returns_buffer_too_small() {
        let file = write_temp(b"short");
        let reader = MmapReader::open(file.path()).expect("open must succeed");
        let mut buf = [0u8; 10];
        let err = reader.read_at(0, &mut buf).unwrap_err();
        assert!(
            matches!(err, consus_core::Error::BufferTooSmall { .. }),
            "expected BufferTooSmall, got: {err:?}"
        );
    }

    #[test]
    fn mmap_reader_from_file() {
        let file = write_temp(b"from_file_test");
        let reader = MmapReader::from_file(file.as_file()).expect("from_file must succeed");
        let mut buf = [0u8; 4];
        reader.read_at(0, &mut buf).expect("read_at must succeed");
        assert_eq!(&buf, b"from");
        assert_eq!(reader.as_slice(), b"from_file_test");
    }

    #[test]
    fn mmap_reader_as_slice_matches_read_at() {
        let payload = b"slice_vs_read_at";
        let file = write_temp(payload);
        let reader = MmapReader::open(file.path()).expect("open must succeed");
        let slice = reader.as_slice();
        let mut buf = vec![0u8; payload.len()];
        reader.read_at(0, &mut buf).expect("read_at must succeed");
        assert_eq!(slice, buf.as_slice());
        assert_eq!(slice, payload);
    }

    #[test]
    fn mmap_reader_is_send_sync() {
        fn _assert_send<T: Send>() {}
        fn _assert_sync<T: Sync>() {}
        _assert_send::<MmapReader>();
        _assert_sync::<MmapReader>();
    }
}
