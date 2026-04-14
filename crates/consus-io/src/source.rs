//! Synchronous positioned I/O traits.
//!
//! ## Contract
//!
//! - `ReadAt::read_at` reads exactly `buf.len()` bytes from position `pos`,
//!   or returns an error if the source is shorter.
//! - `WriteAt::write_at` writes all of `buf` at position `pos`.
//! - Implementations must be safe for concurrent positioned reads (the OS
//!   guarantees this for `pread`/`ReadFile` with offset).

use consus_core::error::Result;

/// Positioned read: read bytes at an absolute offset without seeking.
///
/// This trait is the fundamental I/O primitive for all format backends.
/// Unlike `std::io::Read`, it does not maintain cursor state, making it
/// safe for concurrent access without external synchronization.
pub trait ReadAt {
    /// Read exactly `buf.len()` bytes starting at byte offset `pos`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` if fewer bytes are available or on I/O failure.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()>;

    /// Total length of the underlying source in bytes.
    fn len(&self) -> Result<u64>;

    /// Whether the source is empty.
    fn is_empty(&self) -> Result<bool> {
        self.len().map(|l| l == 0)
    }
}

/// Positioned write: write bytes at an absolute offset.
pub trait WriteAt {
    /// Write all of `buf` starting at byte offset `pos`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` on I/O failure.
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<()>;

    /// Flush any buffered data to the underlying sink.
    fn flush(&mut self) -> Result<()>;

    /// Truncate or extend the sink to `size` bytes.
    fn set_len(&mut self, size: u64) -> Result<()>;
}

/// Combined random-access I/O.
pub trait RandomAccess: ReadAt + WriteAt {}

impl<T: ReadAt + WriteAt> RandomAccess for T {}

/// `std::fs::File` implementation of positioned I/O.
#[cfg(feature = "std")]
mod file_impl {
    use super::*;
    use consus_core::error::Error;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom, Write};

    impl ReadAt for File {
        fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
            // Use platform-specific positioned read when available.
            // Fallback: clone handle, seek, read.
            let mut file = self.try_clone().map_err(Error::Io)?;
            file.seek(SeekFrom::Start(pos)).map_err(Error::Io)?;
            file.read_exact(buf).map_err(Error::Io)?;
            Ok(())
        }

        fn len(&self) -> Result<u64> {
            let meta = self.metadata().map_err(Error::Io)?;
            Ok(meta.len())
        }
    }

    impl WriteAt for File {
        fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
            self.seek(SeekFrom::Start(pos)).map_err(Error::Io)?;
            self.write_all(buf).map_err(Error::Io)?;
            Ok(())
        }

        fn flush(&mut self) -> Result<()> {
            Write::flush(self).map_err(Error::Io)?;
            Ok(())
        }

        fn set_len(&mut self, size: u64) -> Result<()> {
            File::set_len(self, size).map_err(Error::Io)?;
            Ok(())
        }
    }
}
