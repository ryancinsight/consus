//! `std::fs::File` implementation of positioned I/O traits.
//!
//! ## Design
//!
//! Implements `ReadAt`, `WriteAt`, `Length`, and `Truncate` for
//! `std::fs::File`. The `ReadAt` implementation uses `try_clone()` +
//! `seek()` as a portable approach. Platform-specific `pread`/`pwrite`
//! optimization is a future enhancement.
//!
//! ## Thread Safety
//!
//! `ReadAt::read_at` clones the file descriptor so concurrent reads
//! from multiple threads do not conflict. `WriteAt` requires `&mut self`
//! and is therefore single-threaded by Rust's borrow rules.

use std::fs::File;
use std::io::{Read, Seek, Write};

use consus_core::{Error, Result};

use crate::io::traits::{Length, ReadAt, Truncate, WriteAt};

impl ReadAt for File {
    /// Read bytes at the given offset using a cloned file descriptor.
    ///
    /// The clone ensures concurrent reads do not interfere with each
    /// other's seek position. Each call performs:
    /// 1. `try_clone()` — duplicate the OS file handle.
    /// 2. `seek(Start(pos))` — position the clone's cursor.
    /// 3. `read_exact(buf)` — read exactly `buf.len()` bytes.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let mut file = self.try_clone().map_err(Error::Io)?;
        file.seek(std::io::SeekFrom::Start(pos))
            .map_err(Error::Io)?;
        file.read_exact(buf).map_err(Error::Io)?;
        Ok(())
    }
}

impl WriteAt for File {
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        self.seek(std::io::SeekFrom::Start(pos))
            .map_err(Error::Io)?;
        self.write_all(buf).map_err(Error::Io)?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Write::flush(self).map_err(Error::Io)?;
        Ok(())
    }
}

impl Length for File {
    fn len(&self) -> Result<u64> {
        let meta = self.metadata().map_err(Error::Io)?;
        Ok(meta.len())
    }
}

impl Truncate for File {
    fn set_len(&mut self, size: u64) -> Result<()> {
        File::set_len(self, size).map_err(Error::Io)?;
        Ok(())
    }
}

// File now satisfies RandomAccess via blanket impl.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::traits::RandomAccess;

    /// `File` satisfies `RandomAccess`.
    #[test]
    fn file_is_random_access() {
        fn _assert(_: &dyn RandomAccess) {}
    }

    /// Write-then-read round-trip via temp file.
    #[test]
    fn file_write_read_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("consus_io_test_roundtrip.bin");
        // Ensure cleanup
        let _cleanup = scopeguard(&path);

        let mut file = File::create(&path).expect("create must succeed");
        let payload = b"consus file I/O test payload";
        file.write_at(0, payload).expect("write must succeed");
        WriteAt::flush(&mut file).expect("flush must succeed");

        let file_r = File::open(&path).expect("open must succeed");
        let mut buf = vec![0u8; payload.len()];
        file_r.read_at(0, &mut buf).expect("read must succeed");
        assert_eq!(&buf, payload);
    }

    /// `Length::len` returns file size.
    #[test]
    fn file_length() {
        let dir = std::env::temp_dir();
        let path = dir.join("consus_io_test_length.bin");
        let _cleanup = scopeguard(&path);

        let mut file = File::create(&path).expect("create must succeed");
        file.write_at(0, &[0u8; 128]).expect("write must succeed");
        WriteAt::flush(&mut file).expect("flush must succeed");

        assert_eq!(Length::len(&file).unwrap(), 128);
    }

    /// `Truncate::set_len` truncates file.
    #[test]
    fn file_truncate() {
        let dir = std::env::temp_dir();
        let path = dir.join("consus_io_test_truncate.bin");
        let _cleanup = scopeguard(&path);

        let mut file = File::create(&path).expect("create must succeed");
        file.write_at(0, &[0xAA; 100]).expect("write must succeed");
        WriteAt::flush(&mut file).expect("flush must succeed");

        file.set_len(50).expect("set_len must succeed");
        assert_eq!(Length::len(&file).unwrap(), 50);
    }

    /// Helper: remove a temp file on drop.
    fn scopeguard(path: &std::path::Path) -> impl Drop + '_ {
        struct Guard<'a>(&'a std::path::Path);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(self.0);
            }
        }
        Guard(path)
    }
}
