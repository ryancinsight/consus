//! HDF5 file-level API.
//!
//! ## Design
//!
//! `Hdf5File` is the entry point for reading and writing HDF5 files.
//! It owns a reference to the I/O source and the parsed superblock,
//! providing navigation through the object hierarchy.
//!
//! ### Lifecycle
//!
//! 1. Open: locate superblock, parse root group
//! 2. Navigate: traverse groups via B-tree/heap
//! 3. Read: resolve dataset metadata, read raw data through selection
//! 4. Close: flush and release resources

use consus_core::error::Result;
use consus_io::source::ReadAt;

use crate::superblock::Superblock;

/// An open HDF5 file for reading.
///
/// Parameterized over the I/O source to support both file and in-memory backends.
pub struct Hdf5File<R: ReadAt> {
    /// Underlying I/O source.
    source: R,
    /// Parsed superblock.
    superblock: Superblock,
}

impl<R: ReadAt> Hdf5File<R> {
    /// Open an HDF5 file from a positioned I/O source.
    ///
    /// Locates and parses the superblock. Returns an error if the source
    /// does not contain a valid HDF5 file.
    pub fn open(source: R) -> Result<Self> {
        let superblock = Superblock::read_from(&source)?;
        Ok(Self { source, superblock })
    }

    /// Access the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Access the underlying I/O source.
    pub fn source(&self) -> &R {
        &self.source
    }

    /// Read raw bytes from a specific file offset.
    ///
    /// Low-level utility for format parsing code.
    pub fn read_bytes(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.source.read_at(offset, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HDF5_MAGIC;
    use byteorder::{ByteOrder, LittleEndian};
    use consus_io::cursor::MemCursor;

    /// Build a minimal v2 HDF5 file image and open it.
    fn make_minimal_hdf5() -> Vec<u8> {
        let mut data = vec![0u8; 4096];
        // Superblock at offset 0
        data[0..8].copy_from_slice(&HDF5_MAGIC);
        data[8] = 2; // version
        data[9] = 8; // offset size
        data[10] = 8; // length size
        data[11] = 0; // consistency flags
        LittleEndian::write_u64(&mut data[12..20], 0); // base address
        LittleEndian::write_u64(&mut data[20..28], u64::MAX); // extension
        LittleEndian::write_u64(&mut data[28..36], 4096); // EOF
        LittleEndian::write_u64(&mut data[36..44], 96); // root group OH
        data
    }

    #[test]
    fn open_minimal_file() {
        let data = make_minimal_hdf5();
        let cursor = MemCursor::from_bytes(data);
        let file = Hdf5File::open(cursor).expect("must open");
        assert_eq!(file.superblock().version, 2);
        assert_eq!(file.superblock().root_group_address, 96);
    }

    #[test]
    fn reject_non_hdf5() {
        let cursor = MemCursor::from_bytes(vec![0u8; 4096]);
        assert!(Hdf5File::open(cursor).is_err());
    }
}
