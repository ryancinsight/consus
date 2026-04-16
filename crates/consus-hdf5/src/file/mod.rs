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

pub mod reader;
#[cfg(feature = "alloc")]
pub mod writer;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::{NodeType, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;
use crate::attribute::Hdf5Attribute;
use crate::dataset::Hdf5Dataset;
use crate::group::Hdf5Group;
use crate::object_header::ObjectHeader;
use crate::superblock::Superblock;

#[cfg(feature = "alloc")]
use consus_core::Error;

/// An open HDF5 file for reading.
///
/// Parameterized over the I/O source to support both file and in-memory backends.
pub struct Hdf5File<R: ReadAt> {
    /// Underlying I/O source.
    source: R,
    /// Parsed superblock.
    superblock: Superblock,
    /// Parsing context derived from the superblock.
    ctx: ParseContext,
}

impl<R: ReadAt> Hdf5File<R> {
    /// Open an HDF5 file from a positioned I/O source.
    ///
    /// Locates and parses the superblock. Returns an error if the source
    /// does not contain a valid HDF5 file.
    pub fn open(source: R) -> Result<Self> {
        let superblock = Superblock::read_from(&source)?;
        let ctx = ParseContext::new(superblock.offset_size, superblock.length_size);
        Ok(Self {
            source,
            superblock,
            ctx,
        })
    }

    /// Access the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Access the underlying I/O source.
    pub fn source(&self) -> &R {
        &self.source
    }

    /// Access the parsing context derived from the superblock.
    pub fn context(&self) -> &ParseContext {
        &self.ctx
    }

    /// Read and parse the root object header.
    #[cfg(feature = "alloc")]
    pub fn root_object_header(&self) -> Result<ObjectHeader> {
        reader::read_object_header(&self.source, self.superblock.root_group_address, &self.ctx)
    }

    /// Classify the root object.
    #[cfg(feature = "alloc")]
    pub fn root_node_type(&self) -> Result<NodeType> {
        let header = self.root_object_header()?;
        Ok(reader::classify_object(&header))
    }

    /// Return a handle for the root group.
    #[cfg(feature = "alloc")]
    pub fn root_group(&self) -> Hdf5Group {
        Hdf5Group {
            path: String::from("/"),
            object_header_address: self.superblock.root_group_address,
        }
    }

    /// List direct children of the root group.
    #[cfg(feature = "alloc")]
    pub fn list_root_group(&self) -> Result<Vec<(String, u64, consus_core::LinkType)>> {
        let header = self.root_object_header()?;
        let children = reader::list_group_v2(&self.source, &header, &self.ctx)?;
        if !children.is_empty() {
            return Ok(children);
        }

        let v1_children = reader::list_group_v1(&self.source, &header, &self.ctx)?;
        Ok(v1_children
            .into_iter()
            .map(|(name, address)| (name, address, consus_core::LinkType::Hard))
            .collect())
    }

    /// Read dataset metadata from an object header address.
    #[cfg(feature = "alloc")]
    pub fn dataset_at(&self, object_header_address: u64) -> Result<Hdf5Dataset> {
        let header = reader::read_object_header(&self.source, object_header_address, &self.ctx)?;
        let mut dataset = reader::read_dataset_metadata(&header, &self.ctx)?;
        dataset.object_header_address = object_header_address;
        Ok(dataset)
    }

    /// Read attributes attached to an object header address.
    #[cfg(feature = "alloc")]
    pub fn attributes_at(&self, object_header_address: u64) -> Result<Vec<Hdf5Attribute>> {
        let header = reader::read_object_header(&self.source, object_header_address, &self.ctx)?;
        reader::read_attributes(&header, &self.ctx)
    }

    /// Read raw bytes from a contiguous dataset region.
    #[cfg(feature = "alloc")]
    pub fn read_contiguous_dataset_bytes(
        &self,
        data_address: u64,
        byte_offset: u64,
        buf: &mut [u8],
    ) -> Result<()> {
        reader::read_contiguous_raw(&self.source, data_address, byte_offset, buf)
    }

    /// Read raw bytes from a specific file offset.
    ///
    /// Low-level utility for format parsing code.
    pub fn read_bytes(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.source.read_at(offset, buf)
    }

    /// Navigate to an object by slash-separated path, returning its
    /// object header address.
    ///
    /// Leading `/` is accepted and ignored. Empty components (double
    /// slashes) are skipped. Returns `Error::NotFound` if any component
    /// is absent.
    ///
    /// ## Algorithm
    ///
    /// Starting from `root_group_address`, each path component is
    /// matched against the hard links reported by `list_group_at`.
    /// The first matching hard link address becomes the current object.
    ///
    /// ## Errors
    ///
    /// - [`Error::NotFound`] if any path component is missing.
    /// - [`Error::InvalidFormat`] if an object header is malformed.
    #[cfg(feature = "alloc")]
    pub fn open_path(&self, path: &str) -> Result<u64> {
        let mut current = self.superblock.root_group_address;
        for component in path.split('/').filter(|s| !s.is_empty()) {
            let header = reader::read_object_header(&self.source, current, &self.ctx)?;
            let mut found: Option<u64> = None;

            // Try v2 link messages first (dense or compact).
            let v2 = reader::list_group_v2(&self.source, &header, &self.ctx)?;
            for (name, addr, _) in &v2 {
                if name == component {
                    found = Some(*addr);
                    break;
                }
            }

            // Fall back to v1 symbol table. v2 groups have no SYMBOL_TABLE
            // message, so list_group_v1 returns InvalidFormat in that case —
            // treat failure as an empty list rather than propagating the error.
            if found.is_none() {
                if let Ok(v1) = reader::list_group_v1(&self.source, &header, &self.ctx) {
                    for (name, addr) in &v1 {
                        if name.as_str() == component {
                            found = Some(*addr);
                            break;
                        }
                    }
                }
            }

            current = found.ok_or_else(|| Error::NotFound {
                path: alloc::string::String::from(component),
            })?;
        }
        Ok(current)
    }

    /// List children of a group at the given object header address.
    ///
    /// Returns `(name, object_header_address, link_type)` triples.
    /// Tries v2 links first; falls back to v1 symbol table.
    #[cfg(feature = "alloc")]
    pub fn list_group_at(&self, address: u64) -> Result<Vec<(String, u64, consus_core::LinkType)>> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;

        let v2 = reader::list_group_v2(&self.source, &header, &self.ctx)?;
        if !v2.is_empty() {
            return Ok(v2);
        }

        let v1 = reader::list_group_v1(&self.source, &header, &self.ctx)?;
        Ok(v1
            .into_iter()
            .map(|(name, addr)| (name, addr, consus_core::LinkType::Hard))
            .collect())
    }

    /// Classify the object at the given object header address.
    #[cfg(feature = "alloc")]
    pub fn node_type_at(&self, address: u64) -> Result<NodeType> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;
        Ok(reader::classify_object(&header))
    }

    /// Read the fill value from the object header at `address`, if present.
    #[cfg(feature = "alloc")]
    pub fn fill_value_at(&self, address: u64) -> Result<Option<Vec<u8>>> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;
        Ok(reader::read_fill_value(&header))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::HDF5_MAGIC;
    use byteorder::{ByteOrder, LittleEndian};
    use consus_io::MemCursor;

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
        assert_eq!(file.context().offset_size, 8);
        assert_eq!(file.context().length_size, 8);
    }

    #[test]
    fn reject_non_hdf5() {
        let cursor = MemCursor::from_bytes(vec![0u8; 4096]);
        assert!(Hdf5File::open(cursor).is_err());
    }
}
