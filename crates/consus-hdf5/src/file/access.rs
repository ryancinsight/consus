//! Object navigation and top-level accessors for `Hdf5File` (open, root group, contexts).

use super::{Hdf5File, reader};
use crate::address::ParseContext;
use crate::group::Hdf5Group;
use crate::object_header::ObjectHeader;
use crate::superblock::Superblock;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use consus_core::{NodeType, Result};
use consus_io::ReadAt;

impl<R: ReadAt + Sync> Hdf5File<R> {
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
        self.list_group_children_from_header(&header, true)
    }

    /// Shared v2/v1 group-listing fallback used by both root and by-address
    /// group listing.
    ///
    /// Tries v2 compact/dense link messages first. Falls back to the v1
    /// symbol-table path only when `allow_v1_without_symbol_table` is set or
    /// the object header carries a `SYMBOL_TABLE` message; v2 groups with no
    /// children have no such message and correctly return an empty list
    /// rather than an error.
    #[cfg(feature = "alloc")]
    pub(super) fn list_group_children_from_header(
        &self,
        header: &ObjectHeader,
        allow_v1_without_symbol_table: bool,
    ) -> Result<Vec<(String, u64, consus_core::LinkType)>> {
        let v2 = reader::list_group_v2(&self.source, header, &self.ctx)?;
        if !v2.is_empty() {
            return Ok(v2
                .into_iter()
                .map(|(name, addr, link_type, _)| (name, addr, link_type))
                .collect());
        }

        let has_symbol_table =
            reader::find_message(header, crate::object_header::message_types::SYMBOL_TABLE)
                .is_some();
        if !allow_v1_without_symbol_table && !has_symbol_table {
            return Ok(Vec::new());
        }

        let v1 = reader::list_group_v1(&self.source, header, &self.ctx)?;
        Ok(v1
            .into_iter()
            .map(|(name, addr)| (name, addr, consus_core::LinkType::Hard))
            .collect())
    }
}
