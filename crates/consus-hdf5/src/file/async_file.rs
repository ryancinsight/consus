//! Async HDF5 file reader.
//!
//! Provides [`AsyncHdf5File`], which mirrors the [`super::Hdf5File`] public API
//! over an [`AsyncReadAt`] + [`AsyncLength`] source.
//!
//! ## Design
//!
//! Async pre-fetching: each method issues one or more `await`-ed positioned
//! reads to collect the bytes needed by a structure, then delegates to the
//! existing sync parsers operating on a [`MultiRegionBuffer`]. Format logic
//! is not duplicated; only the I/O coordination layer is async.
//!
//! ## Invariant
//!
//! Every structure loaded by an async method fits within a finite number of
//! `AsyncReadAt::read_at` calls bounded by the continuation chain depth
//! limit (256 hops).

#![cfg(all(feature = "async-io", feature = "alloc"))]

use alloc::vec::Vec;

use consus_core::Result;
use consus_io::{AsyncLength, AsyncReadAt};

use crate::address::ParseContext;
use crate::superblock::Superblock;

use super::async_reader;
use super::reader;

// ---------------------------------------------------------------------------
// AsyncHdf5File
// ---------------------------------------------------------------------------

/// An open HDF5 file for async reading.
///
/// Parameterized over the I/O source to support both file and object-store
/// backends. All read operations are async; format parsing is delegated to
/// the sync parsers via [`MultiRegionBuffer`].
///
/// [`MultiRegionBuffer`]: super::async_reader::MultiRegionBuffer
#[cfg(feature = "async-io")]
#[cfg(feature = "alloc")]
pub struct AsyncHdf5File<R>
where
    R: consus_io::AsyncReadAt + consus_io::AsyncLength,
{
    /// Underlying I/O source.
    source: R,
    /// Parsed superblock.
    superblock: Superblock,
    /// Parsing context derived from the superblock.
    ctx: ParseContext,
}

#[cfg(feature = "async-io")]
#[cfg(feature = "alloc")]
impl<R> AsyncHdf5File<R>
where
    R: AsyncReadAt + AsyncLength,
{
    /// Open an HDF5 file from an async positioned I/O source.
    ///
    /// Reads and parses the superblock. Returns an error if the source
    /// does not contain a valid HDF5 file.
    pub async fn open(source: R) -> Result<Self> {
        let superblock = async_reader::async_read_superblock(&source).await?;
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
    pub async fn root_object_header(&self) -> Result<crate::object_header::ObjectHeader> {
        async_reader::async_read_object_header(
            &self.source,
            self.superblock.root_group_address,
            &self.ctx,
        )
        .await
    }

    /// Classify the root object as Group, Dataset, or NamedDatatype.
    pub async fn root_node_type(&self) -> Result<consus_core::NodeType> {
        let header = self.root_object_header().await?;
        Ok(reader::classify_object(&header))
    }

    /// Read raw bytes from the source at the given offset.
    pub async fn read_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        async_reader::read_region(&self.source, offset, len).await
    }

    /// Classify the object at `address` as Group, Dataset, or NamedDatatype.
    pub async fn node_type_at(&self, address: u64) -> Result<consus_core::NodeType> {
        let header =
            async_reader::async_read_object_header(&self.source, address, &self.ctx).await?;
        Ok(reader::classify_object(&header))
    }

    /// Read and resolve dataset metadata from the object header at `object_header_address`.
    pub async fn dataset_at(
        &self,
        object_header_address: u64,
    ) -> Result<crate::dataset::Hdf5Dataset> {
        let header = async_reader::async_read_object_header(
            &self.source,
            object_header_address,
            &self.ctx,
        )
        .await?;
        let mut dataset = reader::read_dataset_metadata(&header, &self.ctx)?;
        dataset.object_header_address = object_header_address;
        Ok(dataset)
    }
}
