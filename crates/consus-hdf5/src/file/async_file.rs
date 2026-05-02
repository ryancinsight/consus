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
use crate::btree::v1::{BTreeV1Header, BTreeV1Type};
use crate::btree::v2::{BTreeV2Header, async_collect_all_records};
use crate::btree::btree_v2_record_type;
use crate::file::ChunkIndexEntry;
use consus_core::Error;

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

    /// Read all bytes from a chunked dataset.
    pub async fn read_chunked_dataset_all_bytes(&self, object_header_address: u64) -> Result<Vec<u8>> {
        let header = async_reader::async_read_object_header(&self.source, object_header_address, &self.ctx).await?;
        let dataset = reader::read_dataset_metadata(&header, &self.ctx)?;

        if dataset.layout != crate::dataset::StorageLayout::Chunked {
            return Err(Error::InvalidFormat {
                message: alloc::string::String::from("dataset is not chunked"),
            });
        }

        let layout_msg = reader::find_message(&header, crate::object_header::message_types::DATA_LAYOUT)
            .ok_or_else(|| Error::InvalidFormat {
                message: alloc::string::String::from("dataset object header missing layout message"),
            })?;
        let layout = crate::dataset::layout::DataLayout::parse(&layout_msg.data, &self.ctx)?;

        let chunk_dims_u32 = layout.chunk_dims.ok_or_else(|| Error::InvalidFormat {
            message: alloc::string::String::from("chunked dataset missing chunk dimensions"),
        })?;
        let chunk_dims: Vec<usize> = chunk_dims_u32.iter().map(|&d| d as usize).collect();

        let element_size = dataset.datatype.element_size().ok_or_else(|| Error::UnsupportedFeature {
            feature: alloc::string::String::from("chunked full read requires fixed-size element datatype"),
        })?;
        let dataset_dims = dataset.shape.current_dims();
        let total_bytes = dataset
            .shape
            .num_elements()
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        let mut out = vec![0u8; total_bytes];

        let fill_value = reader::read_fill_value(&header);
        let filter_ids = dataset.filters;
        let registry = consus_compression::DefaultCodecRegistry::new();

        match (
            layout.version,
            layout.chunk_btree_address,
            layout.chunk_index_type,
            layout.chunk_index_address,
        ) {
            (3, Some(chunk_btree_address), _, _) => {
                if dataset_dims.is_empty() {
                    let entries = self.async_read_v1_chunk_btree_leaf_entries(chunk_btree_address, 0).await?;
                    let entry = entries.first().ok_or_else(|| Error::InvalidFormat {
                        message: alloc::string::String::from("scalar chunked dataset has no chunk entries"),
                    })?;
                    let chunk = crate::dataset::chunk::async_read_chunk_raw(
                        &self.source,
                        &crate::dataset::chunk::ChunkLocation {
                            address: entry.chunk_address,
                            size: entry.chunk_size as u64,
                            filter_mask: entry.filter_mask,
                        },
                        element_size,
                        &filter_ids,
                        &registry,
                        fill_value.as_deref(),
                    ).await?;
                    out.copy_from_slice(&chunk[..element_size]);
                    return Ok(out);
                }

                let entries = self.async_read_v1_chunk_btree_leaf_entries(chunk_btree_address, chunk_dims.len()).await?;
                for entry in entries {
                    let chunk_uncompressed_size = chunk_dims.iter().product::<usize>() * element_size;
                    let chunk = crate::dataset::chunk::async_read_chunk_raw(
                        &self.source,
                        &crate::dataset::chunk::ChunkLocation {
                            address: entry.chunk_address,
                            size: entry.chunk_size as u64,
                            filter_mask: entry.filter_mask,
                        },
                        chunk_uncompressed_size,
                        &filter_ids,
                        &registry,
                        fill_value.as_deref(),
                    ).await?;

                    let chunk_coords: Vec<usize> = entry.dimension_offsets.iter().map(|&o| o as usize).collect();

                    // Same layout copy logic as sync. 
                    // This could be optimized later.
                    let mut chunk_pos = 0;
                    let mut dataset_pos = vec![0usize; dataset_dims.len()];

                    while chunk_pos < chunk.len() {
                        let mut ds_coords = chunk_coords.clone();
                        for (i, &o) in dataset_pos.iter().enumerate() {
                            ds_coords[i] += o;
                        }

                        let valid = ds_coords.iter().zip(dataset_dims.iter()).all(|(&c, &d)| c < d);
                        if valid {
                            let mut linear_idx = 0usize;
                            for (&coord, &dim) in ds_coords.iter().zip(dataset_dims.iter()) {
                                linear_idx = linear_idx * dim + coord;
                            }
                            let out_offset = linear_idx * element_size;
                            out[out_offset..out_offset + element_size]
                                .copy_from_slice(&chunk[chunk_pos..chunk_pos + element_size]);
                        }

                        chunk_pos += element_size;
                        for i in (0..dataset_pos.len()).rev() {
                            dataset_pos[i] += 1;
                            if dataset_pos[i] < chunk_dims[i] {
                                break;
                            }
                            dataset_pos[i] = 0;
                        }
                    }
                }
            }
            (4, _, Some(index_type), Some(index_address)) => {
                if index_type != crate::dataset::layout::chunk_index_type::BTREE_V2 {
                    return Err(Error::UnsupportedFeature {
                        feature: alloc::format!("v4 chunk index type {index_type:?}"),
                    });
                }
                
                let entries = self.async_read_v4_chunk_btree_entries(index_address).await?;
                let chunk_uncompressed_size = chunk_dims.iter().product::<usize>() * element_size;

                for entry in entries {
                    let chunk = crate::dataset::chunk::async_read_chunk_raw(
                        &self.source,
                        &crate::dataset::chunk::ChunkLocation {
                            address: entry.chunk_address,
                            size: entry.chunk_size as u64,
                            filter_mask: entry.filter_mask,
                        },
                        chunk_uncompressed_size,
                        &filter_ids,
                        &registry,
                        fill_value.as_deref(),
                    ).await?;

                    let chunk_coords: Vec<usize> = entry.dimension_offsets.iter().map(|&o| o as usize).collect();
                    let mut chunk_pos = 0;
                    let mut dataset_pos = vec![0usize; dataset_dims.len()];

                    while chunk_pos < chunk.len() {
                        let mut ds_coords = chunk_coords.clone();
                        for (i, &o) in dataset_pos.iter().enumerate() {
                            ds_coords[i] += o;
                        }

                        let valid = ds_coords.iter().zip(dataset_dims.iter()).all(|(&c, &d)| c < d);
                        if valid {
                            let mut linear_idx = 0usize;
                            for (&coord, &dim) in ds_coords.iter().zip(dataset_dims.iter()) {
                                linear_idx = linear_idx * dim + coord;
                            }
                            let out_offset = linear_idx * element_size;
                            out[out_offset..out_offset + element_size]
                                .copy_from_slice(&chunk[chunk_pos..chunk_pos + element_size]);
                        }

                        chunk_pos += element_size;
                        for i in (0..dataset_pos.len()).rev() {
                            dataset_pos[i] += 1;
                            if dataset_pos[i] < chunk_dims[i] {
                                break;
                            }
                            dataset_pos[i] = 0;
                        }
                    }
                }
            }
            _ => {
                return Err(Error::UnsupportedFeature {
                    feature: alloc::string::String::from("unsupported chunked layout version or missing B-tree address"),
                });
            }
        }

        Ok(out)
    }

    async fn async_read_v1_chunk_btree_leaf_entries(
        &self,
        btree_address: u64,
        rank: usize,
    ) -> Result<Vec<ChunkIndexEntry>> {
        let header = BTreeV1Header::async_parse(&self.source, btree_address, &self.ctx).await?;
        if header.node_type != BTreeV1Type::RawDataChunk {
            return Err(Error::InvalidFormat {
                message: alloc::string::String::from("chunk index B-tree is not a raw-data chunk tree"),
            });
        }
        if header.level != 0 {
            return Err(Error::UnsupportedFeature {
                feature: alloc::string::String::from("chunked full read currently supports only leaf chunk B-trees"),
            });
        }

        let s = self.ctx.offset_bytes();
        let key_size = 4 + 4 + 8 * (rank + 1);
        let header_size = 8 + 2 * s;
        let data_size = key_size + header.entries_used as usize * (s + key_size);
        let mut data = vec![0u8; data_size];
        self.source.read_at(btree_address + header_size as u64, &mut data).await?;

        let mut entries = Vec::with_capacity(header.entries_used as usize);
        use byteorder::{ByteOrder, LittleEndian};
        let mut pos = key_size;

        for _ in 0..header.entries_used as usize {
            let chunk_address = self.ctx.read_offset(&data[pos..pos + s]);
            pos += s;

            let chunk_size = LittleEndian::read_u32(&data[pos..pos + 4]);
            pos += 4;
            let filter_mask = LittleEndian::read_u32(&data[pos..pos + 4]);
            pos += 4;
            let mut dimension_offsets = Vec::with_capacity(rank);
            for _ in 0..rank {
                dimension_offsets.push(LittleEndian::read_u64(&data[pos..pos + 8]));
                pos += 8;
            }
            pos += 8;

            entries.push(ChunkIndexEntry {
                dimension_offsets,
                filter_mask,
                chunk_size,
                chunk_address,
            });
        }

        Ok(entries)
    }

    async fn async_read_v4_chunk_btree_entries(&self, index_address: u64) -> Result<Vec<ChunkIndexEntry>> {
        let header = BTreeV2Header::async_parse(&self.source, index_address, &self.ctx).await?;
        if header.record_type != btree_v2_record_type::CHUNK_V4_NON_FILTERED
            && header.record_type != btree_v2_record_type::CHUNK_V4_FILTERED
        {
            return Err(Error::InvalidFormat {
                message: alloc::string::String::from("v4 chunk index is not a chunked-data B-tree v2 tree"),
            });
        }

        let records = async_collect_all_records(&self.source, &header, &self.ctx).await?;

        // Extract rank
        let rank = if header.record_type == btree_v2_record_type::CHUNK_V4_NON_FILTERED {
            let r = (header.record_size as usize).saturating_sub(12) / 8;
            if r * 8 + 12 == header.record_size as usize {
                Some(r)
            } else {
                None
            }
        } else {
            let r = (header.record_size as usize).saturating_sub(16) / 8;
            if r * 8 + 16 == header.record_size as usize {
                Some(r)
            } else {
                None
            }
        }.ok_or_else(|| Error::InvalidFormat {
            message: alloc::string::String::from("unable to determine v4 chunk rank from record size"),
        })?;

        let mut entries = Vec::with_capacity(records.len());
        use byteorder::{ByteOrder, LittleEndian};

        for record in &records {
            let data = &record.data;
            let mut dimension_offsets = Vec::with_capacity(rank);
            let mut pos = 4; // Skip dimensional chunk size (size of chunk in elements scaled)

            let is_filtered = header.record_type == btree_v2_record_type::CHUNK_V4_FILTERED;
            
            // Scaled dimensional coordinates
            for _ in 0..rank {
                if pos + 8 > data.len() {
                    break; // Will error later if invalid
                }
                dimension_offsets.push(LittleEndian::read_u64(&data[pos..pos + 8]));
                pos += 8;
            }

            let filter_mask = if is_filtered {
                if pos + 4 <= data.len() {
                    let fm = LittleEndian::read_u32(&data[pos..pos + 4]);
                    pos += 4;
                    fm
                } else {
                    0
                }
            } else {
                0
            };

            let chunk_size = if is_filtered {
                if pos + 4 <= data.len() {
                    let cs = LittleEndian::read_u32(&data[pos..pos + 4]);
                    pos += 4;
                    cs
                } else {
                    0
                }
            } else {
                // Not strictly correct since uncompressed size is implied, but good enough.
                0
            };

            let chunk_address = if pos + self.ctx.offset_bytes() <= data.len() {
                self.ctx.read_offset(&data[pos..pos + self.ctx.offset_bytes()])
            } else {
                crate::constants::UNDEFINED_ADDRESS
            };

            entries.push(ChunkIndexEntry {
                dimension_offsets,
                filter_mask,
                chunk_size,
                chunk_address,
            });
        }

        Ok(entries)
    }
}
