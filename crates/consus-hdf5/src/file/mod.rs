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

#[cfg(all(feature = "async-io", feature = "alloc"))]
pub mod async_file;
#[cfg(all(feature = "async-io", feature = "alloc"))]
pub mod async_reader;
pub mod reader;
#[cfg(feature = "alloc")]
pub mod writer;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use byteorder::{ByteOrder, LittleEndian};
use consus_core::{Datatype, NodeType, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;
use crate::attribute::Hdf5Attribute;
#[cfg(feature = "alloc")]
use crate::btree::v1::{BTreeV1Header, BTreeV1Type};
#[cfg(feature = "alloc")]
use crate::btree::{BTreeV2Header, btree_v2_record_type, collect_all_btree_v2_records};
#[cfg(feature = "alloc")]
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
        let children = reader::list_group_v2(&self.source, &header, &self.ctx)?;
        if !children.is_empty() {
            return Ok(children
                .into_iter()
                .map(|(n, a, lt, _)| (n, a, lt))
                .collect());
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
        reader::read_attributes(&self.source, &header, &self.ctx)
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

    /// Read the full raw byte payload of a chunked dataset.
    ///
    /// This helper is intended for end-to-end roundtrip verification of the
    /// current writer path. It supports version-3 chunked layout with a
    /// single-leaf raw-data chunk B-tree and version-4 chunked layout with
    /// a B-tree v2 chunk index leaf.
    #[cfg(feature = "alloc")]
    pub fn read_chunked_dataset_all_bytes(&self, object_header_address: u64) -> Result<Vec<u8>> {
        let header = reader::read_object_header(&self.source, object_header_address, &self.ctx)?;
        let dataset = reader::read_dataset_metadata(&header, &self.ctx)?;

        if dataset.layout != crate::dataset::StorageLayout::Chunked {
            return Err(Error::InvalidFormat {
                message: String::from("dataset is not chunked"),
            });
        }

        let layout_msg =
            reader::find_message(&header, crate::object_header::message_types::DATA_LAYOUT)
                .ok_or_else(|| Error::InvalidFormat {
                    message: String::from("dataset object header missing layout message"),
                })?;
        let layout = crate::dataset::layout::DataLayout::parse(&layout_msg.data, &self.ctx)?;

        let chunk_dims_u32 = layout.chunk_dims.ok_or_else(|| Error::InvalidFormat {
            message: String::from("chunked dataset missing chunk dimensions"),
        })?;
        let chunk_dims: Vec<usize> = chunk_dims_u32.iter().map(|&d| d as usize).collect();

        let element_size =
            dataset
                .datatype
                .element_size()
                .ok_or_else(|| Error::UnsupportedFeature {
                    feature: String::from("chunked full read requires fixed-size element datatype"),
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
                    let entries = self.read_v1_chunk_btree_leaf_entries(chunk_btree_address, 0)?;
                    let entry = entries.first().ok_or_else(|| Error::InvalidFormat {
                        message: String::from("scalar chunked dataset has no chunk entries"),
                    })?;
                    let chunk = crate::dataset::chunk::read_chunk_raw(
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
                    )?;
                    out.copy_from_slice(&chunk[..element_size]);
                    return Ok(out);
                }

                let grid_dims: Vec<usize> = dataset_dims
                    .iter()
                    .zip(chunk_dims.iter())
                    .map(|(&dataset_dim, &chunk_dim)| dataset_dim.div_ceil(chunk_dim))
                    .collect();

                let entries =
                    self.read_v1_chunk_btree_leaf_entries(chunk_btree_address, chunk_dims.len())?;

                #[cfg(feature = "alloc")]
                {
                    use crate::dataset::parallel::ChunkTask;

                    let tasks: Vec<ChunkTask> = entries
                        .iter()
                        .map(|entry| {
                            let chunk_coord = self.decode_chunk_coord(
                                entry.dimension_offsets.as_slice(),
                                &chunk_dims,
                                &grid_dims,
                            )?;
                            let actual_chunk_dims = crate::dataset::chunk::edge_chunk_dims(
                                &chunk_coord,
                                &chunk_dims,
                                &dataset_dims,
                            );
                            let uncompressed_size = actual_chunk_dims
                                .iter()
                                .product::<usize>()
                                .checked_mul(element_size)
                                .ok_or(Error::Overflow)?;

                            Ok(ChunkTask {
                                chunk_coord,
                                location: crate::dataset::chunk::ChunkLocation {
                                    address: entry.chunk_address,
                                    size: entry.chunk_size as u64,
                                    filter_mask: entry.filter_mask,
                                },
                                actual_chunk_dims,
                                uncompressed_size,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;

                    #[cfg(feature = "parallel-io")]
                    let results = crate::dataset::parallel::execute_parallel(
                        &self.source,
                        tasks,
                        &filter_ids,
                        &registry,
                        fill_value.as_deref(),
                    )?;

                    #[cfg(not(feature = "parallel-io"))]
                    let results = crate::dataset::parallel::execute_serial(
                        &self.source,
                        tasks,
                        &filter_ids,
                        &registry,
                        fill_value.as_deref(),
                    )?;

                    for result in results {
                        self.copy_chunk_into_dataset(
                            &result.data,
                            &mut out,
                            &dataset_dims,
                            &chunk_dims,
                            &result.chunk_coord,
                            &result.actual_chunk_dims,
                            element_size,
                        )?;
                    }

                    return Ok(out);
                }
            }
            (4, _, Some(indexing_type), Some(index_address))
                if indexing_type == crate::dataset::layout::chunk_index_type::BTREE_V2 =>
            {
                let entries = self.read_v4_chunk_btree_entries(index_address)?;
                self.read_v4_chunk_entries(
                    &entries,
                    &dataset_dims,
                    &chunk_dims,
                    element_size,
                    &filter_ids,
                    fill_value.as_deref(),
                    &registry,
                    &mut out,
                )?;
                Ok(out)
            }
            _ => Err(Error::UnsupportedFeature {
                feature: String::from(
                    "chunked dataset read requires v3 v1-tree or v4 B-tree v2 index",
                ),
            }),
        }
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
    /// Leading  is accepted and ignored. Empty components (double
    /// slashes) are skipped. Returns  if any component
    /// is absent.
    ///
    /// ## Soft Link Resolution
    ///
    /// Soft links (type 1) are resolved recursively up to a depth of 40
    /// hops to break potential cycles. Absolute soft link targets (beginning
    /// with ) are resolved from the root group. Relative targets are
    /// resolved within the current group.
    ///
    /// External links return .
    ///
    /// ## Errors
    ///
    /// -  if any path component is missing.
    /// -  if an object header is malformed or a
    ///   soft link cycle exceeds the maximum depth of 40.
    /// -  if an external link is traversed.
    #[cfg(feature = "alloc")]
    pub fn open_path(&self, path: &str) -> Result<u64> {
        self.open_path_from(self.superblock.root_group_address, path, 0)
    }

    /// Resolve a path from a given group address with cycle-break depth tracking.
    ///
    /// Called by [] and recursively for soft link resolution.
    ///  increments on each soft link hop; exceeding
    /// returns [] to break cycles.
    #[cfg(feature = "alloc")]
    #[cfg(feature = "alloc")]
    fn open_path_from(&self, start: u64, path: &str, depth: usize) -> Result<u64> {
        const MAX_LINK_DEPTH: usize = 40;
        if depth > MAX_LINK_DEPTH {
            return Err(Error::InvalidFormat {
                message: alloc::string::String::from(
                    "soft link cycle detected: maximum link depth exceeded",
                ),
            });
        }

        let mut current = start;
        for component in path.split('/').filter(|s| !s.is_empty()) {
            let header = reader::read_object_header(&self.source, current, &self.ctx)?;
            let mut found: Option<u64> = None;

            // Try v2 link messages first (dense or compact).
            let v2 = reader::list_group_v2(&self.source, &header, &self.ctx)?;
            for (name, addr, link_type, soft_target) in &v2 {
                if name == component {
                    match link_type {
                        consus_core::LinkType::Hard => {
                            found = Some(*addr);
                        }
                        consus_core::LinkType::Soft => {
                            if let Some(target) = soft_target {
                                let resolved = if target.starts_with('/') {
                                    self.open_path_from(
                                        self.superblock.root_group_address,
                                        target,
                                        depth + 1,
                                    )?
                                } else {
                                    self.open_path_from(current, target, depth + 1)?
                                };
                                found = Some(resolved);
                            }
                        }
                        consus_core::LinkType::External => {
                            return Err(Error::UnsupportedFeature {
                                feature: alloc::string::String::from("external link resolution"),
                            });
                        }
                    }
                    break;
                }
            }

            // Fall back to v1 symbol table. v2 groups have no SYMBOL_TABLE
            // message, so list_group_v1 returns InvalidFormat in that case
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
    ///
    /// Tries v2 compact/dense link messages first.  Falls back to the v1
    /// symbol-table path only when a `SYMBOL_TABLE` message is present in
    /// the object header; v2 groups with no children have no such message
    /// and correctly return an empty list rather than an error.
    #[cfg(feature = "alloc")]
    pub fn list_group_at(&self, address: u64) -> Result<Vec<(String, u64, consus_core::LinkType)>> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;

        let v2 = reader::list_group_v2(&self.source, &header, &self.ctx)?;
        if !v2.is_empty() {
            return Ok(v2.into_iter().map(|(n, a, lt, _)| (n, a, lt)).collect());
        }

        // Only attempt the v1 symbol-table path when the object header
        // contains a SYMBOL_TABLE message.  v2 groups with zero children
        // produce an empty v2 list and carry no SYMBOL_TABLE message; for
        // them an empty result is correct and the v1 path must not be tried.
        use crate::object_header::message_types;
        if reader::find_message(&header, message_types::SYMBOL_TABLE).is_none() {
            return Ok(Vec::new());
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

    /// Read the canonical datatype from a committed (named) datatype object at `address`.
    ///
    /// ## HDF5 specification
    ///
    /// A committed datatype object header (§IV.A.2.3) carries exactly one
    /// Datatype message (0x0003) without Dataspace or Data Layout.
    /// `classify_object` returns `NodeType::NamedDatatype` for such objects.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the object header is missing a Datatype message
    ///   or the datatype cannot be decoded.
    #[cfg(feature = "alloc")]
    pub fn named_datatype_at(&self, address: u64) -> Result<Datatype> {
        let header = reader::read_object_header(&self.source, address, &self.ctx)?;
        let dt_msg = reader::find_message(&header, crate::object_header::message_types::DATATYPE)
            .ok_or_else(|| Error::InvalidFormat {
            message: String::from("committed datatype object header missing datatype message"),
        })?;
        crate::datatype::compound::parse_datatype(&dt_msg.data)
    }

    #[cfg(feature = "alloc")]
    fn read_v1_chunk_btree_leaf_entries(
        &self,
        btree_address: u64,
        rank: usize,
    ) -> Result<Vec<ChunkIndexEntry>> {
        let header = BTreeV1Header::parse(&self.source, btree_address, &self.ctx)?;
        if header.node_type != BTreeV1Type::RawDataChunk {
            return Err(Error::InvalidFormat {
                message: String::from("chunk index B-tree is not a raw-data chunk tree"),
            });
        }
        if header.level != 0 {
            return Err(Error::UnsupportedFeature {
                feature: String::from(
                    "chunked full read currently supports only leaf chunk B-trees",
                ),
            });
        }

        let s = self.ctx.offset_bytes();
        let key_size = 4 + 4 + 8 * (rank + 1);
        let header_size = 8 + 2 * s;
        let data_size = key_size + header.entries_used as usize * (s + key_size);
        let mut data = vec![0u8; data_size];
        self.source
            .read_at(btree_address + header_size as u64, &mut data)?;

        let mut entries = Vec::with_capacity(header.entries_used as usize);
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

    #[cfg(feature = "alloc")]
    fn read_v4_chunk_btree_entries(&self, index_address: u64) -> Result<Vec<ChunkIndexEntry>> {
        let header = BTreeV2Header::parse(&self.source, index_address, &self.ctx)?;
        if header.record_type != btree_v2_record_type::CHUNK_V4_NON_FILTERED
            && header.record_type != btree_v2_record_type::CHUNK_V4_FILTERED
        {
            return Err(Error::InvalidFormat {
                message: String::from("v4 chunk index is not a chunked-data B-tree v2 tree"),
            });
        }

        let records = collect_all_btree_v2_records(&self.source, &header, &self.ctx)?;

        let rank = self
            .read_v4_chunk_rank(&header)?
            .ok_or_else(|| Error::InvalidFormat {
                message: String::from("unable to determine v4 chunk rank from record size"),
            })?;

        let mut entries = Vec::with_capacity(records.len());
        for record in &records {
            let (dimension_offsets, filter_mask, chunk_address, chunk_size) =
                self.parse_v4_chunk_record(&record.data, rank, &header)?;
            entries.push(ChunkIndexEntry {
                dimension_offsets,
                filter_mask,
                chunk_size,
                chunk_address,
            });
        }

        Ok(entries)
    }

    #[cfg(feature = "alloc")]
    fn read_v4_chunk_rank(&self, header: &BTreeV2Header) -> Result<Option<usize>> {
        if header.total_records == 0 {
            return Ok(None);
        }

        let record_size = header.record_size as usize;
        let o = self.ctx.offset_bytes();

        let overhead = if header.record_type == btree_v2_record_type::CHUNK_V4_FILTERED {
            // Type 11: address(O) + chunk_size(L) + filter_mask(4)
            o + self.ctx.length_bytes() + 4
        } else {
            // Type 10: address(O)
            o
        };

        if record_size < overhead {
            return Err(Error::InvalidFormat {
                message: String::from("v4 chunk record too small for context sizes"),
            });
        }

        let payload = record_size - overhead;
        if payload % 8 != 0 {
            return Err(Error::InvalidFormat {
                message: String::from(
                    "v4 chunk record scaled-offset payload is not aligned to 8-byte offsets",
                ),
            });
        }

        Ok(Some(payload / 8))
    }

    #[cfg(feature = "alloc")]
    fn parse_v4_chunk_record(
        &self,
        data: &[u8],
        rank: usize,
        header: &BTreeV2Header,
    ) -> Result<(Vec<u64>, u32, u64, u32)> {
        let expected = header.record_size as usize;
        if data.len() != expected {
            return Err(Error::InvalidFormat {
                message: String::from("unexpected v4 chunk record length"),
            });
        }

        let o = self.ctx.offset_bytes();
        let mut pos = 0usize;

        // Address (offset_size bytes)
        let chunk_address = self.ctx.read_offset(&data[pos..]);
        pos += o;

        // For filtered records (type 11): chunk size + filter mask
        let (chunk_size, filter_mask) =
            if header.record_type == btree_v2_record_type::CHUNK_V4_FILTERED {
                let l = self.ctx.length_bytes();
                // Chunk size after filtering (length_size bytes)
                let size = self.ctx.read_length(&data[pos..]);
                pos += l;
                // Filter mask (4 bytes)
                let mask = LittleEndian::read_u32(&data[pos..pos + 4]);
                pos += 4;
                (size as u32, mask)
            } else {
                // Non-filtered: no size or mask in record
                (0u32, 0u32)
            };

        // Scaled dimension offsets (rank x 8 bytes)
        let mut dimension_offsets = Vec::with_capacity(rank);
        for _ in 0..rank {
            dimension_offsets.push(LittleEndian::read_u64(&data[pos..pos + 8]));
            pos += 8;
        }

        Ok((dimension_offsets, filter_mask, chunk_address, chunk_size))
    }

    #[cfg(feature = "alloc")]
    fn read_v4_chunk_entries(
        &self,
        entries: &[ChunkIndexEntry],
        dataset_dims: &[usize],
        chunk_dims: &[usize],
        element_size: usize,
        filter_ids: &[u16],
        fill_value: Option<&[u8]>,
        registry: &dyn consus_compression::CompressionRegistry,
        out: &mut [u8],
    ) -> Result<()> {
        let grid_dims: Vec<usize> = dataset_dims
            .iter()
            .zip(chunk_dims.iter())
            .map(|(&dataset_dim, &chunk_dim)| dataset_dim.div_ceil(chunk_dim))
            .collect();

        #[cfg(all(feature = "parallel-io", feature = "alloc"))]
        {
            use crate::dataset::parallel::{ChunkTask, execute_parallel};

            let tasks: Vec<ChunkTask> = entries
                .iter()
                .map(|entry| {
                    let chunk_coord = Self::decode_v4_scaled_offsets(
                        entry.dimension_offsets.as_slice(),
                        &grid_dims,
                    )?;
                    let actual_chunk_dims = crate::dataset::chunk::edge_chunk_dims(
                        &chunk_coord,
                        chunk_dims,
                        dataset_dims,
                    );
                    let uncompressed_size = actual_chunk_dims
                        .iter()
                        .product::<usize>()
                        .checked_mul(element_size)
                        .ok_or(Error::Overflow)?;

                    Ok(ChunkTask {
                        chunk_coord,
                        location: crate::dataset::chunk::ChunkLocation {
                            address: entry.chunk_address,
                            size: if entry.chunk_size == 0 {
                                uncompressed_size as u64
                            } else {
                                entry.chunk_size as u64
                            },
                            filter_mask: entry.filter_mask,
                        },
                        actual_chunk_dims,
                        uncompressed_size,
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            let results = execute_parallel(&self.source, tasks, filter_ids, registry, fill_value)?;

            for result in results {
                self.copy_chunk_into_dataset(
                    &result.data,
                    out,
                    dataset_dims,
                    chunk_dims,
                    &result.chunk_coord,
                    &result.actual_chunk_dims,
                    element_size,
                )?;
            }

            return Ok(());
        }

        for entry in entries {
            let chunk_coord =
                Self::decode_v4_scaled_offsets(entry.dimension_offsets.as_slice(), &grid_dims)?;
            let actual_chunk_dims =
                crate::dataset::chunk::edge_chunk_dims(&chunk_coord, chunk_dims, dataset_dims);
            let chunk_elements = actual_chunk_dims.iter().product::<usize>();
            let uncompressed_size = chunk_elements
                .checked_mul(element_size)
                .ok_or(Error::Overflow)?;

            let chunk = crate::dataset::chunk::read_chunk_raw(
                &self.source,
                &crate::dataset::chunk::ChunkLocation {
                    address: entry.chunk_address,
                    size: if entry.chunk_size == 0 {
                        uncompressed_size as u64
                    } else {
                        entry.chunk_size as u64
                    },
                    filter_mask: entry.filter_mask,
                },
                uncompressed_size,
                filter_ids,
                registry,
                fill_value,
            )?;

            self.copy_chunk_into_dataset(
                &chunk,
                out,
                dataset_dims,
                chunk_dims,
                &chunk_coord,
                &actual_chunk_dims,
                element_size,
            )?;
        }

        Ok(())
    }

    /// Decode v4 B-tree v2 scaled offsets into chunk grid coordinates.
    ///
    /// V4 chunk index records store scaled offsets that are already chunk
    /// grid indices (unlike v3 byte offsets which must be divided by chunk
    /// dimensions). This method validates the indices against the grid
    /// dimensions.
    #[cfg(feature = "alloc")]
    fn decode_v4_scaled_offsets(scaled_offsets: &[u64], grid_dims: &[usize]) -> Result<Vec<usize>> {
        if scaled_offsets.len() != grid_dims.len() {
            return Err(Error::ShapeError {
                message: String::from("v4 chunk record rank mismatch with dataset grid dimensions"),
            });
        }

        let mut coord = Vec::with_capacity(grid_dims.len());
        for (dim, &scaled) in scaled_offsets.iter().enumerate() {
            let idx = usize::try_from(scaled).map_err(|_| Error::Overflow)?;
            if idx >= grid_dims[dim] {
                return Err(Error::SelectionOutOfBounds);
            }
            coord.push(idx);
        }

        Ok(coord)
    }

    #[cfg(feature = "alloc")]
    fn decode_chunk_coord(
        &self,
        dimension_offsets: &[u64],
        chunk_dims: &[usize],
        grid_dims: &[usize],
    ) -> Result<Vec<usize>> {
        if dimension_offsets.len() != chunk_dims.len() || chunk_dims.len() != grid_dims.len() {
            return Err(Error::ShapeError {
                message: String::from("chunk rank mismatch while decoding chunk coordinates"),
            });
        }

        let mut coord = Vec::with_capacity(chunk_dims.len());
        for dim in 0..chunk_dims.len() {
            let chunk_dim = chunk_dims[dim];
            if chunk_dim == 0 {
                return Err(Error::ShapeError {
                    message: String::from("chunk dimension must be strictly positive"),
                });
            }

            let offset = usize::try_from(dimension_offsets[dim]).map_err(|_| Error::Overflow)?;
            if offset % chunk_dim != 0 {
                return Err(Error::InvalidFormat {
                    message: String::from(
                        "chunk logical offset is not aligned to chunk dimensions",
                    ),
                });
            }

            let chunk_index = offset / chunk_dim;
            if chunk_index >= grid_dims[dim] {
                return Err(Error::SelectionOutOfBounds);
            }

            coord.push(chunk_index);
        }

        Ok(coord)
    }

    #[cfg(feature = "alloc")]
    fn copy_chunk_into_dataset(
        &self,
        chunk: &[u8],
        out: &mut [u8],
        dataset_dims: &[usize],
        chunk_dims: &[usize],
        chunk_coord: &[usize],
        actual_chunk_dims: &[usize],
        element_size: usize,
    ) -> Result<()> {
        let rank = dataset_dims.len();
        let expected_chunk_bytes = actual_chunk_dims
            .iter()
            .product::<usize>()
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        if chunk.len() != expected_chunk_bytes {
            return Err(Error::InvalidFormat {
                message: String::from("decoded chunk byte length does not match edge chunk shape"),
            });
        }

        let mut local_coord = vec![0usize; rank];
        let mut done = rank == 0;

        while !done {
            let mut dataset_coord = vec![0usize; rank];
            for d in 0..rank {
                dataset_coord[d] = chunk_coord[d] * chunk_dims[d] + local_coord[d];
            }

            let dataset_linear = linear_index(&dataset_coord, dataset_dims);
            let chunk_linear = linear_index(&local_coord, actual_chunk_dims);

            let src_start = chunk_linear
                .checked_mul(element_size)
                .ok_or(Error::Overflow)?;
            let src_end = src_start.checked_add(element_size).ok_or(Error::Overflow)?;
            let dst_start = dataset_linear
                .checked_mul(element_size)
                .ok_or(Error::Overflow)?;
            let dst_end = dst_start.checked_add(element_size).ok_or(Error::Overflow)?;

            out[dst_start..dst_end].copy_from_slice(&chunk[src_start..src_end]);

            for dim in (0..rank).rev() {
                local_coord[dim] += 1;
                if local_coord[dim] < actual_chunk_dims[dim] {
                    break;
                }
                local_coord[dim] = 0;
                if dim == 0 {
                    done = true;
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub(crate) struct ChunkIndexEntry {
    pub(crate) dimension_offsets: Vec<u64>,
    pub(crate) filter_mask: u32,
    pub(crate) chunk_size: u32,
    pub(crate) chunk_address: u64,
}

#[cfg(feature = "alloc")]
fn linear_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        index = index * dim + coord;
    }
    index
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
