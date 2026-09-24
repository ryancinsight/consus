//! v1/v4 chunk index leaf/record readers and chunk-coordinate decoding for `Hdf5File`.

use super::{ChunkIndexEntry, Hdf5File, checked_v1_chunk_btree_data_size};
use crate::btree::v1::{BTreeV1Header, BTreeV1Type};
#[cfg(feature = "alloc")]
use crate::btree::{BTreeV2Header, btree_v2_record_type, collect_all_btree_v2_records};
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use byteorder::{ByteOrder, LittleEndian};
use consus_core::{Error, Result};
use consus_io::ReadAt;

impl<R: ReadAt + Sync> Hdf5File<R> {
    #[cfg(feature = "alloc")]
    pub(super) fn read_v1_chunk_btree_leaf_entries(
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
        let header_size = 8usize
            .checked_add(s.checked_mul(2).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let data_size = checked_v1_chunk_btree_data_size(&self.ctx, rank, header.entries_used)?;
        let read_offset = btree_address
            .checked_add(u64::try_from(header_size).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let mut data = self.ctx.budget.zeroed(
            u64::try_from(data_size).map_err(|_| Error::Overflow)?,
            "HDF5 v1 chunk B-tree records",
        )?;
        self.source.read_at(read_offset, &mut data)?;

        let mut entries = self.ctx.budget.vec_with_capacity(
            u64::from(header.entries_used),
            "HDF5 v1 chunk B-tree entries",
        )?;
        // Per HDF5 spec §IV.A.2.b (B-tree v1 raw-data chunk leaf node):
        // The node layout is key[0] | addr[0] | key[1] | addr[1] | ... | key[N-1] | addr[N-1] | key[N]
        // where key[i] describes the chunk at addr[i].  Read key first, then address.
        let mut pos = 0;

        for _ in 0..header.entries_used as usize {
            let chunk_size = LittleEndian::read_u32(&data[pos..pos + 4]);
            pos += 4;
            let filter_mask = LittleEndian::read_u32(&data[pos..pos + 4]);
            pos += 4;
            let mut dimension_offsets = self.ctx.budget.vec_with_capacity(
                u64::try_from(rank).map_err(|_| Error::Overflow)?,
                "HDF5 v1 chunk rank",
            )?;
            for _ in 0..rank {
                dimension_offsets.push(LittleEndian::read_u64(&data[pos..pos + 8]));
                pos += 8;
            }
            pos += 8; // skip the extra +1 terminator dimension

            let chunk_address = self.ctx.read_offset(&data[pos..pos + s]);
            pos += s;

            entries.push(ChunkIndexEntry {
                dimension_offsets,
                filter_mask,
                chunk_size,
                chunk_address,
            });
        }
        // key[N] (sentinel) is not read; pos lands at its start within `data`.

        Ok(entries)
    }
    #[cfg(feature = "alloc")]
    pub(super) fn read_v4_chunk_farray_entries(
        &self,
        index_address: u64,
        dataset_dims: &[usize],
        chunk_dims: &[usize],
    ) -> Result<Vec<ChunkIndexEntry>> {
        // Read FAHD (Fixed Array Header Descriptor, 28 bytes).
        let mut fahd = [0u8; 28];
        self.source.read_at(index_address, &mut fahd)?;
        if &fahd[0..4] != b"FAHD" {
            return Err(Error::InvalidFormat {
                message: String::from("invalid Fixed Array header signature (expected FAHD)"),
            });
        }
        let entry_size = fahd[6] as usize; // bytes per chunk record (= offset_size for no-filter)
        if entry_size == 0 {
            return Err(Error::InvalidFormat {
                message: String::from("Fixed Array entry size must be non-zero"),
            });
        }
        let nelmts = self.ctx.budget.checked_elements(
            LittleEndian::read_u64(&fahd[8..16]),
            entry_size,
            "fixed-array chunk entry count",
        )?;
        let fadb_addr = self.ctx.read_offset(&fahd[16..]);

        // Read FADB (Fixed Array Data Block).
        // Layout: sig(4) + ver(1) + cid(1) + hdr_addr(8) + nelmts*entry_size + checksum(4)
        let content_size = 14usize
            .checked_add(nelmts.checked_mul(entry_size).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let fadb_size = content_size.checked_add(4).ok_or(Error::Overflow)?; // include checksum
        let mut fadb = self.ctx.budget.zeroed(
            u64::try_from(fadb_size).map_err(|_| Error::Overflow)?,
            "fixed-array data block",
        )?;
        self.source.read_at(fadb_addr, &mut fadb)?;
        if &fadb[0..4] != b"FADB" {
            return Err(Error::InvalidFormat {
                message: String::from("invalid Fixed Array data block signature (expected FADB)"),
            });
        }

        // Chunk grid dimensions (number of chunks per spatial axis).
        let grid_dims: Vec<usize> = dataset_dims
            .iter()
            .zip(chunk_dims.iter())
            .map(|(&ds, &cs)| {
                if cs == 0 {
                    Err(Error::InvalidFormat {
                        message: String::from("Fixed Array chunk dimension must be non-zero"),
                    })
                } else {
                    Ok(ds.div_ceil(cs))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        if nelmts != 0 && grid_dims.contains(&0) {
            return Err(Error::InvalidFormat {
                message: String::from(
                    "Fixed Array contains entries for a dataset with an empty chunk grid",
                ),
            });
        }

        // Entries are stored in row-major (C) order: decompose linear index i
        // into grid coordinates using row-major decomposition.
        let mut entries = Vec::with_capacity(nelmts);
        for i in 0..nelmts {
            let off = 14 + i * entry_size;
            let chunk_address = self.ctx.read_offset(&fadb[off..]);

            let mut coord = vec![0u64; grid_dims.len()];
            let mut remaining = i;
            for d in (0..grid_dims.len()).rev() {
                coord[d] = (remaining % grid_dims[d]) as u64;
                remaining /= grid_dims[d];
            }

            entries.push(ChunkIndexEntry {
                dimension_offsets: coord,
                filter_mask: 0,
                chunk_size: 0, // no per-chunk size stored for non-filtered FARRAY
                chunk_address,
            });
        }

        Ok(entries)
    }
    #[cfg(feature = "alloc")]
    pub(super) fn read_v4_chunk_btree_entries(
        &self,
        index_address: u64,
    ) -> Result<Vec<ChunkIndexEntry>> {
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
        if !payload.is_multiple_of(8) {
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
        let mut dimension_offsets = self.ctx.budget.vec_with_capacity(
            u64::try_from(rank).map_err(|_| Error::Overflow)?,
            "HDF5 v4 chunk rank",
        )?;
        for _ in 0..rank {
            dimension_offsets.push(LittleEndian::read_u64(&data[pos..pos + 8]));
            pos += 8;
        }

        Ok((dimension_offsets, filter_mask, chunk_address, chunk_size))
    }
    #[cfg(feature = "alloc")]
    pub(super) fn read_v4_chunk_entries(
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

            let results = execute_parallel(
                &self.source,
                tasks,
                filter_ids,
                element_size,
                registry,
                fill_value,
            )?;

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

            Ok(())
        }

        // Serial fallback when parallel-io is disabled (mutually exclusive with the
        // parallel block above).
        #[cfg(not(all(feature = "parallel-io", feature = "alloc")))]
        {
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
                    element_size,
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
    /// Decode B-tree v1 chunk dimension offsets into chunk grid coordinates.
    ///
    /// HDF5 spec §IV.A.2.b: dimension offsets in B-tree v1 chunk keys are
    /// **byte** offsets (first element position × element size) for each
    /// dimension. Divide by `chunk_dims[d] * element_size` to obtain the
    /// zero-based chunk grid index.
    #[cfg(feature = "alloc")]
    pub(super) fn decode_chunk_coord(
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
            if !offset.is_multiple_of(chunk_dim) {
                return Err(Error::InvalidFormat {
                    message: String::from("chunk element offset is not aligned to chunk dimension"),
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
}
