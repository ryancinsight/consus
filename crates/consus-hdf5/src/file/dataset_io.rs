//! Dataset and attribute read paths for `Hdf5File`.

use super::{Hdf5File, checked_dataset_byte_count, linear_index, reader};
use crate::attribute::Hdf5Attribute;
use crate::dataset::{Hdf5Dataset, StorageLayout};
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use consus_core::{Datatype, Error, Result};
use consus_io::ReadAt;

impl<R: ReadAt + Sync> Hdf5File<R> {
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
        let dataset_dims = dataset.shape.current_dims();
        // Trim chunk_dims to the spatial rank (v4 layout may carry a trailing
        // element-size dimension that must be excluded from spatial operations).
        let chunk_dims: Vec<usize> = chunk_dims_u32
            .iter()
            .take(dataset_dims.len().max(1))
            .map(|&d| d as usize)
            .collect();

        let element_size =
            dataset
                .datatype
                .element_size()
                .ok_or_else(|| Error::UnsupportedFeature {
                    feature: String::from("chunked full read requires fixed-size element datatype"),
                })?;
        let total_bytes = checked_dataset_byte_count(&self.ctx, &dataset.shape, element_size)?;
        let mut out = self.ctx.budget.zeroed(
            u64::try_from(total_bytes).map_err(|_| Error::Overflow)?,
            "chunked dataset output",
        )?;

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
                        element_size,
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
                        element_size,
                        &registry,
                        fill_value.as_deref(),
                    )?;

                    #[cfg(not(feature = "parallel-io"))]
                    let results = crate::dataset::parallel::execute_serial(
                        &self.source,
                        tasks,
                        &filter_ids,
                        element_size,
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

                    Ok(out)
                }
            }
            (4, _, Some(indexing_type), Some(index_address))
                if indexing_type == crate::dataset::layout::chunk_index_type::FIXED_ARRAY =>
            {
                let entries =
                    self.read_v4_chunk_farray_entries(index_address, &dataset_dims, &chunk_dims)?;
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
    /// Read the full raw byte payload of a dataset, dispatching on its
    /// storage layout.
    ///
    /// This is the single layout-dispatch point shared by the high-level
    /// format crates (HDMF, NWB): contiguous datasets are read directly from
    /// their data address, chunked datasets delegate to
    /// [`Self::read_chunked_dataset_all_bytes`], and compact / virtual
    /// layouts fail closed with [`Error::UnsupportedFeature`].
    ///
    /// For variable-length element types (`VariableString`, `VarLen`) the
    /// element size is unknown; `variable_element_size` supplies the on-disk
    /// element width when the caller knows it (e.g. the fixed-size VL string
    /// reference layout), and is ignored for fixed-size datatypes.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] — contiguous dataset has no data address.
    /// - [`Error::UnsupportedFeature`] — compact or virtual layout.
    /// - Propagates HDF5 I/O and chunk-index errors.
    #[cfg(feature = "alloc")]
    pub fn read_dataset_raw(
        &self,
        dataset: &Hdf5Dataset,
        variable_element_size: Option<usize>,
    ) -> Result<Vec<u8>> {
        match dataset.layout {
            StorageLayout::Contiguous => {
                let element_size = match &dataset.datatype {
                    Datatype::VariableString { .. } => {
                        variable_element_size.ok_or_else(|| Error::UnsupportedFeature {
                            feature: String::from(
                                "variable-length contiguous dataset read needs an element size",
                            ),
                        })?
                    }
                    other => other.element_size().unwrap_or(0),
                };
                let n_bytes = checked_dataset_byte_count(&self.ctx, &dataset.shape, element_size)?;
                if n_bytes == 0 {
                    return Ok(Vec::new());
                }
                let data_addr = dataset.data_address.ok_or_else(|| Error::InvalidFormat {
                    message: String::from("contiguous dataset has no data address"),
                })?;
                let mut buf = self.ctx.budget.zeroed(
                    u64::try_from(n_bytes).map_err(|_| Error::Overflow)?,
                    "contiguous dataset output",
                )?;
                self.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)?;
                Ok(buf)
            }
            StorageLayout::Chunked => {
                self.read_chunked_dataset_all_bytes(dataset.object_header_address)
            }
            StorageLayout::Compact => Err(Error::UnsupportedFeature {
                feature: String::from("compact dataset layout is not supported"),
            }),
            StorageLayout::Virtual => Err(Error::UnsupportedFeature {
                feature: String::from("virtual dataset layout is not supported"),
            }),
        }
    }
    /// Read raw bytes from a specific file offset.
    ///
    /// Low-level utility for format parsing code.
    pub fn read_bytes(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.source.read_at(offset, buf)
    }
    #[cfg(feature = "alloc")]
    pub(super) fn copy_chunk_into_dataset(
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
