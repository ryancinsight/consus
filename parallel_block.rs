
#[cfg(all(feature = "parallel-io", feature = "alloc"))]
impl<R: ReadAt + Sync> Hdf5File<R> {
    pub fn read_chunked_dataset_all_bytes(&self, object_header_address: u64) -> Result<Vec<u8>> {
        use crate::dataset::chunk::{edge_chunk_dims, ChunkLocation};
        use crate::dataset::parallel::{execute_parallel, ChunkTask};

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

        let element_size = dataset
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
                        &ChunkLocation {
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
