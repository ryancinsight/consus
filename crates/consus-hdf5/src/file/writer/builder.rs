//! High-level file-creation builder composing the low-level writer primitives.

#[cfg(feature = "alloc")]
use super::{
    ChildDatasetSpec, ChildGroupSpec, DatasetCreationProps, FileCreationProps, SubGroupBuilder,
    WriteState, dataset_filter_ids, encode_attribute, encode_dataspace, encode_datatype,
    encode_filter_pipeline, encode_group_info, encode_hard_link, encode_layout,
    encode_layout_with_chunk_index, encode_link_info, update_superblock_eof, write_chunked_data,
    write_chunked_data_with_element_size, write_contiguous_data, write_dataset_header,
    write_group_node, write_object_header_v2, write_superblock, write_v1_group_header,
};
#[cfg(feature = "alloc")]
use crate::object_header::message_types;
#[cfg(feature = "alloc")]
use crate::property_list::DatasetLayout;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use consus_core::{Datatype, Error, Result, Shape};

// ---------------------------------------------------------------------------
// High-level file builder
// ---------------------------------------------------------------------------

/// High-level HDF5 file builder.
///
/// Accumulates datasets and root-group attributes in memory, then writes a
/// well-formed HDF5 v2 file on [`finish`][Self::finish].
///
/// ### Write Order (bottom-up; no back-patching required)
///
/// 1. Superblock space reserved at offset 0.
/// 2. Dataset raw data blocks (addresses known before headers).
/// 3. Dataset object headers (data address known).
/// 4. Root group object header (all child addresses known).
/// 5. Actual superblock written at offset 0; EOF address patched.
///
/// This ordering guarantees every address reference in each structure
/// points to an already-allocated region.
#[cfg(feature = "alloc")]
pub struct Hdf5FileBuilder {
    sink: consus_io::MemCursor,
    state: WriteState,
    /// Encoded hard-link (name, address) pairs for the root group.
    root_links: Vec<(String, u64)>,
    /// Encoded attribute message payloads for the root group.
    root_attr_bytes: Vec<Vec<u8>>,
}

#[cfg(feature = "alloc")]
impl Hdf5FileBuilder {
    /// Create a new builder with the given file creation properties.
    pub fn new(props: FileCreationProps) -> Self {
        let mut state = WriteState::new(props);
        // Reserve superblock space at offset 0 so subsequent allocations
        // do not overwrite it.
        let sb_size = 12 + 4 * state.ctx.offset_bytes() + 4;
        state.eof = sb_size as u64;
        Self {
            sink: consus_io::MemCursor::new(),
            state,
            root_links: Vec::new(),
            root_attr_bytes: Vec::new(),
        }
    }

    /// Add a contiguous dataset to the root group.
    ///
    /// The dataset is written immediately (data block + object header).
    /// Returns the object header address.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] or [`Error::UnsupportedFeature`] if the
    ///   datatype is not encodable.
    pub fn add_dataset(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
    ) -> Result<u64> {
        let vlen_ref_size = 4 + self.state.ctx.offset_bytes() + 4;
        let data_addr = match dcpl.layout {
            DatasetLayout::Chunked => {
                if matches!(dt, Datatype::VariableString { .. }) {
                    write_chunked_data_with_element_size(
                        &mut self.sink,
                        &mut self.state,
                        vlen_ref_size,
                        shape,
                        raw_data,
                        dcpl,
                    )?
                } else {
                    write_chunked_data(&mut self.sink, &mut self.state, dt, shape, raw_data, dcpl)?
                }
            }
            _ => write_contiguous_data(&mut self.sink, &mut self.state, raw_data)?,
        };
        let header_addr =
            write_dataset_header(&mut self.sink, &mut self.state, dt, shape, data_addr, dcpl)?;
        self.root_links.push((String::from(name), header_addr));
        Ok(header_addr)
    }

    /// Add a dataset to the root group, pointing to a predefined data address.
    ///
    /// This writes the dataset object header (datatype, dataspace, layout) but
    /// does **not** write the raw data bytes. Useful for generating layout messages
    /// pointing to external or virtual regions.
    pub fn add_virtual_dataset(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        data_addr: u64,
        dcpl: &DatasetCreationProps,
    ) -> Result<u64> {
        let header_addr =
            write_dataset_header(&mut self.sink, &mut self.state, dt, shape, data_addr, dcpl)?;
        self.root_links.push((String::from(name), header_addr));
        Ok(header_addr)
    }

    /// Add a dataset with attached attributes to the root group.
    ///
    /// `attributes` is a slice of `(name, datatype, shape, raw_data)`.
    /// Attributes are encoded into the dataset's object header.
    pub fn add_dataset_with_attributes(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
        attributes: &[(&str, &Datatype, &Shape, &[u8])],
    ) -> Result<u64> {
        let vlen_ref_size = 4 + self.state.ctx.offset_bytes() + 4;
        let data_addr = match dcpl.layout {
            DatasetLayout::Chunked => {
                if matches!(dt, Datatype::VariableString { .. }) {
                    write_chunked_data_with_element_size(
                        &mut self.sink,
                        &mut self.state,
                        vlen_ref_size,
                        shape,
                        raw_data,
                        dcpl,
                    )?
                } else {
                    write_chunked_data(&mut self.sink, &mut self.state, dt, shape, raw_data, dcpl)?
                }
            }
            _ => write_contiguous_data(&mut self.sink, &mut self.state, raw_data)?,
        };

        let dt_bytes = encode_datatype(dt)?;
        let ds_bytes = encode_dataspace(shape)?;
        let ctx = self.state.ctx;
        let filter_ids = dataset_filter_ids(dcpl);
        let layout_bytes = match dcpl.layout {
            DatasetLayout::Chunked => {
                let element_size = if matches!(dt, Datatype::VariableString { .. }) {
                    4 + ctx.offset_bytes() + 4
                } else {
                    dt.element_size().ok_or_else(|| Error::UnsupportedFeature {
                        feature: String::from("chunked write requires fixed-size element datatype"),
                    })?
                };
                encode_layout_with_chunk_index(
                    data_addr,
                    dcpl,
                    &ctx,
                    Some(data_addr),
                    Some(element_size as u32),
                )?
            }
            _ => encode_layout(data_addr, dcpl, &ctx)?,
        };

        let mut msgs: Vec<(u16, Vec<u8>)> = vec![
            (message_types::DATATYPE, dt_bytes),
            (message_types::DATASPACE, ds_bytes),
            (message_types::DATA_LAYOUT, layout_bytes),
        ];

        if !filter_ids.is_empty() {
            msgs.push((
                message_types::FILTER_PIPELINE,
                encode_filter_pipeline(&filter_ids, &dcpl.compression)?,
            ));
        }

        for (attr_name, attr_dt, attr_shape, attr_data) in attributes {
            let attr_bytes = encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?;
            msgs.push((message_types::ATTRIBUTE, attr_bytes));
        }

        let msg_refs: Vec<(u16, &[u8])> = msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let header_addr = write_object_header_v2(&mut self.sink, &mut self.state, &msg_refs)?;

        self.root_links.push((String::from(name), header_addr));
        Ok(header_addr)
    }

    /// Add a named datatype to the root group.
    ///
    /// Emits an object header containing only a Datatype message (no Dataspace,
    /// no Layout). The object is linked from the root group under `name`.
    /// Returns the object header address.
    ///
    /// ## HDF5 specification
    ///
    /// HDF5 spec §IV.A.2.3: a committed datatype object header carries a
    /// single Datatype message (0x0003) without Dataspace or Data Layout.
    /// `classify_object` returns `NodeType::NamedDatatype` for such headers.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] or [`Error::UnsupportedFeature`] if the
    ///   datatype cannot be encoded.
    pub fn add_named_datatype(&mut self, name: &str, datatype: &Datatype) -> Result<u64> {
        let dt_bytes = encode_datatype(datatype)?;
        let msgs: &[(u16, &[u8])] = &[(message_types::DATATYPE, &dt_bytes)];
        let header_addr = write_object_header_v2(&mut self.sink, &mut self.state, msgs)?;
        self.root_links.push((String::from(name), header_addr));
        Ok(header_addr)
    }

    /// Add a v1 group linked from the root with explicit child link names.
    ///
    /// The writer emits a local heap, SNOD symbol-table node, and B-tree v1
    /// group index so `list_group_v1` can resolve the child names.
    pub fn add_v1_group_with_children(
        &mut self,
        name: &str,
        children: &[(&str, u64)],
    ) -> Result<u64> {
        let (names, addresses): (Vec<&str>, Vec<u64>) = children.iter().copied().unzip();
        let group_addr =
            write_v1_group_header(&mut self.sink, &mut self.state, &names, &addresses)?;
        self.root_links.push((String::from(name), group_addr));
        Ok(group_addr)
    }

    /// Add an attribute to the root group.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] or [`Error::UnsupportedFeature`] if the
    ///   datatype is not encodable.
    pub fn add_root_attribute(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
    ) -> Result<()> {
        let bytes = encode_attribute(name, dt, shape, raw_data)?;
        self.root_attr_bytes.push(bytes);
        Ok(())
    }

    /// Finalise the file and return the complete HDF5 image as bytes.
    ///
    /// Writes the root group object header (with all accumulated links and
    /// attributes), then writes and patches the superblock.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        let ctx = self.state.ctx;

        // Root group messages: LINK_INFO + GROUP_INFO first, then links and attributes.
        let mut msgs: Vec<(u16, Vec<u8>)> = vec![
            (
                message_types::LINK_INFO,
                encode_link_info(ctx.offset_bytes()),
            ),
            (message_types::GROUP_INFO, encode_group_info()),
        ];

        for (name, addr) in &self.root_links {
            let link_bytes = encode_hard_link(name, *addr, &ctx)?;
            msgs.push((message_types::LINK, link_bytes));
        }

        for attr_bytes in &self.root_attr_bytes {
            msgs.push((message_types::ATTRIBUTE, attr_bytes.clone()));
        }

        let msg_refs: Vec<(u16, &[u8])> = msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let root_addr = write_object_header_v2(&mut self.sink, &mut self.state, &msg_refs)?;

        write_superblock(&mut self.sink, &mut self.state, root_addr)?;
        update_superblock_eof(&mut self.sink, &self.state)?;

        Ok(self.sink.into_bytes())
    }

    /// Add a named group to the root group with attached attributes and child datasets.
    ///
    /// ## Write model
    ///
    /// For each child in `children`:
    /// 1. Data bytes are written as a contiguous block.
    /// 2. A dataset object header is written with Datatype + Dataspace + Layout
    ///    messages and optional attribute messages.
    ///
    /// Then the group object header is written with:
    /// - One LINK message per child mapping the child name to its header address.
    /// - One ATTRIBUTE message per entry in `group_attributes`.
    ///
    /// The group is linked from the root group.
    ///
    /// ## Errors
    ///
    /// Returns an error if any datatype or layout cannot be encoded.
    /// `ChildDatasetSpec::dcpl` must specify `DatasetLayout::Contiguous`.
    pub fn add_group_with_attributes(
        &mut self,
        group_name: &str,
        group_attributes: &[(&str, &Datatype, &Shape, &[u8])],
        children: &[ChildDatasetSpec<'_>],
    ) -> Result<u64> {
        let group_addr = write_group_node(
            &mut self.sink,
            &mut self.state,
            group_attributes,
            children,
            &[],
        )?;
        self.root_links.push((String::from(group_name), group_addr));
        Ok(group_addr)
    }

    /// Add a named group to the root group with both child datasets and child sub-groups.
    ///
    /// Sub-groups support arbitrary nesting depth: each [`ChildGroupSpec`] can
    /// contain further `ChildGroupSpec` values in its `sub_groups` field.
    /// The hierarchy is written depth-first (leaf nodes first).
    ///
    /// ## Errors
    ///
    /// Returns an error if any datatype or layout cannot be encoded.
    pub fn add_group_with_children(
        &mut self,
        group_name: &str,
        group_attributes: &[(&str, &Datatype, &Shape, &[u8])],
        datasets: &[ChildDatasetSpec<'_>],
        sub_groups: &[ChildGroupSpec<'_>],
    ) -> Result<u64> {
        let group_addr = write_group_node(
            &mut self.sink,
            &mut self.state,
            group_attributes,
            datasets,
            sub_groups,
        )?;
        self.root_links.push((String::from(group_name), group_addr));
        Ok(group_addr)
    }

    /// Open an incremental write context for a named group linked from the root.
    ///
    /// Datasets can be written one at a time, and the returned address from each
    /// write is immediately available for constructing attributes on subsequent
    /// datasets (e.g. `DIMENSION_LIST` referencing dimension-scale addresses).
    ///
    /// The group must be finished with
    /// [`SubGroupBuilder::finish_with_attributes`] before [`finish`][Self::finish]
    /// is called.
    pub fn begin_group(&mut self, name: &str) -> SubGroupBuilder<'_> {
        SubGroupBuilder {
            sink: &mut self.sink,
            state: &mut self.state,
            parent_links: &mut self.root_links,
            name: String::from(name),
            child_links: Vec::new(),
        }
    }
}
