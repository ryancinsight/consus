//! Nested sub-group builder used by the file-creation builder.

#[cfg(feature = "alloc")]
use super::{
    DatasetCreationProps, WriteState, encode_attribute, encode_dataspace, encode_datatype,
    encode_group_info, encode_hard_link, encode_layout, encode_link_info, write_contiguous_data,
    write_object_header_v2,
};
#[cfg(feature = "alloc")]
use crate::object_header::message_types;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use consus_core::{Datatype, Result, Shape};

// ---------------------------------------------------------------------------
// Incremental group write context
// ---------------------------------------------------------------------------

/// An incremental write context for a named HDF5 group.
///
/// Obtained from [`Hdf5FileBuilder::begin_group`][super::Hdf5FileBuilder::begin_group] or
/// [`SubGroupBuilder::begin_sub_group`].  Datasets are written one at a time
/// via [`add_dataset_with_attributes`][Self::add_dataset_with_attributes]; the
/// returned address is available immediately for use in subsequent attributes
/// (e.g. `DIMENSION_LIST`).  Nested sub-groups are opened with
/// [`begin_sub_group`][Self::begin_sub_group] and must be finished before
/// the parent is used again.
///
/// ## Invariants
///
/// - Datasets within this group are written in the order `add_dataset_with_attributes`
///   is called.
/// - `finish_with_attributes` must be called exactly once; it consumes `self`,
///   writes the group object header, and registers the group link in the parent.
/// - Nested `SubGroupBuilder` values borrow the parent exclusively (standard
///   Rust exclusion) and must be finished before the parent can be used.
#[cfg(feature = "alloc")]
pub struct SubGroupBuilder<'a> {
    pub(crate) sink: &'a mut consus_io::MemCursor,
    pub(crate) state: &'a mut WriteState,
    /// Link destination table in the parent (root_links or parent child_links).
    pub(crate) parent_links: &'a mut Vec<(String, u64)>,
    pub(crate) name: String,
    pub(crate) child_links: Vec<(String, u64)>,
}
#[cfg(feature = "alloc")]
impl<'a> SubGroupBuilder<'a> {
    /// Write a contiguous dataset with attached attributes into this group.
    ///
    /// The dataset is written immediately (data block + object header).
    /// Returns the object-header address, which can be used in subsequent
    /// `DIMENSION_LIST` attribute bytes for other datasets in the same group.
    pub fn add_dataset_with_attributes(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
        attributes: &[(&str, &Datatype, &Shape, &[u8])],
    ) -> Result<u64> {
        let data_addr = write_contiguous_data(&mut *self.sink, &mut *self.state, raw_data)?;

        let dt_bytes = encode_datatype(dt)?;
        let ds_bytes = encode_dataspace(shape)?;
        let layout_bytes = encode_layout(data_addr, dcpl, &self.state.ctx)?;

        let mut msgs: Vec<(u16, Vec<u8>)> = vec![
            (message_types::DATATYPE, dt_bytes),
            (message_types::DATASPACE, ds_bytes),
            (message_types::DATA_LAYOUT, layout_bytes),
        ];

        for (attr_name, attr_dt, attr_shape, attr_data) in attributes {
            msgs.push((
                message_types::ATTRIBUTE,
                encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?,
            ));
        }

        let msg_refs: Vec<(u16, &[u8])> = msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let header_addr = write_object_header_v2(&mut *self.sink, &mut *self.state, &msg_refs)?;
        self.child_links.push((String::from(name), header_addr));
        Ok(header_addr)
    }

    /// Open a nested sub-group write context within this group.
    ///
    /// The returned `SubGroupBuilder` borrows `self` exclusively.  It must be
    /// finished with [`finish_with_attributes`][SubGroupBuilder::finish_with_attributes]
    /// before `self` can be used again.
    pub fn begin_sub_group<'b>(&'b mut self, name: &str) -> SubGroupBuilder<'b> {
        SubGroupBuilder {
            sink: &mut *self.sink,
            state: &mut *self.state,
            parent_links: &mut self.child_links,
            name: String::from(name),
            child_links: Vec::new(),
        }
    }

    /// Finalise this group: write its object header (links + attributes) and
    /// register it as a child of the parent group.
    ///
    /// Consumes `self`.  After this call the parent group context is usable again.
    pub fn finish_with_attributes(
        self,
        group_attrs: &[(&str, &Datatype, &Shape, &[u8])],
    ) -> Result<()> {
        let ctx = self.state.ctx;
        // LINK_INFO and GROUP_INFO must precede LINK messages so libhdf5
        // recognises this object header as a group.
        let mut msgs: Vec<(u16, Vec<u8>)> = vec![
            (
                message_types::LINK_INFO,
                encode_link_info(ctx.offset_bytes()),
            ),
            (message_types::GROUP_INFO, encode_group_info()),
        ];

        for (child_name, child_addr) in &self.child_links {
            msgs.push((
                message_types::LINK,
                encode_hard_link(child_name, *child_addr, &ctx)?,
            ));
        }

        for (attr_name, attr_dt, attr_shape, attr_data) in group_attrs {
            msgs.push((
                message_types::ATTRIBUTE,
                encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?,
            ));
        }

        let msg_refs: Vec<(u16, &[u8])> = msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let group_addr = write_object_header_v2(&mut *self.sink, &mut *self.state, &msg_refs)?;
        self.parent_links.push((self.name, group_addr));
        Ok(())
    }
}
