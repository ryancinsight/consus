//! Dataset object-header assembly and contiguous data emission.

#[cfg(feature = "alloc")]
use super::{
    DatasetCreationProps, WriteState, dataset_filter_ids, encode_dataspace, encode_datatype,
    encode_filter_pipeline, encode_layout, encode_layout_with_chunk_index, write_object_header_v2,
};
#[cfg(feature = "alloc")]
use crate::object_header::message_types;
#[cfg(feature = "alloc")]
use crate::property_list::DatasetLayout;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use consus_core::{Datatype, Error, Result, Shape};
#[cfg(feature = "alloc")]
use consus_io::WriteAt;

// ---------------------------------------------------------------------------
// Dataset header
// ---------------------------------------------------------------------------

/// Write a v2 object header for a dataset.
///
/// Emits three header messages:
/// 1. Datatype (0x0003)
/// 2. Dataspace (0x0001)
/// 3. Data Layout (0x0008)
///
/// Returns the file address of the written object header.
#[cfg(feature = "alloc")]
pub fn write_dataset_header<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    datatype: &Datatype,
    shape: &Shape,
    data_address: u64,
    props: &DatasetCreationProps,
) -> Result<u64> {
    let dt_bytes = encode_datatype(datatype)?;
    let ds_bytes = encode_dataspace(shape)?;
    let filter_ids = dataset_filter_ids(props);
    let layout_bytes = match props.layout {
        DatasetLayout::Chunked => {
            let element_size =
                datatype
                    .element_size()
                    .ok_or_else(|| Error::UnsupportedFeature {
                        feature: String::from("chunked write requires fixed-size element datatype"),
                    })?;
            encode_layout_with_chunk_index(
                data_address,
                props,
                &state.ctx,
                Some(data_address),
                Some(element_size as u32),
            )?
        }
        _ => encode_layout(data_address, props, &state.ctx)?,
    };

    let filter_bytes = if filter_ids.is_empty() {
        None
    } else {
        Some(encode_filter_pipeline(&filter_ids, &props.compression)?)
    };

    let mut messages: Vec<(u16, Vec<u8>)> = vec![
        (message_types::DATATYPE, dt_bytes),
        (message_types::DATASPACE, ds_bytes),
        (message_types::DATA_LAYOUT, layout_bytes),
    ];

    if let Some(filter_bytes) = filter_bytes {
        messages.push((message_types::FILTER_PIPELINE, filter_bytes));
    }

    let msg_refs: Vec<(u16, &[u8])> = messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
    write_object_header_v2(sink, state, &msg_refs)
}

// ---------------------------------------------------------------------------
// Contiguous data write
// ---------------------------------------------------------------------------

/// Write raw contiguous data to the file.
///
/// Allocates space at the current EOF and writes `data` verbatim.
/// Returns the file address where the data was written.
#[cfg(feature = "alloc")]
pub fn write_contiguous_data<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    data: &[u8],
) -> Result<u64> {
    let addr = state.allocate_aligned(data.len() as u64);
    sink.write_at(addr, data)?;
    Ok(addr)
}
