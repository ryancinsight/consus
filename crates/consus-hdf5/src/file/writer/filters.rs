//! Filter list ordering and the v1 filter-pipeline message encoder.

#[cfg(feature = "alloc")]
use super::DatasetCreationProps;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_core::{Compression, Result};

#[cfg(feature = "alloc")]
const DEFLATE_FILTER_ID: u16 = 1;
#[cfg(feature = "alloc")]
const FLETCHER32_FILTER_ID: u16 = 3;
#[cfg(feature = "alloc")]
const LZ4_FILTER_ID: u16 = 32004;
#[cfg(feature = "alloc")]
const ZSTD_FILTER_ID: u16 = 32015;

#[cfg(feature = "alloc")]
fn compression_filter_id(compression: &Compression) -> Option<u16> {
    match compression {
        Compression::None => None,
        Compression::Deflate { .. } | Compression::Gzip { .. } => Some(DEFLATE_FILTER_ID),
        Compression::Zstd { .. } => Some(ZSTD_FILTER_ID),
        Compression::Lz4 => Some(LZ4_FILTER_ID),
    }
}

#[cfg(feature = "alloc")]
fn deflate_level(compression: &Compression) -> u32 {
    match compression {
        Compression::Deflate { level } | Compression::Gzip { level } => *level,
        _ => 6,
    }
}

#[cfg(feature = "alloc")]
fn push_unique(filter_ids: &mut Vec<u16>, filter_id: u16) {
    if !filter_ids.contains(&filter_id) {
        filter_ids.push(filter_id);
    }
}

#[cfg(feature = "alloc")]
fn is_compression_filter(filter_id: u16) -> bool {
    matches!(
        filter_id,
        DEFLATE_FILTER_ID | ZSTD_FILTER_ID | LZ4_FILTER_ID
    )
}

#[cfg(feature = "alloc")]
pub(crate) fn dataset_filter_ids(props: &DatasetCreationProps) -> Vec<u16> {
    let mut integrity_filters = Vec::new();
    let mut transform_filters = Vec::new();
    let mut compression_filters = Vec::new();

    if let Some(filter_id) = compression_filter_id(&props.compression) {
        compression_filters.push(filter_id);
    }

    for &filter_id in &props.filters {
        if filter_id == FLETCHER32_FILTER_ID {
            push_unique(&mut integrity_filters, filter_id);
        } else if is_compression_filter(filter_id) {
            push_unique(&mut compression_filters, filter_id);
        } else if !transform_filters.contains(&filter_id) {
            transform_filters.push(filter_id);
        }
    }

    integrity_filters
        .into_iter()
        .chain(transform_filters)
        .chain(compression_filters)
        .collect()
}

#[cfg(feature = "alloc")]
pub(crate) fn encode_filter_pipeline(
    filter_ids: &[u16],
    compression: &Compression,
) -> Result<Vec<u8>> {
    // Filter pipeline message version 1. The HDF5 C library (1.14.x) does not
    // correctly extract cd_nelmts from version 2 pipelines: H5Pget_filter2 returns
    // cd_nelmts=0 regardless of the encoded value, so H5Z__filter_deflate never
    // receives the compression level and fails during read. Version 1 is what the
    // HDF5 C library itself writes (via h5py) and reads correctly.
    //
    // Version 1 layout (HDF5 spec §IV.A.2.j):
    //   Header (8 bytes): version(1)=1, nfilters(1)=N, reserved(6)=0
    //   Per-filter entry (each padded to 8-byte boundary):
    //     filter_id (2), name_len (2, includes null), flags (2), nclient (2)
    //     name (padded to 8-byte boundary, null-terminated)
    //     client_data (4 * nclient bytes)
    //     padding to 8-byte boundary

    let mut buf = vec![
        1u8,                    // version = 1
        filter_ids.len() as u8, // nfilters
        0,
        0,
        0,
        0,
        0,
        0, // 6 reserved bytes
    ];

    for &filter_id in filter_ids {
        let (name, client_data_owned, flags): (&[u8], alloc::vec::Vec<u32>, u16) = match filter_id {
            DEFLATE_FILTER_ID => {
                // Deflate filter: name="deflate\0" (8 bytes), flags=H5Z_FLAG_OPTIONAL(1),
                // one client data element = compression level.
                let level = deflate_level(compression);
                (b"deflate\0", alloc::vec![level], 1u16)
            }
            _ => (b"", alloc::vec::Vec::new(), 0u16),
        };

        let name_len = name.len() as u16;
        // Name must be padded up to the next 8-byte boundary.
        let name_padded_len = if name.is_empty() {
            0
        } else {
            name.len().div_ceil(8) * 8
        };
        let nclient = client_data_owned.len() as u16;

        let entry_start = buf.len();

        // Entry header: 8 bytes (filter_id + name_len + flags + nclient)
        let header_start = buf.len();
        buf.resize(header_start + 8, 0);
        LittleEndian::write_u16(&mut buf[header_start..header_start + 2], filter_id);
        LittleEndian::write_u16(&mut buf[header_start + 2..header_start + 4], name_len);
        LittleEndian::write_u16(&mut buf[header_start + 4..header_start + 6], flags);
        LittleEndian::write_u16(&mut buf[header_start + 6..header_start + 8], nclient);

        // Name field (padded to 8-byte boundary).
        if !name.is_empty() {
            buf.extend_from_slice(name);
            buf.resize(entry_start + 8 + name_padded_len, 0);
        }

        // Client data values.
        for &v in &client_data_owned {
            let mut tmp = [0u8; 4];
            LittleEndian::write_u32(&mut tmp, v);
            buf.extend_from_slice(&tmp);
        }

        // Pad entire entry to 8-byte boundary.
        while buf.len() % 8 != 0 {
            buf.push(0);
        }
    }

    Ok(buf)
}
