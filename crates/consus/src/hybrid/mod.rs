//! Hybrid Parquet-in-Consus container embedding.
//!
//! Provides the canonical mechanism to embed a Parquet physical payload
//! into a hierarchical Consus container, complete with metadata annotations.

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec::Vec};

#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};

#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
use core::num::NonZeroUsize;

#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
use consus_hdf5::file::writer::{DatasetCreationProps, Hdf5FileBuilder};

#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
use consus_parquet::HybridStorageDescriptor;

#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
use consus_hdf5::file::Hdf5File;

/// Encodes a `&str` into a `FixedString` datatype and byte payload.
#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
fn encode_string_attr(s: &str) -> (Datatype, Shape, Vec<u8>) {
    let bytes = s.as_bytes().to_vec();
    let dt = Datatype::FixedString {
        length: bytes.len().max(1),
        encoding: StringEncoding::Utf8,
    };
    (dt, Shape::scalar(), if bytes.is_empty() { vec![0] } else { bytes })
}

/// Helper struct for managing multiple encoded attributes.
#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
struct AttributeStorage {
    buffers: Vec<(String, Datatype, Shape, Vec<u8>)>,
}

#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
impl AttributeStorage {
    fn new() -> Self {
        Self { buffers: Vec::new() }
    }

    fn push(&mut self, name: &str, value: &str) {
        let (dt, shape, data) = encode_string_attr(value);
        self.buffers.push((String::from(name), dt, shape, data));
    }

    fn as_refs(&self) -> Vec<(&str, &Datatype, &Shape, &[u8])> {
        self.buffers
            .iter()
            .map(|(n, dt, sh, d)| (n.as_str(), dt, sh, d.as_slice()))
            .collect()
    }
}

/// Embeds a Parquet payload into an HDF5 file builder at the root group.
///
/// Converts the `ParquetWriter` output to a 1D byte array dataset, adding
/// `HybridStorageDescriptor` metadata via HDF5 attributes.
#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
pub fn write_embedded_parquet(
    builder: &mut Hdf5FileBuilder,
    parquet_payload: &[u8],
    descriptor: &HybridStorageDescriptor,
    dataset_name: &str,
) -> consus_core::Result<u64> {

    let mut attrs = AttributeStorage::new();
    attrs.push("consus_hybrid_mode", "Embedded");
    
    if let Some(ref layout) = descriptor.table_layout {
        attrs.push("consus_hybrid_table_name", &layout.table_name);
        attrs.push("consus_hybrid_payload_path", &layout.payload_path);
        attrs.push("consus_hybrid_relation", &format!("{:?}", layout.relation));
        attrs.push("consus_hybrid_encoding", &format!("{:?}", layout.encoding));
    }
    
    if let Some(ref part) = descriptor.partitioning {
        if part.is_partitioned() {
            attrs.push("consus_hybrid_partition_keys", &part.keys.join(","));
            attrs.push("consus_hybrid_partition_paths", &part.paths.join(","));
        }
    }

    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(8).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let shape = Shape::fixed(&[parquet_payload.len()]);
    let dcpl = DatasetCreationProps::default();

    let attr_refs = attrs.as_refs();
    
    builder.add_dataset_with_attributes(
        dataset_name,
        &dt,
        &shape,
        parquet_payload,
        &dcpl,
        &attr_refs,
    )
}

/// Reads an embedded Parquet payload from an HDF5 file and reconstructs the `ParquetReader`.
///
/// Also extracts the `HybridStorageDescriptor` metadata attributes.
#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
pub fn read_embedded_parquet<R: consus_io::ReadAt + Sync>(
    file: &Hdf5File<R>,
    path: &str,
) -> consus_core::Result<(Vec<u8>, HybridStorageDescriptor)> {
    use consus_core::{AttributeValue, Error};
    use consus_parquet::{
        HybridPartitioning, HybridTableLayout, HybridStorageEncoding,
        HybridTableRelation, HybridMode,
    };

    let addr = file.open_path(path)?;
    let dataset = file.dataset_at(addr)?;
    
    if dataset.shape.rank() != 1 {
        return Err(Error::InvalidFormat {
            message: String::from("Embedded Parquet dataset must be a 1D byte array"),
        });
    }

    let mut payload = vec![0u8; dataset.shape.num_elements()];
    file.read_contiguous_dataset_bytes(dataset.data_address.unwrap_or(0), 0, &mut payload)?;

    let attrs = file.attributes_at(addr)?;
    
    let mut descriptor = HybridStorageDescriptor::default();
    descriptor.mode = HybridMode::Embedded;
    
    let mut table_name = String::new();
    let mut payload_path = String::new();
    let mut relation = HybridTableRelation::Primary;
    let mut encoding = HybridStorageEncoding::ColumnarParquet;
    let mut has_layout = false;

    let mut part_keys = Vec::new();
    let mut part_paths = Vec::new();
    let mut has_part = false;

    for attr in attrs {
        if let Ok(AttributeValue::String(val)) = attr.decode_value() {
            match attr.name.as_str() {
                "consus_hybrid_table_name" => {
                    table_name = val;
                    has_layout = true;
                }
                "consus_hybrid_payload_path" => {
                    payload_path = val;
                    has_layout = true;
                }
                "consus_hybrid_relation" => {
                    has_layout = true;
                    if val == "MaterializedView" { relation = HybridTableRelation::MaterializedView; }
                    else if val == "Cache" { relation = HybridTableRelation::Cache; }
                    else if val == "Partition" { relation = HybridTableRelation::Partition; }
                }
                "consus_hybrid_encoding" => {
                    has_layout = true;
                    if val == "RowGroupSegmentedParquet" { encoding = HybridStorageEncoding::RowGroupSegmentedParquet; }
                    else if val == "ArrowIntermediate" { encoding = HybridStorageEncoding::ArrowIntermediate; }
                }
                "consus_hybrid_partition_keys" => {
                    part_keys = val.split(',').filter(|s| !s.is_empty()).map(String::from).collect();
                    has_part = true;
                }
                "consus_hybrid_partition_paths" => {
                    part_paths = val.split(',').filter(|s| !s.is_empty()).map(String::from).collect();
                    has_part = true;
                }
                _ => {}
            }
        }
    }

    if has_layout {
        descriptor.table_layout = Some(HybridTableLayout::new(table_name, payload_path, relation, encoding));
    }
    if has_part {
        descriptor.partitioning = Some(HybridPartitioning { keys: part_keys, paths: part_paths });
    }

    Ok((payload, descriptor))
}
