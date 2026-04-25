//! Extract `NetcdfVariable` descriptors from HDF5 dataset objects and read
//! their full byte payloads through the HDF5 bridge.
//!
//! ## Spec
//!
//! A netCDF-4 variable maps 1-to-1 to an HDF5 dataset. The caller resolves
//! the dataset metadata, fill-value bytes, decoded HDF5 attributes, and
//! object-header address before calling `build_variable`; this module performs
//! the semantic mapping into the canonical netCDF variable descriptor.
//!
//! Full-byte reads are supported for:
//! - contiguous datasets via `Hdf5File::read_contiguous_dataset_bytes`
//! - chunked datasets via `Hdf5File::read_chunked_dataset_all_bytes`
//!
//! ## Invariants
//!
//! - `name` is the dataset name from the parent group traversal (non-empty).
//! - `datatype` and `shape` are taken directly from `Hdf5Dataset`.
//! - `fill_value` is stored as raw bytes; an empty byte vector is treated
//!   as absent (HDF5 fill-value messages may carry a zero-length payload).
//! - Decodable HDF5 attributes are attached to the variable, excluding
//!   dimension-scale marker attributes `CLASS` and `NAME`.
//! - `object_header_address`, when provided, is preserved on the variable for
//!   later data retrieval through the HDF5 bridge.
//! - `unlimited` is derived from `dataset.shape.has_unlimited()`.
//! - `dimension_names.len()` must equal `dataset.shape.rank()` for
//!   non-scalar datasets; pass an empty slice for rank-0 datasets.
//! - Full-byte reads require a fixed-size datatype.
//! - Compact and virtual layouts are rejected until a real bridge exists.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

use consus_core::{Error, Result};
use consus_hdf5::{
    attribute::{Hdf5Attribute, decode_attribute_value},
    dataset::{Hdf5Dataset, StorageLayout},
    file::Hdf5File,
};
use consus_io::ReadAt;

use crate::variable::NetcdfVariable;

/// Build a `NetcdfVariable` from resolved `Hdf5Dataset` metadata.
///
/// ## Parameters
///
/// - `name`: the dataset name as it appears in the parent group.
/// - `dataset`: the resolved dataset metadata from `file.dataset_at(addr)`.
/// - `fill_value_bytes`: optional raw fill-value bytes from
///   `file.fill_value_at(addr)`. An `Some(empty_vec)` is treated as
///   absent.
/// - `dimension_names`: ordered dimension names in the same order as
///   the shape dimensions. Pass `vec![]` for scalar datasets.
/// - `attrs`: HDF5 attributes attached to the dataset object header.
/// - `object_header_address`: optional HDF5 object-header address used
///   for later data retrieval.
///
/// ## Invariants
///
/// - The returned variable has `shape` set from `dataset.shape`.
/// - `fill_value` is `Some(bytes)` only when `fill_value_bytes` is
///   `Some(non_empty_bytes)`.
/// - Decodable attributes other than `CLASS` and `NAME` are attached.
/// - `unlimited` is derived from `dataset.shape.has_unlimited()`.
#[cfg(feature = "alloc")]
pub fn build_variable(
    name: String,
    dataset: &Hdf5Dataset,
    fill_value_bytes: Option<Vec<u8>>,
    dimension_names: Vec<String>,
    attrs: &[Hdf5Attribute],
    object_header_address: Option<u64>,
) -> NetcdfVariable {
    let decoded_attributes: Vec<(String, consus_core::AttributeValue)> = attrs
        .iter()
        .filter(|attr| attr.name != "CLASS" && attr.name != "NAME")
        .filter_map(|attr| {
            decode_attribute_value(&attr.raw_data, &attr.datatype, &attr.shape)
                .ok()
                .map(|value| (attr.name.clone(), value))
        })
        .collect();

    let mut var = NetcdfVariable::new(name, dataset.datatype.clone(), dimension_names)
        .with_shape(dataset.shape.clone())
        .with_attributes(decoded_attributes)
        .unlimited(dataset.shape.has_unlimited());

    if let Some(addr) = object_header_address {
        var = var.with_object_header_address(addr);
    }

    if let Some(fv) = fill_value_bytes {
        if !fv.is_empty() {
            var = var.with_fill_value(fv);
        }
    }

    var
}

/// Read the full raw byte payload for a netCDF variable backed by an HDF5 dataset.
///
/// ## Contract
///
/// - `variable.object_header_address` must be present.
/// - The referenced HDF5 dataset must use a fixed-size datatype.
/// - Contiguous datasets are read directly from their data address.
/// - Chunked datasets are assembled through the HDF5 chunk reader.
/// - Compact and virtual datasets are rejected until a real bridge exists.
#[cfg(feature = "std")]
pub fn read_variable_bytes<R>(file: &Hdf5File<R>, variable: &NetcdfVariable) -> Result<Vec<u8>>
where
    R: ReadAt + Sync,
{
    let object_header_address = variable.object_header_address.ok_or_else(|| {
        Error::InvalidFormat {
            message: String::from(
                "netCDF variable is missing the HDF5 object header address required for data reads",
            ),
        }
    })?;

    let dataset = file.dataset_at(object_header_address)?;
    let element_size =
        dataset
            .datatype
            .element_size()
            .ok_or_else(|| Error::UnsupportedFeature {
                feature: String::from(
                    "netCDF variable byte reads require a fixed-size HDF5 datatype",
                ),
            })?;

    match dataset.layout {
        StorageLayout::Contiguous => {
            let data_address = dataset.data_address.ok_or_else(|| Error::InvalidFormat {
                message: String::from("contiguous HDF5 dataset is missing its data address"),
            })?;
            let total_bytes = dataset.shape.num_elements() * element_size;
            let mut buf = vec![0u8; total_bytes];
            file.read_contiguous_dataset_bytes(data_address, 0, &mut buf)?;
            Ok(buf)
        }
        StorageLayout::Chunked => file.read_chunked_dataset_all_bytes(object_header_address),
        StorageLayout::Compact => Err(Error::UnsupportedFeature {
            feature: String::from(
                "netCDF variable byte reads for compact HDF5 datasets are not implemented",
            ),
        }),
        StorageLayout::Virtual => Err(Error::UnsupportedFeature {
            feature: String::from(
                "netCDF variable byte reads for virtual HDF5 datasets are not implemented",
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use consus_core::{ByteOrder, Datatype, Shape};
    use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
    use consus_io::SliceReader;
    use core::num::NonZeroUsize;

    fn f32_dt() -> Datatype {
        Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        }
    }

    fn i32_dt() -> Datatype {
        Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }
    }

    fn make_dataset(shape: Shape) -> Hdf5Dataset {
        Hdf5Dataset {
            path: String::from("/test"),
            object_header_address: 0,
            datatype: f32_dt(),
            shape,
            layout: consus_hdf5::dataset::StorageLayout::Contiguous,
            chunk_shape: None,
            data_address: None,
            filters: Vec::new(),
        }
    }

    #[test]
    fn build_variable_sets_name_and_datatype() {
        let ds = make_dataset(Shape::fixed(&[10usize]));
        let var = build_variable(
            String::from("temperature"),
            &ds,
            None,
            vec![String::from("time")],
            &[] as &[Hdf5Attribute],
            None,
        );
        assert_eq!(var.name, "temperature");
        assert_eq!(var.datatype, f32_dt());
        assert_eq!(var.dimensions, vec![String::from("time")]);
    }

    #[test]
    fn build_variable_attaches_shape() {
        let ds = make_dataset(Shape::fixed(&[5usize, 3usize]));
        let var = build_variable(
            String::from("data"),
            &ds,
            None,
            vec![String::from("time"), String::from("station")],
            &[] as &[Hdf5Attribute],
            None,
        );
        let shape = var.shape.expect("shape must be set");
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.current_dims().as_slice(), &[5, 3]);
    }

    #[test]
    fn build_variable_stores_fill_value() {
        let fill = vec![0x00u8, 0x00, 0x80, 0x3F]; // 1.0f32 LE
        let ds = make_dataset(Shape::fixed(&[4usize]));
        let var = build_variable(
            String::from("v"),
            &ds,
            Some(fill.clone()),
            vec![String::from("x")],
            &[] as &[Hdf5Attribute],
            None,
        );
        assert_eq!(var.fill_value, Some(fill));
    }

    #[test]
    fn build_variable_ignores_empty_fill_value() {
        let ds = make_dataset(Shape::fixed(&[4usize]));
        let var = build_variable(
            String::from("v"),
            &ds,
            Some(vec![]),
            vec![String::from("x")],
            &[] as &[Hdf5Attribute],
            None,
        );
        assert!(var.fill_value.is_none());
    }

    #[test]
    fn build_variable_scalar_has_no_dimensions() {
        let ds = make_dataset(Shape::scalar());
        let var = build_variable(
            String::from("scalar"),
            &ds,
            None,
            vec![],
            &[] as &[Hdf5Attribute],
            None,
        );
        assert!(var.is_scalar());
        assert_eq!(var.rank(), 0);
    }

    #[test]
    fn read_variable_bytes_reads_contiguous_dataset_payload() {
        let raw: Vec<u8> = (0i32..6).flat_map(|v| v.to_le_bytes()).collect();
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let dataset_addr = builder
            .add_dataset(
                "temperature",
                &i32_dt(),
                &Shape::fixed(&[6usize]),
                &raw,
                &DatasetCreationProps::default(),
            )
            .expect("add dataset");
        let bytes = builder.finish().expect("finish");

        let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
        let dataset = file.dataset_at(dataset_addr).expect("dataset");
        let variable = build_variable(
            String::from("temperature"),
            &dataset,
            None,
            vec![String::from("time")],
            &[] as &[Hdf5Attribute],
            Some(dataset_addr),
        );

        let read = read_variable_bytes(&file, &variable).expect("read variable bytes");
        assert_eq!(read, raw);
    }

    #[test]
    fn read_variable_bytes_reads_chunked_dataset_payload() {
        let raw: Vec<u8> = (0i32..16).flat_map(|v| v.to_le_bytes()).collect();
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let dataset_addr = builder
            .add_dataset(
                "temperature",
                &i32_dt(),
                &Shape::fixed(&[4usize, 4usize]),
                &raw,
                &DatasetCreationProps {
                    layout: consus_hdf5::property_list::DatasetLayout::Chunked,
                    chunk_dims: Some(vec![2, 2]),
                    ..DatasetCreationProps::default()
                },
            )
            .expect("add dataset");
        let bytes = builder.finish().expect("finish");

        let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");
        let dataset = file.dataset_at(dataset_addr).expect("dataset");
        let variable = build_variable(
            String::from("temperature"),
            &dataset,
            None,
            vec![String::from("y"), String::from("x")],
            &[] as &[Hdf5Attribute],
            Some(dataset_addr),
        );

        let read = read_variable_bytes(&file, &variable).expect("read variable bytes");
        assert_eq!(read, raw);
    }

    #[test]
    fn read_variable_bytes_rejects_missing_object_header_address() {
        let variable = NetcdfVariable::new(
            String::from("temperature"),
            i32_dt(),
            vec![String::from("time")],
        )
        .with_shape(Shape::fixed(&[4usize]));

        let bytes = Hdf5FileBuilder::new(FileCreationProps::default())
            .finish()
            .expect("finish");
        let file = Hdf5File::open(SliceReader::new(&bytes)).expect("open");

        let err = read_variable_bytes(&file, &variable).expect_err("must reject missing address");
        match err {
            Error::InvalidFormat { message } => {
                assert!(
                    message.contains("missing the HDF5 object header address"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }
}
