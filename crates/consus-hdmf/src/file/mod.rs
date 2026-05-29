//! HDF5-backed reader and writer for HDMF DynamicTable objects.
//!
//! ## Reader — [`HdmfFile`]
//!
//! Opens an HDF5 byte slice, validates that the root group carries
//! `data_type = "DynamicTable"`, and exposes [`HdmfFile::read_table`] to
//! extract the full [`DynamicTable`] with all columns.
//!
//! ## Writer — [`HdmfFileBuilder`]
//!
//! Accumulates columns via the builder API, then serialises a conformant
//! HDMF HDF5 image with the required `VectorData`, `ElementIdentifiers`, and
//! optional `VectorIndex` datasets and all mandatory root-group attributes.
//!
//! ## HDF5 file layout written by [`HdmfFileBuilder`]
//!
//! ```text
//! / (root group)
//!   attrs: data_type="DynamicTable", namespace="hdmf-common",
//!          description=..., colnames=[...], object_id=UUID
//!   id          — int64 1-D  [ElementIdentifiers]
//!   <col>       — typed 1-D  [VectorData]
//!   <col>_index — uint64 1-D [VectorIndex]  (ragged columns only)
//! ```

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec, vec::Vec};

use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype, Result, Shape, StringEncoding};
use consus_hdf5::file::writer::Hdf5FileBuilder;
use consus_hdf5::file::Hdf5File;
use consus_hdf5::property_list::{DatasetCreationProps, FileCreationProps};
use consus_io::SliceReader;

use crate::storage::{
    detect_column_data, read_i64_dataset, read_string_array_attr_any, read_string_attr_any,
    read_u64_dataset,
};
use crate::table::{Column, ColumnData, DynamicTable};

// ---------------------------------------------------------------------------
// HDMF namespace constants
// ---------------------------------------------------------------------------

const HDMF_COMMON_NS: &str = "hdmf-common";
const TYPE_DYNAMIC_TABLE: &str = "DynamicTable";
const TYPE_VECTOR_DATA: &str = "VectorData";
const TYPE_VECTOR_INDEX: &str = "VectorIndex";
const TYPE_ELEMENT_IDENTIFIERS: &str = "ElementIdentifiers";

// ---------------------------------------------------------------------------
// UUID generation (deterministic, no external deps)
// ---------------------------------------------------------------------------

/// Generate a UUID-format string deterministic on `counter`.
///
/// Format: `{counter:08x}-0000-4000-8000-{counter:012x}`
fn make_object_id(counter: u32) -> String {
    format!(
        "{:08x}-0000-4000-8000-{:012x}",
        counter, counter
    )
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

/// Encode a fixed-length ASCII string attribute value.
#[cfg(feature = "alloc")]
fn fixed_string_bytes(value: &str) -> (Datatype, Vec<u8>) {
    let len = value.len().max(1);
    let dt = Datatype::FixedString {
        length: len,
        encoding: StringEncoding::Ascii,
    };
    let mut raw = value.as_bytes().to_vec();
    raw.resize(len, 0u8);
    (dt, raw)
}

/// Encode a 1-D fixed-string attribute for the `colnames` array.
///
/// Each entry is padded to the maximum column-name length so all elements
/// share a uniform stride.
#[cfg(feature = "alloc")]
fn colnames_attr_bytes(colnames: &[String]) -> (Datatype, Shape, Vec<u8>) {
    let max_len = colnames.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    let dt = Datatype::FixedString {
        length: max_len,
        encoding: StringEncoding::Ascii,
    };
    let shape = Shape::fixed(&[colnames.len()]);
    let mut raw = vec![0u8; colnames.len() * max_len];
    for (i, name) in colnames.iter().enumerate() {
        let src = name.as_bytes();
        let start = i * max_len;
        raw[start..start + src.len().min(max_len)].copy_from_slice(&src[..src.len().min(max_len)]);
    }
    (dt, shape, raw)
}

/// Integer datatype helper (little-endian, signed).
fn int64_le() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        signed: true,
        byte_order: ByteOrder::LittleEndian,
    }
}

/// Unsigned 64-bit integer datatype helper (little-endian).
fn uint64_le() -> Datatype {
    Datatype::Integer {
        bits: NonZeroUsize::new(64).unwrap(),
        signed: false,
        byte_order: ByteOrder::LittleEndian,
    }
}

/// Float 64-bit datatype helper (little-endian).
fn float64_le() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Reader for an HDMF DynamicTable stored as the root object of an HDF5 file.
///
/// ## Lifetime
///
/// `'a` binds the reader to the byte slice it was opened from.
pub struct HdmfFile<'a> {
    hdf5: Hdf5File<SliceReader<'a>>,
}

impl core::fmt::Debug for HdmfFile<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HdmfFile").finish_non_exhaustive()
    }
}

#[cfg(feature = "alloc")]
impl<'a> HdmfFile<'a> {
    /// Open an HDMF file from a byte slice.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` when the root group does not carry
    /// `data_type = "DynamicTable"`.
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        let reader = SliceReader::new(bytes);
        let hdf5 = Hdf5File::open(reader)?;
        Ok(Self { hdf5 })
    }

    /// Read the DynamicTable from the root group of the opened HDF5 file.
    ///
    /// Reads the root-group attributes for `data_type`, `description`,
    /// `colnames`, and `object_id`.  Then reads the `id` dataset and all
    /// column datasets listed in `colnames`, including their optional
    /// `VectorIndex` companions (`<col>_index`).
    ///
    /// ## Errors
    ///
    /// - `Error::InvalidFormat` — root group has no `data_type` attribute or
    ///   it is not `"DynamicTable"`.
    /// - `Error::NotFound` — a column listed in `colnames` is absent.
    /// - Propagates HDF5 I/O errors.
    pub fn read_table(&self) -> Result<DynamicTable> {
        let root_addr = self.hdf5.superblock().root_group_address;
        let attrs = self.hdf5.attributes_at(root_addr)?;

        let data_type = read_string_attr_any(&attrs, "data_type", &self.hdf5)
            .unwrap_or_default();
        if data_type != TYPE_DYNAMIC_TABLE {
            return Err(consus_core::Error::InvalidFormat {
                message: format!(
                    "HDMF: root data_type is '{}', expected 'DynamicTable'",
                    data_type
                ),
            });
        }

        let description = read_string_attr_any(&attrs, "description", &self.hdf5)
            .unwrap_or_default();
        let colnames = read_string_array_attr_any(&attrs, "colnames", &self.hdf5)
            .unwrap_or_default();

        // Row IDs
        let id: Vec<i64> = match self.hdf5.open_path("id") {
            Ok(addr) => read_i64_dataset(&self.hdf5, addr)?,
            Err(_) => vec![],
        };

        // Columns
        let mut columns: Vec<Column> = Vec::with_capacity(colnames.len());
        for col_name in &colnames {
            let col_addr = self.hdf5.open_path(col_name)?;
            let col_attrs = self.hdf5.attributes_at(col_addr)?;
            let col_description =
                read_string_attr_any(&col_attrs, "description", &self.hdf5).unwrap_or_default();

            let data = detect_column_data(&self.hdf5, col_addr)?;

            // Look for a VectorIndex companion: `{name}_index`
            let index_path = format!("{}_index", col_name);
            let index: Option<Vec<u64>> = match self.hdf5.open_path(&index_path) {
                Ok(idx_addr) => {
                    let idx_attrs = self.hdf5.attributes_at(idx_addr)?;
                    let idx_dt =
                        read_string_attr_any(&idx_attrs, "data_type", &self.hdf5).unwrap_or_default();
                    if idx_dt == TYPE_VECTOR_INDEX {
                        Some(read_u64_dataset(&self.hdf5, idx_addr)?)
                    } else {
                        None
                    }
                }
                Err(_) => None,
            };

            columns.push(Column {
                name: col_name.clone(),
                description: col_description,
                data,
                index,
            });
        }

        Ok(DynamicTable {
            name: String::from("root"),
            description,
            colnames,
            id,
            columns,
        })
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Builder for an HDF5 file containing a single HDMF DynamicTable at root.
///
/// Columns are accumulated with [`add_column`][Self::add_column] and
/// [`add_ragged_column`][Self::add_ragged_column].  The table is serialised
/// by calling [`finish`][Self::finish].
///
/// ## Example
///
/// ```no_run
/// use consus_hdmf::{ColumnData, HdmfFileBuilder};
///
/// let bytes = HdmfFileBuilder::new("trials", "trial metadata")
///     .add_column("start_time", "trial onset (s)", ColumnData::F64(vec![0.0, 1.0]))
///     .add_column("correct", "whether trial was correct", ColumnData::Bool(vec![true, false]))
///     .finish()
///     .unwrap();
/// ```
#[cfg(feature = "alloc")]
pub struct HdmfFileBuilder {
    name: String,
    description: String,
    /// (name, description, data, cumulative_index)
    columns: Vec<(String, String, ColumnData, Option<Vec<u64>>)>,
}

#[cfg(feature = "alloc")]
impl core::fmt::Debug for HdmfFileBuilder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HdmfFileBuilder")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("columns", &self.columns.len())
            .finish()
    }
}

#[cfg(feature = "alloc")]
impl HdmfFileBuilder {
    /// Create a new builder for a DynamicTable with the given `name` and `description`.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            columns: Vec::new(),
        }
    }

    /// Add a dense column.
    ///
    /// Consumes and returns `self` to allow method chaining.
    pub fn add_column(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        data: ColumnData,
    ) -> Self {
        self.columns.push((name.into(), description.into(), data, None));
        self
    }

    /// Add a ragged (variable-length) column with a cumulative `VectorIndex`.
    ///
    /// `index` must have length equal to the number of rows; each element is
    /// the exclusive end position in `data` for that row.
    pub fn add_ragged_column(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        data: ColumnData,
        index: Vec<u64>,
    ) -> Self {
        self.columns.push((name.into(), description.into(), data, Some(index)));
        self
    }

    /// Serialise the DynamicTable into a complete HDF5 byte image.
    ///
    /// ## Errors
    ///
    /// Returns an error if any column data cannot be encoded or if the HDF5
    /// writer encounters a structural error.
    pub fn finish(self) -> Result<Vec<u8>> {
        let mut hdf5 = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        let dcpl = DatasetCreationProps::default();

        let n_rows: usize = self
            .columns
            .first()
            .map(|(_, _, data, _)| data.len())
            .unwrap_or(0);

        // --- id dataset (ElementIdentifiers) ---
        let id_dt = int64_le();
        let id_raw: Vec<u8> = (0i64..n_rows as i64)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let id_shape = Shape::fixed(&[n_rows]);
        let id_uuid = make_object_id(0);
        let (id_dt_attr, id_dt_raw) = fixed_string_bytes(TYPE_ELEMENT_IDENTIFIERS);
        let (id_ns_dt, id_ns_raw) = fixed_string_bytes(HDMF_COMMON_NS);
        let (id_oid_dt, id_oid_raw) = fixed_string_bytes(&id_uuid);
        hdf5.add_dataset_with_attributes(
            "id",
            &id_dt,
            &id_shape,
            &id_raw,
            &dcpl,
            &[
                ("data_type", &id_dt_attr, &scalar, &id_dt_raw),
                ("namespace", &id_ns_dt, &scalar, &id_ns_raw),
                ("object_id", &id_oid_dt, &scalar, &id_oid_raw),
            ],
        )?;

        // --- column datasets ---
        let mut col_counter: u32 = 1;
        for (col_name, col_desc, col_data, col_index) in &self.columns {
            let (dtype, raw_bytes, shape) = encode_column_data(col_data)?;
            let col_uuid = make_object_id(col_counter);
            col_counter += 1;

            let (vd_dt, vd_raw) = fixed_string_bytes(TYPE_VECTOR_DATA);
            let (ns_dt, ns_raw) = fixed_string_bytes(HDMF_COMMON_NS);
            let (desc_dt, desc_raw) = fixed_string_bytes(col_desc);
            let (oid_dt, oid_raw) = fixed_string_bytes(&col_uuid);

            hdf5.add_dataset_with_attributes(
                col_name,
                &dtype,
                &shape,
                &raw_bytes,
                &dcpl,
                &[
                    ("data_type", &vd_dt, &scalar, &vd_raw),
                    ("namespace", &ns_dt, &scalar, &ns_raw),
                    ("description", &desc_dt, &scalar, &desc_raw),
                    ("object_id", &oid_dt, &scalar, &oid_raw),
                ],
            )?;

            // --- VectorIndex (ragged column) ---
            if let Some(index) = col_index {
                let idx_name = format!("{}_index", col_name);
                let idx_uuid = make_object_id(col_counter);
                col_counter += 1;

                let idx_dt = uint64_le();
                let idx_raw: Vec<u8> = index.iter().flat_map(|v| v.to_le_bytes()).collect();
                let idx_shape = Shape::fixed(&[index.len()]);

                let idx_desc_str =
                    format!("Index for VectorData '{}'", col_name);
                let (vi_dt, vi_raw) = fixed_string_bytes(TYPE_VECTOR_INDEX);
                let (vi_ns_dt, vi_ns_raw) = fixed_string_bytes(HDMF_COMMON_NS);
                let (vi_desc_dt, vi_desc_raw) = fixed_string_bytes(&idx_desc_str);
                let (vi_oid_dt, vi_oid_raw) = fixed_string_bytes(&idx_uuid);

                hdf5.add_dataset_with_attributes(
                    &idx_name,
                    &idx_dt,
                    &idx_shape,
                    &idx_raw,
                    &dcpl,
                    &[
                        ("data_type", &vi_dt, &scalar, &vi_raw),
                        ("namespace", &vi_ns_dt, &scalar, &vi_ns_raw),
                        ("description", &vi_desc_dt, &scalar, &vi_desc_raw),
                        ("object_id", &vi_oid_dt, &scalar, &vi_oid_raw),
                    ],
                )?;
            }
        }

        // --- root group attributes ---
        let colnames: Vec<String> = self.columns.iter().map(|(n, _, _, _)| n.clone()).collect();
        let table_uuid = make_object_id(col_counter);

        let (dt_dt, dt_raw) = fixed_string_bytes(TYPE_DYNAMIC_TABLE);
        hdf5.add_root_attribute("data_type", &dt_dt, &scalar, &dt_raw)?;

        let (ns_dt, ns_raw) = fixed_string_bytes(HDMF_COMMON_NS);
        hdf5.add_root_attribute("namespace", &ns_dt, &scalar, &ns_raw)?;

        let (desc_dt, desc_raw) = fixed_string_bytes(&self.description);
        hdf5.add_root_attribute("description", &desc_dt, &scalar, &desc_raw)?;

        let (oid_dt, oid_raw) = fixed_string_bytes(&table_uuid);
        hdf5.add_root_attribute("object_id", &oid_dt, &scalar, &oid_raw)?;

        // colnames as 1-D fixed-string array
        if !colnames.is_empty() {
            let (cn_dt, cn_shape, cn_raw) = colnames_attr_bytes(&colnames);
            hdf5.add_root_attribute("colnames", &cn_dt, &cn_shape, &cn_raw)?;
        } else {
            // Empty colnames: write a zero-element FixedString(1) array
            let empty_dt = Datatype::FixedString {
                length: 1,
                encoding: StringEncoding::Ascii,
            };
            let empty_shape = Shape::fixed(&[0]);
            hdf5.add_root_attribute("colnames", &empty_dt, &empty_shape, &[])?;
        }

        hdf5.finish()
    }
}

// ---------------------------------------------------------------------------
// Column encoding
// ---------------------------------------------------------------------------

/// Encode column data into (Datatype, raw_bytes, Shape).
#[cfg(feature = "alloc")]
fn encode_column_data(data: &ColumnData) -> Result<(Datatype, Vec<u8>, Shape)> {
    match data {
        ColumnData::F64(v) => {
            let dtype = float64_le();
            let raw: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            let shape = Shape::fixed(&[v.len()]);
            Ok((dtype, raw, shape))
        }
        ColumnData::I64(v) => {
            let dtype = int64_le();
            let raw: Vec<u8> = v.iter().flat_map(|i| i.to_le_bytes()).collect();
            let shape = Shape::fixed(&[v.len()]);
            Ok((dtype, raw, shape))
        }
        ColumnData::U64(v) => {
            let dtype = uint64_le();
            let raw: Vec<u8> = v.iter().flat_map(|u| u.to_le_bytes()).collect();
            let shape = Shape::fixed(&[v.len()]);
            Ok((dtype, raw, shape))
        }
        ColumnData::Bool(v) => {
            let dtype = Datatype::Boolean;
            let raw: Vec<u8> = v.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
            let shape = Shape::fixed(&[v.len()]);
            Ok((dtype, raw, shape))
        }
        ColumnData::Str(v) => {
            let max_len = v.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
            let dtype = Datatype::FixedString {
                length: max_len,
                encoding: StringEncoding::Ascii,
            };
            let mut raw = vec![0u8; v.len() * max_len];
            for (i, s) in v.iter().enumerate() {
                let src = s.as_bytes();
                let n = src.len().min(max_len);
                raw[i * max_len..i * max_len + n].copy_from_slice(&src[..n]);
            }
            let shape = Shape::fixed(&[v.len()]);
            Ok((dtype, raw, shape))
        }
    }
}
