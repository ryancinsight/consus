use alloc::{format, string::String, vec, vec::Vec};

use consus_core::Result;
use consus_hdf5::file::Hdf5File;
use consus_io::SliceReader;

use super::{TYPE_DYNAMIC_TABLE, TYPE_VECTOR_INDEX};
use crate::storage::{
    detect_column_data, read_i64_dataset, read_string_array_attr_any, read_string_attr_any,
    read_u64_dataset,
};
use crate::table::{Column, DynamicTable};

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

        let data_type = read_string_attr_any(&attrs, "data_type", &self.hdf5).unwrap_or_default();
        if data_type != TYPE_DYNAMIC_TABLE {
            return Err(consus_core::Error::InvalidFormat {
                message: format!(
                    "HDMF: root data_type is '{}', expected 'DynamicTable'",
                    data_type
                ),
            });
        }

        let description =
            read_string_attr_any(&attrs, "description", &self.hdf5).unwrap_or_default();
        let colnames =
            read_string_array_attr_any(&attrs, "colnames", &self.hdf5).unwrap_or_default();

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
                    let idx_dt = read_string_attr_any(&idx_attrs, "data_type", &self.hdf5)
                        .unwrap_or_default();
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
