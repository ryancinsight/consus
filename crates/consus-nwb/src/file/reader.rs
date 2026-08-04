#[cfg(feature = "alloc")]
use alloc::{format, string::String};
#[cfg(feature = "alloc")]
use consus_core::Result;
use consus_hdf5::file::Hdf5File;
use consus_io::SliceReader;

#[cfg(feature = "alloc")]
use crate::metadata::{NwbSessionMetadata, NwbSubjectMetadata};
#[cfg(feature = "alloc")]
use crate::model::TimeSeries;
#[cfg(feature = "alloc")]
use crate::version::NwbVersion;

/// Top-level NWB 2.x file reader.
///
/// Wraps an [`Hdf5File`] opened over a borrowed byte slice and exposes
/// typed accessors for NWB session metadata and neurodata objects.
///
/// ## Lifetime
///
/// The `'a` parameter binds the `NwbFile` to the slice it was opened from.
/// The slice must remain valid for the lifetime of the `NwbFile`.
pub struct NwbFile<'a> {
    hdf5: Hdf5File<SliceReader<'a>>,
}

impl core::fmt::Debug for NwbFile<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NwbFile").finish_non_exhaustive()
    }
}

#[cfg(feature = "alloc")]
impl<'a> NwbFile<'a> {
    /// Open an NWB file from a byte slice.
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        let reader = SliceReader::new(bytes);
        let hdf5 = Hdf5File::open(reader)?;
        crate::validation::validate_root_attributes(&hdf5)?;
        Ok(Self { hdf5 })
    }

    /// Detect the NWB specification version from the root group's
    /// `nwb_version` attribute.
    pub fn nwb_version(&self) -> Result<NwbVersion> {
        crate::version::detect_version(&self.hdf5)
    }

    /// Read the required NWB session-level metadata from the root group.
    pub fn session_metadata(&self) -> Result<NwbSessionMetadata> {
        let root_addr = self.hdf5.superblock().root_group_address;
        let attrs = self.hdf5.attributes_at(root_addr)?;

        let identifier = crate::storage::read_string_attr(&attrs, "identifier")?;
        let session_description = crate::storage::read_string_attr(&attrs, "session_description")?;
        let session_start_time = crate::storage::read_string_attr(&attrs, "session_start_time")?;

        Ok(NwbSessionMetadata::new(
            identifier,
            session_description,
            session_start_time,
        ))
    }

    /// Read a `TimeSeries` neurodata group at the given HDF5 path.
    pub fn time_series(&self, path: &str) -> Result<TimeSeries> {
        let data_path = format!("{}/data", path);
        let data_addr = self.hdf5.open_path(&data_path)?;
        let data = crate::storage::read_f64_dataset(&self.hdf5, data_addr)?;

        let timestamps_path = format!("{}/timestamps", path);
        let timestamps = match self.hdf5.open_path(&timestamps_path) {
            Ok(ts_addr) => Some(crate::storage::read_f64_dataset(&self.hdf5, ts_addr)?),
            Err(_) => None,
        };

        let (starting_time, rate) = if timestamps.is_none() {
            let st_dataset_path = format!("{}/starting_time", path);
            let st_val =
                self.hdf5.open_path(&st_dataset_path).ok().and_then(|addr| {
                    crate::storage::read_scalar_f64_dataset(&self.hdf5, addr).ok()
                });
            let rate_val = if st_val.is_some() {
                self.hdf5
                    .open_path(&st_dataset_path)
                    .ok()
                    .and_then(|addr| self.hdf5.attributes_at(addr).ok())
                    .and_then(|attrs| crate::storage::read_f64_attr(&attrs, "rate").ok())
            } else {
                None
            };
            (st_val, rate_val)
        } else {
            (None, None)
        };

        let name = String::from(path.split('/').rfind(|s| !s.is_empty()).unwrap_or(path));

        Ok(TimeSeries::from_parts(
            name,
            data,
            timestamps,
            starting_time,
            rate,
        ))
    }

    /// Read the `spike_times` dataset from the `Units` group.
    pub fn units_spike_times(&self) -> Result<alloc::vec::Vec<f64>> {
        let addr = self.hdf5.open_path("Units/spike_times")?;
        crate::storage::read_f64_dataset(&self.hdf5, addr)
    }

    /// Read the `Units` DynamicTable from the HDMF VectorData + VectorIndex representation.
    pub fn units_table(&self) -> Result<crate::model::units::UnitsTable> {
        let flat_addr = self.hdf5.open_path("Units/spike_times")?;
        let flat_times = crate::storage::read_f64_dataset(&self.hdf5, flat_addr)?;

        let idx_addr = self.hdf5.open_path("Units/spike_times_index")?;
        let index = crate::storage::read_u64_dataset(&self.hdf5, idx_addr)?;

        let ids = match self.hdf5.open_path("Units/id") {
            Ok(id_addr) => Some(crate::storage::read_u64_dataset(&self.hdf5, id_addr)?),
            Err(_) => None,
        };

        crate::model::units::UnitsTable::from_vectordata(flat_times, index, ids)
    }

    /// Read the `electrodes` DynamicTable.
    pub fn electrode_table(&self) -> Result<crate::model::electrode::ElectrodeTable> {
        let id_addr = self.hdf5.open_path("electrodes/id")?;
        let ids = crate::storage::read_u64_dataset(&self.hdf5, id_addr)?;

        let loc_addr = self.hdf5.open_path("electrodes/location")?;
        let locations = crate::storage::read_string_dataset(&self.hdf5, loc_addr)?;

        let grp_addr = self.hdf5.open_path("electrodes/group_name")?;
        let group_names = crate::storage::read_string_dataset(&self.hdf5, grp_addr)?;

        crate::model::electrode::ElectrodeTable::from_columns(ids, locations, group_names)
    }

    /// List HDF5 paths of all `TimeSeries` (and known subtype) groups inside
    /// the container group at `group_path`.
    pub fn list_time_series(&self, group_path: &str) -> Result<alloc::vec::Vec<String>> {
        use crate::conventions::is_timeseries_type;
        let children = crate::group::list_typed_group_children(&self.hdf5, group_path)?;
        let prefix = group_path.trim_end_matches('/');
        let paths: alloc::vec::Vec<String> = children
            .into_iter()
            .filter(|c| {
                is_timeseries_type(
                    c.neurodata_type_def.as_deref().unwrap_or(""),
                    c.neurodata_type_inc.as_deref(),
                )
            })
            .map(|c| format!("{}/{}", prefix, c.name))
            .collect();
        Ok(paths)
    }

    /// Read subject metadata from the `general/subject` group.
    pub fn subject(&self) -> Result<NwbSubjectMetadata> {
        let addr = self.hdf5.open_path("general/subject")?;
        let attrs = self.hdf5.attributes_at(addr)?;
        let read_opt = |name: &str| crate::storage::read_string_attr(&attrs, name).ok();
        Ok(NwbSubjectMetadata::from_parts(
            read_opt("subject_id"),
            read_opt("species"),
            read_opt("sex"),
            read_opt("age"),
            read_opt("description"),
        ))
    }

    /// List HDF5 paths of all TimeSeries groups inside `acquisition/`.
    pub fn list_acquisition(&self) -> Result<alloc::vec::Vec<String>> {
        self.list_time_series("acquisition")
    }

    /// List HDF5 paths of all TimeSeries groups inside a processing module.
    pub fn list_processing(&self, module_name: &str) -> Result<alloc::vec::Vec<String>> {
        self.list_time_series(&alloc::format!("processing/{}", module_name))
    }

    /// List the namespace names present in the `/specifications/` group.
    ///
    /// Returns the direct children of `/specifications/` as namespace name
    /// strings (e.g. `["core", "hdmf-common"]`).  Returns an empty `Vec`
    /// when the `/specifications/` group is absent from the file, which is
    /// structurally valid for NWB files written without embedded spec YAML.
    ///
    /// ## Errors
    ///
    /// Propagates HDF5 format or I/O errors from group traversal.
    pub fn list_specifications(&self) -> Result<alloc::vec::Vec<String>> {
        let spec_addr = match self.hdf5.open_path("specifications") {
            Ok(a) => a,
            Err(consus_core::Error::NotFound { .. }) => return Ok(alloc::vec![]),
            Err(e) => return Err(e),
        };
        let children = self.hdf5.list_group_at(spec_addr)?;
        Ok(children.into_iter().map(|(name, _, _)| name).collect())
    }

    /// Read and parse the namespace specification YAML stored at
    /// `/specifications/{namespace}/{version}/namespace`.
    ///
    /// Reads the scalar `FixedString` dataset at the canonical NWB path and
    /// parses it with [`crate::namespace::parse_nwb_spec_yaml`].
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::NotFound`] when the path does not exist.
    /// - [`consus_core::Error::InvalidFormat`] when the dataset cannot be
    ///   decoded as a string or the YAML is malformed.
    /// - Propagates HDF5 I/O errors.
    pub fn read_specification(
        &self,
        namespace: &str,
        version: &str,
    ) -> Result<alloc::vec::Vec<crate::namespace::NwbNamespaceSpec>> {
        let path = alloc::format!("specifications/{}/{}/namespace", namespace, version);
        let addr = self.hdf5.open_path(&path)?;
        let yaml_text = crate::storage::read_scalar_string_dataset(&self.hdf5, addr)?;
        crate::namespace::parse_nwb_spec_yaml(&yaml_text).map_err(|e| {
            consus_core::Error::InvalidFormat {
                message: alloc::format!(
                    "NWB specification YAML parse error at '{}': {:?}",
                    path,
                    e
                ),
            }
        })
    }

    /// Run all NWB 2.x conformance checks and return a collected violation report.
    ///
    /// ## Validation layers
    ///
    /// 1. **Root identity** (fail-fast): `neurodata_type_def == "NWBFile"` and
    ///    `nwb_version` present — delegates to
    ///    [`crate::validation::validate_root_attributes`].
    /// 2. **Session attributes**: `identifier` (non-empty string),
    ///    `session_description` (non-empty string), `session_start_time`
    ///    (present, ISO 8601 format `YYYY-MM-DDTHH:MM:SS[Z|±HH:MM]`).
    /// 3. **Required top-level groups**: `/acquisition`, `/analysis`,
    ///    `/processing`, `/stimulus`, `/general`.
    /// 4. **TimeSeries constraints** for each child group under `/acquisition`
    ///    that is identified as a TimeSeries type: `neurodata_type_def`
    ///    attribute must be present; `data` sub-dataset must be present.
    ///
    /// Layers 2–4 collect all violations without short-circuiting so that
    /// the caller can inspect the full list in one pass.
    ///
    /// ## Errors
    ///
    /// Returns `Err` only when layer 1 fails (fatal format identity error) or
    /// when an I/O error occurs during HDF5 navigation.  Conformance
    /// violations in layers 2–4 are accumulated into the report.
    pub fn validate_conformance(&self) -> Result<crate::validation::NwbConformanceReport> {
        use crate::validation::{
            check_root_session_attrs, ConformanceViolation, NwbConformanceReport,
        };
        use consus_core::Error;

        // Layer 1: fail-fast identity + version gatekeeper.
        crate::validation::validate_root_attributes(&self.hdf5)?;

        let mut report = NwbConformanceReport::new();

        // Layer 2: session attributes (identifier, session_description,
        // session_start_time ISO 8601 format).
        check_root_session_attrs(&self.hdf5, &mut report)?;

        // Layer 3: required top-level NWB groups.
        const REQUIRED_GROUPS: &[&str] = &[
            "acquisition",
            "analysis",
            "processing",
            "stimulus",
            "general",
        ];
        for group_name in REQUIRED_GROUPS {
            match self.hdf5.open_path(group_name) {
                Ok(_) => {}
                Err(Error::NotFound { .. }) => {
                    report.push(ConformanceViolation::MissingRequiredGroup {
                        path: alloc::string::String::from(*group_name),
                    });
                }
                Err(e) => return Err(e),
            }
        }

        // Layer 4: per-child constraints for groups under /acquisition.
        // Only children identified as TimeSeries types are checked for `data`.
        let acq_children = match crate::group::list_typed_group_children(&self.hdf5, "acquisition")
        {
            Ok(children) => children,
            // Already reported as missing in layer 3; skip layer 4 for it.
            Err(Error::NotFound { .. }) => alloc::vec::Vec::new(),
            Err(e) => return Err(e),
        };
        for child in &acq_children {
            let child_path = alloc::format!("acquisition/{}", child.name);

            // Every neurodata object must carry neurodata_type_def.
            if child.neurodata_type_def.is_none() {
                report.push(ConformanceViolation::GroupMissingAttribute {
                    group_path: child_path.clone(),
                    attr_name: alloc::string::String::from("neurodata_type_def"),
                });
            }

            // TimeSeries types must have a `data` sub-dataset.
            let type_def = child.neurodata_type_def.as_deref().unwrap_or("");
            let type_inc = child.neurodata_type_inc.as_deref();
            if crate::conventions::is_timeseries_type(type_def, type_inc) {
                let data_path = alloc::format!("{}/data", child_path);
                match self.hdf5.open_path(&data_path) {
                    Ok(_) => {}
                    Err(Error::NotFound { .. }) => {
                        report.push(ConformanceViolation::TimeSeriesMissingData {
                            group_path: child_path,
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Layer 5: DynamicTable groups must carry a `colnames` attribute.
        crate::validation::check_dynamic_table_colnames(&self.hdf5, &mut report)?;

        // Layer 6: DynamicTable column-content consistency — each name in
        // `colnames` must correspond to a child dataset.
        crate::validation::check_dynamic_table_column_content(&self.hdf5, &mut report)?;

        Ok(report)
    }
}
