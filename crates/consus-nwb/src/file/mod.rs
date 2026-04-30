//! NWBFile top-level container.
//!
//! An NWBFile is an HDF5 file conforming to the NWB 2.x specification.
//! This module provides the entry point for opening and reading NWB files
//! and the builder for writing them.
//!
//! ## Specification
//!
//! Reference: *NWB 2.x Format Specification*
//! <https://nwb-schema.readthedocs.io/en/latest/format.html>
//!
//! ## Architecture
//!
//! ```text
//! NwbFile<'a>
//!   ├── open(bytes)              — validate HDF5 + NWB root attributes
//!   ├── nwb_version()            — detect NWB spec version
//!   ├── session_metadata()       — read required session-level attributes
//!   ├── time_series(path)        — read a TimeSeries neurodata group
//!   ├── units_spike_times()      — read flat Units spike times
//!   ├── units_table()            — read Units VectorIndex table
//!   ├── electrode_table()        — read electrodes DynamicTable
//!   ├── subject()                — read subject metadata
//!   ├── list_specifications()    — list namespace names from /specifications/
//!   └── read_specification(ns, ver) — read and parse namespace spec YAML
//!
//! NwbFileBuilder
//!   ├── new(...)                 — create root NWB metadata
//!   ├── write_time_series(ts)    — emit TimeSeries group
//!   ├── write_units(spikes)      — emit flat Units spike times
//!   ├── write_units_table(...)   — emit Units VectorData + VectorIndex table
//!   ├── write_electrode_table(...)— emit electrodes DynamicTable
//!   ├── write_subject(...)       — emit general/subject
//!   ├── write_namespace_specs(specs) — emit /specifications/{ns}/{ver}/namespace datasets
//!   └── finish()                 — return HDF5 bytes
//! ```
//!
//! ## Invariants
//!
//! - [`NwbFile::open`] only succeeds when the file passes
//!   [`crate::validation::validate_root_attributes`].
//! - All read methods are pure with respect to the file image: they never
//!   mutate the underlying byte slice.
//! - The `'a` lifetime ties the `NwbFile` to the byte slice it was opened
//!   from; the file cannot outlive its source data.

#[cfg(feature = "alloc")]
use alloc::format;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
#[cfg(feature = "alloc")]
use consus_hdf5::file::writer::{
    ChildDatasetSpec, ChildGroupSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder,
};
#[cfg(feature = "alloc")]
use core::num::NonZeroUsize;

use consus_core::Result;
use consus_hdf5::file::Hdf5File;
use consus_io::SliceReader;

use crate::metadata::{NwbSessionMetadata, NwbSubjectMetadata};
use crate::model::TimeSeries;
use crate::version::NwbVersion;

#[cfg(feature = "alloc")]
fn fixed_string_bytes(value: &str) -> (Datatype, alloc::vec::Vec<u8>) {
    let len = value.len().max(1);
    let dt = Datatype::FixedString {
        length: len,
        encoding: StringEncoding::Ascii,
    };
    let mut raw = value.as_bytes().to_vec();
    while raw.len() < len {
        raw.push(0u8);
    }
    (dt, raw)
}

#[cfg(feature = "alloc")]
fn f64_le_datatype() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

#[cfg(feature = "alloc")]
fn f32_le_datatype() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

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

        let name = String::from(
            path.split('/')
                .filter(|s| !s.is_empty())
                .last()
                .unwrap_or(path),
        );

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

        Ok(report)
    }
}

/// Builder for constructing NWB 2.x files.
pub struct NwbFileBuilder {
    hdf5: Hdf5FileBuilder,
}

impl core::fmt::Debug for NwbFileBuilder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NwbFileBuilder").finish_non_exhaustive()
    }
}

#[cfg(feature = "alloc")]
impl NwbFileBuilder {
    pub fn new(
        nwb_version: &str,
        identifier: impl Into<String>,
        session_description: impl Into<String>,
        session_start_time: impl Into<String>,
    ) -> Result<Self> {
        let identifier = identifier.into();
        let session_description = session_description.into();
        let session_start_time = session_start_time.into();

        if identifier.is_empty() {
            return Err(consus_core::Error::InvalidFormat {
                message: String::from("NWB: identifier must be non-empty"),
            });
        }
        if session_description.is_empty() {
            return Err(consus_core::Error::InvalidFormat {
                message: String::from("NWB: session_description must be non-empty"),
            });
        }

        let mut hdf5 = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        // Scalar fixed-string attributes: identity, version, session metadata,
        // and reference time.  NWB 2.x §4.1 requires timestamps_reference_time;
        // for new files with no explicit reference epoch it defaults to
        // session_start_time per NWB convention.
        let scalar_attrs: &[(&str, &str)] = &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", nwb_version),
            ("identifier", identifier.as_str()),
            ("session_description", session_description.as_str()),
            ("session_start_time", session_start_time.as_str()),
            ("timestamps_reference_time", session_start_time.as_str()),
        ];
        for (name, value) in scalar_attrs {
            let (dt, raw) = fixed_string_bytes(value);
            hdf5.add_root_attribute(name, &dt, &scalar, &raw)?;
        }

        // file_create_date: NWB 2.x §4.1 specifies a list of ISO 8601
        // timestamps recording when the file was created or appended to.
        // Encoded as a 1-D FixedString array of length 1 so it decodes as
        // AttributeValue::StringArray — the representation expected by
        // HDMF-compliant readers.
        let (fcd_dt, fcd_raw) = fixed_string_bytes(session_start_time.as_str());
        let fcd_shape = Shape::fixed(&[1]);
        hdf5.add_root_attribute("file_create_date", &fcd_dt, &fcd_shape, &fcd_raw)?;

        Ok(Self { hdf5 })
    }

    pub fn write_time_series(&mut self, ts: &TimeSeries) -> Result<()> {
        ts.validate()?;
        crate::validation::validate_time_series_for_write(ts)?;

        let scalar = Shape::scalar();
        let f64_dt = f64_le_datatype();
        let dcpl = DatasetCreationProps::default();
        let (ndt_dt, ndt_raw) = fixed_string_bytes("TimeSeries");

        let data_raw: alloc::vec::Vec<u8> =
            ts.data().iter().flat_map(|v| v.to_le_bytes()).collect();
        let data_shape = Shape::fixed(&[ts.len()]);

        if ts.has_timestamps() {
            let timestamps = ts.timestamps().expect("has_timestamps guarantees Some");
            let ts_raw: alloc::vec::Vec<u8> =
                timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();
            let ts_shape = Shape::fixed(&[timestamps.len()]);

            self.hdf5.add_group_with_attributes(
                ts.name(),
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[
                    ChildDatasetSpec {
                        name: "data",
                        datatype: &f64_dt,
                        shape: &data_shape,
                        raw_data: &data_raw,
                        dcpl: dcpl.clone(),
                        attributes: &[],
                    },
                    ChildDatasetSpec {
                        name: "timestamps",
                        datatype: &f64_dt,
                        shape: &ts_shape,
                        raw_data: &ts_raw,
                        dcpl: dcpl.clone(),
                        attributes: &[],
                    },
                ],
            )?;
        } else {
            let starting_time = ts.starting_time().unwrap_or(0.0);
            let rate = ts
                .rate()
                .expect("validate_time_series_for_write guarantees rate is Some")
                as f32;
            let st_raw = starting_time.to_le_bytes().to_vec();
            let rate_raw = rate.to_le_bytes().to_vec();
            let f32_dt = f32_le_datatype();

            let rate_attrs: &[(&str, &Datatype, &Shape, &[u8])] =
                &[("rate", &f32_dt, &scalar, &rate_raw)];

            self.hdf5.add_group_with_attributes(
                ts.name(),
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[
                    ChildDatasetSpec {
                        name: "data",
                        datatype: &f64_dt,
                        shape: &data_shape,
                        raw_data: &data_raw,
                        dcpl: dcpl.clone(),
                        attributes: &[],
                    },
                    ChildDatasetSpec {
                        name: "starting_time",
                        datatype: &f64_dt,
                        shape: &scalar,
                        raw_data: &st_raw,
                        dcpl: dcpl.clone(),
                        attributes: rate_attrs,
                    },
                ],
            )?;
        }

        Ok(())
    }

    pub fn write_units(&mut self, spike_times: &[f64]) -> Result<()> {
        let scalar = Shape::scalar();
        let f64_dt = f64_le_datatype();
        let dcpl = DatasetCreationProps::default();

        let (units_ndt_dt, units_ndt_raw) = fixed_string_bytes("Units");
        let (vd_ndt_dt, vd_ndt_raw) = fixed_string_bytes("VectorData");
        let (desc_dt, desc_raw) = fixed_string_bytes("spike times");

        let st_raw: alloc::vec::Vec<u8> =
            spike_times.iter().flat_map(|v| v.to_le_bytes()).collect();
        let st_shape = Shape::fixed(&[spike_times.len()]);

        let spike_attrs: &[(&str, &Datatype, &Shape, &[u8])] = &[
            ("neurodata_type_def", &vd_ndt_dt, &scalar, &vd_ndt_raw),
            ("description", &desc_dt, &scalar, &desc_raw),
        ];

        self.hdf5.add_group_with_attributes(
            "Units",
            &[("neurodata_type_def", &units_ndt_dt, &scalar, &units_ndt_raw)],
            &[ChildDatasetSpec {
                name: "spike_times",
                datatype: &f64_dt,
                shape: &st_shape,
                raw_data: &st_raw,
                dcpl,
                attributes: spike_attrs,
            }],
        )?;

        Ok(())
    }

    pub fn write_units_table(&mut self, units: &crate::model::units::UnitsTable) -> Result<()> {
        let scalar = Shape::scalar();
        let f64_dt = f64_le_datatype();
        let u64_dt = Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            signed: false,
            byte_order: ByteOrder::LittleEndian,
        };
        let dcpl = DatasetCreationProps::default();

        let (units_ndt_dt, units_ndt_raw) = fixed_string_bytes("Units");
        let (vd_ndt_dt, vd_ndt_raw) = fixed_string_bytes("VectorData");
        let (vi_ndt_dt, vi_ndt_raw) = fixed_string_bytes("VectorIndex");
        let (st_desc_dt, st_desc_raw) = fixed_string_bytes("spike times");
        let (si_desc_dt, si_desc_raw) = fixed_string_bytes("spike times index");

        let flat_times = units.flat_spike_times();
        let cumulative_index = units.cumulative_index();

        let st_raw: alloc::vec::Vec<u8> = flat_times.iter().flat_map(|v| v.to_le_bytes()).collect();
        let st_shape = Shape::fixed(&[flat_times.len()]);

        let idx_raw: alloc::vec::Vec<u8> = cumulative_index
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let idx_shape = Shape::fixed(&[cumulative_index.len()]);

        let id_raw: alloc::vec::Vec<u8>;
        let id_shape: Shape;
        let has_ids = units.ids().is_some();
        if let Some(ids) = units.ids() {
            id_raw = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
            id_shape = Shape::fixed(&[ids.len()]);
        } else {
            id_raw = alloc::vec![];
            id_shape = Shape::fixed(&[0]);
        }

        let st_attrs: &[(&str, &Datatype, &Shape, &[u8])] = &[
            ("neurodata_type_def", &vd_ndt_dt, &scalar, &vd_ndt_raw),
            ("description", &st_desc_dt, &scalar, &st_desc_raw),
        ];
        let idx_attrs: &[(&str, &Datatype, &Shape, &[u8])] = &[
            ("neurodata_type_def", &vi_ndt_dt, &scalar, &vi_ndt_raw),
            ("description", &si_desc_dt, &scalar, &si_desc_raw),
        ];

        if has_ids {
            self.hdf5.add_group_with_attributes(
                "Units",
                &[("neurodata_type_def", &units_ndt_dt, &scalar, &units_ndt_raw)],
                &[
                    ChildDatasetSpec {
                        name: "spike_times",
                        datatype: &f64_dt,
                        shape: &st_shape,
                        raw_data: &st_raw,
                        dcpl: dcpl.clone(),
                        attributes: st_attrs,
                    },
                    ChildDatasetSpec {
                        name: "spike_times_index",
                        datatype: &u64_dt,
                        shape: &idx_shape,
                        raw_data: &idx_raw,
                        dcpl: dcpl.clone(),
                        attributes: idx_attrs,
                    },
                    ChildDatasetSpec {
                        name: "id",
                        datatype: &u64_dt,
                        shape: &id_shape,
                        raw_data: &id_raw,
                        dcpl,
                        attributes: &[],
                    },
                ],
            )?;
        } else {
            self.hdf5.add_group_with_attributes(
                "Units",
                &[("neurodata_type_def", &units_ndt_dt, &scalar, &units_ndt_raw)],
                &[
                    ChildDatasetSpec {
                        name: "spike_times",
                        datatype: &f64_dt,
                        shape: &st_shape,
                        raw_data: &st_raw,
                        dcpl: dcpl.clone(),
                        attributes: st_attrs,
                    },
                    ChildDatasetSpec {
                        name: "spike_times_index",
                        datatype: &u64_dt,
                        shape: &idx_shape,
                        raw_data: &idx_raw,
                        dcpl,
                        attributes: idx_attrs,
                    },
                ],
            )?;
        }

        Ok(())
    }

    pub fn write_electrode_table(
        &mut self,
        table: &crate::model::electrode::ElectrodeTable,
    ) -> Result<()> {
        let scalar = Shape::scalar();
        let u64_dt = Datatype::Integer {
            bits: NonZeroUsize::new(64).unwrap(),
            signed: false,
            byte_order: ByteOrder::LittleEndian,
        };
        let dcpl = DatasetCreationProps::default();

        let (dyn_ndt_dt, dyn_ndt_raw) = fixed_string_bytes("DynamicTable");
        let (desc_dt, desc_raw) = fixed_string_bytes("Electrode metadata");
        let (colnames_dt, colnames_raw) = fixed_string_bytes("location,group_name");

        let n = table.len();

        let id_raw: alloc::vec::Vec<u8> = table.id_column().flat_map(|v| v.to_le_bytes()).collect();
        let id_shape = Shape::fixed(&[n]);

        let locs: alloc::vec::Vec<String> = table.location_column().map(String::from).collect();
        let loc_max = locs.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let mut loc_raw: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(n * loc_max);
        for s in &locs {
            loc_raw.extend_from_slice(s.as_bytes());
            for _ in s.len()..loc_max {
                loc_raw.push(0u8);
            }
        }
        let loc_dt = Datatype::FixedString {
            length: loc_max,
            encoding: StringEncoding::Ascii,
        };
        let loc_shape = Shape::fixed(&[n]);

        let grps: alloc::vec::Vec<String> = table.group_name_column().map(String::from).collect();
        let grp_max = grps.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let mut grp_raw: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(n * grp_max);
        for s in &grps {
            grp_raw.extend_from_slice(s.as_bytes());
            for _ in s.len()..grp_max {
                grp_raw.push(0u8);
            }
        }
        let grp_dt = Datatype::FixedString {
            length: grp_max,
            encoding: StringEncoding::Ascii,
        };
        let grp_shape = Shape::fixed(&[n]);

        self.hdf5.add_group_with_attributes(
            "electrodes",
            &[
                ("neurodata_type_def", &dyn_ndt_dt, &scalar, &dyn_ndt_raw),
                ("description", &desc_dt, &scalar, &desc_raw),
                ("colnames", &colnames_dt, &scalar, &colnames_raw),
            ],
            &[
                ChildDatasetSpec {
                    name: "id",
                    datatype: &u64_dt,
                    shape: &id_shape,
                    raw_data: &id_raw,
                    dcpl: dcpl.clone(),
                    attributes: &[],
                },
                ChildDatasetSpec {
                    name: "location",
                    datatype: &loc_dt,
                    shape: &loc_shape,
                    raw_data: &loc_raw,
                    dcpl: dcpl.clone(),
                    attributes: &[],
                },
                ChildDatasetSpec {
                    name: "group_name",
                    datatype: &grp_dt,
                    shape: &grp_shape,
                    raw_data: &grp_raw,
                    dcpl,
                    attributes: &[],
                },
            ],
        )?;

        Ok(())
    }

    /// Write NWB namespace specifications to `/specifications/{ns}/{ver}/namespace`.
    ///
    /// Each [`crate::namespace::NwbNamespaceSpec`] is serialized to the
    /// canonical flat-key YAML format via
    /// [`crate::namespace::format_nwb_spec_yaml`] and stored as a scalar
    /// `FixedString` dataset at the path
    /// `/specifications/{name}/{version}/namespace`.
    ///
    /// All specs are written in a single `/specifications/` group call.
    /// This method must not be called more than once per builder instance.
    ///
    /// ## Errors
    ///
    /// Returns an HDF5 encoding error if any datatype cannot be encoded.
    pub fn write_namespace_specs(
        &mut self,
        specs: &[crate::namespace::NwbNamespaceSpec],
    ) -> Result<()> {
        if specs.is_empty() {
            return Ok(());
        }

        let scalar = Shape::scalar();

        // Step 1: build all YAML texts and their FixedString datatypes (owned).
        let yaml_texts: alloc::vec::Vec<String> = specs
            .iter()
            .map(|s| crate::namespace::format_nwb_spec_yaml(core::slice::from_ref(s)))
            .collect();

        let yaml_dts: alloc::vec::Vec<Datatype> = yaml_texts
            .iter()
            .map(|t| Datatype::FixedString {
                length: t.len().max(1),
                encoding: StringEncoding::Utf8,
            })
            .collect();

        let yaml_raws: alloc::vec::Vec<alloc::vec::Vec<u8>> =
            yaml_texts.iter().map(|t| t.as_bytes().to_vec()).collect();

        // Step 2: build ChildDatasetSpec for each spec's "namespace" dataset.
        let ns_dataset_specs: alloc::vec::Vec<ChildDatasetSpec<'_>> = (0..specs.len())
            .map(|i| ChildDatasetSpec {
                name: "namespace",
                datatype: &yaml_dts[i],
                shape: &scalar,
                raw_data: &yaml_raws[i],
                dcpl: DatasetCreationProps::default(),
                attributes: &[],
            })
            .collect();

        // Step 3: build version-level ChildGroupSpec for each spec.
        let version_group_specs: alloc::vec::Vec<ChildGroupSpec<'_>> = (0..specs.len())
            .map(|i| ChildGroupSpec {
                name: specs[i].version.as_str(),
                attributes: &[],
                datasets: core::slice::from_ref(&ns_dataset_specs[i]),
                sub_groups: &[],
            })
            .collect();

        // Step 4: build namespace-level ChildGroupSpec for each spec.
        let ns_group_specs: alloc::vec::Vec<ChildGroupSpec<'_>> = (0..specs.len())
            .map(|i| ChildGroupSpec {
                name: specs[i].name.as_str(),
                attributes: &[],
                datasets: &[],
                sub_groups: core::slice::from_ref(&version_group_specs[i]),
            })
            .collect();

        // Step 5: write /specifications/ root group containing all namespace groups.
        self.hdf5
            .add_group_with_children("specifications", &[], &[], &ns_group_specs)?;

        Ok(())
    }

    pub fn write_subject(&mut self, subject: &NwbSubjectMetadata) -> Result<()> {
        let scalar = Shape::scalar();
        let mut attrs_owned: alloc::vec::Vec<(
            alloc::string::String,
            Datatype,
            alloc::vec::Vec<u8>,
        )> = alloc::vec::Vec::new();

        let (ndt_dt, ndt_raw) = fixed_string_bytes("Subject");
        attrs_owned.push((
            alloc::string::String::from("neurodata_type_def"),
            ndt_dt,
            ndt_raw,
        ));

        for (name, val_opt) in &[
            ("subject_id", subject.subject_id()),
            ("species", subject.species()),
            ("sex", subject.sex()),
            ("age", subject.age()),
            ("description", subject.description()),
        ] {
            if let Some(val) = val_opt {
                let (dt, raw) = fixed_string_bytes(val);
                attrs_owned.push((alloc::string::String::from(*name), dt, raw));
            }
        }

        let attr_refs: alloc::vec::Vec<(&str, &Datatype, &Shape, &[u8])> = attrs_owned
            .iter()
            .map(|(name, dt, raw)| (name.as_str(), dt, &scalar, raw.as_slice()))
            .collect();

        let subject_spec = ChildGroupSpec {
            name: "subject",
            attributes: &attr_refs,
            datasets: &[],
            sub_groups: &[],
        };

        self.hdf5
            .add_group_with_children("general", &[], &[], &[subject_spec])?;

        Ok(())
    }

    /// Create an empty HDF5 group at `path` with no attributes or datasets.
    ///
    /// Use this to satisfy NWB 2.x required group structure when no content
    /// is written to a mandatory group (`acquisition`, `analysis`, `processing`,
    /// `stimulus`, `general`).
    ///
    /// ## Errors
    ///
    /// Propagates HDF5 format errors from the underlying group writer.
    pub fn write_empty_group(&mut self, path: &str) -> Result<&mut Self> {
        self.hdf5.add_group_with_attributes(path, &[], &[])?;
        Ok(self)
    }

    pub fn finish(self) -> Result<alloc::vec::Vec<u8>> {
        self.hdf5.finish()
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
    use consus_hdf5::file::writer::{
        ChildDatasetSpec, ChildGroupSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder,
    };
    use consus_hdf5::file::Hdf5File;
    use consus_io::SliceReader;
    use core::num::NonZeroUsize;

    fn fixed_string_dt(value: &str) -> (Datatype, alloc::vec::Vec<u8>) {
        let len = value.len().max(1);
        let dt = Datatype::FixedString {
            length: len,
            encoding: StringEncoding::Ascii,
        };
        let mut raw = value.as_bytes().to_vec();
        while raw.len() < len {
            raw.push(0u8);
        }
        (dt, raw)
    }

    fn make_minimal_nwb(id: &str, desc: &str, ts: &str) -> alloc::vec::Vec<u8> {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", id),
            ("session_description", desc),
            ("session_start_time", ts),
            ("timestamps_reference_time", ts),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        // file_create_date as 1-D array (matches NwbFileBuilder output).
        let (fcd_dt, fcd_raw) = fixed_string_dt(ts);
        let fcd_shape = Shape::fixed(&[1]);
        builder
            .add_root_attribute("file_create_date", &fcd_dt, &fcd_shape, &fcd_raw)
            .unwrap();

        builder.finish().unwrap()
    }

    fn make_nwb_with_timeseries(
        id: &str,
        ts_name: &str,
        data: &[f64],
        timestamps: &[f64],
    ) -> alloc::vec::Vec<u8> {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", id),
            ("session_description", "test"),
            ("session_start_time", "2023-01-01T00:00:00+00:00"),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }

        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let data_raw: alloc::vec::Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ts_raw: alloc::vec::Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data_shape = Shape::fixed(&[data.len()]);
        let ts_shape = Shape::fixed(&[timestamps.len()]);
        let (ndt_dt, ndt_raw) = fixed_string_dt("TimeSeries");

        builder
            .add_group_with_attributes(
                ts_name,
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[
                    ChildDatasetSpec {
                        name: "data",
                        datatype: &f64_dt,
                        shape: &data_shape,
                        raw_data: &data_raw,
                        dcpl: DatasetCreationProps::default(),
                        attributes: &[],
                    },
                    ChildDatasetSpec {
                        name: "timestamps",
                        datatype: &f64_dt,
                        shape: &ts_shape,
                        raw_data: &ts_raw,
                        dcpl: DatasetCreationProps::default(),
                        attributes: &[],
                    },
                ],
            )
            .unwrap();

        builder.finish().unwrap()
    }

    #[test]
    fn open_valid_nwb_file_succeeds() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let version = nwb.nwb_version().unwrap();
        assert_eq!(version.as_str(), "2.7");
    }

    #[test]
    fn open_non_hdf5_bytes_returns_error() {
        let bytes = b"not hdf5".to_vec();
        let err = NwbFile::open(&bytes).unwrap_err();
        assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
    }

    #[test]
    fn open_hdf5_without_neurodata_type_def_returns_invalid_format() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        let (dt, raw) = fixed_string_dt("2.7.0");
        builder
            .add_root_attribute("nwb_version", &dt, &scalar, &raw)
            .unwrap();
        let bytes = builder.finish().unwrap();
        let err = NwbFile::open(&bytes).unwrap_err();
        assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
    }

    #[test]
    fn open_hdf5_without_nwb_version_returns_not_found() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        let (dt, raw) = fixed_string_dt("NWBFile");
        builder
            .add_root_attribute("neurodata_type_def", &dt, &scalar, &raw)
            .unwrap();
        let bytes = builder.finish().unwrap();
        let err = NwbFile::open(&bytes).unwrap_err();
        assert!(matches!(err, consus_core::Error::NotFound { .. }));
    }

    #[test]
    fn nwb_version_returns_v2_7_for_2_7_0() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let version = nwb.nwb_version().unwrap();
        assert_eq!(version.as_str(), "2.7");
    }

    #[test]
    fn session_metadata_returns_correct_fields() {
        let bytes = make_minimal_nwb("session-123", "desc", "2023-01-01T12:34:56+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let meta = nwb.session_metadata().unwrap();
        assert_eq!(meta.identifier(), "session-123");
        assert_eq!(meta.session_description(), "desc");
        assert_eq!(meta.session_start_time(), "2023-01-01T12:34:56+00:00");
    }

    #[test]
    fn session_metadata_missing_identifier_returns_not_found() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00+00:00"),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let bytes = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes).unwrap();
        let err = nwb.session_metadata().unwrap_err();
        assert!(matches!(err, consus_core::Error::NotFound { .. }));
    }

    #[test]
    fn time_series_reads_data_and_timestamps() {
        let bytes = make_nwb_with_timeseries("id", "ts", &[1.0, 2.0, 3.0], &[0.0, 0.1, 0.2]);
        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("ts").unwrap();
        assert_eq!(ts.name(), "ts");
        assert_eq!(ts.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(ts.timestamps(), Some([0.0, 0.1, 0.2].as_slice()));
    }

    #[test]
    fn time_series_validates_length_invariant() {
        let ts = TimeSeries::from_parts("bad", vec![1.0, 2.0], Some(vec![0.0]), None, None);
        let err = ts.validate().unwrap_err();
        assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
    }

    #[test]
    fn time_series_missing_group_returns_not_found() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let err = nwb.time_series("missing").unwrap_err();
        assert!(matches!(err, consus_core::Error::NotFound { .. }));
    }

    #[test]
    fn time_series_name_derived_from_last_path_component() {
        let mut builder =
            NwbFileBuilder::new("2.7.0", "id", "desc", "2023-01-01T00:00:00+00:00").unwrap();
        let ts = TimeSeries::with_timestamps("my_ts", vec![1.0], vec![0.0]);
        builder.write_time_series(&ts).unwrap();
        let bytes = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("my_ts").unwrap();
        assert_eq!(ts.name(), "my_ts");
    }

    #[test]
    fn time_series_reads_starting_time_and_rate_from_dataset() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00+00:00"),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let f32_dt = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let (ndt_dt, ndt_raw) = fixed_string_dt("TimeSeries");
        let data = [1.0f64, 2.0f64];
        let st = [0.5f64];
        let rate = [2.0f32];
        builder
            .add_group_with_attributes(
                "rate_ts",
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[
                    ChildDatasetSpec {
                        name: "data",
                        datatype: &f64_dt,
                        shape: &Shape::fixed(&[data.len()]),
                        raw_data: &data
                            .iter()
                            .copied()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<alloc::vec::Vec<u8>>(),
                        dcpl: DatasetCreationProps::default(),
                        attributes: &[],
                    },
                    ChildDatasetSpec {
                        name: "starting_time",
                        datatype: &f64_dt,
                        shape: &scalar,
                        raw_data: &st
                            .iter()
                            .copied()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<alloc::vec::Vec<u8>>(),
                        dcpl: DatasetCreationProps::default(),
                        attributes: &[(
                            "rate",
                            &f32_dt,
                            &scalar,
                            &rate
                                .iter()
                                .copied()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<alloc::vec::Vec<u8>>(),
                        )],
                    },
                ],
            )
            .unwrap();
        let bytes = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("rate_ts").unwrap();
        assert_eq!(ts.starting_time(), Some(0.5));
        assert_eq!(ts.rate(), Some(2.0));
    }

    #[test]
    fn time_series_with_timestamps_does_not_read_starting_time() {
        let bytes = make_nwb_with_timeseries("id", "ts", &[1.0, 2.0], &[0.0, 0.1]);
        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("ts").unwrap();
        assert_eq!(ts.timestamps(), Some([0.0, 0.1].as_slice()));
        assert!(ts.starting_time().is_none());
        assert!(ts.rate().is_none());
    }

    // Additional tests for list_time_series, subject, builder roundtrips, units table,
    // electrode table, proptests, and negative paths should be restored alongside the
    // corresponding model and storage modules.

    // ── list_specifications ───────────────────────────────────────────────

    #[test]
    fn list_specifications_returns_empty_when_group_absent() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let names = nwb.list_specifications().unwrap();
        assert!(
            names.is_empty(),
            "no /specifications/ group should yield empty list, got {:?}",
            names
        );
    }

    #[test]
    fn list_specifications_returns_namespace_names_after_write() {
        use crate::namespace::NwbNamespaceSpec;
        let mut builder =
            NwbFileBuilder::new("2.8.0", "spec-list", "test", "2024-01-01T00:00:00").unwrap();
        let specs = alloc::vec![
            NwbNamespaceSpec {
                name: alloc::string::String::from("core"),
                version: alloc::string::String::from("2.8.0"),
                doc_url: None,
                neurodata_types: alloc::vec![],
            },
            NwbNamespaceSpec {
                name: alloc::string::String::from("hdmf-common"),
                version: alloc::string::String::from("1.8.0"),
                doc_url: None,
                neurodata_types: alloc::vec![],
            },
        ];
        builder.write_namespace_specs(&specs).unwrap();
        let bytes = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes).unwrap();
        let mut names = nwb.list_specifications().unwrap();
        names.sort();
        assert_eq!(names, &["core", "hdmf-common"]);
    }

    // ── read_specification ────────────────────────────────────────────────

    #[test]
    fn read_specification_returns_not_found_when_absent() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let err = nwb
            .read_specification("core", "2.8.0")
            .expect_err("absent spec must return error");
        assert!(
            matches!(err, consus_core::Error::NotFound { .. }),
            "expected NotFound, got {:?}",
            err
        );
    }

    #[test]
    fn read_specification_roundtrip_core_spec_with_neurodata_types() {
        use crate::namespace::NwbNamespaceSpec;
        let original = NwbNamespaceSpec {
            name: alloc::string::String::from("core"),
            version: alloc::string::String::from("2.8.0"),
            doc_url: Some(alloc::string::String::from(
                "https://nwb-schema.readthedocs.io/en/latest/",
            )),
            neurodata_types: alloc::vec![
                crate::namespace::NwbTypeSpec {
                    name: alloc::string::String::from("TimeSeries"),
                    neurodata_type_inc: None,
                },
                crate::namespace::NwbTypeSpec {
                    name: alloc::string::String::from("ElectricalSeries"),
                    neurodata_type_inc: None,
                },
                crate::namespace::NwbTypeSpec {
                    name: alloc::string::String::from("SpatialSeries"),
                    neurodata_type_inc: None,
                },
            ],
        };
        let mut builder =
            NwbFileBuilder::new("2.8.0", "spec-rt", "desc", "2024-01-01T00:00:00").unwrap();
        builder
            .write_namespace_specs(core::slice::from_ref(&original))
            .unwrap();
        let bytes = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes).unwrap();
        let restored = nwb.read_specification("core", "2.8.0").unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0], original);
    }

    #[test]
    fn read_specification_roundtrip_hdmf_common_spec() {
        use crate::namespace::NwbNamespaceSpec;
        let original = NwbNamespaceSpec {
            name: alloc::string::String::from("hdmf-common"),
            version: alloc::string::String::from("1.8.0"),
            doc_url: None,
            neurodata_types: alloc::vec![
                crate::namespace::NwbTypeSpec {
                    name: alloc::string::String::from("VectorData"),
                    neurodata_type_inc: None,
                },
                crate::namespace::NwbTypeSpec {
                    name: alloc::string::String::from("DynamicTable"),
                    neurodata_type_inc: None,
                },
            ],
        };
        let mut builder =
            NwbFileBuilder::new("2.8.0", "spec-hdmf", "desc", "2024-01-01T00:00:00").unwrap();
        builder
            .write_namespace_specs(core::slice::from_ref(&original))
            .unwrap();
        let bytes = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes).unwrap();
        let restored = nwb.read_specification("hdmf-common", "1.8.0").unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0], original);
    }

    #[test]
    fn write_namespace_specs_empty_slice_is_noop() {
        let mut builder =
            NwbFileBuilder::new("2.8.0", "spec-empty", "desc", "2024-01-01T00:00:00").unwrap();
        builder.write_namespace_specs(&[]).unwrap();
        let bytes = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes).unwrap();
        let names = nwb.list_specifications().unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn nwb_version_returns_v2_8_for_2_8_0() {
        use crate::version::NwbVersion;
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        // Build a file with version "2.8.0" to verify V2_8 parsing.
        let builder =
            NwbFileBuilder::new("2.8.0", "id-v28", "desc", "2024-01-01T00:00:00").unwrap();
        let bytes28 = builder.finish().unwrap();
        let nwb = NwbFile::open(&bytes28).unwrap();
        assert_eq!(nwb.nwb_version().unwrap(), NwbVersion::V2_8);
        let _ = bytes; // suppress unused warning
    }

    // -----------------------------------------------------------------------
    // validate_conformance tests
    // -----------------------------------------------------------------------

    /// A file with only root attributes (no required groups) must report
    /// MissingRequiredGroup violations for all 5 mandatory groups.
    #[test]
    fn validate_conformance_reports_missing_required_groups() {
        let bytes = make_minimal_nwb("test-id", "session desc", "2023-01-01T00:00:00Z");
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        assert!(
            !report.is_conformant(),
            "minimal file must have group violations"
        );
        let missing: alloc::vec::Vec<&str> = report
            .violations()
            .iter()
            .filter_map(|v| match v {
                crate::validation::ConformanceViolation::MissingRequiredGroup { path } => {
                    Some(path.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            missing.contains(&"acquisition"),
            "acquisition must be reported missing: {missing:?}"
        );
        assert!(
            missing.contains(&"analysis"),
            "analysis must be reported missing: {missing:?}"
        );
        assert!(
            missing.contains(&"processing"),
            "processing must be reported missing: {missing:?}"
        );
    }

    /// All 5 required groups must be reported when none are present.
    #[test]
    fn validate_conformance_collects_all_five_missing_groups() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00Z");
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        let missing_count = report
            .violations()
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    crate::validation::ConformanceViolation::MissingRequiredGroup { .. }
                )
            })
            .count();
        assert_eq!(
            missing_count,
            5,
            "all 5 required groups must be reported; got: {:?}",
            report.violations()
        );
    }

    /// A fully conformant file (all required groups present, valid session
    /// attributes) must pass validation with zero violations.
    #[test]
    fn validate_conformance_passes_with_all_required_groups() {
        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "test-id",
            "test description",
            "2023-01-01T00:00:00Z",
        )
        .unwrap();
        for group in &[
            "acquisition",
            "analysis",
            "processing",
            "stimulus",
            "general",
        ] {
            builder.write_empty_group(group).unwrap();
        }
        let bytes = builder.finish().unwrap();
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        assert!(
            report.is_conformant(),
            "file with all required groups must be conformant: {:?}",
            report.violations()
        );
    }

    /// A file with a bad `session_start_time` format must report exactly one
    /// InvalidRootAttributeValue violation for that attribute.
    #[test]
    fn validate_conformance_reports_bad_session_start_time_format() {
        // NwbFile::open only checks neurodata_type_def + nwb_version, so a file
        // with a bad timestamp format opens successfully but fails full conformance.
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id-001"),
            ("session_description", "desc"),
            ("session_start_time", "not-a-date"),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        for group in &[
            "acquisition",
            "analysis",
            "processing",
            "stimulus",
            "general",
        ] {
            builder.add_group_with_attributes(group, &[], &[]).unwrap();
        }
        let bytes = builder.finish().unwrap();
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        assert!(!report.is_conformant());
        let bad_ts_violations: alloc::vec::Vec<_> = report
            .violations()
            .iter()
            .filter(|v| {
                matches!(v,
                    crate::validation::ConformanceViolation::InvalidRootAttributeValue { name, .. }
                    if name == "session_start_time"
                )
            })
            .collect();
        assert_eq!(
            bad_ts_violations.len(),
            1,
            "exactly one bad-format violation expected: {:?}",
            report.violations()
        );
    }

    /// A TimeSeries group under /acquisition that is missing a `data` dataset
    /// must produce a TimeSeriesMissingData violation naming the group path.
    #[test]
    fn validate_conformance_reports_timeseries_missing_data() {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id-001"),
            ("session_description", "desc"),
            ("session_start_time", "2023-01-01T00:00:00Z"),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        // Add the 4 non-acquisition required groups directly.
        for group in &["analysis", "processing", "stimulus", "general"] {
            builder.add_group_with_attributes(group, &[], &[]).unwrap();
        }
        // Build /acquisition with a TimeSeries child that has NO `data` dataset.
        let (ndt_dt, ndt_raw) = fixed_string_dt("TimeSeries");
        let ts_group = ChildGroupSpec {
            name: "test_ts",
            attributes: &[("neurodata_type_def", &ndt_dt, &scalar, ndt_raw.as_slice())],
            datasets: &[],
            sub_groups: &[],
        };
        builder
            .add_group_with_children("acquisition", &[], &[], &[ts_group])
            .unwrap();
        let bytes = builder.finish().unwrap();
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        let missing_data: alloc::vec::Vec<_> = report
            .violations()
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    crate::validation::ConformanceViolation::TimeSeriesMissingData { .. }
                )
            })
            .collect();
        assert!(
            !missing_data.is_empty(),
            "missing-data violation must be present: {:?}",
            report.violations()
        );
        match &missing_data[0] {
            crate::validation::ConformanceViolation::TimeSeriesMissingData { group_path } => {
                assert!(
                    group_path.contains("test_ts"),
                    "violation must name the group: {group_path}"
                );
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    /// validate_conformance must not short-circuit: a file missing all 5
    /// required groups must report exactly 5 MissingRequiredGroup violations,
    /// not stop after the first.
    #[test]
    fn validate_conformance_does_not_short_circuit() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00Z");
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        // Layer 2 finds no session-attribute violations (they are all valid).
        let session_violations = report
            .violations()
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    crate::validation::ConformanceViolation::MissingRootAttribute { .. }
                        | crate::validation::ConformanceViolation::InvalidRootAttributeValue { .. }
                )
            })
            .count();
        assert_eq!(
            session_violations,
            0,
            "no session-attr violations expected: {:?}",
            report.violations()
        );
        // Layer 3 must find 5 missing groups — not short-circuit after 1.
        let group_violations = report
            .violations()
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    crate::validation::ConformanceViolation::MissingRequiredGroup { .. }
                )
            })
            .count();
        assert_eq!(
            group_violations,
            5,
            "all 5 missing-group violations must be collected: {:?}",
            report.violations()
        );
    }

    // -----------------------------------------------------------------------
    // M-048: NWB extended conformance — timestamps_reference_time,
    // file_create_date, and DynamicTable colnames.
    // -----------------------------------------------------------------------

    /// NwbFileBuilder::new must write timestamps_reference_time as a scalar
    /// FixedString equal to session_start_time.
    #[test]
    fn nwb_file_builder_writes_timestamps_reference_time() {
        let ts = "2024-03-15T08:00:00Z";
        let bytes = NwbFileBuilder::new("2.7.0", "uid1", "desc", ts)
            .unwrap()
            .finish()
            .unwrap();
        // NwbFile::hdf5 is private; open via Hdf5File directly for raw attr access.
        let hdf5 = Hdf5File::open(SliceReader::new(&bytes)).unwrap();
        let root_addr = hdf5.superblock().root_group_address;
        let attrs = hdf5.attributes_at(root_addr).unwrap();
        let trt_attr = attrs.iter().find(|a| a.name == "timestamps_reference_time");
        assert!(
            trt_attr.is_some(),
            "timestamps_reference_time attribute must be written"
        );
        match trt_attr.unwrap().decode_value().unwrap() {
            consus_core::AttributeValue::String(ref s) => {
                assert_eq!(
                    s.as_str(),
                    ts,
                    "timestamps_reference_time must equal session_start_time"
                );
            }
            other => panic!("expected String, got {:?}", other),
        }
    }

    /// NwbFileBuilder::new must write file_create_date as a 1-D FixedString
    /// array with exactly one entry equal to session_start_time.
    #[test]
    fn nwb_file_builder_writes_file_create_date() {
        let ts = "2024-03-15T08:00:00Z";
        let bytes = NwbFileBuilder::new("2.7.0", "uid2", "desc", ts)
            .unwrap()
            .finish()
            .unwrap();
        let hdf5 = Hdf5File::open(SliceReader::new(&bytes)).unwrap();
        let root_addr = hdf5.superblock().root_group_address;
        let attrs = hdf5.attributes_at(root_addr).unwrap();
        let fcd_attr = attrs.iter().find(|a| a.name == "file_create_date");
        assert!(
            fcd_attr.is_some(),
            "file_create_date attribute must be written"
        );
        match fcd_attr.unwrap().decode_value().unwrap() {
            consus_core::AttributeValue::String(ref s) => {
                assert_eq!(s.as_str(), ts);
            }
            consus_core::AttributeValue::StringArray(ref v) => {
                assert_eq!(v.len(), 1, "file_create_date must have exactly 1 entry");
                assert_eq!(v[0].as_str(), ts);
            }
            other => panic!("expected String or StringArray, got {:?}", other),
        }
    }

    /// A file built with NwbFileBuilder::new and all required groups must
    /// report no violations for timestamps_reference_time or file_create_date.
    #[test]
    fn validate_conformance_passes_with_all_extended_attrs() {
        let ts = "2023-07-04T12:00:00Z";
        let mut builder = NwbFileBuilder::new("2.7.0", "full-id", "full desc", ts).unwrap();
        for group in &[
            "acquisition",
            "analysis",
            "processing",
            "stimulus",
            "general",
        ] {
            builder.write_empty_group(group).unwrap();
        }
        let bytes = builder.finish().unwrap();
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        let trt_violations: alloc::vec::Vec<_> = report
            .violations()
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    crate::validation::ConformanceViolation::MissingRootAttribute { name }
                    | crate::validation::ConformanceViolation::InvalidRootAttributeValue {
                        name, ..
                    } if name == "timestamps_reference_time" || name == "file_create_date"
                )
            })
            .collect();
        assert!(
            trt_violations.is_empty(),
            "unexpected timestamps_reference_time or file_create_date violations: {:?}",
            report.violations()
        );
    }

    /// A file whose root group lacks timestamps_reference_time must report
    /// MissingRootAttribute for that attribute.
    #[test]
    fn validate_conformance_reports_missing_timestamps_reference_time() {
        let ts = "2023-01-01T00:00:00Z";
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id"),
            ("session_description", "desc"),
            ("session_start_time", ts),
            // timestamps_reference_time intentionally omitted
            ("file_create_date", ts),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        for group in &[
            "acquisition",
            "analysis",
            "processing",
            "stimulus",
            "general",
        ] {
            builder.add_group_with_attributes(group, &[], &[]).unwrap();
        }
        let bytes = builder.finish().unwrap();
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        assert!(
            report.violations().iter().any(|v| matches!(
                v,
                crate::validation::ConformanceViolation::MissingRootAttribute { name }
                    if name == "timestamps_reference_time"
            )),
            "expected MissingRootAttribute(timestamps_reference_time): {:?}",
            report.violations()
        );
    }

    /// A file whose root group lacks file_create_date must report
    /// MissingRootAttribute for that attribute.
    #[test]
    fn validate_conformance_reports_missing_file_create_date() {
        let ts = "2023-01-01T00:00:00Z";
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id"),
            ("session_description", "desc"),
            ("session_start_time", ts),
            ("timestamps_reference_time", ts),
            // file_create_date intentionally omitted
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        for group in &[
            "acquisition",
            "analysis",
            "processing",
            "stimulus",
            "general",
        ] {
            builder.add_group_with_attributes(group, &[], &[]).unwrap();
        }
        let bytes = builder.finish().unwrap();
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        assert!(
            report.violations().iter().any(|v| matches!(
                v,
                crate::validation::ConformanceViolation::MissingRootAttribute { name }
                    if name == "file_create_date"
            )),
            "expected MissingRootAttribute(file_create_date): {:?}",
            report.violations()
        );
    }

    /// A DynamicTable root-level group without a `colnames` attribute must
    /// produce a GroupMissingAttribute violation for that group.
    ///
    /// The file is built manually using Hdf5FileBuilder because NwbFileBuilder
    /// does not expose its internal hdf5 field (private invariant).
    #[test]
    fn validate_conformance_reports_dynamic_table_missing_colnames() {
        let ts = "2023-01-01T00:00:00Z";
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        // Write all 7 required root attributes.
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", "id"),
            ("session_description", "desc"),
            ("session_start_time", ts),
            ("timestamps_reference_time", ts),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }
        // file_create_date as 1-D array.
        let (fcd_dt, fcd_raw) = fixed_string_dt(ts);
        let fcd_shape = Shape::fixed(&[1]);
        builder
            .add_root_attribute("file_create_date", &fcd_dt, &fcd_shape, &fcd_raw)
            .unwrap();
        // Required top-level groups.
        for group in &[
            "acquisition",
            "analysis",
            "processing",
            "stimulus",
            "general",
        ] {
            builder.add_group_with_attributes(group, &[], &[]).unwrap();
        }
        // DynamicTable group without colnames — layer 5 must report this.
        let (ndt_dt, ndt_raw) = fixed_string_dt("DynamicTable");
        builder
            .add_group_with_attributes(
                "my_table",
                &[("neurodata_type_def", &ndt_dt, &scalar, &ndt_raw)],
                &[],
            )
            .unwrap();
        let bytes = builder.finish().unwrap();
        let file = NwbFile::open(&bytes).unwrap();
        let report = file.validate_conformance().unwrap();
        assert!(
            report.violations().iter().any(|v| matches!(
                v,
                crate::validation::ConformanceViolation::GroupMissingAttribute {
                    group_path,
                    attr_name,
                } if group_path == "my_table" && attr_name == "colnames"
            )),
            "expected GroupMissingAttribute(my_table, colnames): {:?}",
            report.violations()
        );
    }
}
