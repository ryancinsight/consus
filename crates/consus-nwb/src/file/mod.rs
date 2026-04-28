//! NWBFile top-level container.
//!
//! An NWBFile is an HDF5 file conforming to the NWB 2.x specification.
//! This module provides the entry point for opening and reading NWB files.
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
//!   └── time_series(path)        — read a TimeSeries neurodata group
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
    ChildDatasetSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder,
};
#[cfg(feature = "alloc")]
use core::num::NonZeroUsize;

use consus_core::Result;
use consus_hdf5::file::Hdf5File;
use consus_io::SliceReader;

use crate::metadata::NwbSessionMetadata;
use crate::model::TimeSeries;
use crate::version::NwbVersion;

// ---------------------------------------------------------------------------
// Module-level write helpers
// ---------------------------------------------------------------------------

/// Encode `value` as a fixed-width ASCII [`Datatype::FixedString`] and return
/// the datatype + null-padded raw bytes pair.
///
/// `length` is `value.len().max(1)` so empty strings produce a 1-byte slot.
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

/// Return the canonical HDF5 datatype for 64-bit IEEE 754 little-endian floats.
#[cfg(feature = "alloc")]
fn f64_le_datatype() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

/// Return the canonical HDF5 datatype for 32-bit IEEE 754 little-endian floats.
#[cfg(feature = "alloc")]
fn f32_le_datatype() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    }
}

// ---------------------------------------------------------------------------
// NwbFile
// ---------------------------------------------------------------------------

/// Top-level NWB 2.x file reader.
///
/// Wraps an [`Hdf5File`] opened over a borrowed byte slice and exposes
/// typed accessors for NWB session metadata and neurodata objects.
///
/// ## Lifetime
///
/// The `'a` parameter binds the `NwbFile` to the slice it was opened from.
/// The slice must remain valid for the lifetime of the `NwbFile`.
///
/// ## Example
///
/// ```ignore
/// # #[cfg(feature = "alloc")] {
/// use consus_nwb::file::NwbFile;
///
/// let bytes: &[u8] = /* NWB file bytes */;
/// let nwb = NwbFile::open(bytes).unwrap();
/// let meta = nwb.session_metadata().unwrap();
/// println!("{}", meta.identifier());
/// # }
/// ```
pub struct NwbFile<'a> {
    hdf5: Hdf5File<SliceReader<'a>>,
}

impl<'a> core::fmt::Debug for NwbFile<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NwbFile").finish_non_exhaustive()
    }
}

#[cfg(feature = "alloc")]
impl<'a> NwbFile<'a> {
    /// Open an NWB file from a byte slice.
    ///
    /// Parses the HDF5 superblock, validates the root group attributes
    /// required by the NWB 2.x specification, and returns a ready-to-use
    /// `NwbFile` handle.
    ///
    /// ## Validation
    ///
    /// Delegates to [`crate::validation::validate_root_attributes`], which
    /// checks:
    /// - Root group has `neurodata_type_def == "NWBFile"`.
    /// - Root group has a `nwb_version` attribute.
    ///
    /// ## Errors
    ///
    /// - Propagates any HDF5 parse error from [`Hdf5File::open`].
    /// - [`consus_core::Error::InvalidFormat`] when root-group validation fails.
    /// - [`consus_core::Error::NotFound`] when a required attribute is absent.
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        let reader = SliceReader::new(bytes);
        let hdf5 = Hdf5File::open(reader)?;
        crate::validation::validate_root_attributes(&hdf5)?;
        Ok(Self { hdf5 })
    }

    // -----------------------------------------------------------------------
    // Version
    // -----------------------------------------------------------------------

    /// Detect the NWB specification version from the root group's
    /// `nwb_version` attribute.
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::NotFound`] when the attribute is absent.
    /// - [`consus_core::Error::InvalidFormat`] when the attribute value cannot
    ///   be decoded as a string.
    pub fn nwb_version(&self) -> Result<NwbVersion> {
        crate::version::detect_version(&self.hdf5)
    }

    // -----------------------------------------------------------------------
    // Session metadata
    // -----------------------------------------------------------------------

    /// Read the required NWB session-level metadata from the root group.
    ///
    /// Reads and decodes the three required scalar string attributes:
    /// `identifier`, `session_description`, and `session_start_time`.
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::NotFound`] when any required attribute is absent.
    /// - [`consus_core::Error::InvalidFormat`] when an attribute value cannot be
    ///   decoded as a string.
    /// - Propagates any I/O or format error from the underlying HDF5 reader.
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

    // -----------------------------------------------------------------------
    // Neurodata objects
    // -----------------------------------------------------------------------

    /// Read a `TimeSeries` neurodata group at the given HDF5 path.
    ///
    /// Navigates to `/{path}/data` and reads the primary data array as
    /// `Vec<f64>`.  If `/{path}/timestamps` is present it is read as the
    /// per-sample timestamp array; otherwise `starting_time` and `rate`
    /// attributes on `/{path}/starting_time` are attempted.
    ///
    /// The `name` of the returned `TimeSeries` is the last path component of
    /// `path` (the text after the final `/`, or `path` itself if it contains
    /// no `/`).
    ///
    /// ## Arguments
    ///
    /// - `path` — HDF5 path to the TimeSeries group, relative to the file
    ///   root (e.g. `"acquisition/my_ts"` or `"my_ts"`).
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::NotFound`] when the group or its `data`
    ///   dataset does not exist at `path`.
    /// - [`consus_core::Error::UnsupportedFeature`] when the `data` dataset
    ///   uses a type that cannot be decoded as `f64` (e.g. integer datasets).
    /// - Propagates any I/O or format error from the underlying HDF5 reader.
    pub fn time_series(&self, path: &str) -> Result<TimeSeries> {
        // Resolve the data dataset.
        let data_path = format!("{}/data", path);
        let data_addr = self.hdf5.open_path(&data_path)?;
        let data = crate::storage::read_f64_dataset(&self.hdf5, data_addr)?;

        // Attempt to read timestamps.
        let timestamps_path = format!("{}/timestamps", path);
        let timestamps = match self.hdf5.open_path(&timestamps_path) {
            Ok(ts_addr) => Some(crate::storage::read_f64_dataset(&self.hdf5, ts_addr)?),
            Err(_) => None,
        };

        // Attempt to read starting_time + rate when no timestamps are present.
        // Per NWB 2.x spec: `starting_time` is a scalar dataset at
        // `{path}/starting_time`; `rate` is a float32 attribute on that dataset.
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

        // Derive the TimeSeries name from the last path component.
        let name = path
            .split('/')
            .filter(|s| !s.is_empty())
            .last()
            .unwrap_or(path)
            .to_owned();

        Ok(TimeSeries::from_parts(
            name,
            data,
            timestamps,
            starting_time,
            rate,
        ))
    }

    // -----------------------------------------------------------------------
    // Units
    // -----------------------------------------------------------------------

    /// Read the `spike_times` dataset from the `Units` group.
    ///
    /// Reads `Units/spike_times` as a `Vec<f64>`, promoting integer datasets
    /// via the same path as [`Self::time_series`].
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::NotFound`] when the `Units` group or
    ///   `spike_times` dataset is absent.
    /// - Propagates I/O errors from the underlying HDF5 reader.
    pub fn units_spike_times(&self) -> Result<Vec<f64>> {
        let addr = self.hdf5.open_path("Units/spike_times")?;
        crate::storage::read_f64_dataset(&self.hdf5, addr)
    }

    // -----------------------------------------------------------------------
    // Container group traversal
    // -----------------------------------------------------------------------

    /// List HDF5 paths of all `TimeSeries` (and known subtype) groups inside
    /// the container group at `group_path`.
    ///
    /// Scans the direct children of `group_path`, retaining those whose
    /// `neurodata_type_def` or `neurodata_type_inc` attribute identifies them
    /// as a TimeSeries type via [`crate::conventions::is_timeseries_type`].
    ///
    /// Returns paths relative to the file root.  Example: if
    /// `group_path = "acquisition"` and there is a child named `"lick_times"`,
    /// the returned entry is `"acquisition/lick_times"`.
    ///
    /// Pass `""` to scan the root group.
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::NotFound`] when `group_path` does not exist.
    /// - Propagates I/O errors from listing children.
    pub fn list_time_series(&self, group_path: &str) -> Result<Vec<String>> {
        use crate::conventions::is_timeseries_type;
        let children = crate::group::list_typed_group_children(&self.hdf5, group_path)?;
        let prefix = group_path.trim_end_matches('/');
        let paths: Vec<String> = children
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
}

// ---------------------------------------------------------------------------
// NwbFileBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing NWB 2.x files.
///
/// Wraps [`consus_hdf5::file::writer::Hdf5FileBuilder`] and abstracts the
/// NWB-specific group and attribute structure.  The five required root-group
/// attributes are written on construction; TimeSeries and Units groups are
/// added via [`write_time_series`][NwbFileBuilder::write_time_series] and
/// [`write_units`][NwbFileBuilder::write_units].
///
/// ## NWB 2.x Root Attribute Contract
///
/// | Attribute              | Value                                   |
/// |------------------------|-----------------------------------------|
/// | `neurodata_type_def`   | `"NWBFile"`                             |
/// | `nwb_version`          | caller-supplied string, e.g. `"2.7.0"` |
/// | `identifier`           | caller-supplied non-empty string        |
/// | `session_description`  | caller-supplied non-empty string        |
/// | `session_start_time`   | ISO 8601 string                         |
///
/// ## Example
///
/// ```
/// # #[cfg(feature = "alloc")] {
/// use consus_nwb::file::{NwbFile, NwbFileBuilder};
/// use consus_nwb::model::TimeSeries;
///
/// let mut builder = NwbFileBuilder::new(
///     "2.7.0",
///     "sub-001_ses-20230101",
///     "Freely moving mouse on linear track",
///     "2023-01-01T09:00:00+00:00",
/// ).unwrap();
///
/// let ts = TimeSeries::with_timestamps(
///     "lick_times",
///     vec![0.1, 0.2, 0.3],
///     vec![0.0, 0.1, 0.2],
/// );
/// builder.write_time_series(&ts).unwrap();
///
/// let bytes = builder.finish().unwrap();
/// let nwb = NwbFile::open(&bytes).unwrap();
/// let read_ts = nwb.time_series("lick_times").unwrap();
/// assert_eq!(read_ts.data(), ts.data());
/// # }
/// ```
#[cfg(feature = "alloc")]
pub struct NwbFileBuilder {
    hdf5: Hdf5FileBuilder,
}

#[cfg(feature = "alloc")]
impl core::fmt::Debug for NwbFileBuilder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NwbFileBuilder").finish_non_exhaustive()
    }
}

#[cfg(feature = "alloc")]
impl NwbFileBuilder {
    /// Construct an `NwbFileBuilder` with the five required NWB root attributes.
    ///
    /// Writes `neurodata_type_def = "NWBFile"`, `nwb_version`, `identifier`,
    /// `session_description`, and `session_start_time` as scalar FixedString
    /// ASCII attributes on the root HDF5 group.
    ///
    /// ## Conformance
    ///
    /// `identifier` and `session_description` must be non-empty.
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::InvalidFormat`] when `identifier` or
    ///   `session_description` is empty.
    /// - Propagates HDF5 encoding errors.
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

        let attrs: &[(&str, &str)] = &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", nwb_version),
            ("identifier", identifier.as_str()),
            ("session_description", session_description.as_str()),
            ("session_start_time", session_start_time.as_str()),
        ];
        for (name, value) in attrs {
            let (dt, raw) = fixed_string_bytes(value);
            hdf5.add_root_attribute(name, &dt, &scalar, &raw)?;
        }

        Ok(Self { hdf5 })
    }

    /// Write a [`TimeSeries`] group into the file.
    ///
    /// Calls [`TimeSeries::validate`] and
    /// [`crate::validation::validate_time_series_for_write`] before any
    /// bytes are written.
    ///
    /// ## TimeSeries with explicit timestamps
    ///
    /// Emits:
    /// - `{name}/data`        — f64 LE contiguous array
    /// - `{name}/timestamps`  — f64 LE contiguous array
    ///
    /// ## TimeSeries with uniform rate
    ///
    /// Emits:
    /// - `{name}/data`            — f64 LE contiguous array
    /// - `{name}/starting_time`   — scalar f64 LE dataset with `rate` f32 LE attribute
    ///
    /// Both variants attach `neurodata_type_def = "TimeSeries"` to the group.
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::InvalidFormat`] when structural or conformance
    ///   validation fails.
    /// - Propagates HDF5 encoding errors.
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
            // validate_time_series_for_write guarantees rate is Some here.
            #[allow(clippy::cast_possible_truncation)]
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

    /// Write a `Units` group with a `spike_times` VectorData dataset.
    ///
    /// Emits:
    /// - `Units` group with `neurodata_type_def = "Units"` attribute.
    /// - `Units/spike_times` — f64 LE contiguous array with
    ///   `neurodata_type_def = "VectorData"` and `description = "spike times"` attributes.
    ///
    /// ## Errors
    ///
    /// Propagates HDF5 encoding errors.
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

    /// Finalise the file and return the complete HDF5 image as a byte vector.
    ///
    /// ## Errors
    ///
    /// Propagates HDF5 encoding errors from
    /// [`consus_hdf5::file::writer::Hdf5FileBuilder::finish`].
    pub fn finish(self) -> Result<alloc::vec::Vec<u8>> {
        self.hdf5.finish()
    }
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
    use consus_hdf5::file::writer::{
        ChildDatasetSpec, DatasetCreationProps, FileCreationProps, Hdf5FileBuilder,
    };
    use core::num::NonZeroUsize;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Build the raw bytes for a FixedString HDF5 attribute.
    ///
    /// Produces a version-1 attribute message suitable for passing to
    /// `Hdf5FileBuilder::add_root_attribute`.
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

    /// Build a minimal NWB HDF5 file that passes `NwbFile::open` validation.
    ///
    /// Root group attributes added:
    /// - `neurodata_type_def` = `"NWBFile"`
    /// - `nwb_version`        = `"2.7.0"`
    /// - `identifier`         = `id`
    /// - `session_description`= `desc`
    /// - `session_start_time` = `ts`
    fn make_minimal_nwb(id: &str, desc: &str, ts: &str) -> alloc::vec::Vec<u8> {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", id),
            ("session_description", desc),
            ("session_start_time", ts),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }

        builder.finish().unwrap()
    }

    /// Extend `make_minimal_nwb` with a TimeSeries group `ts_name` containing
    /// a `data` dataset (f64 LE, contiguous) and a `timestamps` dataset.
    fn make_nwb_with_timeseries(
        id: &str,
        ts_name: &str,
        data: &[f64],
        timestamps: &[f64],
    ) -> alloc::vec::Vec<u8> {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        // Root group attributes.
        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", id),
            ("session_description", "TimeSeries test session"),
            ("session_start_time", "2023-01-01T00:00:00+00:00"),
        ] {
            let (dt, raw) = fixed_string_dt(value);
            builder
                .add_root_attribute(name, &dt, &scalar, &raw)
                .unwrap();
        }

        // Encode data as f64 LE bytes.
        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let data_shape = Shape::fixed(&[data.len()]);
        let data_raw: alloc::vec::Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ts_shape = Shape::fixed(&[timestamps.len()]);
        let ts_raw: alloc::vec::Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();

        let dcpl = DatasetCreationProps::default();

        // TimeSeries group: attributes + child datasets (data, timestamps).
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
            )
            .unwrap();

        builder.finish().unwrap()
    }

    // ── NwbFile::open ─────────────────────────────────────────────────────

    #[test]
    fn open_valid_nwb_file_succeeds() {
        let bytes = make_minimal_nwb("test-id-001", "A test session", "2023-01-01T00:00:00+00:00");
        NwbFile::open(&bytes).unwrap();
    }

    #[test]
    fn open_non_hdf5_bytes_returns_error() {
        let bytes = b"not an HDF5 file at all";
        let err = NwbFile::open(bytes).unwrap_err();
        // Must be an HDF5 format error, not a panic.
        match err {
            consus_core::Error::InvalidFormat { .. } | consus_core::Error::NotFound { .. } => {}
            other => panic!("expected format error, got {:?}", other),
        }
    }

    #[test]
    fn open_hdf5_without_neurodata_type_def_returns_invalid_format() {
        // Build a valid HDF5 file but missing the neurodata_type_def attribute.
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();
        let (dt, raw) = fixed_string_dt("2.7.0");
        builder
            .add_root_attribute("nwb_version", &dt, &scalar, &raw)
            .unwrap();
        let bytes = builder.finish().unwrap();

        let err = NwbFile::open(&bytes).unwrap_err();
        assert!(
            matches!(err, consus_core::Error::InvalidFormat { .. }),
            "expected InvalidFormat, got {:?}",
            err
        );
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
        assert!(
            matches!(err, consus_core::Error::NotFound { .. }),
            "expected NotFound, got {:?}",
            err
        );
    }

    // ── NwbFile::nwb_version ──────────────────────────────────────────────

    #[test]
    fn nwb_version_returns_v2_7_for_2_7_0() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let ver = nwb.nwb_version().unwrap();
        assert_eq!(ver, NwbVersion::V2_7);
        assert!(ver.is_supported());
    }

    // ── NwbFile::session_metadata ─────────────────────────────────────────

    #[test]
    fn session_metadata_returns_correct_fields() {
        let bytes = make_minimal_nwb(
            "sub-001_ses-20230101",
            "Freely moving mouse on linear track",
            "2023-01-01T09:00:00+00:00",
        );
        let nwb = NwbFile::open(&bytes).unwrap();
        let meta = nwb.session_metadata().unwrap();

        assert_eq!(meta.identifier(), "sub-001_ses-20230101");
        assert_eq!(
            meta.session_description(),
            "Freely moving mouse on linear track"
        );
        assert_eq!(meta.session_start_time(), "2023-01-01T09:00:00+00:00");
    }

    #[test]
    fn session_metadata_missing_identifier_returns_not_found() {
        // File has all attributes except identifier.
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
        assert!(
            matches!(err, consus_core::Error::NotFound { .. }),
            "expected NotFound for missing identifier, got {:?}",
            err
        );
    }

    // ── NwbFile::time_series ──────────────────────────────────────────────

    #[test]
    fn time_series_reads_data_and_timestamps() {
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let timestamps = [0.0f64, 0.1, 0.2, 0.3, 0.4];
        let bytes = make_nwb_with_timeseries("ses-001", "test_ts", &data, &timestamps);

        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("test_ts").unwrap();

        assert_eq!(ts.name(), "test_ts");
        assert_eq!(ts.data(), data.as_slice());
        assert_eq!(ts.timestamps(), Some(timestamps.as_slice()));
        assert!(ts.starting_time().is_none());
        assert!(ts.rate().is_none());
    }

    #[test]
    fn time_series_validates_length_invariant() {
        let data = [1.0f64, 2.0, 3.0];
        let timestamps = [0.0f64, 0.1, 0.2];
        let bytes = make_nwb_with_timeseries("ses-001", "ts1", &data, &timestamps);
        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("ts1").unwrap();

        // timestamps.len() == data.len() is required by the NWB spec.
        ts.validate().unwrap();
    }

    #[test]
    fn time_series_missing_group_returns_not_found() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let err = nwb.time_series("nonexistent_ts").unwrap_err();
        assert!(
            matches!(err, consus_core::Error::NotFound { .. }),
            "expected NotFound, got {:?}",
            err
        );
    }

    #[test]
    fn time_series_name_derived_from_last_path_component() {
        let data = [0.0f64, 1.0];
        let timestamps = [0.0f64, 0.01];
        let bytes = make_nwb_with_timeseries("ses-001", "my_ts", &data, &timestamps);
        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("my_ts").unwrap();
        assert_eq!(ts.name(), "my_ts");
    }

    // ── helper: NWB TimeSeries with rate-based timing ─────────────────────

    /// Build an NWB file with a TimeSeries group that uses the rate-based
    /// timing representation.
    ///
    /// The group contains:
    /// - `data`          — f64 array dataset
    /// - `starting_time` — scalar f64 dataset with a `rate` float32 attribute
    ///
    /// No `timestamps` dataset is present.
    fn make_nwb_with_rate_timeseries(
        id: &str,
        ts_name: &str,
        data: &[f64],
        starting_time: f64,
        rate: f32,
    ) -> alloc::vec::Vec<u8> {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        let scalar = Shape::scalar();

        for (name, value) in &[
            ("neurodata_type_def", "NWBFile"),
            ("nwb_version", "2.7.0"),
            ("identifier", id),
            ("session_description", "Rate TimeSeries test"),
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
        let data_shape = Shape::fixed(&[data.len()]);
        let data_raw: alloc::vec::Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let st_raw = starting_time.to_le_bytes().to_vec();
        let rate_raw = rate.to_le_bytes().to_vec();
        let dcpl = DatasetCreationProps::default();
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
                        dcpl: dcpl.clone(),
                        attributes: &[],
                    },
                    ChildDatasetSpec {
                        name: "starting_time",
                        datatype: &f64_dt,
                        shape: &scalar,
                        raw_data: &st_raw,
                        dcpl: dcpl.clone(),
                        attributes: &[("rate", &f32_dt, &scalar, &rate_raw)],
                    },
                ],
            )
            .unwrap();

        builder.finish().unwrap()
    }

    // ── NwbFile::time_series — rate-based timing ──────────────────────────

    #[test]
    fn time_series_reads_starting_time_and_rate_from_dataset() {
        // Theorem: starting_time is read from a scalar dataset at
        // `{ts_name}/starting_time`; rate is a float32 attribute on that dataset.
        let data = [0.0f64, -1.0, 1.0];
        let bytes =
            make_nwb_with_rate_timeseries("ses-002", "lfp_ts", &data, 0.5_f64, 30_000.0_f32);

        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("lfp_ts").unwrap();

        assert_eq!(ts.name(), "lfp_ts");
        assert_eq!(ts.data(), data.as_slice());
        assert!(ts.timestamps().is_none(), "should use rate-based timing");

        let st = ts
            .starting_time()
            .expect("starting_time must be present from dataset");
        assert!(
            (st - 0.5_f64).abs() < 1e-9,
            "starting_time should be 0.5, got {st}"
        );

        let r = ts
            .rate()
            .expect("rate must be present from dataset attribute");
        // Stored as f32 30000.0, widened to f64; tolerance covers f32→f64 cast.
        assert!(
            (r - 30_000.0_f64).abs() < 1.0,
            "rate should be ~30000.0 Hz, got {r}"
        );
    }

    #[test]
    fn time_series_with_timestamps_does_not_read_starting_time() {
        // When timestamps are present, starting_time and rate must be None.
        let data = [1.0f64, 2.0];
        let timestamps = [0.0f64, 0.1];
        let bytes = make_nwb_with_timeseries("ses-003", "ts_with_ts", &data, &timestamps);
        let nwb = NwbFile::open(&bytes).unwrap();
        let ts = nwb.time_series("ts_with_ts").unwrap();

        assert!(ts.timestamps().is_some());
        assert!(
            ts.starting_time().is_none(),
            "starting_time must be None when timestamps are present"
        );
        assert!(
            ts.rate().is_none(),
            "rate must be None when timestamps are present"
        );
    }

    // ── NwbFile::list_time_series ─────────────────────────────────────────

    #[test]
    fn list_time_series_returns_only_timeseries_typed_children() {
        // Root has two groups: "lick_ts" (TimeSeries) and "subject" (Subject).
        // list_time_series("") must return exactly ["lick_ts"] (or "/lick_ts").
        let bytes = {
            let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
            let scalar = Shape::scalar();

            for (name, value) in &[
                ("neurodata_type_def", "NWBFile"),
                ("nwb_version", "2.7.0"),
                ("identifier", "ls-test"),
                ("session_description", "list_ts test"),
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
            let data_raw: alloc::vec::Vec<u8> =
                [1.0f64, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
            let data_shape = Shape::fixed(&[2]);
            let dcpl = DatasetCreationProps::default();

            let (ndt_ts_dt, ndt_ts_raw) = fixed_string_dt("TimeSeries");
            builder
                .add_group_with_attributes(
                    "lick_ts",
                    &[("neurodata_type_def", &ndt_ts_dt, &scalar, &ndt_ts_raw)],
                    &[ChildDatasetSpec {
                        name: "data",
                        datatype: &f64_dt,
                        shape: &data_shape,
                        raw_data: &data_raw,
                        dcpl: dcpl.clone(),
                        attributes: &[],
                    }],
                )
                .unwrap();

            let (ndt_subj_dt, ndt_subj_raw) = fixed_string_dt("Subject");
            builder
                .add_group_with_attributes(
                    "subject",
                    &[("neurodata_type_def", &ndt_subj_dt, &scalar, &ndt_subj_raw)],
                    &[],
                )
                .unwrap();

            builder.finish().unwrap()
        };

        let nwb = NwbFile::open(&bytes).unwrap();
        let mut paths = nwb.list_time_series("").unwrap();
        paths.sort();

        assert_eq!(paths.len(), 1, "expected 1 TimeSeries path, got: {paths:?}");
        assert!(
            paths[0].ends_with("lick_ts"),
            "path should end with 'lick_ts', got: {}",
            paths[0]
        );
    }

    #[test]
    fn list_time_series_known_subtype_is_included() {
        // A group with neurodata_type_def = "ElectricalSeries" is a TimeSeries subtype.
        let bytes = {
            let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
            let scalar = Shape::scalar();
            for (name, value) in &[
                ("neurodata_type_def", "NWBFile"),
                ("nwb_version", "2.7.0"),
                ("identifier", "subtype-test"),
                ("session_description", "subtype test"),
                ("session_start_time", "2023-01-01T00:00:00+00:00"),
            ] {
                let (dt, raw) = fixed_string_dt(value);
                builder
                    .add_root_attribute(name, &dt, &scalar, &raw)
                    .unwrap();
            }
            let (eseries_dt, eseries_raw) = fixed_string_dt("ElectricalSeries");
            builder
                .add_group_with_attributes(
                    "ecog_ts",
                    &[("neurodata_type_def", &eseries_dt, &scalar, &eseries_raw)],
                    &[],
                )
                .unwrap();
            builder.finish().unwrap()
        };

        let nwb = NwbFile::open(&bytes).unwrap();
        let paths = nwb.list_time_series("").unwrap();
        assert_eq!(paths.len(), 1, "ElectricalSeries is a TimeSeries subtype");
        assert!(paths[0].ends_with("ecog_ts"));
    }

    #[test]
    fn list_time_series_nwbfile_root_group_not_returned_as_child() {
        // The root group itself has neurodata_type_def = "NWBFile"; it must not
        // appear in the list — only children of the specified group are returned.
        let bytes = make_minimal_nwb("id2", "desc2", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        // Root has no child groups → list must be empty, not containing itself.
        let paths = nwb.list_time_series("").unwrap();
        assert!(
            paths.is_empty(),
            "root has no TimeSeries children: {paths:?}"
        );
    }

    #[test]
    fn list_time_series_missing_group_returns_not_found() {
        let bytes = make_minimal_nwb("id", "desc", "2023-01-01T00:00:00+00:00");
        let nwb = NwbFile::open(&bytes).unwrap();
        let err = nwb.list_time_series("nonexistent_acquisition").unwrap_err();
        assert!(
            matches!(err, consus_core::Error::NotFound { .. }),
            "expected NotFound, got {:?}",
            err
        );
    }

    // ── NwbFileBuilder ────────────────────────────────────────────────────

    #[test]
    fn nwb_file_builder_minimal_file_opens_successfully() {
        // Theorem: NwbFileBuilder::new writes all five required root attributes;
        // NwbFile::open succeeds and session_metadata returns the correct fields.
        let builder = NwbFileBuilder::new(
            "2.7.0",
            "sub-001_ses-20230101",
            "Freely moving mouse on linear track",
            "2023-01-01T09:00:00+00:00",
        )
        .unwrap();
        let bytes = builder.finish().unwrap();

        let nwb = NwbFile::open(&bytes).unwrap();
        let meta = nwb.session_metadata().unwrap();
        assert_eq!(meta.identifier(), "sub-001_ses-20230101");
        assert_eq!(
            meta.session_description(),
            "Freely moving mouse on linear track"
        );
        assert_eq!(meta.session_start_time(), "2023-01-01T09:00:00+00:00");
    }

    #[test]
    fn nwb_file_builder_empty_identifier_returns_error() {
        // Theorem: empty identifier is rejected before any HDF5 bytes are written.
        let err = NwbFileBuilder::new("2.7.0", "", "valid desc", "2023-01-01T00:00:00+00:00")
            .unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { ref message } => {
                assert!(
                    message.contains("identifier"),
                    "error must mention 'identifier': {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn nwb_file_builder_empty_session_description_returns_error() {
        // Theorem: empty session_description is rejected before any bytes are written.
        let err =
            NwbFileBuilder::new("2.7.0", "valid-id", "", "2023-01-01T00:00:00+00:00").unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { ref message } => {
                assert!(
                    message.contains("session_description"),
                    "error must mention 'session_description': {message}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn write_time_series_with_timestamps_roundtrip() {
        // Theorem: write_time_series → finish → NwbFile::open → time_series
        // preserves name, data, and timestamps exactly.
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let timestamps = [0.0f64, 0.1, 0.2, 0.3, 0.4];
        let ts = TimeSeries::with_timestamps("neural_data", data.to_vec(), timestamps.to_vec());

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-001",
            "test session",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        builder.write_time_series(&ts).unwrap();
        let bytes = builder.finish().unwrap();

        let nwb = NwbFile::open(&bytes).unwrap();
        let read_ts = nwb.time_series("neural_data").unwrap();

        assert_eq!(read_ts.name(), "neural_data");
        assert_eq!(read_ts.data(), data.as_slice());
        assert_eq!(read_ts.timestamps(), Some(timestamps.as_slice()));
        assert!(read_ts.starting_time().is_none());
        assert!(read_ts.rate().is_none());
        read_ts.validate().unwrap();
    }

    #[test]
    fn write_time_series_with_rate_roundtrip() {
        // Theorem: write_time_series with rate representation → finish →
        // NwbFile::open → time_series preserves name, data, starting_time, and rate.
        let data = [0.0f64, -1.0, 1.0, -0.5, 0.5];
        let ts = TimeSeries::with_rate("lfp", data.to_vec(), 1.5_f64, 30_000.0_f64);

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-002",
            "LFP session",
            "2023-06-15T00:00:00+00:00",
        )
        .unwrap();
        builder.write_time_series(&ts).unwrap();
        let bytes = builder.finish().unwrap();

        let nwb = NwbFile::open(&bytes).unwrap();
        let read_ts = nwb.time_series("lfp").unwrap();

        assert_eq!(read_ts.name(), "lfp");
        assert_eq!(read_ts.data(), data.as_slice());
        assert!(read_ts.timestamps().is_none());

        let st = read_ts
            .starting_time()
            .expect("starting_time must be present");
        assert!(
            (st - 1.5_f64).abs() < 1e-9,
            "starting_time should be 1.5, got {st}"
        );

        let r = read_ts.rate().expect("rate must be present");
        // Stored as f32 30000.0, widened to f64; f32 exact range: 30000.0 is representable.
        assert!(
            (r - 30_000.0_f64).abs() < 1.0,
            "rate should be ~30000 Hz, got {r}"
        );
    }

    #[test]
    fn write_multiple_time_series_roundtrip() {
        // Theorem: two write_time_series calls produce independently readable groups.
        let ts1 = TimeSeries::with_timestamps("lick", vec![0.1, 0.2], vec![0.0, 0.5]);
        let ts2 = TimeSeries::with_rate("ecog", vec![1.0, -1.0, 0.5], 0.0_f64, 1000.0_f64);

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-multi",
            "multi-ts session",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        builder.write_time_series(&ts1).unwrap();
        builder.write_time_series(&ts2).unwrap();
        let bytes = builder.finish().unwrap();

        let nwb = NwbFile::open(&bytes).unwrap();

        let r1 = nwb.time_series("lick").unwrap();
        assert_eq!(r1.data(), ts1.data());
        assert_eq!(r1.timestamps(), ts1.timestamps());

        let r2 = nwb.time_series("ecog").unwrap();
        assert_eq!(r2.data(), ts2.data());
        assert!(r2.rate().is_some());
    }

    #[test]
    fn write_empty_time_series_with_timestamps_roundtrip() {
        // Theorem: a TimeSeries with zero samples is structurally valid and round-trips.
        let ts = TimeSeries::with_timestamps("empty_ts", vec![], vec![]);

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-empty",
            "empty session",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        builder.write_time_series(&ts).unwrap();
        let bytes = builder.finish().unwrap();

        let nwb = NwbFile::open(&bytes).unwrap();
        let read_ts = nwb.time_series("empty_ts").unwrap();
        assert!(read_ts.is_empty());
        assert_eq!(read_ts.timestamps(), Some([].as_slice()));
        read_ts.validate().unwrap();
    }

    #[test]
    fn write_time_series_without_timing_returns_conformance_error() {
        // Theorem: a TimeSeries with no timing representation is rejected before
        // any bytes are written via validate_time_series_for_write.
        let ts = TimeSeries::without_timing("bare", vec![1.0, 2.0]);

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-notime",
            "no-timing session",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        let err = builder.write_time_series(&ts).unwrap_err();
        assert!(
            matches!(err, consus_core::Error::InvalidFormat { .. }),
            "expected InvalidFormat for missing timing, got {:?}",
            err
        );
    }

    #[test]
    fn write_time_series_with_zero_rate_returns_error() {
        // Theorem: rate = 0.0 is physically invalid and must be rejected.
        let ts = TimeSeries::with_rate("bad_rate", vec![1.0], 0.0_f64, 0.0_f64);

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-zerorate",
            "zero rate session",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        let err = builder.write_time_series(&ts).unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { ref message } => {
                assert!(
                    message.contains("rate"),
                    "error must mention 'rate': {message}"
                );
            }
            other => panic!("expected InvalidFormat for zero rate, got {:?}", other),
        }
    }

    #[test]
    fn write_time_series_with_negative_rate_returns_error() {
        // Theorem: negative rate is physically invalid and must be rejected.
        let ts = TimeSeries::with_rate("neg_rate", vec![1.0, 2.0], 0.0_f64, -100.0_f64);

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-negrate",
            "negative rate session",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        let err = builder.write_time_series(&ts).unwrap_err();
        assert!(
            matches!(err, consus_core::Error::InvalidFormat { .. }),
            "expected InvalidFormat for negative rate, got {:?}",
            err
        );
    }

    #[test]
    fn write_units_spike_times_roundtrip() {
        // Theorem: write_units → finish → NwbFile::open → units_spike_times
        // returns the original spike times vector exactly.
        let spike_times = [0.01_f64, 0.15, 0.33, 0.72, 1.01, 1.44];

        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-units",
            "spike sorting session",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        builder.write_units(&spike_times).unwrap();
        let bytes = builder.finish().unwrap();

        let nwb = NwbFile::open(&bytes).unwrap();
        let read_st = nwb.units_spike_times().unwrap();
        assert_eq!(read_st.len(), spike_times.len());
        for (got, expected) in read_st.iter().zip(spike_times.iter()) {
            assert!(
                (got - expected).abs() < 1e-12,
                "spike_time mismatch: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn write_units_empty_spike_times_roundtrip() {
        // Theorem: zero-spike Units group round-trips to an empty Vec.
        let mut builder = NwbFileBuilder::new(
            "2.7.0",
            "ses-nospike",
            "no spikes",
            "2023-01-01T00:00:00+00:00",
        )
        .unwrap();
        builder.write_units(&[]).unwrap();
        let bytes = builder.finish().unwrap();

        let nwb = NwbFile::open(&bytes).unwrap();
        let read_st = nwb.units_spike_times().unwrap();
        assert!(read_st.is_empty());
    }
}
