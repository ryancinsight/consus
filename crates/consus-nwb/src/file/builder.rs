#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use consus_core::{ByteOrder, Datatype, Result, Shape, StringEncoding};
use consus_hdf5::file::writer::Hdf5FileBuilder;
#[cfg(feature = "alloc")]
use consus_hdf5::file::writer::{
    ChildDatasetSpec, ChildGroupSpec, DatasetCreationProps, FileCreationProps,
};
#[cfg(feature = "alloc")]
use core::num::NonZeroUsize;

#[cfg(feature = "alloc")]
use crate::metadata::NwbSubjectMetadata;
#[cfg(feature = "alloc")]
use crate::model::TimeSeries;

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
        bits: NonZeroUsize::new(64).expect("nonzero literal"),
        byte_order: ByteOrder::LittleEndian,
    }
}

#[cfg(feature = "alloc")]
fn f32_le_datatype() -> Datatype {
    Datatype::Float {
        bits: NonZeroUsize::new(32).expect("nonzero literal"),
        byte_order: ByteOrder::LittleEndian,
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
            bits: NonZeroUsize::new(64).expect("nonzero literal"),
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
            bits: NonZeroUsize::new(64).expect("nonzero literal"),
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
            loc_raw.resize(loc_raw.len() + (loc_max - s.len()), 0u8);
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
            grp_raw.resize(grp_raw.len() + (grp_max - s.len()), 0u8);
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
