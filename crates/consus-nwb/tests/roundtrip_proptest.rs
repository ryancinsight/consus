use consus_core::Error as ConsusError;
use consus_nwb::file::{NwbFile, NwbFileBuilder};
use consus_nwb::{ElectrodeTable, UnitsTable};
use proptest::prelude::*;
use std::path::{Path, PathBuf};

fn build_file(bytes_label: &str) -> NwbFileBuilder {
    NwbFileBuilder::new(
        "2.7.0",
        "roundtrip-proptest",
        bytes_label,
        "2024-01-01T00:00:00",
    )
    .expect("builder")
}

fn nwb_fixture_dir() -> PathBuf {
    PathBuf::from(r"D:\consus\data\nwb")
}

fn nwb_fixture_manifest_path() -> PathBuf {
    nwb_fixture_dir().join("manifest.txt")
}

fn nwb_real_sample_path() -> PathBuf {
    nwb_fixture_dir().join("allen_brain_observatory_sample.nwb")
}

fn ascii_string_strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(prop::char::range('\u{20}', '\u{7e}'), 0..=8)
        .prop_map(|chars| chars.into_iter().collect::<String>())
}

fn units_strategy() -> impl Strategy<Value = (Vec<Vec<f64>>, Option<Vec<u64>>)> {
    prop::collection::vec(prop::collection::vec(any::<f64>(), 0..=4), 0..=6).prop_flat_map(
        |spike_times_per_unit| {
            let unit_count = spike_times_per_unit.len();
            let ids_strategy = prop_oneof![
                Just(None),
                prop::collection::vec(any::<u64>(), unit_count).prop_map(Some),
            ];
            ids_strategy.prop_map(move |ids| (spike_times_per_unit.clone(), ids))
        },
    )
}

fn electrodes_strategy() -> impl Strategy<Value = (Vec<u64>, Vec<String>, Vec<String>)> {
    prop::collection::vec(
        (
            any::<u64>(),
            ascii_string_strategy(),
            ascii_string_strategy(),
        ),
        0..=6,
    )
    .prop_map(|rows| {
        let mut ids = Vec::with_capacity(rows.len());
        let mut locations = Vec::with_capacity(rows.len());
        let mut group_names = Vec::with_capacity(rows.len());
        for (id, location, group_name) in rows {
            ids.push(id);
            locations.push(location);
            group_names.push(group_name);
        }
        (ids, locations, group_names)
    })
}

proptest! {
    #[test]
    fn prop_units_table_roundtrip_preserves_partitioning_and_ids(
        (spike_times_per_unit, ids) in units_strategy()
    ) {
        let units = UnitsTable::from_parts(spike_times_per_unit.clone(), ids.clone())
            .expect("matching ids length");

        let mut builder = build_file("units roundtrip");
        builder.write_units_table(&units).expect("write units");
        let bytes = builder.finish().expect("finish file");

        let file = NwbFile::open(&bytes).expect("open file");
        let restored = file.units_table().expect("read units");

        prop_assert_eq!(restored.num_units(), units.num_units());
        prop_assert_eq!(restored.spike_times_per_unit(), units.spike_times_per_unit());
        prop_assert_eq!(restored.ids(), units.ids());
        prop_assert_eq!(restored.flat_spike_times(), units.flat_spike_times());
        prop_assert_eq!(restored.cumulative_index(), units.cumulative_index());

        let reconstructed_flat: Vec<f64> = restored.spike_times_per_unit()
            .iter()
            .flatten()
            .copied()
            .collect();
        prop_assert_eq!(reconstructed_flat, restored.flat_spike_times());

        let mut cumulative = Vec::with_capacity(restored.num_units());
        let mut sum = 0u64;
        for unit in restored.spike_times_per_unit().iter() {
            sum += unit.len() as u64;
            cumulative.push(sum);
        }
        prop_assert_eq!(cumulative, restored.cumulative_index());

        if let Some(restored_ids) = restored.ids() {
            prop_assert_eq!(restored_ids.len(), restored.num_units());
        } else {
            prop_assert!(ids.is_none());
        }
    }

    #[test]
    fn prop_electrode_table_roundtrip_preserves_rows(
        (ids, locations, group_names) in electrodes_strategy()
    ) {
        let table = ElectrodeTable::from_columns(ids.clone(), locations.clone(), group_names.clone())
            .expect("matching column lengths");

        let mut builder = build_file("electrodes roundtrip");
        builder.write_electrode_table(&table).expect("write electrodes");
        let bytes = builder.finish().expect("finish file");

        let file = NwbFile::open(&bytes).expect("open file");
        let restored = file.electrode_table().expect("read electrodes");

        prop_assert_eq!(restored.rows().len(), table.rows().len());
        prop_assert_eq!(restored.rows(), table.rows());

        let restored_ids: Vec<u64> = restored.id_column().collect();
        let restored_locations: Vec<&str> = restored.location_column().collect();
        let restored_group_names: Vec<&str> = restored.group_name_column().collect();

        prop_assert_eq!(restored_ids, ids);
        prop_assert_eq!(restored_locations, locations.iter().map(String::as_str).collect::<Vec<_>>());
        prop_assert_eq!(restored_group_names, group_names.iter().map(String::as_str).collect::<Vec<_>>());
    }
}

#[test]
fn units_table_from_parts_rejects_id_length_mismatch() {
    let err = UnitsTable::from_parts(
        vec![vec![0.1_f64], vec![0.2_f64, 0.3_f64]],
        Some(vec![7_u64]),
    )
    .expect_err("length mismatch must fail");

    match err {
        consus_core::Error::InvalidFormat { message } => {
            assert!(message.contains("ids.len()=1"));
            assert!(message.contains("spike_times_per_unit.len()=2"));
        }
        other => panic!("expected InvalidFormat, got {other:?}"),
    }
}

#[test]
fn electrode_table_from_columns_rejects_length_mismatch() {
    let err = ElectrodeTable::from_columns(
        vec![1_u64, 2_u64],
        vec!["CA1".to_owned()],
        vec!["grp0".to_owned(), "grp1".to_owned()],
    )
    .expect_err("length mismatch must fail");

    match err {
        ConsusError::InvalidFormat { message } => {
            assert!(message.contains("ids=2"));
            assert!(message.contains("location=1"));
            assert!(message.contains("group_name=2"));
        }
        other => panic!("expected InvalidFormat, got {other:?}"),
    }
}

#[test]
fn nwb_fixture_directory_guard_fails_when_real_sample_is_absent() {
    let dir = nwb_fixture_dir();
    let manifest = nwb_fixture_manifest_path();
    let sample = nwb_real_sample_path();

    assert_eq!(dir, Path::new(r"D:\consus\data\nwb"));
    assert_eq!(manifest, Path::new(r"D:\consus\data\nwb\manifest.txt"));
    assert_eq!(
        sample,
        Path::new(r"D:\consus\data\nwb\allen_brain_observatory_sample.nwb")
    );
    assert!(
        !sample.exists(),
        "real NWB sample must be acquired into D:\\consus\\data\\nwb before this test can pass"
    );
}
