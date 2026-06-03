//! Integration tests for consus-hdmf: roundtrip, multi-column, ragged, Python-compat.

use consus_hdmf::{ColumnData, HdmfFile, HdmfFileBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_basic_table() -> Vec<u8> {
    HdmfFileBuilder::new("test_table", "a test table")
        .add_column("x", "x values", ColumnData::F64(vec![1.0, 2.0, 3.0]))
        .add_column(
            "y",
            "y labels",
            ColumnData::Str(vec![
                String::from("a"),
                String::from("bb"),
                String::from("ccc"),
            ]),
        )
        .add_column(
            "flag",
            "boolean flags",
            ColumnData::Bool(vec![true, false, true]),
        )
        .add_column("idx", "integer index", ColumnData::I64(vec![10, 20, 30]))
        .add_column("uid", "uint values", ColumnData::U64(vec![100, 200, 300]))
        .finish()
        .expect("build_basic_table failed")
}

// ---------------------------------------------------------------------------
// Roundtrip
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_basic_table() {
    let bytes = build_basic_table();
    let file = HdmfFile::open(&bytes).expect("open failed");
    let table = file.read_table().expect("read_table failed");

    assert_eq!(table.description, "a test table");
    assert_eq!(table.colnames, &["x", "y", "flag", "idx", "uid"]);
    assert_eq!(table.id, &[0i64, 1, 2]);
    assert_eq!(table.columns.len(), 5);

    // x — f64
    assert_eq!(table.columns[0].name, "x");
    assert_eq!(table.columns[0].description, "x values");
    match &table.columns[0].data {
        ColumnData::F64(v) => assert_eq!(v, &[1.0, 2.0, 3.0]),
        other => panic!("expected F64, got {:?}", other),
    }

    // y — str
    assert_eq!(table.columns[1].name, "y");
    match &table.columns[1].data {
        ColumnData::Str(v) => assert_eq!(v, &["a", "bb", "ccc"]),
        other => panic!("expected Str, got {:?}", other),
    }

    // flag — bool (encoded as uint8, reads back as U64)
    assert_eq!(table.columns[2].name, "flag");
    match &table.columns[2].data {
        ColumnData::U64(v) => assert_eq!(v, &[1u64, 0, 1]),
        other => panic!("expected U64 (bool-as-uint8), got {:?}", other),
    }

    // idx — i64
    assert_eq!(table.columns[3].name, "idx");
    match &table.columns[3].data {
        ColumnData::I64(v) => assert_eq!(v, &[10i64, 20, 30]),
        other => panic!("expected I64, got {:?}", other),
    }

    // uid — u64
    assert_eq!(table.columns[4].name, "uid");
    match &table.columns[4].data {
        ColumnData::U64(v) => assert_eq!(v, &[100u64, 200, 300]),
        other => panic!("expected U64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Empty table
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_empty_table() {
    let bytes = HdmfFileBuilder::new("empty_table", "no rows")
        .finish()
        .expect("finish failed");

    let file = HdmfFile::open(&bytes).expect("open failed");
    let table = file.read_table().expect("read_table failed");
    assert_eq!(table.description, "no rows");
    assert!(table.colnames.is_empty());
    assert!(table.id.is_empty());
    assert!(table.columns.is_empty());
}

// ---------------------------------------------------------------------------
// Single-row table
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_single_row() {
    let bytes = HdmfFileBuilder::new("single", "one row")
        .add_column("val", "a value", ColumnData::F64(vec![3.14]))
        .finish()
        .expect("finish failed");

    let file = HdmfFile::open(&bytes).expect("open failed");
    let table = file.read_table().expect("read_table failed");
    assert_eq!(table.id, &[0i64]);
    match &table.columns[0].data {
        ColumnData::F64(v) => {
            assert!((v[0] - 3.14).abs() < 1e-12);
        }
        other => panic!("expected F64, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Ragged column with VectorIndex
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_ragged_column() {
    // 3 rows with variable-length groups: [1,2], [3,4,5], [6]
    let flat_data = ColumnData::I64(vec![1, 2, 3, 4, 5, 6]);
    // cumulative end indices
    let index = vec![2u64, 5, 6];

    let bytes = HdmfFileBuilder::new("ragged_table", "ragged test")
        .add_ragged_column("events", "event ids per row", flat_data, index.clone())
        .finish()
        .expect("finish failed");

    let file = HdmfFile::open(&bytes).expect("open failed");
    let table = file.read_table().expect("read_table failed");

    assert_eq!(table.colnames, &["events"]);
    match &table.columns[0].data {
        ColumnData::I64(v) => assert_eq!(v, &[1, 2, 3, 4, 5, 6]),
        other => panic!("expected I64, got {:?}", other),
    }
    assert_eq!(table.columns[0].index.as_deref(), Some(index.as_slice()));
}

// ---------------------------------------------------------------------------
// HDF5 identity attributes validation
// ---------------------------------------------------------------------------

#[test]
fn root_attributes_are_correct() {
    use consus_hdf5::file::Hdf5File;
    use consus_io::SliceReader;

    let bytes = build_basic_table();
    let raw = Hdf5File::open(SliceReader::new(&bytes)).expect("hdf5 open");
    let root = raw.superblock().root_group_address;
    let attrs = raw.attributes_at(root).expect("attrs");

    let data_type = attrs
        .iter()
        .find(|a| a.name == "data_type")
        .expect("no data_type")
        .decode_value()
        .expect("decode");
    assert_eq!(format!("{:?}", data_type), "String(\"DynamicTable\")");

    let namespace = attrs
        .iter()
        .find(|a| a.name == "namespace")
        .expect("no namespace")
        .decode_value()
        .expect("decode");
    assert_eq!(format!("{:?}", namespace), "String(\"hdmf-common\")");
}
