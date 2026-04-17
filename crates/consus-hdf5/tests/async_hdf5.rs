//! Async HDF5 file reader integration tests.
//!
//! ## Coverage
//!
//! - AsyncHdf5File::open on builder-produced in-memory HDF5 files
//! - Superblock fields (version, offset_size, length_size)
//! - root_node_type() on a real group object header
//! - node_type_at(root_group_address) dispatch
//! - read_bytes returning the HDF5 magic bytes
//! - dataset_at returning correct address, shape, and layout
//! - Rejection of non-HDF5 data and empty sources

#![cfg(feature = "async-io")]

use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype, NodeType, Shape};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_hdf5::file::async_file::AsyncHdf5File;
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_io::{AsyncMemCursor, MemCursor};

fn build_scalar_hdf5() -> Vec<u8> {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let shape = Shape::scalar();
    let raw = 42i32.to_le_bytes();
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset("scalar_value", &dt, &shape, &raw, &DatasetCreationProps::default())
        .expect("add dataset");
    builder.finish().expect("finish")
}

fn dataset_addr_sync(bytes: &[u8], name: &str) -> u64 {
    let cursor = MemCursor::from_bytes(bytes.to_vec());
    let file = Hdf5File::open(cursor).expect("sync open");
    let children = file.list_root_group().expect("list root");
    children
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, addr, _)| *addr)
        .unwrap_or_else(|| panic!("dataset {} not found", name))
}

#[tokio::test]
async fn async_open_reads_correct_superblock_version() {
    let bytes = build_scalar_hdf5();
    let cursor = AsyncMemCursor::from_bytes(bytes);
    let file = AsyncHdf5File::open(cursor).await.expect("must open");
    assert_eq!(file.superblock().version, 2);
}

#[tokio::test]
async fn async_superblock_offset_and_length_size() {
    let bytes = build_scalar_hdf5();
    let cursor = AsyncMemCursor::from_bytes(bytes);
    let file = AsyncHdf5File::open(cursor).await.expect("must open");
    assert_eq!(file.superblock().offset_size, 8);
    assert_eq!(file.superblock().length_size, 8);
    assert_eq!(file.context().offset_size, 8);
    assert_eq!(file.context().length_size, 8);
}

#[tokio::test]
async fn async_superblock_eof_address_nonzero() {
    let bytes = build_scalar_hdf5();
    let cursor = AsyncMemCursor::from_bytes(bytes);
    let file = AsyncHdf5File::open(cursor).await.expect("must open");
    assert!(file.superblock().eof_address > 0, "eof_address must be non-zero");
}

#[tokio::test]
async fn async_root_node_type_is_group() {
    let bytes = build_scalar_hdf5();
    let cursor = AsyncMemCursor::from_bytes(bytes);
    let file = AsyncHdf5File::open(cursor).await.expect("must open");
    let nt = file.root_node_type().await.expect("root_node_type");
    assert_eq!(nt, NodeType::Group);
}

#[tokio::test]
async fn async_node_type_at_root_matches_root_node_type() {
    let bytes = build_scalar_hdf5();
    let cursor = AsyncMemCursor::from_bytes(bytes);
    let file = AsyncHdf5File::open(cursor).await.expect("must open");
    let root_addr = file.superblock().root_group_address;
    let nt_via_method = file.root_node_type().await.expect("root_node_type");
    let nt_via_addr = file.node_type_at(root_addr).await.expect("node_type_at");
    assert_eq!(nt_via_method, nt_via_addr);
    assert_eq!(nt_via_addr, NodeType::Group);
}

#[tokio::test]
async fn async_read_bytes_returns_hdf5_magic() {
    let bytes = build_scalar_hdf5();
    let cursor = AsyncMemCursor::from_bytes(bytes);
    let file = AsyncHdf5File::open(cursor).await.expect("must open");
    let magic = file.read_bytes(0, 8).await.expect("read_bytes");
    assert_eq!(magic.as_slice(), b"\x89HDF\r\n\x1a\n", "first 8 bytes must be HDF5 magic");
}

#[tokio::test]
async fn async_dataset_at_scalar_returns_correct_metadata() {
    let bytes = build_scalar_hdf5();
    let ds_addr = dataset_addr_sync(&bytes, "scalar_value");
    let cursor = AsyncMemCursor::from_bytes(bytes);
    let file = AsyncHdf5File::open(cursor).await.expect("must open");
    let dataset = file.dataset_at(ds_addr).await.expect("dataset_at");
    assert_eq!(dataset.object_header_address, ds_addr, "object_header_address must match");
    assert!(dataset.shape.is_scalar(), "shape must be scalar");
    assert_eq!(dataset.layout, StorageLayout::Contiguous, "must use contiguous layout");
}

#[tokio::test]
async fn async_dataset_at_matches_sync_path() {
    let bytes = build_scalar_hdf5();
    let ds_addr = dataset_addr_sync(&bytes, "scalar_value");
    let sync_cursor = MemCursor::from_bytes(bytes.clone());
    let sync_file = Hdf5File::open(sync_cursor).expect("sync open");
    let sync_dataset = sync_file.dataset_at(ds_addr).expect("sync dataset_at");
    let async_cursor = AsyncMemCursor::from_bytes(bytes);
    let async_file = AsyncHdf5File::open(async_cursor).await.expect("async open");
    let async_dataset = async_file.dataset_at(ds_addr).await.expect("async dataset_at");
    assert_eq!(async_dataset.object_header_address, sync_dataset.object_header_address);
    assert_eq!(async_dataset.layout, sync_dataset.layout);
    assert_eq!(async_dataset.shape.is_scalar(), sync_dataset.shape.is_scalar());
    assert_eq!(async_dataset.shape.num_elements(), sync_dataset.shape.num_elements());
}

#[tokio::test]
async fn async_open_rejects_non_hdf5() {
    let cursor = AsyncMemCursor::from_bytes(vec![0u8; 4096]);
    let result = AsyncHdf5File::open(cursor).await;
    assert!(result.is_err(), "must reject a buffer containing no HDF5 superblock");
}

#[tokio::test]
async fn async_open_rejects_empty_source() {
    let cursor = AsyncMemCursor::new();
    let result = AsyncHdf5File::open(cursor).await;
    assert!(result.is_err(), "must reject an empty source");
}
