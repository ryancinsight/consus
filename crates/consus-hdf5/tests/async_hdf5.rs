//! Async HDF5 reader integration tests using Moirai's native async surface.

#![cfg(feature = "async")]

use core::future::Future;
use core::num::NonZeroUsize;

use consus_core::{ByteOrder, Datatype, Error, NodeType, ParseBudget, Shape};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use consus_hdf5::file::async_file::AsyncHdf5File;
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_io::MemCursor;
use moirai_async::{AsyncExecutor, AsyncMemReader};

fn run_async<F, T>(future: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    AsyncExecutor::new()
        .expect("Moirai executor must initialize")
        .block_on(future)
}

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
        .add_dataset(
            "scalar_value",
            &dt,
            &shape,
            &raw,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");
    builder.finish().expect("finish")
}

fn dataset_addr_sync(bytes: &[u8], name: &str) -> u64 {
    let cursor = MemCursor::from_bytes(bytes.to_vec());
    let file = Hdf5File::open(cursor).expect("sync open");
    let children = file.list_root_group().expect("list root");
    children
        .iter()
        .find(|(child_name, _, _)| child_name == name)
        .map(|(_, address, _)| *address)
        .unwrap_or_else(|| panic!("dataset {name} not found"))
}

#[test]
fn async_open_reads_correct_superblock_version() {
    run_async(async {
        let file = AsyncHdf5File::open(AsyncMemReader::from_bytes(build_scalar_hdf5()))
            .await
            .expect("must open");
        assert_eq!(file.superblock().version, 2);
        assert_eq!(file.superblock().offset_size, 8);
        assert_eq!(file.superblock().length_size, 8);
        assert!(file.superblock().eof_address > 0);
    });
}

#[test]
fn async_root_node_type_is_group() {
    run_async(async {
        let file = AsyncHdf5File::open(AsyncMemReader::from_bytes(build_scalar_hdf5()))
            .await
            .expect("must open");
        assert_eq!(
            file.root_node_type().await.expect("root_node_type"),
            NodeType::Group
        );
        let root = file.superblock().root_group_address;
        assert_eq!(
            file.node_type_at(root).await.expect("node_type_at"),
            NodeType::Group
        );
    });
}

#[test]
fn async_read_bytes_returns_hdf5_magic() {
    run_async(async {
        let file = AsyncHdf5File::open(AsyncMemReader::from_bytes(build_scalar_hdf5()))
            .await
            .expect("must open");
        assert_eq!(
            file.read_bytes(0, 8).await.expect("read_bytes"),
            b"\x89HDF\r\n\x1a\n"
        );
    });
}

#[test]
fn async_read_bytes_rejects_region_beyond_budget() {
    run_async(async {
        let file = AsyncHdf5File::open(AsyncMemReader::from_bytes(build_scalar_hdf5()))
            .await
            .expect("must open");
        let result = file
            .read_bytes(0, ParseBudget::DEFAULT.max_alloc_bytes + 1)
            .await;
        assert!(matches!(
            result,
            Err(Error::ResourceLimit {
                what: "async HDF5 read region",
                ..
            })
        ));
    });
}

#[test]
fn async_dataset_at_matches_sync_path() {
    run_async(async {
        let bytes = build_scalar_hdf5();
        let address = dataset_addr_sync(&bytes, "scalar_value");
        let sync_file = Hdf5File::open(MemCursor::from_bytes(bytes.clone())).expect("sync open");
        let sync_dataset = sync_file.dataset_at(address).expect("sync dataset_at");
        let async_file = AsyncHdf5File::open(AsyncMemReader::from_bytes(bytes))
            .await
            .expect("async open");
        let async_dataset = async_file
            .dataset_at(address)
            .await
            .expect("async dataset_at");
        assert_eq!(
            async_dataset.object_header_address,
            sync_dataset.object_header_address
        );
        assert_eq!(async_dataset.layout, sync_dataset.layout);
        assert_eq!(
            async_dataset.shape.is_scalar(),
            sync_dataset.shape.is_scalar()
        );
        assert_eq!(
            async_dataset.shape.num_elements(),
            sync_dataset.shape.num_elements()
        );
        assert_eq!(async_dataset.layout, StorageLayout::Contiguous);
    });
}

#[test]
fn async_open_rejects_non_hdf5_and_empty_sources() {
    run_async(async {
        assert!(
            AsyncHdf5File::open(AsyncMemReader::from_bytes(vec![0; 4096]))
                .await
                .is_err()
        );
        assert!(AsyncHdf5File::open(AsyncMemReader::new()).await.is_err());
    });
}
