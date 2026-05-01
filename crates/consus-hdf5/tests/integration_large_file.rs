use consus_core::{Datatype, Shape};
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use consus_hdf5::file::Hdf5File;
use consus_io::{Length, ReadAt};

/// A virtual file that simulates a file > 4 GiB in size without allocating memory.
///
/// It does this by exposing the HDF5 Superblock (first 48 bytes) at physical offset 0,
/// and shifting the rest of the HDF5 data (object headers, B-trees, datasets)
/// to a virtual offset (e.g. 5 GiB).
struct VirtualLargeFile {
    original: Vec<u8>,
    shift: u64,
    raw_dataset_bytes: Vec<u8>,
}

impl ReadAt for VirtualLargeFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> consus_core::Result<()> {
        // Superblock V2 with 8-byte offsets and 8-byte lengths is 48 bytes.
        let sb_size = 48;

        for i in 0..buf.len() {
            let pos = offset + i as u64;
            if pos < self.original.len() as u64 {
                // Serve metadata from 0..original.len()
                buf[i] = self.original[pos as usize];
            } else if pos >= self.shift && pos < self.shift + self.raw_dataset_bytes.len() as u64 {
                // Serve shifted data at 5GiB boundary
                let orig_pos = pos - self.shift;
                buf[i] = self.raw_dataset_bytes[orig_pos as usize];
            } else {
                // Serve zeroes for the gap
                buf[i] = 0;
            }
        }
        Ok(())
    }
}

impl Length for VirtualLargeFile {
    fn len(&self) -> consus_core::Result<u64> {
        Ok(self.original.len() as u64 + self.shift)
    }
}

#[test]
fn test_large_file_address_resolution() {
    // 1. Build a normal HDF5 file in memory.
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());

    let dt = Datatype::Integer {
        byte_order: consus_core::ByteOrder::LittleEndian,
        bits: std::num::NonZeroUsize::new(32).unwrap(),
        signed: true,
    };
    let shape = Shape::fixed(&[3]);
    let raw: Vec<u8> = vec![
        1, 0, 0, 0, // 1
        2, 0, 0, 0, // 2
        3, 0, 0, 0, // 3
    ];

    let shift: u64 = 5 * 1024 * 1024 * 1024; // 5 GiB

    builder
        .add_virtual_dataset("test_data", &dt, &shape, shift, &DatasetCreationProps::default())
        .expect("add dataset");

    let original_bytes = builder.finish().expect("finish builder");

    // 3. Create virtual file and open it.
    let virtual_file = VirtualLargeFile {
        original: original_bytes,
        shift,
        raw_dataset_bytes: raw.clone(),
    };

    let file = Hdf5File::open(virtual_file).expect("open large file");

    // 4. Verify we can read the dataset located past the 5 GiB boundary.
    let addr = file.open_path("/test_data").expect("path exists");
    let dataset = file.dataset_at(addr).expect("dataset exists");

    assert_eq!(dataset.data_address.unwrap(), shift, "address must be 5 GiB");

    let mut out = vec![0u8; 12];
    file.read_contiguous_dataset_bytes(dataset.data_address.unwrap(), 0, &mut out)
        .expect("read dataset bytes");

    assert_eq!(out, raw, "dataset bytes must match");
}
