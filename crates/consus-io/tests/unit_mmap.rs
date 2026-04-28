//! Integration tests for MmapReader.
//!
//! These tests verify that MmapReader integrates correctly with the
//! consus-io trait hierarchy when opened from real files.

#[cfg(feature = "mmap")]
mod mmap_tests {
    use consus_io::{Length, MmapReader, ReadAt};
    use std::io::Write;

    fn write_temp(payload: &[u8]) -> (std::path::PathBuf, impl Drop) {
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let path = std::env::temp_dir().join(format!(
            "consus_mmap_integration_{}.bin",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let mut f = std::fs::File::create(&path).expect("create temp");
        f.write_all(payload).expect("write temp");
        f.flush().expect("flush");
        let guard = Guard(path.clone());
        (path, guard)
    }

    #[test]
    fn integration_mmap_read_large_payload() {
        // 64 KiB sequential payload for mmap coverage.
        let payload: Vec<u8> = (0u8..=255).cycle().take(65536).collect();
        let (path, _g) = write_temp(&payload);
        let reader = MmapReader::open(&path).expect("open must succeed");
        assert_eq!(reader.len().unwrap(), 65536);

        // Read a 256-byte window at offset 1024.
        let mut buf = vec![0u8; 256];
        reader
            .read_at(1024, &mut buf)
            .expect("read_at must succeed");
        assert_eq!(&buf, &payload[1024..1280]);
    }

    #[test]
    fn integration_mmap_read_last_bytes() {
        let payload = b"end_marker";
        let (path, _g) = write_temp(payload);
        let reader = MmapReader::open(&path).expect("open must succeed");
        let n = payload.len() as u64;
        // Read the last 3 bytes.
        let mut buf = [0u8; 3];
        reader
            .read_at(n - 3, &mut buf)
            .expect("read_at must succeed");
        assert_eq!(&buf, b"ker");
    }

    #[test]
    fn integration_mmap_length_matches_file_size() {
        let payload = vec![0xABu8; 4096];
        let (path, _g) = write_temp(&payload);
        let reader = MmapReader::open(&path).expect("open must succeed");
        assert_eq!(reader.len().unwrap(), 4096);
    }
}
