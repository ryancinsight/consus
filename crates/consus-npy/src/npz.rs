//! Named NPY arrays stored in ZIP archives.

use std::io::{Read, Seek, Write};

use crate::{NpyArray, NpyElement, Result, read_npy, write_npy};

/// Typed NPZ archive reader.
pub struct NpzReader<R: Read + Seek> {
    archive: zip::ZipArchive<R>,
}

impl<R: Read + Seek> NpzReader<R> {
    /// Opens an NPZ archive.
    pub fn new(reader: R) -> Result<Self> {
        Ok(Self {
            archive: zip::ZipArchive::new(reader)?,
        })
    }

    /// Reads a named NPY member.
    pub fn by_name<T: NpyElement>(&mut self, name: &str) -> Result<NpyArray<T>> {
        let member = if name.ends_with(".npy") {
            name.to_owned()
        } else {
            format!("{name}.npy")
        };
        read_npy(self.archive.by_name(&member)?)
    }
}

/// Typed NPZ archive writer.
pub struct NpzWriter<W: Write + Seek> {
    archive: zip::ZipWriter<W>,
}

impl<W: Write + Seek> NpzWriter<W> {
    /// Creates an NPZ archive writer.
    pub fn new(writer: W) -> Self {
        Self {
            archive: zip::ZipWriter::new(writer),
        }
    }

    /// Adds a named typed array.
    pub fn add_array<T: NpyElement>(&mut self, name: &str, array: &NpyArray<T>) -> Result<()> {
        let member = if name.ends_with(".npy") {
            name.to_owned()
        } else {
            format!("{name}.npy")
        };
        self.archive.start_file(
            member,
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated),
        )?;
        write_npy(&mut self.archive, array)
    }

    /// Finishes the ZIP central directory and returns the writer.
    pub fn finish(self) -> Result<W> {
        Ok(self.archive.finish()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn archive_round_trip_is_name_and_value_semantic() {
        let array = NpyArray::new([3], [2_i64, 1, 1]).unwrap();
        let mut writer = NpzWriter::new(Cursor::new(Vec::new()));
        writer.add_array("focus_idx", &array).unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        let mut reader = NpzReader::new(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.by_name::<i64>("focus_idx").unwrap(), array);
    }
}
