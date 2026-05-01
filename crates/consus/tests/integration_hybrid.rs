#[cfg(all(feature = "alloc", feature = "hdf5", feature = "parquet"))]
mod tests {
    use consus::core::Result;
    use consus::hybrid::{read_embedded_parquet, write_embedded_parquet};
    use consus_hdf5::file::writer::{FileCreationProps, Hdf5FileBuilder};
    use consus_hdf5::file::Hdf5File;
    use consus_io::MemCursor;
    use consus_parquet::{
        HybridMode, HybridPartitioning, HybridStorageDescriptor, HybridStorageEncoding,
        HybridTableLayout, HybridTableRelation, ParquetReader, ParquetWriter,
        FieldDescriptor, FieldId, ParquetPhysicalType, ColumnChunkDescriptor,
    };

    #[test]
    fn test_hybrid_mode_roundtrip() -> Result<()> {
        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());

        // 1. Create a Parquet writer and write a simple file.
        let parquet_writer = ParquetWriter::new();
        // Just writing an empty plan to get a valid Parquet payload.
        let schema = consus_parquet::SchemaDescriptor::new(vec![
            FieldDescriptor::required(FieldId::new(1), "col1", ParquetPhysicalType::Int32)
        ]);
        let dataset_desc = consus_parquet::ParquetDatasetDescriptor::new(
            schema,
            vec![consus_parquet::RowGroupDescriptor::new(1, vec![
                ColumnChunkDescriptor::new(FieldId::new(1), 1, 4).unwrap()
            ]).unwrap()],
        ).unwrap();
        
        struct EmptyRowSource;
        impl consus_parquet::RowSource for EmptyRowSource {
            fn row_count(&self) -> usize { 1 }
            fn row(&self, _index: usize) -> consus_core::Result<consus_parquet::RowValue<'_>> {
                Ok(consus_parquet::RowValue::new(vec![consus_parquet::CellValue::Int32(42)]))
            }
        }
        let payload = parquet_writer.write_dataset_bytes(&dataset_desc, &EmptyRowSource).unwrap();
        
        let descriptor = HybridStorageDescriptor::new()
            .with_table_layout(HybridTableLayout::new(
                "my_table".to_string(),
                "/data/payload".to_string(),
                HybridTableRelation::MaterializedView,
                HybridStorageEncoding::ArrowIntermediate,
            ))
            .with_partitioning(HybridPartitioning {
                keys: vec!["year".to_string(), "month".to_string()],
                paths: vec!["2024/01".to_string(), "2024/02".to_string()],
            });

        // 2. Embed the Parquet payload into the HDF5 file.
        write_embedded_parquet(&mut builder, &payload, &descriptor, "embedded_parquet")?;

        // 3. Finalize HDF5.
        let hdf5_bytes = builder.finish()?;

        // 4. Read the embedded Parquet payload from the HDF5 file.
        let cursor = MemCursor::from_bytes(hdf5_bytes);
        let hdf5_file = Hdf5File::open(cursor)?;

        let (reconstructed_payload, reconstructed_descriptor) =
            read_embedded_parquet(&hdf5_file, "/embedded_parquet")?;

        let reader = ParquetReader::new(&reconstructed_payload)?;

        assert_eq!(reconstructed_descriptor.mode, HybridMode::Embedded);
        
        let layout = reconstructed_descriptor.table_layout().unwrap();
        assert_eq!(layout.table_name, "my_table");
        assert_eq!(layout.payload_path, "/data/payload");
        assert_eq!(layout.relation, HybridTableRelation::MaterializedView);
        assert_eq!(layout.encoding, HybridStorageEncoding::ArrowIntermediate);

        let part = reconstructed_descriptor.partitioning.unwrap();
        assert_eq!(part.keys, vec!["year", "month"]);
        assert_eq!(part.paths, vec!["2024/01", "2024/02"]);

        // Reader contains valid bytes (which is just the Parquet magic + footer if empty).
        let _md = reader.metadata();

        Ok(())
    }
}
