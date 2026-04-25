    //! Parquet footer byte-validation and canonical trailer metadata model.
    //!
    //! ## Specification
    //!
    //! This module validates the Parquet trailer prelude:
    //! - file length is at least 8 bytes
    //! - the final 4 bytes equal the ASCII magic `PAR1`
    //! - the preceding 4 bytes encode the footer length as little-endian `u32`
    //! - the computed footer start does not underflow the file length
    //!
    //! This increment does not parse the Thrift footer payload. It provides the
    //! canonical validated trailer metadata required before real footer decoding.
    //!
    //! ## Invariants
    //!
    //! - `footer_len <= file_len - 8`
    //! - `footer_offset + footer_len + 8 == file_len`
    //! - row-group and column-chunk locations are explicit and non-overlapping by construction
    //!
    //! ## Non-goals
    //!
    //! - No Thrift footer decoding
    //! - No schema extraction from bytes
    //! - No public file-read API
    //!
    //! ## Architecture
    //!
    //! ```text
    //! wire/
    //! ├── FooterPrelude           # Validated trailer prelude
    //! ├── ColumnChunkLocation     # Canonical byte-range metadata
    //! ├── RowGroupLocation        # Canonical row-group byte-range metadata
    //! ├── ParquetFooterDescriptor # Validated footer/trailer metadata envelope
    //! └── validate_footer_prelude # Byte-level trailer validation
    //! ```

pub mod thrift;
pub mod page;
pub mod metadata;
pub mod payload;

    use alloc::{string::String, vec::Vec};

    use consus_core::{Error, Result};

    const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
    const FOOTER_TRAILER_LEN: usize = 8;

    /// Validated Parquet trailer prelude.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FooterPrelude {
        footer_len: u32,
        footer_offset: usize,
        file_len: usize,
    }

    impl FooterPrelude {
        /// Footer payload length in bytes.
        #[must_use]
        pub fn footer_len(&self) -> u32 {
            self.footer_len
        }

        /// Byte offset of the footer payload start.
        #[must_use]
        pub fn footer_offset(&self) -> usize {
            self.footer_offset
        }

        /// Total file length in bytes.
        #[must_use]
        pub fn file_len(&self) -> usize {
            self.file_len
        }

        /// Byte offset immediately after the footer payload.
        #[must_use]
        pub fn footer_end_offset(&self) -> usize {
            self.file_len - FOOTER_TRAILER_LEN
        }
    }

    /// Canonical byte-range metadata for one column chunk.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ColumnChunkLocation {
        column_ordinal: usize,
        offset: usize,
        length: usize,
    }

    impl ColumnChunkLocation {
        /// Create a validated column-chunk location.
        pub fn new(column_ordinal: usize, offset: usize, length: usize) -> Result<Self> {
            if length == 0 {
                return Err(Error::InvalidFormat {
                    message: String::from("parquet column chunk length must be positive"),
                });
            }
            offset.checked_add(length).ok_or(Error::Overflow)?;
            Ok(Self {
                column_ordinal,
                offset,
                length,
            })
        }

        /// Source column ordinal.
        #[must_use]
        pub fn column_ordinal(&self) -> usize {
            self.column_ordinal
        }

        /// Byte offset of the chunk start.
        #[must_use]
        pub fn offset(&self) -> usize {
            self.offset
        }

        /// Byte length of the chunk.
        #[must_use]
        pub fn length(&self) -> usize {
            self.length
        }

        /// Byte offset immediately after the chunk.
        #[must_use]
        pub fn end_offset(&self) -> usize {
            self.offset + self.length
        }
    }

    /// Canonical byte-range metadata for one row group.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RowGroupLocation {
        row_count: usize,
        columns: Vec<ColumnChunkLocation>,
    }

    impl RowGroupLocation {
        /// Create a validated row-group location.
        pub fn new(row_count: usize, columns: Vec<ColumnChunkLocation>) -> Result<Self> {
            if row_count == 0 {
                return Err(Error::InvalidFormat {
                    message: String::from("parquet row group row_count must be positive"),
                });
            }
            if columns.is_empty() {
                return Err(Error::InvalidFormat {
                    message: String::from("parquet row group must contain column chunks"),
                });
            }

            let mut i = 1;
            while i < columns.len() {
                if columns[i - 1].end_offset() > columns[i].offset() {
                    return Err(Error::InvalidFormat {
                        message: String::from(
                            "parquet row group column chunk byte ranges must not overlap",
                        ),
                    });
                }
                i += 1;
            }

            Ok(Self { row_count, columns })
        }

        /// Number of rows in the row group.
        #[must_use]
        pub fn row_count(&self) -> usize {
            self.row_count
        }

        /// Borrow column chunk locations in byte-order sequence.
        #[must_use]
        pub fn columns(&self) -> &[ColumnChunkLocation] {
            &self.columns
        }

        /// Byte offset of the first chunk in the row group.
        #[must_use]
        pub fn start_offset(&self) -> usize {
            self.columns[0].offset()
        }

        /// Byte offset immediately after the last chunk in the row group.
        #[must_use]
        pub fn end_offset(&self) -> usize {
            self.columns[self.columns.len() - 1].end_offset()
        }
    }

    /// Canonical validated footer/trailer metadata envelope.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ParquetFooterDescriptor {
        prelude: FooterPrelude,
        row_groups: Vec<RowGroupLocation>,
    }

    impl ParquetFooterDescriptor {
        /// Create a validated footer descriptor.
        pub fn new(prelude: FooterPrelude, row_groups: Vec<RowGroupLocation>) -> Result<Self> {
            let mut i = 0;
            while i < row_groups.len() {
                if row_groups[i].end_offset() > prelude.footer_offset() {
                    return Err(Error::InvalidFormat {
                        message: String::from(
                            "parquet row group byte range must end before footer payload",
                        ),
                    });
                }
                i += 1;
            }

            Ok(Self {
                prelude,
                row_groups,
            })
        }

        /// Borrow the validated trailer prelude.
        #[must_use]
        pub fn prelude(&self) -> FooterPrelude {
            self.prelude
        }

        /// Borrow row-group locations.
        #[must_use]
        pub fn row_groups(&self) -> &[RowGroupLocation] {
            &self.row_groups
        }

        /// Number of row groups.
        #[must_use]
        pub fn row_group_count(&self) -> usize {
            self.row_groups.len()
        }

        /// Total rows across all row groups.
        #[must_use]
        pub fn total_rows(&self) -> usize {
            self.row_groups
                .iter()
                .map(RowGroupLocation::row_count)
                .sum()
        }
    }

    /// Canonical footer validation error alias.
    pub type ParquetFooterError = Error;

    /// Validate the Parquet trailer prelude from the final 8 bytes of a file.
    pub fn validate_footer_prelude(bytes: &[u8]) -> Result<FooterPrelude> {
        if bytes.len() < FOOTER_TRAILER_LEN {
            return Err(Error::BufferTooSmall {
                required: FOOTER_TRAILER_LEN,
                provided: bytes.len(),
            });
        }

        let trailer = &bytes[bytes.len() - FOOTER_TRAILER_LEN..];
        if &trailer[4..8] != PARQUET_MAGIC {
            return Err(Error::InvalidFormat {
                message: String::from("parquet trailer magic must equal PAR1"),
            });
        }

        let footer_len = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
        let footer_len_usize = footer_len as usize;
        let footer_offset = bytes
            .len()
            .checked_sub(FOOTER_TRAILER_LEN)
            .and_then(|n| n.checked_sub(footer_len_usize))
            .ok_or(Error::InvalidFormat {
                message: String::from("parquet footer length exceeds file payload"),
            })?;

        Ok(FooterPrelude {
            footer_len,
            footer_offset,
            file_len: bytes.len(),
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn validate_footer_prelude_accepts_valid_trailer() {
            let bytes = b"abcdwxyz\x04\0\0\0PAR1";
            let prelude = validate_footer_prelude(bytes).unwrap();
            assert_eq!(prelude.footer_len(), 4);
            assert_eq!(prelude.footer_offset(), 4);
            assert_eq!(prelude.footer_end_offset(), 8);
            assert_eq!(prelude.file_len(), 16);
        }

        #[test]
        fn validate_footer_prelude_rejects_short_input() {
            let err = validate_footer_prelude(b"PAR1").unwrap_err();
            assert!(matches!(
                err,
                Error::BufferTooSmall {
                    required: 8,
                    provided: 4
                }
            ));
        }

        #[test]
        fn validate_footer_prelude_rejects_invalid_magic() {
            let err = validate_footer_prelude(b"abcdwxyz\x04\0\0\0XXXX").unwrap_err();
            assert!(matches!(err, Error::InvalidFormat { .. }));
        }

        #[test]
        fn validate_footer_prelude_rejects_footer_length_overflow() {
            let err = validate_footer_prelude(b"abcd\x10\0\0\0PAR1").unwrap_err();
            assert!(matches!(err, Error::InvalidFormat { .. }));
        }

        #[test]
        fn row_group_location_rejects_overlapping_columns() {
            let err = RowGroupLocation::new(
                2,
                vec![
                    ColumnChunkLocation::new(0, 0, 8).unwrap(),
                    ColumnChunkLocation::new(1, 4, 8).unwrap(),
                ],
            )
            .unwrap_err();
            assert!(matches!(err, Error::InvalidFormat { .. }));
        }

        #[test]
        fn footer_descriptor_rejects_row_group_past_footer() {
            let prelude = validate_footer_prelude(b"abcdefgh\x04\0\0\0PAR1").unwrap();
            let row_groups = vec![
                RowGroupLocation::new(2, vec![ColumnChunkLocation::new(0, 0, 10).unwrap()])
                    .unwrap(),
            ];
            let err = ParquetFooterDescriptor::new(prelude, row_groups).unwrap_err();
            assert!(matches!(err, Error::InvalidFormat { .. }));
        }

        #[test]
        fn footer_descriptor_computes_total_rows() {
            let prelude = validate_footer_prelude(b"abcdefghijkl\x04\0\0\0PAR1").unwrap();
            let row_groups = vec![
                RowGroupLocation::new(
                    2,
                    vec![
                        ColumnChunkLocation::new(0, 0, 2).unwrap(),
                        ColumnChunkLocation::new(1, 2, 2).unwrap(),
                    ],
                )
                .unwrap(),
            ];
            let footer = ParquetFooterDescriptor::new(prelude, row_groups).unwrap();
            assert_eq!(footer.row_group_count(), 1);
            assert_eq!(footer.total_rows(), 2);
            assert_eq!(footer.prelude().footer_offset(), 8);
        }
    }
