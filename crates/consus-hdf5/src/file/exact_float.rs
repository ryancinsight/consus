//! Opt-in representation-checked dataset lookup, without a second header read.

use super::{Hdf5File, reader, writer};
use crate::dataset::Hdf5Dataset;
use crate::datatype::IeeeFloatEncoding;
use crate::object_header::message_types;
use alloc::{format, string::String};
use consus_core::{Error, Result};
use consus_io::ReadAt;

impl<R: ReadAt + Sync> Hdf5File<R> {
    /// Read dataset metadata only if its actual numeric representation is IEEE.
    ///
    /// Checks the original inline datatype message, not the lossy canonical
    /// `Datatype::Float` projection. The expected representation comes from the
    /// provider's IEEE writer, so byte order (including the VAX order bit), sign
    /// location, normalization, storage size, bit offset, precision, exponent and
    /// mantissa fields, and exponent bias must agree.
    ///
    /// Supported datatype-message versions 1 through 5 share these float fields.
    /// Message version, reserved bits and unused padding preferences are not
    /// numeric-identity fields. Full-width IEEE has no unused bits, so its low,
    /// high and internal padding preferences do not alter decoded values.
    /// This method is not a general file-format validity checker.
    ///
    /// Existing [`Self::dataset_at`] and non-IEEE parsing remain unchanged. This
    /// method reads metadata once, never reads the scalar payload, and imposes no
    /// new dataset allocation ceiling. It does not decode or convert values.
    ///
    /// # Errors
    /// Returns the existing metadata/I/O errors, `UnsupportedFeature` for shared
    /// datatype messages or unsupported message versions, and `InvalidFormat`
    /// when the dataset does not use the requested numeric representation.
    pub fn dataset_at_with_ieee_float_encoding(
        &self,
        object_header_address: u64,
        expected: IeeeFloatEncoding,
    ) -> Result<Hdf5Dataset> {
        let header = reader::read_object_header(&self.source, object_header_address, &self.ctx)?;
        let message = reader::find_message(&header, message_types::DATATYPE).ok_or_else(|| {
            Error::InvalidFormat {
                message: String::from("dataset object header missing datatype message"),
            }
        })?;
        if message.flags & 0x02 != 0 {
            return Err(Error::UnsupportedFeature {
                feature: String::from("IEEE admission of shared datatype messages"),
            });
        }
        let tag = message
            .data
            .first()
            .copied()
            .ok_or_else(|| Error::InvalidFormat {
                message: String::from("dataset datatype message has no class/version"),
            })?;
        if !(1..=5).contains(&(tag >> 4)) {
            return Err(Error::UnsupportedFeature {
                feature: format!("IEEE admission for datatype message version {}", tag >> 4),
            });
        }
        let canonical = writer::encode_datatype(&expected.datatype())?;
        let matches = tag & 0x0f == 1
            && message.data.get(1).map(|flags| flags & 0x71) == canonical.get(1).copied()
            && message.data.get(2) == canonical.get(2)
            && message.data.get(4..20) == canonical.get(4..20);
        if !matches {
            return Err(Error::InvalidFormat {
                message: format!(
                    "dataset datatype does not use requested {expected:?} representation"
                ),
            });
        }
        let mut dataset = reader::read_dataset_metadata(&header, &self.ctx)?;
        dataset.object_header_address = object_header_address;
        Ok(dataset)
    }
}
