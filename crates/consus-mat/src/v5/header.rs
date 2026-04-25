//! MAT v5 file-level header parser (128 bytes).
#[cfg(feature = "alloc")]
use alloc::string::String;
use crate::error::MatError;

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct V5FileHeader {
    pub description: String,
    pub big_endian: bool,
}

#[cfg(feature = "alloc")]
impl V5FileHeader {
    pub fn parse(data: &[u8]) -> Result<Self, MatError> {
        if data.len() < 128 {
            return Err(MatError::InvalidFormat(
                String::from("MAT v5 file header requires at least 128 bytes"),
            ));
        }
        let big_endian = match &data[126..128] {
            b"MI" => true,
            b"IM" => false,
            other => return Err(MatError::InvalidFormat(alloc::format!(
                "invalid MAT v5 endian indicator: 0x{:02X} 0x{:02X}", other[0], other[1]
            ))),
        };
        let desc_bytes = &data[..116];
        let nul_end = desc_bytes.iter().position(|&b| b == 0).unwrap_or(116);
        let description = String::from_utf8_lossy(&desc_bytes[..nul_end]).into_owned();
        Ok(V5FileHeader { description, big_endian })
    }
}
