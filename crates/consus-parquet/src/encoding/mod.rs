//! Parquet page encoding decoders.
//!
//! ## Modules
//!
//! - `levels` -- RLE/bit-packing hybrid and raw bit-packed level decoders
//! - `plain` -- PLAIN encoding decoders for all Parquet physical types
//! - `rle_dict` -- RLE_DICTIONARY (encoding ID 8) index decoder
//! - `column` -- Typed column value extraction and dictionary page decoder
//! - `compression` -- Parquet compression codec dispatch and decompression

pub mod column;
pub mod compression;
pub mod levels;
pub mod plain;
pub mod rle_dict;

pub use column::{
    ColumnValues, ColumnValuesWithLevels, decode_column_values, decode_compressed_column_values,
    decode_dictionary_page,
};
pub use compression::{CompressionCodec, decompress_page_values};
pub use levels::{decode_bit_packed_raw, decode_levels, level_bit_width};
pub use plain::{
    decode_plain_boolean, decode_plain_byte_array, decode_plain_f32, decode_plain_f64,
    decode_plain_fixed_byte_array, decode_plain_i32, decode_plain_i64, decode_plain_i96,
};
pub use rle_dict::decode_rle_dict_indices;

#[cfg(test)]
mod compression_proptest;
#[cfg(test)]
mod plain_proptest;
