//! Chunk read/write operation families: chunk addressing, selection copies, fill
//! expansion, single-chunk IO, array-level and sharded paths.

mod arrays;
mod chunk_io;
mod coords;
mod fill;
mod selection;
mod sharded;
#[cfg(test)]
mod tests;

pub use arrays::{read_array, write_array, write_array_selection};
pub use chunk_io::{read_chunk, write_chunk};
pub(crate) use coords::{
    chunk_key_for_array, compute_strides, stored_shape_for_chunk, validate_chunk_coords,
};
pub(crate) use fill::checked_chunk_bytes;
pub use fill::{expand_fill_value, try_expand_fill_value};
pub(crate) use selection::{
    SelectionIndices, chunk_intersects_selection, copy_chunk_selection_to_output,
    copy_selection_input_to_chunk,
};
pub(crate) use sharded::{read_array_sharded, write_array_sharded};
