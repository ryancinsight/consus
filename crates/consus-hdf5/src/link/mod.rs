//! HDF5 link message parsing (header message type 0x0006) and link storage
//! metadata (header message type 0x0002).

pub mod external;

mod info;
mod message;

#[cfg(feature = "alloc")]
pub use info::LinkInfo;
#[cfg(feature = "alloc")]
pub use message::{ExternalLinkData, Hdf5Link};
