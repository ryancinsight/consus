use consus_hdf5::file::async_file::AsyncHdf5File;
use moirai_async::io::{AsyncLength, AsyncMemReader, AsyncReadAt};

fn accepts_existing_reader<R: AsyncReadAt + AsyncLength>() {
    let _ = core::mem::size_of::<AsyncHdf5File<R>>();
}

fn main() {
    accepts_existing_reader::<AsyncMemReader>();
    println!("the public HDF5 file accepts the caller's positioned-reader bounds");
}
