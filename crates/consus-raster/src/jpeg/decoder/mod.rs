mod arithmetic;
mod conditioning;
mod container;
mod entropy;
mod failure;
mod frame;
mod lossless;
mod lossless_arithmetic;
mod model;
mod output;
mod scan;
mod syntax;

pub use container::decode;

use failure::{allocation, malformed, too_large, unsupported};
use model::{Coding, ColorModel, Component, EntropyCoding, Frame, Scan, ScanComponent, UNSEEN};
use syntax::read_word;
