mod container;
mod entropy;
mod failure;
mod frame;
mod lossless;
mod model;
mod output;
mod syntax;

pub use container::decode;

use failure::{allocation, malformed, too_large, unsupported};
use model::{Coding, ColorModel, Component, Frame, Scan, ScanComponent, UNSEEN};
use syntax::read_word;
