pub(super) const UNSEEN: u8 = u8::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Coding {
    Baseline,
    Sequential,
    Progressive,
    Lossless,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EntropyCoding {
    Huffman,
    Arithmetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ColorModel {
    Gray,
    Rgb,
    Ycbcr,
    Cmyk,
    Ycck,
}

pub(super) struct Component {
    pub(super) id: u8,
    pub(super) horizontal: u8,
    pub(super) vertical: u8,
    pub(super) quantization: usize,
    pub(super) quantization_values: Option<[u16; 64]>,
    pub(super) blocks_across: usize,
    pub(super) blocks_down: usize,
    pub(super) stored_across: usize,
    pub(super) stored_down: usize,
    pub(super) coefficient_offset: usize,
    pub(super) approximation: [u8; 64],
}

pub(super) struct Frame {
    pub(super) coding: Coding,
    pub(super) entropy_coding: EntropyCoding,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) precision: u8,
    pub(super) max_horizontal: u8,
    pub(super) max_vertical: u8,
    pub(super) mcu_across: usize,
    pub(super) mcu_down: usize,
    pub(super) components: Vec<Component>,
    pub(super) coefficients: Vec<i32>,
}

#[derive(Clone, Copy)]
pub(super) struct ScanComponent {
    pub(super) frame_index: usize,
    pub(super) dc_table: usize,
    pub(super) ac_table: usize,
}

pub(super) struct Scan {
    pub(super) components: Vec<ScanComponent>,
    pub(super) start: u8,
    pub(super) end: u8,
    pub(super) high: u8,
    pub(super) low: u8,
}
