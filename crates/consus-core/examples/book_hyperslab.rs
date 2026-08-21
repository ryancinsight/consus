//! Hyperslab selections for partial I/O.
//!
//! A [`Hyperslab`] selects a strided rectangular region of an N-dimensional
//! dataset.  Each dimension has `start`, `stride`, `count`, and `block`
//! parameters; the selected indices are
//! `{ start + n·stride + b : n ∈ [0, count), b ∈ [0, block) }`.

extern crate consus_core;

use consus_core::types::selection::{Hyperslab, HyperslabDim};

fn main() {
    // ── Select every other row from a 3D dataset ──
    // Dimension 0: start=0, stride=2, count=5 → rows 0,2,4,6,8
    // Dimension 1: all 8 columns
    // Dimension 2: all 4 channels
    let row_stride = HyperslabDim {
        start: 0,
        stride: 2,
        count: 5,
        block: 1,
    };
    let all_cols = HyperslabDim {
        start: 0,
        stride: 1,
        count: 8,
        block: 1,
    };
    let all_chans = HyperslabDim {
        start: 0,
        stride: 1,
        count: 4,
        block: 1,
    };
    let slab = Hyperslab::new(&[row_stride, all_cols, all_chans]);

    let selected = slab.num_elements();
    let expected = 5 * 8 * 4;
    println!("hyperslab elements: {selected} (expected {expected})");
    assert_eq!(selected, expected);

    // ── Block selection: 2×2 blocks with stride ──
    // Dimension 0: start=0, stride=3, count=2, block=2 → indices [0,1, 3,4]
    // Dimension 1: start=0, stride=1, count=4, block=2 → indices [0,1, 2,3, 4,5, 6,7]
    let block_rows = HyperslabDim {
        start: 0,
        stride: 3,
        count: 2,
        block: 2,
    };
    let block_cols = HyperslabDim {
        start: 0,
        stride: 1,
        count: 4,
        block: 2,
    };
    let block_slab = Hyperslab::new(&[block_rows, block_cols]);

    let block_selected = block_slab.num_elements();
    let block_expected = (2 * 2) * (4 * 2); // count×block per dim, product
    println!("block hyperslab elements: {block_selected} (expected {block_expected})");
    assert_eq!(block_selected, block_expected);

    println!("all hyperslab assertions passed");
}