//! Array shapes, extents, and chunk tiling in Consus.
//!
//! [`Shape`] describes an N-dimensional array's extents.  Dimensions may be
//! fixed (known at creation) or unlimited (growable).  [`ChunkShape`] tiles a
//! dataset into fixed-size storage blocks for chunked I/O.

extern crate consus_core;

use consus_core::types::{ChunkShape, Extent, Layout, Shape};

fn main() {
    // ── Fixed 3×4 shape ──
    let s = Shape::fixed(&[3, 4]);
    println!("rank={} elements={}", s.rank(), s.num_elements());
    assert_eq!(s.rank(), 2);
    assert_eq!(s.num_elements(), 12);

    // ── Shape with one unlimited dimension ──
    let s_growable = Shape::new(&[Extent::Unlimited { current: 0 }, Extent::Fixed(10)]);
    assert_eq!(s_growable.rank(), 2);
    assert!(s_growable.has_unlimited());
    println!("growable: elements={}", s_growable.num_elements()); // 0 * 10 = 0

    // ── Scalar shape (rank 0) ──
    let scalar = Shape::scalar();
    assert!(scalar.is_scalar());
    assert_eq!(scalar.rank(), 0);
    assert_eq!(
        scalar.num_elements(),
        1,
        "empty product of scalar shape is 1"
    );
    println!(
        "scalar: rank={} elements={}",
        scalar.rank(),
        scalar.num_elements()
    );

    // ── Chunk tiling ──
    let shape = Shape::fixed(&[100, 64]);
    let chunk = ChunkShape::new(&[10, 8]).expect("positive chunk dims");
    let tiling = chunk.num_chunks(&shape);
    println!("100×64 tiled by 10×8 = {:?} chunks", &tiling[..]);
    assert_eq!(tiling[0], 10); // 100/10 = 10
    assert_eq!(tiling[1], 8); // 64/8  = 8

    // Non-even tiling rounds up.
    let shape_uneven = Shape::fixed(&[103, 65]);
    let tiling_uneven = chunk.num_chunks(&shape_uneven);
    println!(
        "103×65 tiled by 10×8 = {:?} chunks (ceil)",
        &tiling_uneven[..]
    );
    assert_eq!(tiling_uneven[0], 11); // ⌈103/10⌉ = 11
    assert_eq!(tiling_uneven[1], 9); // ⌈65/8⌉  = 9

    // ── Layout defaults to RowMajor ──
    assert_eq!(Layout::default(), Layout::RowMajor);
    println!("default layout: {:?}", Layout::default());

    println!("all shape assertions passed");
}
