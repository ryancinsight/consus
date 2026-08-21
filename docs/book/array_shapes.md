# Array Shapes

An N-dimensional array in Consus is described by its **shape** — an ordered
sequence of dimension extents — and optionally a **chunk shape** that tiles
the array into fixed-size blocks for partial I/O.

## Extents

Each dimension has an `Extent`, which is either:

| Variant | Description |
|---------|-------------|
| `Fixed(usize)` | Immutable size; known at dataset creation |
| `Unlimited { current: usize }` | Growable; `current` is the present size |

```rust,ignore
use consus::core::{Extent, Shape};

let shape = Shape::from_extents(vec![
    Extent::Fixed(256),
    Extent::Fixed(256),
    Extent::Unlimited { current: 100 },
]);
assert_eq!(shape.rank(), 3);
assert_eq!(shape.num_elements(), 256 * 256 * 100);
```

The `Shape::num_elements()` value is the product of all current sizes.
A rank-0 (scalar) shape has `num_elements() == 1` by the empty-product
convention.

## Memory Layout

`Layout` controls the mapping from an N-dimensional index to a linear memory
offset:

| Variant | Order | Convention |
|---------|-------|------------|
| `RowMajor` (default) | last index varies fastest | C order |
| `ColumnMajor` | first index varies fastest | Fortran order |

For a rank-R array with shape `(d₀, d₁, …, d_{R−1})` in row-major order:

```text
offset = Σᵢ idxᵢ × Πⱼ₌ᵢ₊₁^{R−1} dⱼ
```

## Chunk Shapes

A `ChunkShape` tiles a dataset for partial I/O and compression. Every chunk
dimension must be strictly positive. Given a dataset shape `S` and chunk shape
`C` (both of rank `R`), the number of chunks along dimension `i` is
`⌈S[i] / C[i]⌉`.

```rust,ignore
use consus::core::ChunkShape;

let chunks = ChunkShape::new(vec![64, 64, 10])?;
```

Chunk shapes affect I/O granularity and compression efficiency.
Zarr v2/v3, HDF5, and netCDF all support chunked storage through this common
type.

## Relation to Other Consus Types

`Shape` and `ChunkShape` are consumed by `DatasetConfig` when creating a new
dataset, and returned by `DatasetRead::shape()` when reading an existing one.
`Selection` (see [Hyperslab](./hyperslab.md)) operates within the bounds set by
the shape.
