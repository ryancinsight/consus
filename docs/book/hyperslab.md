# Hyperslab and Partial I/O

A **hyperslab** selects a rectangular sub-region of an N-dimensional dataset
for partial reads or writes. Consus hyperslab semantics follow the HDF5/Zarr
convention.

## `HyperslabDim`

Each dimension of a hyperslab is described by four parameters:

| Field | Type | Description |
|-------|------|-------------|
| `start` | `usize` | Starting index |
| `stride` | `usize` | Step between blocks (≥ 1) |
| `count` | `usize` | Number of blocks |
| `block` | `usize` | Size of each block (≥ 1) |

The selected indices along dimension `i` are:

```text
{ start[i] + n × stride[i] + b : n ∈ [0, count[i]), b ∈ [0, block[i]) }
```

Total selected elements = `Πᵢ count[i] × block[i]`.

### Convenience Constructors

```rust,ignore
use consus::core::HyperslabDim;

// Contiguous range: indices 10..10+50
let dim = HyperslabDim::range(10, 50);   // stride=1, block=1

// Every other row, 4-element blocks: rows 0, 2, 4, …
let dim = HyperslabDim { start: 0, stride: 2, count: 100, block: 4 };
```

### Validity

`HyperslabDim::is_valid_for_extent(extent)` returns `true` when
`max_index() < extent` or `count == 0`.

`max_index()` is `start + (count − 1) × stride + block − 1` (returns `None`
when `count == 0`).

## `Hyperslab`

A `Hyperslab` is a per-dimension `Vec<HyperslabDim>` of the same rank as the
dataset.

## `Selection`

A `Selection` is the union of the supported partial-I/O access patterns:

| Variant | Description |
|---------|-------------|
| `All` | Select the entire dataset |
| `Hyperslab(Hyperslab)` | Rectangular sub-region |
| `Points(PointSelection)` | An unordered set of individual elements |

`SelectionOps` on a dataset object resolves the selection against the stored
shape at I/O time.

## Usage Pattern

```rust,ignore
use consus::core::{HyperslabDim, Hyperslab, Selection};

// Read a 10×10 tile starting at (32, 32) from a 2D dataset
let sel = Selection::Hyperslab(Hyperslab::new(vec![
    HyperslabDim::range(32, 10),
    HyperslabDim::range(32, 10),
]));

dataset.read_selection::<f32>(&sel)?;
```

Hyperslab I/O avoids loading the entire dataset into memory and is essential
for large scientific arrays that exceed available RAM.