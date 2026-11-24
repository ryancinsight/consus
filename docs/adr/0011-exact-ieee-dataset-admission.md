# ADR 0011: Opt-in exact IEEE dataset admission

Status: Accepted

## Context

The ordinary `Datatype::Float` metadata projection preserves storage width
and byte order, but not all fields that determine the stored representation.
For example, a custom HDF5 float bias can compare equal to the projection for
binary64. An archive reader promising binary64 little-endian needs to check the
original metadata before reading or allocating the numerical payload. Consus
owns this check, not a second HDF5 parser in a consumer.

## Decision

Add `IeeeFloatEncoding` and `Hdf5File::dataset_at_with_ieee_float_encoding`.
The selector describes an on-disk contract, not a per-value dispatch. It supports
the native writer's binary16, binary32, and binary64 encodings in explicit
little- and big-endian order. Existing `dataset_at` behavior and public dataset
fields are unchanged.

Read the original object header once. Compare the significant numeric fields
with the writer's canonical IEEE representation, then return the same ordinary
dataset metadata. Compare float class, order (including the VAX order bit),
sign location, mantissa normalization, storage size, significant-bit offset
and precision, exponent and mantissa locations and widths, and exponent bias.
Datatype message versions 1 through 5 are admitted; they share the float
fields above, so one comparison covers all five.
Reserved bits and padding preferences in a full-width encoding do not alter
numeric identity. Shared datatype messages and other message versions remain
unsupported by this opt-in method. This is not a general HDF5 conformance check.

## Compatibility and verification

The lookup reads no scalar payload, performs no numeric conversion, and
adds no dataset size ceiling. The metadata check is independent of voxel count.
The regression compares metadata read counts with ordinary lookup and covers
all six width/order combinations across the five admitted message versions,
noncanonical numeric encodings, and equivalent unused padding preferences.
Ordinary metadata lookups remain covered separately.

This is an additive public API. The [provider-first adoption plan](../typed-float-admission-plan.md)
defines the separate provider release and consumer verification gates.

## Delivery

- Provider: PR 230 (efd03ed) adds `IeeeFloatEncoding`,
  `dataset_at_with_ieee_float_encoding` and `tests/ieee_float_admission.rs`.
- Release: `consus-hdf5` 0.6.0 is tag `crate-consus-hdf5-v0.6.0`, revision
  b14cbd3. PR 235 matched the registry archive to that source and exercised
  the API from a registry-only consumer.
- Consumer: Helios b7bda4cd calls the method at
  `crates/helios-domain/src/storage.rs:154` with `Binary64LittleEndian`.

Revision 2026-11-24: Proposed to Accepted, because every decision above is
what `crates/consus-hdf5/src/datatype/ieee.rs` and
`crates/consus-hdf5/src/file/exact_float.rs` do at b14cbd3, and the
"checked separately" sentence now matches the single comparison at
`exact_float.rs:57-66`.

## Source

- HDF Group, [HDF5 datatype message fields](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html)
  and [atomic datatype field and offset contracts](https://support.hdfgroup.org/documentation/hdf5/latest/group___a_t_o_m.html).
