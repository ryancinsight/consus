# Exact IEEE float admission: provider-first adoption plan

[ADR 0011](adr/0011-exact-ieee-dataset-admission.md) records the provider
contract. The opt-in check is not a replacement for general HDF5 metadata
inspection or a proof of correctness in an archive consumer.

## Consus delivery

1. Verify the exact final provider revision with the unchanged Atlas push gate,
   applicable hosted checks and independent code review. Confirm the public API
   comparison and leave unrelated owner work and service holds untouched.
2. Coordinate the additive API's version with the Consus release owner. This
   change does not select a version, update release configuration, authorize a
   release, or publish a crate. Keep the provider unavailable to downstream
   release integration until the applicable release gates pass.
3. After an authorized release, verify the official `consus-hdf5` package,
   including the registry index checksum, downloaded archive and embedded
   source revision. Do not substitute an unreleased Git revision to skip
   release qualification.

## Helios adoption

1. Compose a fresh candidate against current Helios and the verified provider
   release. Admit archive volume and geometry with
   `IeeeFloatEncoding::Binary64LittleEndian` before reading their scalar
   payload or allocating output.
2. Keep the direct typed conversion, the exact little-endian archive bytes,
   millimetre units and C ordering, and the existing large contiguous read
   path. Test ordinary binary64 and the two independently identified custom-bias
   cases for volume and geometry. Add equivalent full-width padding controls.
3. Check destination f32/f64 values and geometry, canonical archive byte hashes,
   and datasets above 64 MiB without adding a new ceiling. Measure allocation and
   execution costs on the new integrated revision. Older prototype timings or
   provider-only tests do not certify the consumer.
4. Verify native and hosted consumer gates, independent review, and any
   release gates on that exact candidate. Preserve prior rejected or held work;
   this document does not authorize changing any hold, publishing a consumer,
   or deploying an application.
