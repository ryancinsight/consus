# Consus — Backlog

No open items. Every item this board carried has been delivered; each one's
delivery is recorded by its pull request and the `Item:` trailer on its
commits, not here. See `gap_audit.md` for risks nobody is fixing and
`checklist.md` for the historical checklist.

<a id="atlas-pr530-provider-coherence"></a>
## ATLAS-PR530-PROVIDER-COHERENCE — Current public provider contract [major] — in progress

- Target: 0.3.0; ADR0005 precedes the version correction. Preserve PR172's
  owner history and numerical repair; its 0.2.1 release claim is withdrawn.
- The actual published HDF50.2.0/Moirai0.7 generic consumer passes; the same
  caller fails with two E0277 errors on current Moirai0.9; migrated bounds
  pass. A current-main API comparison cannot establish published compatibility.
- Retain all production Rust, features, Rust1.95, provider/registry identities,
  existing workflow gates and test limits. Re-run the unchanged native/package/final-head
  gates for the corrected uniform workspace and internal 0.3.0 requirements.
- The cancelled 0.2.1 publishing job executes no steps. Both earlier upload
  commands are obsolete. Prepare a new exact-source raster0.3.0 handoff only
  after verification; authentication/publisher registration stay separate.
