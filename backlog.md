# Consus — Backlog

No open items. Every item this board carried has been delivered; each one's
delivery is recorded by its pull request and the `Item:` trailer on its
commits, not here. See `gap_audit.md` for risks nobody is fixing and
`checklist.md` for the historical checklist.

<a id="atlas-pr530-provider-coherence"></a>
## ATLAS-PR530-PROVIDER-COHERENCE — Current provider admission and canonical lock [patch] — done

- Priority: correctness; target: 0.2.1; owner: existing PR 172.
- Fresh consumers reject the old Apollo core 0.16 requirement against current
  main 0.17. The corrected private requirement and Moirai 0.9.3 lock preserve
  public contracts, registry identities, features, and Rust 1.95.
- Verified: fresh-consumer JPEG reconstruction; all 2,811 native tests;
  Rust 1.95; strict diagnostics; 76 normalized-package tests; both 223-check
  API comparisons; CI 38787639519 at fda3d838. The final documentation-only
  head retains the same exact-head gate before default-branch adoption.
- Publication: source 3510db09 / raster 0.2.0 is an obsolete handoff. Initial
  registry authentication and publisher configuration remain separate human
  actions; this dependency repair supplies no credentials or permissions.
