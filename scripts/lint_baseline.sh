#!/usr/bin/env bash
# Re-measure the lint ratchet baseline recorded in the root Cargo.toml.
#
# The workspace lint floor denies `clippy::pedantic` and allows only the
# classes that already fire on pre-existing code. Those allow entries are a
# non-increasing baseline: an entry may shrink and be deleted, never grow, and
# a class absent from the table is a hard error.
#
# This script prints the current per-class counts so a burn-down can be
# verified and so a newly-firing class is visible before it is silently
# added to the allow list.
#
# Usage:
#   scripts/lint_baseline.sh                 # pedantic census
#   scripts/lint_baseline.sh --restriction   # trust-boundary restriction census
set -euo pipefail

cd "$(dirname "$0")/.."

LINTS=(-W clippy::pedantic)
if [[ "${1:-}" == "--restriction" ]]; then
    LINTS=(
        -W clippy::unwrap_used
        -W clippy::indexing_slicing
        -W clippy::arithmetic_side_effects
    )
fi

# `-A clippy::…` allows from the workspace table are overridden on the command
# line so the census sees the unratcheted truth.
cargo clippy --workspace --all-targets --message-format=json -- "${LINTS[@]}" 2>/dev/null |
    python3 -c '
import collections, json, sys

counts = collections.Counter()
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        record = json.loads(line)
    except ValueError:
        continue
    if record.get("reason") != "compiler-message":
        continue
    code = ((record.get("message") or {}).get("code") or {}).get("code")
    if code and code.startswith("clippy::"):
        counts[code] += 1

for code, count in counts.most_common():
    print(f"{count:6d}  {code}")
print(f"{sum(counts.values()):6d}  TOTAL")
'
