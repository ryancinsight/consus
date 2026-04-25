
import re

# Update gap_audit.md
with open('gap_audit.md', 'r', encoding='utf-8') as f:
    txt = f.read()

txt = txt.replace(
    'MATLAB .mat remaining roadmap items: (1) non-scalar struct array shape preservation via HDF5 MATLAB_dims attribute; (2) virtual-layout explicit rejection test (blocked on virtual dataset authoring in HDF5 builder); (3) chunked-dataset fixture coverage in v73 test suite. These are tracked in backlog.md as low-risk deferred items.',
    'All three previously-deferred MATLAB v7.3 roadmap items are closed (Sprint 6). See Closed section for M-006a/b/c.'
)
txt = txt.replace('cargo test -p consus-mat (74/74)')
txt = txt.replace('Closed this sprint | 8 (M-001s5a through M-001s5h)', 'Closed this sprint | 3 (M-006a/b/c: MATLAB_dims struct shape, virtual-layout rejection, chunked v7.3 fixture)')

with open('gap_audit.md', 'w', encoding='utf-8') as f:
    f.write(txt)
print('gap_audit updated')

# Update backlog.md P2.6c remaining items
with open('backlog.md', 'r', encoding='utf-8') as f:
    txt = f.read()

txt = txt.replace(
    '- [ ] Non-scalar struct array decoding with authoritative shape preservation',
    '- [x] Non-scalar struct array decoding with authoritative shape preservation'
)
txt = txt.replace(
    '- [ ] Virtual-layout rejection coverage (blocked on current fixture-authoring surface)',
    '- [x] Virtual-layout rejection coverage (DatasetLayout::Virtual added to HDF5 builder; v73 rejection test passing)'
)
txt = txt.replace(
    '- [ ] Chunked-dataset fixture coverage beyond current synthetic cases',
    '- [x] Chunked-dataset fixture coverage: v73_chunked_double_array_roundtrip passing'
)

with open('backlog.md', 'w', encoding='utf-8') as f:
    f.write(txt)
print('backlog updated')
