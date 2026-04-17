import sys
path = r"crates/consus-hdf5/src/file/mod.rs"
lines = open(path, encoding="utf-8").readlines()

# Find the struct ChunkIndexEntry line (marker for insertion point)
insert_idx = None
for i, ln in enumerate(lines):
    if "struct ChunkIndexEntry {" in ln and i >= 2 and "cfg(feature" in lines[i-2]:
        insert_idx = i - 2
        break
assert insert_idx is not None
print(f"Insert at line {insert_idx+1}")
