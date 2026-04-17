import base64, sys
path = r"crates/consus-hdf5/src/file/mod.rs"
src = open(path, encoding="utf-8").read()
marker_serial = "#[cfg(all(not(feature = \"parallel-io\"), feature = \"alloc\"))]"
marker_struct = "
#[cfg(feature = \"alloc\")]
#[derive(Debug, Clone)]
struct ChunkIndexEntry {"
assert marker_struct in src, "struct marker not found"
print("markers ok")
