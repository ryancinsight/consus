# consus-zip

Bounded, pure-Rust ZIP container reader and writer for in-memory members.
It serves the container formats built on ZIP: NumPy NPZ archives
(`consus-npy`) and 3MF meshes.

Scope:

- Stored (method 0) and DEFLATE (method 8) members. DEFLATE runs through the
  `consus-compression` raw DEFLATE codec, and every member is checked
  against its CRC-32.
- ZIP64 sizes, offsets and entry counts on both read and write.
- No filesystem extraction, symbolic links or permission bits. Members are
  returned as bytes, so no archive field can name a path outside the
  caller's control.
- Typed errors. Every declared length and offset is checked against the
  region that must hold it before anything is read or allocated. A DEFLATE
  member's declared size is checked against DEFLATE's 1032:1 expansion bound.
- Encrypted, multi-disk and non-UTF-8, non-ASCII names are rejected, not
  guessed.

Written members carry the fixed date 1980-01-01, so equal inputs produce
byte-identical archives.

```rust
use std::io::Cursor;

use consus_zip::{Method, ZipReader, ZipWriter};

let mut writer = ZipWriter::new(Vec::new());
writer.add_entry("notes.txt", b"stored verbatim", Method::Stored)?;
writer.add_entry("data.bin", &[7u8; 4096], Method::Deflated)?;
let archive = writer.finish()?;

let mut reader = ZipReader::new(Cursor::new(archive))?;
assert_eq!(reader.entries().len(), 2);
assert_eq!(reader.read_by_name("data.bin")?, vec![7u8; 4096]);
# Ok::<(), consus_zip::Error>(())
```

Format reference: PKWARE APPNOTE.TXT 6.3.10,
<https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT>.
