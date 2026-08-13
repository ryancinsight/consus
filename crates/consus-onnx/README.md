# consus-onnx

Bounded, zero-copy ONNX protobuf document reader.

Parses the structure of an ONNX model — graph topology, tensor metadata,
operator sets, and initializers — without pulling in a tensor runtime or a
protobuf code generator.

```toml
[dependencies]
consus-onnx = { version = "0.1", default-features = false }
```

## Copy behavior

Parsing is genuinely borrowing: `ModelDocument<'a>`, `Node<'a>`,
`ValueInfo<'a>`, `Initializer<'a>`, and `OperatorSet<'a>` alias the source
buffer, so names and initializer payloads are never copied out of it.

Parsing is bounded against hostile input by an explicit `ParseLimits` budget —
document size, single length-delimited field size, node count, value count,
initializer count, per-node name count, tensor dimensions, and operator-set
count — so a malformed count field cannot drive unbounded work.

Part of the [Consus](https://github.com/ryancinsight/consus) scientific storage
library; usable standalone.

- Documentation: <https://docs.rs/consus-onnx>

Licensed under MIT OR Apache-2.0.
