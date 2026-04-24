import os, re

BASE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(BASE, 'crates', 'consus-parquet', 'src', 'lib.rs')
WIRE = os.path.join(BASE, 'crates', 'consus-parquet', 'src', 'wire')

def write_file(path, content):
    with open(path, 'w', newline='\n', encoding='utf-8') as f:
        f.write(content)

# ---- Extract inline wire module from lib.rs ----
with open(LIB, 'r', encoding='utf-8') as f:
    lib = f.read()

idx = lib.find('\npub mod wire {')
if idx == -1:
    raise ValueError('pub mod wire { not found')

brace_start = lib.index('{', idx) + 1
depth = 1
i = brace_start
while i < len(lib) and depth > 0:
    if lib[i] == '{':
        depth += 1
    elif lib[i] == '}':
        depth -= 1
    i += 1

inner = lib[brace_start:i-1]
wire_end = i

# ---- Write wire/mod.rs ----
write_file(os.path.join(WIRE, 'mod.rs'),
    'pub mod thrift;\npub mod page;\n' + inner)
print('wire/mod.rs written')

# ---- Update lib.rs ----
write_file(LIB, lib[:idx+1] + 'pub mod wire;\n' + lib[wire_end:])
print('lib.rs updated')
print('inner lines:', len(inner.splitlines()))
