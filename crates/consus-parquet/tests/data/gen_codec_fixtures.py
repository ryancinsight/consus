"""Write the pyarrow-produced Parquet fixtures used by the codec tests.

Usage::

    python tests/data/gen_codec_fixtures.py <output-directory>

Each file holds one required INT32 column ``id`` with ``VALUES`` as PLAIN
data pages (version 1.0), compressed with one specified codec. The
``pyarrow_*_fixture_decodes_with_spec_number`` tests in
``src/reader/tests.rs`` read them and assert the codec numbers of
parquet.thrift ``CompressionCodec``.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

VALUES = [(i * 7) % 31 - 15 for i in range(100)]
CODECS = {"gzip": "parquet_gzip_i32.parquet", "zstd": "parquet_zstd_i32.parquet"}


def main() -> None:
    out = Path(sys.argv[1])
    schema = pa.schema([pa.field("id", pa.int32(), nullable=False)])
    table = pa.table({"id": pa.array(VALUES, type=pa.int32())}, schema=schema)
    for codec, name in CODECS.items():
        pq.write_table(
            table,
            out / name,
            compression=codec,
            use_dictionary=False,
            data_page_version="1.0",
            write_statistics=False,
        )


if __name__ == "__main__":
    main()
