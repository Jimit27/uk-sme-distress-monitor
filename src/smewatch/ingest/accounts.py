"""Parse a Companies House accounts bulk zip into a parquet table.

The monthly archives hold roughly 150-250k instance documents each. Parsing is
spread over worker processes; each worker opens the zip itself and handles a
slice of member names, so nothing large is pickled between processes. Nested
zips (some archives bundle daily zips) are expanded transparently.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from smewatch.ixbrl import FILENAME_RE, OUTPUT_COLUMNS, parse_document

log = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        ("company_number", pa.string()),
        ("balance_sheet_date", pa.date32()),
        ("source_file", pa.string()),
        ("file_format", pa.string()),
        ("parse_ok", pa.bool_()),
        ("n_facts", pa.int32()),
        ("n_officers", pa.int32()),
        ("is_dormant", pa.bool_()),
        ("entity_name", pa.string()),
        *[(c, pa.float64()) for c in OUTPUT_COLUMNS[9:]],
        ("filing_batch", pa.string()),
        ("filing_period_end", pa.date32()),
    ]
)


def _is_instance(name: str) -> bool:
    return bool(FILENAME_RE.search(name))


def list_members(zip_path: Path) -> list[tuple[str, str | None]]:
    """Return (member, inner_member) pairs for every instance document in the zip."""
    out: list[tuple[str, str | None]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".zip"):
                with zf.open(name) as fh, zipfile.ZipFile(io.BytesIO(fh.read())) as inner:
                    out.extend((name, n) for n in inner.namelist() if _is_instance(n))
            elif _is_instance(name):
                out.append((name, None))
    return out


def _parse_slice(zip_path: str, members: list[tuple[str, str | None]]) -> list[dict]:
    records = []
    inner_cache: dict[str, zipfile.ZipFile] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for outer, inner in members:
            try:
                if inner is None:
                    content = zf.read(outer)
                    fname = outer
                else:
                    if outer not in inner_cache:
                        inner_cache[outer] = zipfile.ZipFile(io.BytesIO(zf.read(outer)))
                    content = inner_cache[outer].read(inner)
                    fname = inner
            except (KeyError, zipfile.BadZipFile, OSError):
                continue
            records.append(parse_document(content, os.path.basename(fname)))
    return records


def parse_zip(
    zip_path: Path,
    out_path: Path,
    filing_batch: str,
    filing_period_end: date,
    workers: int | None = None,
    limit: int | None = None,
    chunk_size: int = 2000,
) -> int:
    """Parse every instance document in ``zip_path`` and write ``out_path``.

    ``filing_batch`` labels the source (e.g. ``August2025`` or ``2026-09-26``)
    and ``filing_period_end`` is the last day the documents could have been
    filed, which later bounds the point-in-time feature set.
    Returns the number of rows written.
    """
    members = list_members(zip_path)
    if limit:
        members = members[:limit]
    log.info("%s: %d instance documents", zip_path.name, len(members))
    workers = workers or max(1, (os.cpu_count() or 2))
    chunks = [members[i : i + chunk_size] for i in range(0, len(members), chunk_size)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with pq.ParquetWriter(out_path, SCHEMA, compression="zstd") as writer:
        def flush(rows: list[dict]) -> int:
            for r in rows:
                r["filing_batch"] = filing_batch
                r["filing_period_end"] = filing_period_end
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            return len(rows)

        if workers == 1 or len(chunks) <= 1:
            for chunk in chunks:
                written += flush(_parse_slice(str(zip_path), chunk))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_parse_slice, str(zip_path), c) for c in chunks]
                for i, fut in enumerate(as_completed(futures), 1):
                    written += flush(fut.result())
                    if i % 10 == 0 or i == len(futures):
                        log.info("  parsed %d/%d chunks (%d rows)", i, len(futures), written)
    return written
