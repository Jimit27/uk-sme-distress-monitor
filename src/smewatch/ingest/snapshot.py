"""Load the Companies House 'Free Company Data Product' register snapshot.

The snapshot is a ~2.5 GB CSV (5+ million live companies) shipped as a zip.
DuckDB streams it straight into parquet, keeping every column as text; typing
and cleaning happen in the dbt staging model so the raw layer stays faithful
to the source.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


def snapshot_to_parquet(zip_path: Path, out_path: Path, workdir: Path, limit: int | None = None) -> int:
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"no CSV inside {zip_path}")
        paths = []
        for name in csv_names:
            target = workdir / Path(name).name
            if not target.exists():
                zf.extract(name, workdir)
                (workdir / name).rename(target) if (workdir / name) != target else None
            paths.append(target)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    file_list = ", ".join(f"'{p.as_posix()}'" for p in paths)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    # Header names in the source carry stray leading spaces; normalize_names
    # turns 'Accounts.NextDueDate' into 'accounts_nextduedate' etc.
    # A handful of rows list a fifth SIC code and so carry one extra field.
    # Non-strict parsing keeps those rows (dropping them would make the
    # companies look dissolved); only their trailing confirmation-statement
    # dates shift, and those are not used by the model.
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_csv([{file_list}], header=true, all_varchar=true,
                                   normalize_names=true, union_by_name=true,
                                   strict_mode=false, null_padding=true, parallel=false,
                                   quote='"', escape='"')
            {limit_sql}
        ) TO '{out_path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    log.info("snapshot rows: %d", n)
    return n
