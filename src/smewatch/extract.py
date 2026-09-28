"""Slim extracts of the raw layer.

The full register snapshot is ~5.7 million rows and 55 columns. The warehouse
only needs a dozen of them, so CI writes a slim, typed-as-text extract that is
small enough to move around (and to cache between runs). Parsed accounts are
copied through unchanged.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# normalised source header -> extract column
SNAPSHOT_COLUMNS = {
    "companynumber": "company_number",
    "companycategory": "company_category",
    "companystatus": "company_status",
    "countryoforigin": "country_of_origin",
    "dissolutiondate": "dissolution_date",
    "incorporationdate": "incorporation_date",
    "accountsnextduedate": "accounts_next_due",
    "accountslastmadeupdate": "accounts_last_made_up",
    "accountsaccountcategory": "accounts_category",
    "confstmtnextduedate": "confstmt_next_due",
    "confstmtlastmadeupdate": "confstmt_last_made_up",
    "mortgagesnummortcharges": "mortgage_charges",
    "mortgagesnummortoutstanding": "mortgage_outstanding",
    "siccodesictext1": "sic_text_1",
    "regaddresspostcode": "postcode",
    "regaddressposttown": "post_town",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def snapshot_select_sql(columns: list[str]) -> str:
    """Build a SELECT list mapping whatever the source headers are to extract names."""
    lookup = {_norm(c): c for c in columns}
    parts = []
    for key, alias in SNAPSHOT_COLUMNS.items():
        src = lookup.get(key)
        if src is None:
            parts.append(f"NULL::VARCHAR AS {alias}")
            continue
        col = f'"{src}"'
        if alias == "sic_text_1":
            # '62020 - Information technology consultancy activities' -> '62020'
            parts.append(f"nullif(trim(split_part({col}, '-', 1)), '') AS sic_code_1")
        elif alias == "postcode":
            # outward area letters only: 'CR0 1AB' -> 'CR'
            parts.append(f"nullif(regexp_extract(upper(trim({col})), '^([A-Z]{{1,2}})', 1), '') AS postcode_area")
        else:
            parts.append(f"nullif(trim({col}), '') AS {alias}")
    return ",\n            ".join(parts)


def export_extract(processed: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = processed / "snapshot.parquet"
    con = duckdb.connect()
    if snap.exists():
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{snap.as_posix()}')").fetchall()]
        sel = snapshot_select_sql(cols)
        target = out_dir / "register_snapshot.parquet"
        con.execute(
            f"""COPY (SELECT {sel} FROM read_parquet('{snap.as_posix()}'))
                TO '{target.as_posix()}' (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 19)"""
        )
        log.info("register extract: %.1f MB", target.stat().st_size / 1e6)
    for f in sorted(processed.glob("accounts_*.parquet")):
        shutil.copy2(f, out_dir / f.name)
        log.info("accounts extract %s: %.1f MB", f.name, f.stat().st_size / 1e6)
