"""Build the DuckDB warehouse by running the dbt project programmatically."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from smewatch import config

log = logging.getLogger(__name__)
DBT_DIR = config.ROOT / "dbt"


def build_warehouse(
    select: str | None = None,
    extract_dir: Path | None = None,
    warehouse: Path | None = None,
    dbt_vars: dict | None = None,
    test: bool = True,
) -> None:
    from dbt.cli.main import dbtRunner

    os.environ["SMEWATCH_EXTRACT"] = str((extract_dir or config.DATA / "extract").resolve())
    os.environ["SMEWATCH_WAREHOUSE"] = str((warehouse or config.WAREHOUSE).resolve())
    args = ["build" if test else "run", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    if select:
        args += ["--select", select]
    if dbt_vars:
        args += ["--vars", json.dumps(dbt_vars)]
    res = dbtRunner().invoke(args)
    if not res.success:
        raise RuntimeError(f"dbt {' '.join(args[:1])} failed: {res.exception}")
    log.info("warehouse built at %s", os.environ["SMEWATCH_WAREHOUSE"])
