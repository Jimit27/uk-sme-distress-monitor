"""Write the small, committed files the dashboard reads (data/published/)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import duckdb

from smewatch import config

log = logging.getLogger(__name__)

MARTS = [
    "mart_register_health_by_sector",
    "mart_register_health_by_area",
    "mart_incorporations_by_year",
]


def publish_all(warehouse: Path | None = None, out_dir: Path | None = None) -> None:
    out_dir = out_dir or config.PUBLISHED
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse or config.WAREHOUSE))
    try:
        for mart in MARTS:
            con.execute(f"copy (select * from {mart}) to '{(out_dir / f'{mart}.parquet').as_posix()}' (format parquet)")
        stats = con.execute(
            """
            select
                count(*)                                            as companies,
                sum((status_group = 'insolvency')::int)             as in_insolvency,
                sum((status_group = 'strike_off_proposed')::int)    as strike_off_proposed
            from stg_ch__register
            """
        ).df().iloc[0].to_dict()
    finally:
        con.close()
    (out_dir / "register_summary.json").write_text(json.dumps({k: int(v) for k, v in stats.items()}, indent=2))

    metrics = config.REPORTS / "metrics.json"
    if metrics.exists():
        shutil.copy2(metrics, out_dir / "metrics.json")
    preds = config.REPORTS / "test_predictions_insolvency.parquet"
    if preds.exists():
        # the dashboard only needs label + scores for its curves; keep the committed file small
        import pandas as pd

        p = pd.read_parquet(preds, columns=["label_insolvency", "pd", "pd_logistic"])
        p = p.astype({"label_insolvency": "int8", "pd": "float32", "pd_logistic": "float32"})
        p.to_parquet(out_dir / preds.name, index=False)
    log.info("published dashboard files to %s", out_dir)
