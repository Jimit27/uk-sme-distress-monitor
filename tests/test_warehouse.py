from datetime import date

import duckdb
import pytest

from smewatch.extract import export_extract
from smewatch.ingest.accounts import parse_zip
from smewatch.ingest.snapshot import snapshot_to_parquet
from smewatch.warehouse import build_warehouse




def test_dbt_build_end_to_end(tmp_path, fixtures_dir):
    processed = tmp_path / "processed"
    snapshot_to_parquet(fixtures_dir / "snapshot" / "BasicCompanyData-fixture.zip", processed / "snapshot.parquet", tmp_path / "csv")
    zp = fixtures_dir / "accounts" / "Accounts_fixture.zip"
    parse_zip(zp, processed / "accounts_July2025.parquet", "July2025", date(2025, 7, 31), workers=1, limit=1)
    parse_zip(zp, processed / "accounts_August2025.parquet", "August2025", date(2025, 8, 31), workers=1)
    parse_zip(zp, processed / "accounts_2026-09-25.parquet", "2026-09-25", date(2026, 9, 25), workers=1)
    export_extract(processed, tmp_path / "extract")

    wh = tmp_path / "wh.duckdb"
    build_warehouse(extract_dir=tmp_path / "extract", warehouse=wh)  # runs dbt models + all data tests

    con = duckdb.connect(str(wh))
    cohort = con.execute("select * from fct_training_cohort").df().set_index("company_number")
    # train/test never share a company
    assert cohort.index.is_unique
    assert set(cohort["split"]) == {"train", "test"}
    # outcomes read from the snapshot
    assert cohort.loc["SC312961", "label_insolvency"] == 1
    assert cohort.loc["09847839", "label_failure"] == 0
    assert cohort.loc["05078870", "outcome_group"] == "removed"   # absent from snapshot
    assert cohort.loc["05078870", "label_failure"] == 1
    # features are computed from the accounts alone
    assert cohort.loc["05078870", "equity_to_assets"] > 0
    live = con.execute("select * from fct_live_filings").df()
    assert len(live) == 3 and live["sic_section"].notna().sum() == 2
