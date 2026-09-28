import io
import zipfile
from datetime import date

import duckdb

from smewatch.extract import export_extract
from smewatch.ingest.accounts import list_members, parse_zip
from smewatch.ingest.snapshot import snapshot_to_parquet


def test_parse_zip_skips_non_instances(tmp_path, fixtures_dir):
    out = tmp_path / "a.parquet"
    n = parse_zip(fixtures_dir / "accounts" / "Accounts_fixture.zip", out, "August2025", date(2025, 8, 31), workers=1)
    assert n == 3
    df = duckdb.sql(f"select * from '{out}'").df()
    assert set(df["filing_batch"]) == {"August2025"}
    assert df["parse_ok"].all()


def test_nested_zip(tmp_path, fixtures_dir):
    inner = fixtures_dir / "accounts" / "Accounts_fixture.zip"
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "Accounts_Bulk_Data-2025-08-01.zip")
    assert len(list_members(outer)) == 3
    n = parse_zip(outer, tmp_path / "b.parquet", "August2025", date(2025, 8, 31), workers=2, chunk_size=1)
    assert n == 3


def test_snapshot_and_extract(tmp_path, fixtures_dir):
    processed = tmp_path / "processed"
    n = snapshot_to_parquet(fixtures_dir / "snapshot" / "BasicCompanyData-fixture.zip", processed / "snapshot.parquet", tmp_path / "csv")
    assert n == 5
    export_extract(processed, tmp_path / "extract")
    df = duckdb.sql(f"select * from '{tmp_path}/extract/register_snapshot.parquet'").df()
    assert {"company_number", "company_status", "sic_code_1", "postcode_area", "incorporation_date"} <= set(df.columns)
    row = df.set_index("company_number").loc["SC312961"]
    assert row["company_status"] == "Liquidation"
    assert row["sic_code_1"] == "70100"
    assert row["postcode_area"] == "EH"
