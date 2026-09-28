"""Command line entry point: ``python -m smewatch.cli <command>``.

Commands mirror the pipeline stages:

    ingest-snapshot     download + convert the register snapshot
    ingest-accounts     download + parse a monthly accounts archive
    ingest-daily        download + parse recent daily accounts files
    export-extract      write the slim extracts the warehouse is built from
    build               run the dbt project (staging -> marts)
    train               fit and evaluate the models
    score               score the latest filings with the trained model
    publish             write the small files the dashboard reads
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

from smewatch import config

log = logging.getLogger("smewatch")


def cmd_ingest_snapshot(args: argparse.Namespace) -> None:
    from smewatch.ingest.download import download
    from smewatch.ingest.snapshot import snapshot_to_parquet

    snap = date.fromisoformat(args.date) if args.date else config.DESIGN.outcome_snapshot
    zpath = download(config.snapshot_url(snap), config.RAW / f"BasicCompanyData-{snap}.zip")
    snapshot_to_parquet(zpath, config.PROCESSED / "snapshot.parquet", config.RAW / "snapshot_csv", limit=args.limit)


def cmd_ingest_accounts(args: argparse.Namespace) -> None:
    from smewatch.ingest.accounts import parse_zip
    from smewatch.ingest.download import download

    for month in args.months or config.DESIGN.all_months:
        zpath = download(config.monthly_accounts_url(month), config.RAW / f"Accounts_Monthly_Data-{month}.zip")
        n = parse_zip(
            zpath,
            config.PROCESSED / f"accounts_{month}.parquet",
            filing_batch=month,
            filing_period_end=config.month_end(month),
            workers=args.workers,
            limit=args.limit,
        )
        log.info("%s: %d filings parsed", month, n)
        if args.delete_zip:
            zpath.unlink(missing_ok=True)


def cmd_ingest_daily(args: argparse.Namespace) -> None:
    import requests

    from smewatch.ingest.accounts import parse_zip
    from smewatch.ingest.download import download

    end = date.fromisoformat(args.end) if args.end else date.today()
    got = 0
    day = end
    while got < args.days and (end - day).days < args.days + 14:
        url = config.daily_accounts_url(day)
        try:
            zpath = download(url, config.RAW / "daily" / f"Accounts_Bulk_Data-{day}.zip")
        except requests.HTTPError:
            day -= timedelta(days=1)  # weekends / holidays have no file
            continue
        parse_zip(zpath, config.PROCESSED / "daily" / f"accounts_{day}.parquet", filing_batch=day.isoformat(), filing_period_end=day, workers=args.workers)
        zpath.unlink(missing_ok=True)
        got += 1
        day -= timedelta(days=1)
    log.info("parsed %d daily files", got)


def cmd_export_extract(args: argparse.Namespace) -> None:
    from smewatch.extract import export_extract

    export_extract(config.PROCESSED, config.DATA / "extract")


def cmd_build(args: argparse.Namespace) -> None:
    from smewatch.warehouse import build_warehouse

    build_warehouse(select=args.select)


def cmd_train(args: argparse.Namespace) -> None:
    from smewatch.model.train import train_all

    train_all()


def cmd_score(args: argparse.Namespace) -> None:
    from smewatch.model.score import score_latest

    score_latest()


def cmd_publish(args: argparse.Namespace) -> None:
    from smewatch.publish import publish_all

    publish_all()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.ensure_dirs()
    p = argparse.ArgumentParser(prog="smewatch")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest-snapshot")
    s.add_argument("--date", help="snapshot date YYYY-MM-DD (default: study outcome snapshot)")
    s.add_argument("--limit", type=int)
    s.set_defaults(func=cmd_ingest_snapshot)

    s = sub.add_parser("ingest-accounts")
    s.add_argument("--months", nargs="*", help="e.g. July2025 August2025")
    s.add_argument("--workers", type=int)
    s.add_argument("--limit", type=int)
    s.add_argument("--delete-zip", action="store_true")
    s.set_defaults(func=cmd_ingest_accounts)

    s = sub.add_parser("ingest-daily")
    s.add_argument("--days", type=int, default=5)
    s.add_argument("--end", help="latest date to try, YYYY-MM-DD")
    s.add_argument("--workers", type=int)
    s.set_defaults(func=cmd_ingest_daily)

    sub.add_parser("export-extract").set_defaults(func=cmd_export_extract)
    s = sub.add_parser("build")
    s.add_argument("--select", default=None)
    s.set_defaults(func=cmd_build)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("publish").set_defaults(func=cmd_publish)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
