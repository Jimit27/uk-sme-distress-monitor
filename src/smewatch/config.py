"""Central configuration: data locations, source URLs and the study design."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(os.environ.get("SMEWATCH_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"            # large downloads, never committed
PROCESSED = DATA / "processed"  # parsed parquet, never committed
PUBLISHED = DATA / "published"  # small outputs the app reads, committed by CI
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
WAREHOUSE = DATA / "warehouse.duckdb"

CH_DOWNLOAD = "https://download.companieshouse.gov.uk"


@dataclass(frozen=True)
class StudyDesign:
    """The point-in-time design behind the training data.

    Features come only from accounts filed in the cohort months. Outcomes are
    read from the register snapshot taken roughly 12 months after the end of
    the test cohort month, so nothing the model sees is dated after the filing.
    """

    # Accounts filed in this month train the model ...
    train_months: tuple[str, ...] = ("July2025",)
    # ... and accounts filed in this later month are the out-of-time test set.
    test_months: tuple[str, ...] = ("August2025",)
    # Register snapshot that supplies the outcome labels.
    outcome_snapshot: date = date(2026, 9, 1)

    @property
    def all_months(self) -> tuple[str, ...]:
        return self.train_months + self.test_months


DESIGN = StudyDesign()

# Status strings on the register that mean a formal insolvency process has started.
INSOLVENCY_KEYWORDS = ("liquidation", "administration", "receiver", "receivership", "voluntary arrangement", "insolvency")
STRIKE_OFF_KEYWORDS = ("proposal to strike off",)


def monthly_accounts_url(month: str) -> str:
    return f"{CH_DOWNLOAD}/Accounts_Monthly_Data-{month}.zip"


def daily_accounts_url(day: date) -> str:
    return f"{CH_DOWNLOAD}/Accounts_Bulk_Data-{day.isoformat()}.zip"


def snapshot_url(snapshot: date) -> str:
    return f"{CH_DOWNLOAD}/BasicCompanyDataAsOneFile-{snapshot.isoformat()}.zip"


def month_end(month: str) -> date:
    """'August2025' -> date(2025, 8, 31)."""
    import calendar
    from datetime import datetime

    first = datetime.strptime(month, "%B%Y").date()
    last_day = calendar.monthrange(first.year, first.month)[1]
    return first.replace(day=last_day)


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, PUBLISHED, MODELS, REPORTS / "figures"):
        p.mkdir(parents=True, exist_ok=True)


@dataclass
class Paths:
    raw: Path = field(default_factory=lambda: RAW)
    processed: Path = field(default_factory=lambda: PROCESSED)
    published: Path = field(default_factory=lambda: PUBLISHED)
