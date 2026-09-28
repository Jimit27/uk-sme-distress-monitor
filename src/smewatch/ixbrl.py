"""Parser for Companies House accounts filed as inline XBRL (.html) or XBRL (.xml).

Companies House publishes every electronically filed set of accounts as an
"instance document". Each one tags balance-sheet numbers with concepts from the
FRC taxonomy (FRS 102 / FRS 105), e.g. ``core:Equity`` or
``core:CashBankOnHand``, and attaches each number to a *context* that says which
date (or period) and which dimension slice it belongs to.

This module pulls out a fixed set of concepts for two points in time:

* ``cur``   - the balance sheet date of the filing
* ``prior`` - the comparative year shown alongside it

Only undimensioned contexts are used for headline figures, except creditors,
which FRS 102 reports split by a maturity dimension (within / after one year).

The parser is deliberately tolerant: filings come from hundreds of different
software packages, namespace prefixes vary (``core:``, ``ns6:``, ``d:``) and a
small share of documents are malformed. Anything unparseable yields ``None``
values rather than an exception, and ``parse_ok`` records whether the document
could be read at all.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

from lxml import etree

# Filenames look like Prod223_2911_SC312961_20200930.html
FILENAME_RE = re.compile(
    r"_(?P<company>[A-Z0-9]{8})_(?P<bsdate>\d{8})\.(?P<ext>html|xml)$", re.IGNORECASE
)

# Instant (balance sheet) concepts: local name -> output column stem.
INSTANT_CONCEPTS: dict[str, str] = {
    "Equity": "equity",
    "NetAssetsLiabilities": "net_assets",
    "NetCurrentAssetsLiabilities": "net_current_assets",
    "TotalAssetsLessCurrentLiabilities": "total_assets_less_cl",
    "CurrentAssets": "current_assets",
    "FixedAssets": "fixed_assets",
    "PropertyPlantEquipment": "ppe",
    "IntangibleAssets": "intangibles",
    "CashBankOnHand": "cash",
    "Debtors": "debtors",
    "TotalInventories": "inventories",
    "ProvisionsForLiabilitiesBalanceSheetSubtotal": "provisions",
    "CalledUpShareCapital": "share_capital",
    "RetainedEarningsAccumulatedLosses": "retained_earnings",
    "BankBorrowingsOverdrafts": "bank_borrowings",
    "AmountsOwedToDirectors": "owed_to_directors",
    "TaxationSocialSecurityPayable": "tax_social_security_payable",
    "TradeCreditorsTradePayables": "trade_creditors",
}

# Period (flow) concepts reported for the financial year ending on the BS date.
DURATION_CONCEPTS: dict[str, str] = {
    "AverageNumberEmployeesDuringPeriod": "employees",
    "TurnoverRevenue": "turnover",
    "ProfitLoss": "profit_loss",
}

# Creditors are dimensioned by maturity in FRS 102 filings, either with the
# maturity dimension, the current/non-current instrument dimension, or both.
CREDITOR_BUCKETS = {"WithinOneYear": "creditors_within_1y", "AfterOneYear": "creditors_after_1y"}
_CREDITOR_MEMBERS = {
    "WithinOneYear": "WithinOneYear",
    "CurrentFinancialInstruments": "WithinOneYear",
    "AfterOneYear": "AfterOneYear",
    "Non-currentFinancialInstruments": "AfterOneYear",
}
_CREDITOR_DIMENSIONS = {"MaturitiesOrExpirationPeriodsDimension", "FinancialInstrumentCurrentNon-currentDimension"}


def _creditor_bucket(dims: tuple[tuple[str, str], ...]) -> str | None:
    """Map a creditors context to within/after one year, or None if it is a sub-analysis."""
    buckets = set()
    for dim, member in dims:
        if dim not in _CREDITOR_DIMENSIONS:
            return None  # e.g. split by creditor type - not the headline figure
        mapped = _CREDITOR_MEMBERS.get(member)
        if mapped is None:
            return None  # e.g. BetweenOneFiveYears
        buckets.add(mapped)
    if len(buckets) != 1:
        return None
    return CREDITOR_BUCKETS[buckets.pop()]

TEXT_CONCEPTS: dict[str, str] = {
    "EntityCurrentLegalOrRegisteredName": "entity_name",
    "EntityDormantTruefalse": "dormant_flag_text",
    "UKCompaniesHouseRegisteredNumber": "ch_number_text",
}

NUMERIC_COLUMNS: list[str] = (
    [f"{stem}_{when}" for stem in INSTANT_CONCEPTS.values() for when in ("cur", "prior")]
    + [f"{stem}_{when}" for stem in DURATION_CONCEPTS.values() for when in ("cur", "prior")]
    + [f"{stem}_{when}" for stem in CREDITOR_BUCKETS.values() for when in ("cur", "prior")]
)

OUTPUT_COLUMNS: list[str] = [
    "company_number",
    "balance_sheet_date",
    "source_file",
    "file_format",
    "parse_ok",
    "n_facts",
    "n_officers",
    "is_dormant",
    "entity_name",
    *NUMERIC_COLUMNS,
]


@dataclass
class _Context:
    instant: date | None
    start: date | None
    end: date | None
    dims: tuple[tuple[str, str], ...]


def _local(tag: str | None) -> str:
    if not tag or not isinstance(tag, str):
        return ""
    if "}" in tag:
        tag = tag.rsplit("}", 1)[1]
    if ":" in tag:
        tag = tag.rsplit(":", 1)[1]
    return tag


def _parse_date(text: str | None) -> date | None:
    if not text:
        return None
    text = text.strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_filename(name: str) -> tuple[str | None, date | None, str | None]:
    """Return (company_number, balance_sheet_date, extension) from a CH filename."""
    m = FILENAME_RE.search(name)
    if not m:
        return None, None, None
    bs = datetime.strptime(m.group("bsdate"), "%Y%m%d").date()
    return m.group("company").upper(), bs, m.group("ext").lower()


_NUM_CLEAN = re.compile(r"[^0-9.\-]")


def parse_number(text: str, fmt: str | None, scale: str | None, sign: str | None) -> float | None:
    """Convert an iXBRL displayed number into a float.

    Handles the transformation formats that Companies House filings actually
    use: dot-decimal, comma-decimal, and the various 'zero dash' formats.
    """
    raw = (text or "").strip()
    fmt_l = (fmt or "").lower()
    if "zero" in fmt_l or raw in {"-", "–", "—", "nil", "Nil"}:
        value = 0.0
    else:
        if not raw:
            return None
        if "numcommadecimal" in fmt_l or "num-comma-decimal" in fmt_l:
            raw = raw.replace(".", "").replace(" ", "").replace(",", ".")
        else:
            raw = raw.replace(",", "").replace(" ", "")
        negative_paren = raw.startswith("(") and raw.endswith(")")
        raw = _NUM_CLEAN.sub("", raw)
        if raw in {"", ".", "-"}:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if negative_paren:
            value = -abs(value)
    try:
        power = int(scale) if scale not in (None, "") else 0
    except ValueError:
        power = 0
    value *= 10**power
    if sign == "-":
        value = -value
    return value


def _read_contexts(root: etree._Element) -> dict[str, _Context]:
    contexts: dict[str, _Context] = {}
    for el in root.iter():
        if _local(el.tag) != "context":
            continue
        cid = el.get("id")
        if not cid:
            continue
        instant = start = end = None
        dims: list[tuple[str, str]] = []
        for child in el.iter():
            name = _local(child.tag)
            if name == "instant":
                instant = _parse_date(child.text)
            elif name == "startDate":
                start = _parse_date(child.text)
            elif name == "endDate":
                end = _parse_date(child.text)
            elif name in ("explicitMember", "typedMember"):
                dims.append((_local(child.get("dimension")), _local((child.text or "").strip())))
        contexts[cid] = _Context(instant, start, end, tuple(sorted(dims)))
    return contexts


def _iter_facts(root: etree._Element, is_inline: bool) -> Iterable[tuple[str, str, str, str | None, str | None, str | None, bool]]:
    """Yield (concept, contextRef, text, format, scale, sign, numeric) for each fact."""
    if is_inline:
        for el in root.iter():
            tag = _local(el.tag)
            if tag not in ("nonFraction", "nonNumeric"):
                continue
            name = el.get("name")
            ctx = el.get("contextRef")
            if not name or not ctx:
                continue
            text = "".join(el.itertext())
            yield (
                _local(name),
                ctx,
                text,
                el.get("format"),
                el.get("scale"),
                el.get("sign"),
                tag == "nonFraction",
            )
    else:
        for el in root.iter():
            ctx = el.get("contextRef")
            if not ctx:
                continue
            numeric = el.get("unitRef") is not None
            yield (_local(el.tag), ctx, el.text or "", None, None, None, numeric)


def _choose_periods(contexts: dict[str, _Context], bs_date: date | None) -> tuple[date | None, date | None]:
    """Work out the current and prior balance sheet instants for this filing."""
    instants = sorted({c.instant for c in contexts.values() if c.instant and not c.dims})
    if not instants:
        instants = sorted({c.end for c in contexts.values() if c.end})
    if not instants:
        return bs_date, None
    cur = bs_date if bs_date in instants else (bs_date or instants[-1])
    if cur not in instants:
        cur = instants[-1]
    prior_candidates = [d for d in instants if 250 <= (cur - d).days <= 550]
    prior = max(prior_candidates) if prior_candidates else None
    return cur, prior


def empty_record(source_file: str) -> dict:
    rec = {col: None for col in OUTPUT_COLUMNS}
    company, bs_date, ext = parse_filename(source_file)
    rec.update(
        company_number=company,
        balance_sheet_date=bs_date,
        source_file=source_file,
        file_format=ext,
        parse_ok=False,
        n_facts=0,
        n_officers=0,
        is_dormant=None,
    )
    return rec


_PARSER = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)


def parse_document(content: bytes, source_file: str) -> dict:
    """Parse one accounts instance document into a flat record."""
    rec = empty_record(source_file)
    try:
        root = etree.fromstring(content, parser=_PARSER)
    except (etree.XMLSyntaxError, ValueError):
        root = None
    if root is None:
        return rec

    is_inline = rec["file_format"] != "xml"
    contexts = _read_contexts(root)
    cur, prior = _choose_periods(contexts, rec["balance_sheet_date"])
    if rec["balance_sheet_date"] is None:
        rec["balance_sheet_date"] = cur

    def which_instant(ctx: _Context) -> str | None:
        when = ctx.instant or ctx.end
        if when is None:
            return None
        if when == cur:
            return "cur"
        if prior is not None and when == prior:
            return "prior"
        return None

    officers: set[str] = set()
    n_facts = 0
    for concept, ctx_ref, text, fmt, scale, sign, numeric in _iter_facts(root, is_inline):
        n_facts += 1
        ctx = contexts.get(ctx_ref)
        if concept == "NameEntityOfficer":
            name = " ".join(text.split()).upper()
            if name:
                officers.add(name)
            continue
        if concept in TEXT_CONCEPTS:
            value = " ".join(text.split())
            col = TEXT_CONCEPTS[concept]
            if value and not rec.get(col):
                rec[col] = value[:200]
            continue
        if not numeric or ctx is None:
            continue
        when = which_instant(ctx)
        if when is None:
            continue
        if concept in INSTANT_CONCEPTS and not ctx.dims and ctx.instant is not None:
            col = f"{INSTANT_CONCEPTS[concept]}_{when}"
        elif concept in DURATION_CONCEPTS and not ctx.dims and ctx.end is not None:
            col = f"{DURATION_CONCEPTS[concept]}_{when}"
        elif concept == "Creditors" and ctx.instant is not None and ctx.dims:
            bucket = _creditor_bucket(ctx.dims)
            if bucket is None:
                continue
            col = f"{bucket}_{when}"
        else:
            continue
        if rec.get(col) is not None:
            continue  # first occurrence wins; duplicates are the same fact repeated
        rec[col] = parse_number(text, fmt, scale, sign)

    rec["n_facts"] = n_facts
    rec["n_officers"] = len(officers)
    dormant_text = (rec.pop("dormant_flag_text", None) or "").strip().lower()
    rec["is_dormant"] = True if dormant_text == "true" else (False if dormant_text == "false" else None)
    if rec["company_number"] is None and rec.get("ch_number_text"):
        rec["company_number"] = rec["ch_number_text"].replace(" ", "").upper()[:8]
    rec.pop("ch_number_text", None)
    rec["parse_ok"] = n_facts > 0
    return {col: rec.get(col) for col in OUTPUT_COLUMNS}
