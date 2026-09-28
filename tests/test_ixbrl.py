from datetime import date

import pytest

from smewatch.ixbrl import parse_document, parse_filename, parse_number


@pytest.mark.parametrize(
    "text,fmt,scale,sign,expected",
    [
        ("1,234", "ixt2:numdotdecimal", "0", None, 1234.0),
        ("1.234,5", "ixt:numcommadecimal", "0", None, 1234.5),
        ("12", "ixt2:numdotdecimal", "3", None, 12000.0),
        ("500", "ixt2:numdotdecimal", "0", "-", -500.0),
        ("-", "ixt2:fixed-zero", "0", None, 0.0),
        ("-", "ixt:zerodash", None, None, 0.0),
        ("(2,000)", None, None, None, -2000.0),
        ("", "ixt2:numdotdecimal", "0", None, None),
    ],
)
def test_parse_number(text, fmt, scale, sign, expected):
    assert parse_number(text, fmt, scale, sign) == expected


def test_parse_filename():
    assert parse_filename("Prod223_2911_SC312961_20200930.html") == ("SC312961", date(2020, 9, 30), "html")
    assert parse_filename("readme.txt") == (None, None, None)


def _load(fixtures_dir, name):
    return parse_document((fixtures_dir / "accounts" / name).read_bytes(), name)


def test_real_filing_balance_sheet(fixtures_dir):
    r = _load(fixtures_dir, "Prod223_2911_05078870_20200930.html")
    assert r["parse_ok"]
    assert r["company_number"] == "05078870"
    assert r["balance_sheet_date"] == date(2020, 9, 30)
    assert r["equity_cur"] == 1262403.0
    assert r["equity_prior"] == 650346.0
    assert r["cash_cur"] == 1482657.0
    assert r["current_assets_cur"] == 3009054.0
    assert r["creditors_within_1y_cur"] == 1832165.0
    assert r["employees_cur"] == 138.0
    assert r["is_dormant"] is False


def test_working_capital_identity(fixtures_dir):
    # current assets - creditors due within a year == net current assets
    r = _load(fixtures_dir, "Prod223_2911_09847839_20201031.html")
    assert r["current_assets_cur"] - r["creditors_within_1y_cur"] == pytest.approx(r["net_current_assets_cur"])


def test_scottish_company(fixtures_dir):
    r = _load(fixtures_dir, "Prod223_2911_SC312961_20200930.html")
    assert r["company_number"] == "SC312961"
    assert r["equity_cur"] == 47305.0 and r["equity_prior"] == 48217.0


def test_malformed_document_does_not_raise():
    r = parse_document(b"<html><body>not xbrl", "Prod223_0001_01234567_20240331.html")
    assert r["parse_ok"] is False
    assert r["company_number"] == "01234567"
    r = parse_document(b"", "Prod223_0001_01234567_20240331.html")
    assert r["parse_ok"] is False


def test_plain_xbrl_xml():
    xml = b"""<?xml version="1.0"?>
    <xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:core="http://xbrl.frc.org.uk/fr/2023-01-01/core">
      <xbrli:context id="c1"><xbrli:entity><xbrli:identifier scheme="x">01234567</xbrli:identifier></xbrli:entity>
        <xbrli:period><xbrli:instant>2024-03-31</xbrli:instant></xbrli:period></xbrli:context>
      <xbrli:context id="c0"><xbrli:entity><xbrli:identifier scheme="x">01234567</xbrli:identifier></xbrli:entity>
        <xbrli:period><xbrli:instant>2023-03-31</xbrli:instant></xbrli:period></xbrli:context>
      <core:Equity contextRef="c1" unitRef="GBP" decimals="0">-2500</core:Equity>
      <core:Equity contextRef="c0" unitRef="GBP" decimals="0">1000</core:Equity>
      <core:CashBankOnHand contextRef="c1" unitRef="GBP" decimals="0">300</core:CashBankOnHand>
    </xbrli:xbrl>"""
    r = parse_document(xml, "Prod224_0001_01234567_20240331.xml")
    assert r["parse_ok"]
    assert r["equity_cur"] == -2500.0 and r["equity_prior"] == 1000.0 and r["cash_cur"] == 300.0
