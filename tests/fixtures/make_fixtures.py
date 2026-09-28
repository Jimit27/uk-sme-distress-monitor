"""Build the small zip fixtures used by the tests from the files in this folder.

* snapshot/BasicCompanyData-fixture.zip  - register snapshot in the real CSV layout
* accounts/Accounts_fixture.zip          - the real iXBRL filings in this folder, zipped
"""
import csv
import io
import zipfile
from pathlib import Path

HERE = Path(__file__).parent

HEADER = ["CompanyName", " CompanyNumber", "RegAddress.CareOf", "RegAddress.POBox", "RegAddress.AddressLine1",
          " RegAddress.AddressLine2", "RegAddress.PostTown", "RegAddress.County", "RegAddress.Country",
          "RegAddress.PostCode", "CompanyCategory", "CompanyStatus", "CountryOfOrigin", "DissolutionDate",
          "IncorporationDate", "Accounts.AccountRefDay", "Accounts.AccountRefMonth", "Accounts.NextDueDate",
          "Accounts.LastMadeUpDate", "Accounts.AccountCategory", "Returns.NextDueDate", "Returns.LastMadeUpDate",
          "Mortgages.NumMortCharges", "Mortgages.NumMortOutstanding", "Mortgages.NumMortPartSatisfied",
          "Mortgages.NumMortSatisfied", "SICCode.SicText_1", "SICCode.SicText_2", "SICCode.SicText_3",
          "SICCode.SicText_4", "LimitedPartnerships.NumGenPartners", "LimitedPartnerships.NumLimPartners", "URI",
          "PreviousName_1.CONDATE", " PreviousName_1.CompanyName", "ConfStmtNextDueDate", " ConfStmtLastMadeUpDate"]

# (number, status, incorporated, sic, postcode, next accounts due, category)
ROWS = [
    ("09847839", "Active", "28/10/2015", "88990 - Other social work activities without accommodation n.e.c.", "M4 5JD", "31/07/2027", "SMALL"),
    ("SC312961", "Liquidation", "20/11/2006", "70100 - Activities of head offices", "EH2 4AD", "30/06/2026", "MICRO ENTITY"),
    # 05078870 deliberately absent: dissolved / removed from the register
    ("00000001", "Active - Proposal to Strike off", "01/01/1990", "62020 - Information technology consultancy activities", "CR0 1AB", "31/12/2025", "DORMANT"),
    ("00000002", "In Administration", "05/05/2012", "47110 - Retail sale in non-specialised stores", "B1 1AA", "30/09/2026", "TOTAL EXEMPTION FULL"),
    ("12345678", "Active", "14/02/2019", "56101 - Licensed restaurants", "SW1A 1AA", "30/11/2026", "MICRO ENTITY"),
]


def build() -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    for num, status, inc, sic, pc, due, cat in ROWS:
        row = [""] * len(HEADER)
        row[0] = f"FIXTURE CO {num} LIMITED"
        row[1] = num
        row[6] = "LONDON"
        row[9] = pc
        row[10] = "Private Limited Company"
        row[11] = status
        row[12] = "United Kingdom"
        row[14] = inc
        row[17] = due
        row[19] = cat
        row[22] = "1"
        row[23] = "1"
        row[26] = sic
        row[35] = "01/01/2027"
        w.writerow(row)
    (HERE / "snapshot").mkdir(exist_ok=True)
    with zipfile.ZipFile(HERE / "snapshot" / "BasicCompanyData-fixture.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BasicCompanyDataAsOneFile-fixture.csv", buf.getvalue())
    with zipfile.ZipFile(HERE / "accounts" / "Accounts_fixture.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted((HERE / "accounts").glob("*.html")):
            zf.write(f, f.name)
        zf.writestr("readme.txt", "not an instance document")


if __name__ == "__main__":
    build()
