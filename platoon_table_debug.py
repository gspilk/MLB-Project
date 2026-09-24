"""
platoon_table_debug.py
25 real tables were found on the splits page, but none matched as the
platoon-specific one -- shows every table's real columns and flags
which ones actually contain "LHP"/"RHP" text anywhere, so the real
selection logic in platoon_lineup_optimizer.py can be fixed against
what's actually there instead of guessing again.

Usage:
    python platoon_table_debug.py
"""

import io
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from platoon_lineup_optimizer import _make_session

url = "https://www.baseball-reference.com/players/split.fcgi?id=raleica01&year=2026&t=b"
print(f"  [GET] {url}")
session = _make_session()
resp = session.get(url, timeout=15)
resp.raise_for_status()
resp.encoding = "utf-8"
time.sleep(3)

soup = BeautifulSoup(resp.text, "html.parser")
tables = []
for tag in soup.find_all("table"):
    try:
        tables.append((tag.get("id", "(no id)"), pd.read_html(io.StringIO(str(tag)))[0]))
    except ValueError:
        continue
for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
    if "<table" not in comment:
        continue
    c_soup = BeautifulSoup(comment, "html.parser")
    for tag in c_soup.find_all("table"):
        try:
            tables.append((tag.get("id", "(no id)"), pd.read_html(io.StringIO(str(tag)))[0]))
        except ValueError:
            continue

print(f"\n{len(tables)} tables total\n")
for i, (table_id, df) in enumerate(tables):
    contains_lhp_rhp = df.astype(str).apply(lambda col: col.str.contains("LHP|RHP", na=False, regex=True)).any().any()
    flag = " <-- CONTAINS LHP/RHP TEXT" if contains_lhp_rhp else ""
    print(f"  Table {i} (id={table_id}): {len(df)} rows, columns={df.columns.tolist()[:6]}{flag}")