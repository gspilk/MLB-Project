"""
spotrac_debug.py
One-off diagnostic: prints every table pandas.read_html() actually
finds on the Spotrac payroll page, and its real column names -- so
spotrac_scraper.py's column-matching can be fixed against the real
structure instead of a guess.

Usage:
    python spotrac_debug.py
"""

import io
import pandas as pd
from spotrac_scraper import _fetch_page, PAYROLL_URL

html = _fetch_page(PAYROLL_URL)
tables = pd.read_html(io.StringIO(html))
print(f"\n{len(tables)} tables found\n")

for i, df in enumerate(tables):
    print(f"--- Table {i} ({len(df)} rows) ---")
    print("Columns:", df.columns.tolist())
    print(df.head(2).to_string(index=False))
    print()