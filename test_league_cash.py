"""
test_league_cash.py
Quick, one-off check: does spotrac.com/mlb/cash actually return CURRENT
year data, or does it hit the same stale-fetch issue found on the
team-specific payroll page earlier tonight?

Usage:
    python test_league_cash.py
"""

from spotrac_scraper import get_league_cash_totals

df = get_league_cash_totals(force_refresh=True)
print(f"\n{len(df)} teams found\n")
print(df.to_string(index=False))