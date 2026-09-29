"""
One-off debug script -- prints the REAL columns bbref's league-wide
fielding leaderboard actually returned, so we can see why Rtot/Rdrs
weren't found (0 of 279 free agents matched on defense) instead of
guessing at the page structure a second time.

Uses your existing cache (data/cache/fielding_*.parquet from the last
run) -- doesn't hit bbref again.
"""
from data_builder import build_all

data = build_all(2026)
fld = data.get("fielding", {}).get("all_players")

if fld is None or fld.empty:
    print("fielding.all_players is empty or missing -- something failed upstream")
else:
    print(f"\nColumns ({len(fld.columns)}): {list(fld.columns)}")
    print(f"\nFirst 15 rows:")
    print(fld.head(15).to_string(index=False))

    # also check the raw AL/NL pre-combine tables in case something got
    # lost specifically in the concat step
    al = data["fielding"].get("al_players")
    if al is not None and not al.empty:
        print(f"\nAL-only columns ({len(al.columns)}): {list(al.columns)}")