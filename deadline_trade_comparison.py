"""
deadline_comparison_data.py
Builds a flat comparison table for Power BI:

  BATTING:  full current Mariners roster (starters + bench) vs.
            tool-identified batter targets vs. Taylor Ward (highlighted
            separately even though he's now actually on the roster too,
            so you can see how he stacks up against his own new teammates)

  PITCHING: full current bullpen -- Ferrer, Speier, Bazardo, Munoz (core)
            + Rucker, Wilcox, Simpson, Dominguez -- vs. tool-identified
            pitcher targets

Writes to data/deadline_comparison.parquet for Power BI (Get Data ->
Parquet -> build clustered bar charts with `name` on axis, `value` on
y-axis, `group` as Legend, filtered by `category`).

DATA NOTE
---------
Ward and Dominguez are both genuinely on the roster now but were
acquired too recently to show up in the scraped bbref/Statcast
leaderboards. Both are manually entered below (see ACTUAL_* dicts) and
folded into the "Current Mariners" group in the output, since that's
what they actually are now -- with a separate `highlight` column so you
can still pick them out in Power BI if you want to color them
differently within that group.

Simpson may or may not appear via the live scrape depending on how
recently your cache refreshed relative to his Aug 2 call-up -- if he's
missing from the "Current Mariners" pitching rows below, that's why;
re-run with a fresh cache (`build_all(2026, force_refresh=True)`) to
pick him up once bbref reflects the move.

REQUIRES: seattle_scraper.py in the same folder (used for get_player()
in case you want to spot-check individual players).

Usage:
    python deadline_comparison_data.py
"""

import os
import pandas as pd

from data_builder import build_all
from team_analyzer import analyze_team
from player_grades import grade_players
from recommender import generate_recommendations


# ── config / manual data ───────────────────────────────────────────────────
# full current bullpen -- core high-leverage arms + the depth/newer arms
# filling out the remaining spots. Matches Yahoo Sports' Aug 2026
# breakdown: Ferrer/Speier/Bazardo/Munoz (core) + Rucker/Wilcox/Simpson,
# plus Dominguez (newly acquired, added manually below since he's not
# in the scraped leaderboard data yet).
BULLPEN_NAMES = ["Ferrer", "Speier", "Bazardo", "Muñoz", "Munoz",
                 "Rucker", "Wilcox", "Simpson"]

# manually entered -- not yet in scraped leaderboard data (see DATA NOTE)
ACTUAL_PITCHER = {"name": "Seranthony Dominguez", "ERA": 4.10, "xwOBA_against": 0.327}
ACTUAL_BATTER = {"name": "Taylor Ward", "OPS": 0.729, "xwOBA": 0.346}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "deadline_comparison.parquet")


def _bullpen_pool(grades: dict) -> list:
    return [p for p in grades.get("pitchers", [])
            if any(n in p["name"] for n in BULLPEN_NAMES)]


def build_rows():
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    analysis = analyze_team(data)
    grades = grade_players(data)
    recs = generate_recommendations(data, analysis, grades)

    current_batters = grades.get("batters", [])       # full roster: starters + bench
    current_pitchers = _bullpen_pool(grades)           # full bullpen, live-scraped names
    batter_targets = recs["player_targets"]["batter_targets"]
    pitcher_targets = recs["player_targets"]["pitcher_targets"]

    rows = []

    # -- batting: current Mariners (starters + bench), tagged by role --
    for p in current_batters:
        if p.get("xwOBA"):
            pa = p.get("PA") or 0
            role = "Starter" if pa >= 200 else "Bench"
            rows.append({"category": "Batting", "group": "Current Mariners",
                         "role": role, "name": p["name"], "stat": "xwOBA",
                         "value": p["xwOBA"], "highlight": ""})

    # Ward -- genuinely a current Mariner now, but highlighted so he's
    # easy to pick out against his own new teammates
    already_has_ward = any("Ward" in p["name"] for p in current_batters)
    if not already_has_ward:
        rows.append({"category": "Batting", "group": "Current Mariners",
                     "role": "Starter", "name": ACTUAL_BATTER["name"],
                     "stat": "xwOBA", "value": ACTUAL_BATTER["xwOBA"],
                     "highlight": "Ward (new)"})

    for p in batter_targets:
        rows.append({"category": "Batting", "group": "Tool Targets",
                     "role": "", "name": p["name"], "stat": "xwOBA",
                     "value": p["xwOBA"], "highlight": ""})

    # -- pitching: full current bullpen (live-scraped + Dominguez manual) --
    for p in current_pitchers:
        if p.get("xwOBA_against"):
            rows.append({"category": "Pitching", "group": "Current Mariners",
                         "role": "Bullpen", "name": p["name"], "stat": "xwOBA_against",
                         "value": p["xwOBA_against"], "highlight": ""})

    already_has_dom = any("Dominguez" in p["name"] for p in current_pitchers)
    if not already_has_dom:
        rows.append({"category": "Pitching", "group": "Current Mariners",
                     "role": "Bullpen", "name": ACTUAL_PITCHER["name"],
                     "stat": "xwOBA_against", "value": ACTUAL_PITCHER["xwOBA_against"],
                     "highlight": "Dominguez (new)"})

    for p in pitcher_targets:
        rows.append({"category": "Pitching", "group": "Tool Targets",
                     "role": "", "name": p["name"], "stat": "xwOBA_against",
                     "value": p["xwOBA_against"], "highlight": ""})

    df = pd.DataFrame(rows)

    print(f"\nBatting rows: {len(df[df['category']=='Batting'])} "
          f"({len(df[(df['category']=='Batting') & (df['group']=='Current Mariners')])} current, "
          f"{len(df[(df['category']=='Batting') & (df['group']=='Tool Targets')])} targets)")
    print(f"Pitching rows: {len(df[df['category']=='Pitching'])} "
          f"({len(df[(df['category']=='Pitching') & (df['group']=='Current Mariners')])} current, "
          f"{len(df[(df['category']=='Pitching') & (df['group']=='Tool Targets')])} targets)")
    if "Simpson" not in " ".join(df[df["category"]=="Pitching"]["name"]):
        print("  [note] Josh Simpson not found in scraped data -- re-run "
              "with force_refresh=True if he was just called up (Aug 2)")

    return df


if __name__ == "__main__":
    df = build_rows()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print(df.to_string(index=False))