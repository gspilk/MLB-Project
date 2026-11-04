"""
bullpen_context.py
Two questions about the Mariners' bullpen that season-long ERA cannot answer:

  A. Did they shut games down?  Save conversion, blown saves and inherited runners scored,
     for all 30 teams (MLB Stats API team pitching totals), with the Mariners' rank.
  B. Who was actually available?  For each reliever in the play-by-play cache: batters faced
     in each half, first and last appearance, and every injured-list move the Mariners made
     for him (MLB Stats API transactions). A reliever with no batters after a date and an IL
     move near it was probably hurt, not ineffective.

Run pitch_count_study.py first (it builds the cache of plate appearances this reads).

Usage:  python bullpen_context.py
        python bullpen_context.py --split 2026-07-15 --min-pa 15
"""
import argparse
import datetime as dt
import os
import re
import unicodedata

import numpy as np
import pandas as pd

import mlb_api
import pitch_count_study as pcs
try:
    from bullpen_split import load, relievers
except ImportError:                      # the file may have been saved as bullpen_splits.py
    from bullpen_split import load, relievers

SEA = mlb_api.SEA_TEAM_ID
STAT_KEYS = {"saves": "SV", "saveOpportunities": "SVO", "blownSaves": "BS", "holds": "HLD",
             "inheritedRunners": "IR", "inheritedRunnersScored": "IRS"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


# ---------------------------------------------------------------- A. shutting games down
def team_save_stats(season=2026, get=None):
    """One row per team: saves, save opportunities, blown saves, holds, inherited runners."""
    get = get or (lambda: mlb_api.api_get("teams_stats", {"stats": "season", "group": "pitching",
                                                          "season": season, "sportIds": 1, "gameType": "R"}))
    resp = get()
    rows = []
    for block in resp.get("stats", []):
        for s in block.get("splits", []):
            st, team = s.get("stat", {}), s.get("team") or {}
            row = {"team_id": team.get("id"), "team": team.get("name")}
            for k, col in STAT_KEYS.items():
                row[col] = pd.to_numeric(st.get(k), errors="coerce")
            rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=["team_id"])
    df["team_id"] = df["team_id"].astype(int)
    df["save_pct"] = (100 * df["SV"] / df["SVO"]).round(1)
    df["irs_pct"] = (100 * df["IRS"] / df["IR"]).round(1)
    df["save_rank"] = df["save_pct"].rank(ascending=False, method="min")
    df["irs_rank"] = df["irs_pct"].rank(ascending=True, method="min")
    return df


def print_team_stats(df):
    if df.empty or df["SVO"].isna().all():
        print("  The API did not return save / inherited-runner fields, so section A is skipped.")
        return
    sea = df[df["team_id"] == SEA]
    if sea.empty:
        return
    r = sea.iloc[0]
    n = len(df)
    print(f"  Save conversion: {r.SV:.0f} of {r.SVO:.0f} = {r.save_pct}%  (league avg {df['save_pct'].mean():.1f}%, "
          f"rank {r.save_rank:.0f}/{n}, higher is better)")
    print(f"  Blown saves:     {r.BS:.0f}  (league avg {df['BS'].mean():.1f}, rank "
          f"{df['BS'].rank(method='min')[sea.index[0]]:.0f}/{n} fewest)")
    print(f"  Inherited runners scored: {r.IRS:.0f} of {r.IR:.0f} = {r.irs_pct}%  "
          f"(league avg {df['irs_pct'].mean():.1f}%, rank {r.irs_rank:.0f}/{n}, lower is better)")
    print(f"  Holds:           {r.HLD:.0f}  (league avg {df['HLD'].mean():.1f})")


# ---------------------------------------------------------------- B. who was available
def il_moves(start, end, get=None):
    """Mariners injured-list transactions in the window: date and description."""
    get = get or (lambda: mlb_api.api_get("transactions", {"startDate": start.strftime("%m/%d/%Y"),
                                                           "endDate": end.strftime("%m/%d/%Y"),
                                                           "teamId": SEA, "sportId": 1}))
    resp = get()
    out = []
    for t in resp.get("transactions", []):
        text = t.get("description") or ""
        if re.search(r"injured list|\bIL\b", text, re.I):
            out.append({"date": (t.get("date") or "")[:10], "text": text})
    return pd.DataFrame(out, columns=["date", "text"]).sort_values("date") if out else pd.DataFrame(columns=["date", "text"])


def availability(rel, split, il, min_pa):
    cut = pd.Timestamp(split)
    last_day = rel["date"].max()
    rows = []
    for name, g in rel.groupby("pitcher"):
        if len(g) < min_pa:
            continue
        first, second = g[g["date"] < cut], g[g["date"] >= cut]
        key = _norm(name)
        moves = il[il["text"].map(lambda t: key in _norm(t))] if len(il) else il
        gap = (last_day - g["date"].max()).days
        rows.append({"pitcher": name, "pa_first": len(first), "pa_second": len(second),
                     "first_seen": g["date"].min().date(), "last_seen": g["date"].max().date(),
                     "days_before_season_end": gap, "il_moves": len(moves),
                     "il_dates": ", ".join(moves["date"].tolist()) if len(moves) else ""})
    return pd.DataFrame(rows).sort_values("pa_first", ascending=False)


def run(season=2026, split="2026-07-15", min_pa=15, get_team=None, get_tx=None, export_dir=pcs.EXPORTS):
    print("== A. Did the bullpen shut games down? (all 30 teams, MLB Stats API) ==")
    try:
        team = team_save_stats(season, get_team)
        print_team_stats(team)
        team.to_csv(os.path.join(export_dir, f"team_save_stats_{season}.csv"), index=False)
    except Exception as e:
        print(f"  [warn] team save stats unavailable ({type(e).__name__}: {e})")

    print("\n== B. Who was available? (relievers with batters faced in the play-by-play cache) ==")
    rel = relievers(load(season, export_dir))
    try:
        il = il_moves(dt.date(season, 2, 15), dt.date(season, 10, 5), get_tx)
    except Exception as e:
        print(f"  [warn] injured-list transactions unavailable ({type(e).__name__}: {e}); IL columns will be blank")
        il = pd.DataFrame(columns=["date", "text"])
    tbl = availability(rel, split, il, min_pa)
    print(tbl.to_string(index=False))
    print("\n  days_before_season_end: how long before the last game he last faced a batter.")
    print("  A large number, plus an IL move, usually means hurt; a large number and no IL move can mean "
          "demoted, traded or released.")
    tbl.to_csv(os.path.join(export_dir, f"bullpen_availability_{season}.csv"), index=False)
    if len(il):
        print("\n== Mariners injured-list moves in the window (check these against the table) ==")
        print(il.to_string(index=False, max_colwidth=110))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--split", default="2026-07-15")
    ap.add_argument("--min-pa", type=int, default=15)
    a = ap.parse_args()
    run(a.season, a.split, a.min_pa)