"""
bullpen_split.py
Did the Mariners' bullpen get worse in the second half? Uses the plate appearances that
pitch_count_study.py already cached (data/exports/pitch_count_pas_2026.csv), so it makes no new
API calls. Relievers are every Mariners pitcher who was not the starter that game.

Shows OPS against, strikeout and walk rates, and hard-hit rate by month, then first half vs
second half (split at the All-Star break, July 15), then the relievers with the most batters
faced. A bootstrap over games gives a rough range for the first-half-to-second-half change, so
you can tell a real drop from noise. Runs allowed are not shown: the cache records runs scoring
on each play, not which pitcher is charged with them.

Usage:  python bullpen_split.py             (run pitch_count_study.py first)
        python bullpen_split.py --split 2026-07-15 --min-pa 40
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import pitch_count_study as pcs


def load(season=2026, export_dir=pcs.EXPORTS):
    path = os.path.join(export_dir, pcs.fname(pcs.PA_FILE, season))
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run python pitch_count_study.py first.")
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"])
    for c in ["is_starter", "is_ab", "is_walk", "is_hbp", "is_sac_fly", "is_k"]:
        d[c] = d[c].astype(bool)
    return d


def relievers(pas):
    return pas[~pas["is_starter"]].copy()


def by_month(rel):
    rows = []
    for m, g in rel.groupby(rel["date"].dt.strftime("%Y-%m")):
        rows.append({"month": m, "games": g["game_pk"].nunique(), **pcs._rates(g)})
    return pd.DataFrame(rows).round(3)


def halves(rel, split):
    cut = pd.Timestamp(split)
    first, second = rel[rel["date"] < cut], rel[rel["date"] >= cut]
    out = pd.DataFrame([{"half": "first", "games": first["game_pk"].nunique(), **pcs._rates(first)},
                        {"half": "second", "games": second["game_pk"].nunique(), **pcs._rates(second)}])
    return out.round(3), first, second


def bootstrap_change(first, second, reps=2000, seed=7):
    """Resample whole games within each half; return (change in OPS, 5th, 95th percentile)."""
    rng = np.random.default_rng(seed)
    gf = {k: g for k, g in first.groupby("game_pk")}
    gs = {k: g for k, g in second.groupby("game_pk")}
    kf, ks = list(gf), list(gs)
    if not kf or not ks:
        return None
    diffs = []
    for _ in range(reps):
        a = pd.concat([gf[k] for k in rng.choice(kf, len(kf))])
        b = pd.concat([gs[k] for k in rng.choice(ks, len(ks))])
        diffs.append(pcs._rates(b)["ops"] - pcs._rates(a)["ops"])
    diffs = np.array(diffs)
    point = pcs._rates(second)["ops"] - pcs._rates(first)["ops"]
    return point, float(np.nanpercentile(diffs, 5)), float(np.nanpercentile(diffs, 95))


def by_pitcher(first, second, min_pa):
    rows = []
    for pid, g in pd.concat([first, second]).groupby("pitcher"):
        f, s = first[first["pitcher"] == pid], second[second["pitcher"] == pid]
        if len(g) < min_pa:
            continue
        rf, rs = pcs._rates(f), pcs._rates(s)
        rows.append({"pitcher": pid, "pa_first": len(f), "ops_first": rf["ops"],
                     "pa_second": len(s), "ops_second": rs["ops"],
                     "change": (rs["ops"] - rf["ops"]) if len(f) and len(s) else np.nan})
    return pd.DataFrame(rows).sort_values("pa_first", ascending=False).round(3)


def run(season=2026, split="2026-07-15", min_pa=40, export_dir=pcs.EXPORTS):
    rel = relievers(load(season, export_dir))
    print(f"{len(rel)} plate appearances against Mariners relievers, {rel['game_pk'].nunique()} games\n")
    print("== By month ==")
    print(by_month(rel).to_string(index=False))
    h, first, second = halves(rel, split)
    print(f"\n== First half vs second half (split {split}) ==")
    print(h.to_string(index=False))
    bc = bootstrap_change(first, second)
    if bc:
        print(f"\nChange in OPS against, second half minus first: {bc[0]:+.3f} "
              f"(rough 90% range {bc[1]:+.3f} to {bc[2]:+.3f}, resampling games)")
        if bc[1] > 0:
            print("  The whole range is above zero: the second-half drop is probably real, not noise.")
        elif bc[2] < 0:
            print("  The whole range is below zero: the bullpen was better in the second half.")
        else:
            print("  The range includes zero: with this many batters, the change could be noise.")
    print(f"\n== Relievers with {min_pa}+ batters faced ==")
    print(by_pitcher(first, second, min_pa).to_string(index=False))
    h.to_csv(os.path.join(export_dir, f"bullpen_halves_{season}.csv"), index=False)
    print(f"\nWrote bullpen_halves_{season}.csv in {export_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--split", default="2026-07-15")
    ap.add_argument("--min-pa", type=int, default=40)
    a = ap.parse_args()
    run(a.season, a.split, a.min_pa)