"""Mariners starters: when did the release point move?

Release side / height / extension for each starter's fastballs, start by start, across two seasons.
For each pitcher it finds the single date where the release point shifts most and says whether that
date is the season boundary (offseason change or tracking recalibration) or mid-season (a change he made).

    python release_timeline.py                  # 2025 and 2026
    python release_timeline.py --a 2024 --b 2025

Uses the pitch files already downloaded -- nothing new to fetch.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pitch_count_study as pc

pd.set_option("display.width", 220)
MIN_FB_PER_START = 8
MIN_SIDE = 5                      # starts on each side of a split
METRICS = {"rel_x": "release side", "rel_z": "release height", "ext": "extension"}


def start_table(pitches):
    d = pitches[pitches["is_starter"].astype(bool) & pitches["pitch_type"].isin(pc.FASTBALLS)].copy()
    for c in METRICS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["pitcher", "game_pk", "date"]).agg(n=("pitch_no", "size"), **{c: (c, "mean") for c in METRICS}).reset_index()
    g = g[g["n"] >= MIN_FB_PER_START].sort_values(["pitcher", "date"]).reset_index(drop=True)
    return g


def best_split(vals, dates, season_start_idx=None):
    """Index i (first start AFTER the shift) that maximises |mean(after) - mean(before)|, in units of the
    pooled standard deviation of the start-by-start values. Returns None if there are too few starts."""
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    v, dates = v[ok], np.asarray(dates)[ok]
    n = len(v)
    if n < 2 * MIN_SIDE:
        return None
    best = None
    for i in range(MIN_SIDE, n - MIN_SIDE + 1):
        b, a = v[:i], v[i:]
        sd = np.sqrt(((len(b) - 1) * b.var(ddof=1) + (len(a) - 1) * a.var(ddof=1)) / (n - 2))
        if sd == 0:
            continue
        z = (a.mean() - b.mean()) / (sd * np.sqrt(1 / len(a) + 1 / len(b)))
        if best is None or abs(z) > abs(best["z"]):
            best = {"i": i, "date": dates[i], "before": b.mean(), "after": a.mean(), "chg": a.mean() - b.mean(),
                    "z": z, "n_before": len(b), "n_after": len(a)}
    return best


def changepoints(starts, boundary_date):
    rows = []
    for p, g in starts.groupby("pitcher"):
        for c, label in METRICS.items():
            s = best_split(g[c].values, g["date"].values)
            if s is None:
                continue
            first_after = str(s["date"])
            near_boundary = abs(s["n_before"] - (g["date"] < boundary_date).sum()) <= 2
            rows.append({"pitcher": p, "metric": label, "col": c, "shift_date": first_after, "before": round(s["before"], 2),
                         "after": round(s["after"], 2), "chg_ft": round(s["chg"], 2), "chg_in": round(s["chg"] * 12, 1),
                         "strength": round(abs(s["z"]), 1), "when": "season boundary" if near_boundary else "mid-season",
                         "starts_before": s["n_before"], "starts_after": s["n_after"]})
    return pd.DataFrame(rows)


STRONG = 4.0                  # roughly: the shift is far bigger than start-to-start noise
BIG_FT = {"rel_x": 0.15, "rel_z": 0.15, "ext": 0.15}


def monthly(starts):
    s = starts.copy()
    s["month"] = s["date"].astype(str).str[:7]
    return s.groupby(["pitcher", "month"])[list(METRICS)].mean().round(2)


def run(a_season, b_season, export_dir=pc.EXPORTS, get=pc._get, pause=pc.PAUSE):
    frames = []
    for s in (a_season, b_season):
        print(f"--- {s} ---")
        _, pit = pc.collect(s, export_dir, get, pause=pause)
        if pit.empty or pd.to_numeric(pit["rel_x"], errors="coerce").isna().all():
            print("  No release-point data in the feed for this season.")
            return None
        frames.append(pit)
    starts = start_table(pd.concat(frames, ignore_index=True))
    boundary = f"{b_season}-01-01"
    cp = changepoints(starts, boundary)
    print("\n== Where each starter's release point moved most (fastballs only) ==")
    print("   strength = size of the shift relative to start-to-start wobble (4+ = clear, under 3 = could be noise)")
    for p, g in cp.groupby("pitcher"):
        print(f"\n{p}  ({int(g['starts_before'].iloc[0]) + int(g['starts_after'].iloc[0])} starts)")
        print(g[["metric", "shift_date", "before", "after", "chg_ft", "chg_in", "strength", "when"]].to_string(index=False))
    big = cp[(cp["strength"] >= STRONG) & (cp["chg_ft"].abs() >= cp["col"].map(BIG_FT))]
    print("\n== Clear shifts (strength 4+ and at least 0.15 ft = about 2 inches) ==")
    if big.empty:
        print("  none")
    else:
        print(big[["pitcher", "metric", "shift_date", "chg_in", "strength", "when"]].sort_values("shift_date").to_string(index=False))
        bd = big[big["when"] == "mid-season"]["shift_date"]
        print("\nSo: %d clear shifts, %d at the season boundary, %d mid-season." % (len(big), (big['when'] == 'season boundary').sum(), len(bd)))
        if len(big) >= 3:
            dates = pd.to_datetime(big["shift_date"])
            print("   The shift dates span %d days across %d pitchers. %s" % ((dates.max() - dates.min()).days, big['pitcher'].nunique(),
                  "Same-day shifts for several pitchers would point to a tracking change." if (dates.max() - dates.min()).days <= 10 else
                  "Different dates for different pitchers fits individual changes more than a single recalibration."))
    print("\n== Monthly averages (fastballs; feet) ==")
    print(monthly(starts).to_string())
    os.makedirs(export_dir, exist_ok=True)
    cp.to_csv(os.path.join(export_dir, f"release_changepoints_{a_season}_{b_season}.csv"), index=False)
    starts.to_csv(os.path.join(export_dir, f"release_by_start_{a_season}_{b_season}.csv"), index=False)
    print(f"\nWrote release_changepoints_{a_season}_{b_season}.csv and release_by_start_{a_season}_{b_season}.csv in {export_dir}")
    return cp, starts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=2025)
    ap.add_argument("--b", type=int, default=2026)
    a = ap.parse_args()
    try:
        run(a.a, a.b)
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}")
        sys.exit(1)