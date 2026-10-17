"""Mariners starters, this season vs last: what changed?

Uses the same cached play-by-play as pitch_count_study.py (downloads a season the first time, about 1-2 min).

    python compare_seasons.py                  # 2025 vs 2026
    python compare_seasons.py --a 2024 --b 2025
"""
import argparse, os, sys
import pandas as pd
import pitch_count_study as pc

pd.set_option("display.width", 200)


def season_table(pas, pitches):
    """One row per starter (plus ALL STARTERS): workload, results, velocity."""
    st = pas[pas["is_starter"].astype(bool)].copy()
    fb = pc._fb(pitches[pitches["is_starter"].astype(bool)]) if len(pitches) else pd.DataFrame()
    pps = pc.pitches_per_start(pas)
    rows = []
    groups = [(p, g) for p, g in st.groupby("pitcher")] + [("ALL STARTERS", st)]
    for name, g in groups:
        r = pc._rates(g)
        starts = g["game_pk"].nunique()
        f = fb[fb["pitcher"] == name] if (len(fb) and name != "ALL STARTERS") else fb
        early = f[f["pitch_no"] <= 25]["speed"] if len(f) else pd.Series(dtype=float)
        tto1 = pc._rates(g[g["tto"] == 1])["ops"]
        tto3 = pc._rates(g[g["tto"] >= 3])
        rows.append({
            "pitcher": name, "starts": starts,
            "pitches_per_start": round(pps.loc[name, "avg_pitches"], 1) if name in pps.index else
                                 round((g.assign(e=g.pitches_before + g.pitches_in_pa).groupby(["pitcher", "game_pk"]).e.max()).mean(), 1),
            "pa": r["pa"], "ops": round(r["ops"], 3), "k_pct": round(r["k_pct"], 1), "bb_pct": round(r["bb_pct"], 1),
            "hard_hit_pct": round(r["hard_hit_pct"], 1),
            "runs_per_27": round(g["runs"].sum() / len(g) * 27, 2),
            "fb_avg": round(f["speed"].mean(), 1) if len(f) else float("nan"),
            "fb_first25": round(early.mean(), 1) if len(early) else float("nan"),
            "ops_1st_time": round(tto1, 3), "ops_3rd_time": round(tto3["ops"], 3) if tto3["pa"] >= 30 else float("nan"),
        })
    return pd.DataFrame(rows).set_index("pitcher")


def compare(a, b, label_a, label_b):
    cols = ["starts", "pitches_per_start", "ops", "k_pct", "bb_pct", "hard_hit_pct", "runs_per_27", "fb_avg", "ops_3rd_time"]
    both = a.index.intersection(b.index)
    out = []
    for p in both:
        row = {"pitcher": p}
        for c in cols:
            row[f"{c} {label_a}"] = a.loc[p, c]
            row[f"{c} {label_b}"] = b.loc[p, c]
            row[f"{c} chg"] = round(b.loc[p, c] - a.loc[p, c], 3 if c.startswith("ops") else 1)
        out.append(row)
    return pd.DataFrame(out).set_index("pitcher")


def run(a_season, b_season, export_dir=pc.EXPORTS, get=pc._get, pause=pc.PAUSE):
    data = {}
    for s in (a_season, b_season):
        print(f"--- {s} ---")
        pas, pit = pc.collect(s, export_dir, get, pause=pause)
        data[s] = (pas, pit)
    tabs = {s: season_table(*data[s]) for s in data}
    la, lb = str(a_season), str(b_season)
    for s in (a_season, b_season):
        print(f"\n== {s}: every starter (ops/k/bb/hard-hit are what opponents did against him) ==")
        print(tabs[s].to_string())
    print(f"\n== Same starters, {a_season} -> {b_season} (chg = {b_season} minus {a_season}; for ops/bb/hard-hit/runs a rise is bad) ==")
    cmp_ = compare(tabs[a_season], tabs[b_season], la, lb)
    for p, r in cmp_.iterrows():
        print(f"\n{p}")
        for c in ["starts", "pitches_per_start", "ops", "k_pct", "bb_pct", "hard_hit_pct", "runs_per_27", "fb_avg", "ops_3rd_time"]:
            print(f"  {c:<18} {r[f'{c} {la}']:>8}  ->  {r[f'{c} {lb}']:>8}   ({r[f'{c} chg']:+})")
    gone = [p for p in tabs[a_season].index if p not in tabs[b_season].index]
    new = [p for p in tabs[b_season].index if p not in tabs[a_season].index]
    print(f"\nStarted in {a_season} but not {b_season}: {', '.join(gone) or 'none'}")
    print(f"Started in {b_season} but not {a_season}: {', '.join(new) or 'none'}")
    # by inning, team level
    ia = pc.by_inning(data[a_season][0], min_pa=1)[0].query("pitcher == 'ALL STARTERS'")[["inning", "ops", "ra9"]]
    ib = pc.by_inning(data[b_season][0], min_pa=1)[0].query("pitcher == 'ALL STARTERS'")[["inning", "ops", "ra9"]]
    m = ia.merge(ib, on="inning", suffixes=(f" {la}", f" {lb}"))
    print(f"\n== All starters by inning ==")
    print(m.to_string(index=False))
    ta = pc.by_times_through(data[a_season][0]).set_index("trip")["ops"]
    tb = pc.by_times_through(data[b_season][0]).set_index("trip")["ops"]
    print("\n== All starters by times through the order (OPS against) ==")
    print(pd.DataFrame({la: ta, lb: tb}).to_string())
    os.makedirs(export_dir, exist_ok=True)
    cmp_.to_csv(os.path.join(export_dir, f"starter_compare_{a_season}_{b_season}.csv"))
    print(f"\nWrote starter_compare_{a_season}_{b_season}.csv in {export_dir}")
    return tabs, cmp_


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