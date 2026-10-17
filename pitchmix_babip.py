"""Mariners starters, 2025 vs 2026: what they throw, and how often balls in play fell for hits.

Uses the cache from pitch_count_study.py / compare_seasons.py (nothing new to download).

    python pitch_mix_babip.py                  # 2025 vs 2026
    python pitch_mix_babip.py --a 2024 --b 2025
"""
import argparse, os, sys
import pandas as pd
import pitch_count_study as pc

pd.set_option("display.width", 200)
MIN_PITCHES = 150          # a pitch type with fewer pitches than this in a season is shown but not compared
MIN_BIP = 100
NAMES = {"FF": "4-seam", "SI": "sinker", "FT": "2-seam", "FC": "cutter", "SL": "slider", "ST": "sweeper",
         "SV": "slurve", "CU": "curve", "KC": "knuckle-curve", "CH": "changeup", "FS": "splitter", "FA": "fastball"}


def pitch_mix(pitches):
    """Per starter and pitch type: pitches, usage %, average mph."""
    d = pitches[pitches["is_starter"].astype(bool)].copy()
    d["speed"] = pd.to_numeric(d["speed"], errors="coerce")
    d = d.dropna(subset=["pitch_type"])
    tot = d.groupby("pitcher").size().rename("total")
    g = d.groupby(["pitcher", "pitch_type"]).agg(n=("pitch_no", "size"), mph=("speed", "mean")).reset_index()
    g = g.merge(tot, on="pitcher")
    g["usage"] = (100 * g["n"] / g["total"]).round(1)
    g["mph"] = g["mph"].round(1)
    return g


def compare_mix(a, b, la, lb):
    m = a.merge(b, on=["pitcher", "pitch_type"], how="outer", suffixes=(f"_{la}", f"_{lb}"))
    for c in (f"n_{la}", f"n_{lb}"):
        m[c] = m[c].fillna(0).astype(int)
    for c in (f"usage_{la}", f"usage_{lb}"):
        m[c] = m[c].fillna(0.0)
    m["usage_chg"] = (m[f"usage_{lb}"] - m[f"usage_{la}"]).round(1)
    m["mph_chg"] = (m[f"mph_{lb}"] - m[f"mph_{la}"]).round(1)
    m["pitch"] = m["pitch_type"].map(NAMES).fillna(m["pitch_type"])
    keep = (m[f"n_{la}"] >= MIN_PITCHES) | (m[f"n_{lb}"] >= MIN_PITCHES)
    return m[keep].sort_values(["pitcher", f"usage_{lb}"], ascending=[True, False]).reset_index(drop=True)


def babip_table(pas):
    """BABIP against and its pieces, per starter plus ALL STARTERS."""
    st = pas[pas["is_starter"].astype(bool)]
    rows = []
    for name, g in [(p, x) for p, x in st.groupby("pitcher")] + [("ALL STARTERS", st)]:
        ab, k = int(g["is_ab"].sum()), int(g["is_k"].sum())
        hr = int((g["hit_bases"] == 4).sum())
        h = int((g["hit_bases"] > 0).sum())
        sf = int(g["is_sac_fly"].sum())
        bip = ab - k - hr + sf
        ls = g["launch_speed"].dropna()
        rows.append({"pitcher": name, "bip": bip, "hits_on_bip": h - hr,
                     "babip": round((h - hr) / bip, 3) if bip else float("nan"),
                     "hr": hr, "hr_per_100_pa": round(100 * hr / len(g), 2),
                     "hard_hit_pct": round(100 * (ls >= pc.HARD_HIT_MPH).mean(), 1) if len(ls) else float("nan")})
    return pd.DataFrame(rows).set_index("pitcher")


def run(a_season, b_season, export_dir=pc.EXPORTS, get=pc._get, pause=pc.PAUSE):
    la, lb = str(a_season), str(b_season)
    pa, pi = {}, {}
    for s in (a_season, b_season):
        pa[s], pi[s] = pc.collect(s, export_dir, get, pause=pause, verbose=False)
    mix = compare_mix(pitch_mix(pi[a_season]), pitch_mix(pi[b_season]), la, lb)
    print(f"== Pitch mix, {a_season} -> {b_season} (usage = % of his pitches; types under {MIN_PITCHES} pitches both years are hidden) ==")
    for p, g in mix.groupby("pitcher"):
        print(f"\n{p}")
        print(g[["pitch", f"n_{la}", f"usage_{la}", f"usage_{lb}", "usage_chg", f"mph_{la}", f"mph_{lb}", "mph_chg"]]
              .rename(columns={f"n_{la}": f"pitches {la}", f"usage_{la}": f"use% {la}", f"usage_{lb}": f"use% {lb}",
                               f"mph_{la}": f"mph {la}", f"mph_{lb}": f"mph {lb}"}).to_string(index=False))
    ba, bb = babip_table(pa[a_season]), babip_table(pa[b_season])
    both = ba.join(bb, lsuffix=f"_{la}", rsuffix=f"_{lb}", how="inner")
    both["babip_chg"] = (both[f"babip_{lb}"] - both[f"babip_{la}"]).round(3)
    both["hard_hit_chg"] = (both[f"hard_hit_pct_{lb}"] - both[f"hard_hit_pct_{la}"]).round(1)
    print(f"\n== Balls in play against (BABIP = hits on balls in play, home runs excluded, / balls in play) ==")
    print(both[[f"bip_{la}", f"babip_{la}", f"bip_{lb}", f"babip_{lb}", "babip_chg",
                f"hard_hit_pct_{la}", f"hard_hit_pct_{lb}", "hard_hit_chg",
                f"hr_per_100_pa_{la}", f"hr_per_100_pa_{lb}"]].to_string())
    print(f"\nRead it like this: BABIP up while hard-hit% is flat or down = hits fell in without harder contact"
          f" (defense, positioning or luck). BABIP up with hard-hit% up = he was hit harder. Samples under {MIN_BIP} balls in play are noisy.")
    os.makedirs(export_dir, exist_ok=True)
    mix.to_csv(os.path.join(export_dir, f"pitch_mix_{a_season}_{b_season}.csv"), index=False)
    both.to_csv(os.path.join(export_dir, f"babip_{a_season}_{b_season}.csv"))
    print(f"\nWrote pitch_mix_{a_season}_{b_season}.csv and babip_{a_season}_{b_season}.csv in {export_dir}")
    return mix, both


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