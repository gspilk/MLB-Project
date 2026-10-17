"""Mariners starters, 2025 vs 2026: which pitches get swings and misses?

Per pitcher and pitch type: whiff% (misses per swing), called+swinging strike% (CSW), zone%, chase%.
Per pitcher: first-pitch strike%, zone%, chase%, whiff%.

The first run re-downloads the pitches for both seasons (about 3 minutes); the plate-appearance files are kept.

    python whiff_by_pitch.py                  # 2025 vs 2026
    python whiff_by_pitch.py --a 2024 --b 2025
"""
import argparse, os, sys
import pandas as pd
import pitch_count_study as pc

pd.set_option("display.width", 200)
MISS = {"S", "W", "M", "T"}                  # swinging strike, blocked, missed bunt, foul tip
FOUL = {"F", "L"}
INPLAY = {"X", "D", "E"}
CALLED = {"C"}
SWING = MISS | FOUL | INPLAY
STRIKE = SWING | CALLED
MIN_PITCHES = 150
MIN_SWINGS = 40
NAMES = {"FF": "4-seam", "SI": "sinker", "FT": "2-seam", "FC": "cutter", "SL": "slider", "ST": "sweeper",
         "SV": "slurve", "CU": "curve", "KC": "knuckle-curve", "CH": "changeup", "FS": "splitter", "FA": "fastball"}


def _pct(num, den, min_den=1):
    return round(100 * num / den, 1) if den >= min_den else float("nan")


def _metrics(g):
    call = g["call"]
    z = pd.to_numeric(g["zone"], errors="coerce")
    swings, miss = call.isin(SWING), call.isin(MISS)
    inz, outz = z.between(1, 9), z >= 11
    two = g["strikes"] == 2
    first = g["pa_pitch"] == 1
    return {
        "pitches": len(g),
        "whiff_pct": _pct(miss.sum(), swings.sum(), MIN_SWINGS),
        "csw_pct": _pct((miss | call.isin(CALLED)).sum(), len(g), MIN_PITCHES // 3),
        "zone_pct": _pct(inz.sum(), z.notna().sum(), MIN_PITCHES // 3),
        "chase_pct": _pct((swings & outz).sum(), outz.sum(), 30),
        "whiff_2strk_pct": _pct((miss & two).sum(), (swings & two).sum(), 25),
        "first_pitch_strike_pct": _pct(call[first].isin(STRIKE).sum(), first.sum(), 50),
    }


def by_pitch(pitches):
    d = pitches[pitches["is_starter"].astype(bool)].dropna(subset=["pitch_type", "call"])
    rows = []
    tot = d.groupby("pitcher").size()
    for (p, t), g in d.groupby(["pitcher", "pitch_type"]):
        r = _metrics(g)
        r.update({"pitcher": p, "pitch_type": t, "pitch": NAMES.get(t, t), "usage": _pct(len(g), tot[p])})
        rows.append(r)
    return pd.DataFrame(rows)


def by_pitcher(pitches):
    d = pitches[pitches["is_starter"].astype(bool)].dropna(subset=["call"])
    rows = [{"pitcher": p, **_metrics(g)} for p, g in d.groupby("pitcher")]
    rows.append({"pitcher": "ALL STARTERS", **_metrics(d)})
    return pd.DataFrame(rows).set_index("pitcher")


def compare(a, b, la, lb, keys, cols):
    m = a.merge(b, on=keys, how="outer", suffixes=(f"_{la}", f"_{lb}"))
    for c in cols:
        m[f"{c}_chg"] = (m[f"{c}_{lb}"] - m[f"{c}_{la}"]).round(1)
    return m


def run(a_season, b_season, export_dir=pc.EXPORTS, get=pc._get, pause=pc.PAUSE):
    la, lb = str(a_season), str(b_season)
    pit = {}
    for s in (a_season, b_season):
        print(f"--- {s} ---")
        _, pit[s] = pc.collect(s, export_dir, get, pause=pause)
        if pit[s].empty or pit[s]["call"].isna().all():
            print("  No pitch outcomes found in the feed -- nothing to compare for this season.")
            return None
    pp = compare(by_pitcher(pit[a_season]).reset_index(), by_pitcher(pit[b_season]).reset_index(), la, lb,
                 ["pitcher"], ["whiff_pct", "csw_pct", "zone_pct", "chase_pct", "first_pitch_strike_pct"])
    print("\n== Pitcher level: strike-throwing and swing-and-miss (chg = %s minus %s, points) ==" % (b_season, a_season))
    show = ["pitcher"]
    for c in ("first_pitch_strike_pct", "zone_pct", "chase_pct", "whiff_pct", "csw_pct"):
        show += [f"{c}_{la}", f"{c}_{lb}", f"{c}_chg"]
    print(pp[show].to_string(index=False))
    pt = compare(by_pitch(pit[a_season]), by_pitch(pit[b_season]), la, lb, ["pitcher", "pitch_type", "pitch"],
                 ["usage", "whiff_pct", "csw_pct", "zone_pct", "chase_pct"])
    keep = (pt[f"pitches_{la}"].fillna(0) >= MIN_PITCHES) | (pt[f"pitches_{lb}"].fillna(0) >= MIN_PITCHES)
    pt = pt[keep].sort_values(["pitcher", f"usage_{lb}"], ascending=[True, False])
    print(f"\n== Each pitch: usage, whiff% (misses per swing), CSW% (called + swinging strikes per pitch) ==")
    print(f"   (blank = fewer than {MIN_SWINGS} swings or {MIN_PITCHES // 3} pitches, too few to read)")
    for p, g in pt.groupby("pitcher"):
        print(f"\n{p}")
        print(g[["pitch", f"usage_{la}", f"usage_{lb}", f"whiff_pct_{la}", f"whiff_pct_{lb}", "whiff_pct_chg",
                 f"csw_pct_{la}", f"csw_pct_{lb}", f"zone_pct_{la}", f"zone_pct_{lb}",
                 f"chase_pct_{la}", f"chase_pct_{lb}"]].to_string(index=False))
    os.makedirs(export_dir, exist_ok=True)
    pp.to_csv(os.path.join(export_dir, f"whiff_pitcher_{a_season}_{b_season}.csv"), index=False)
    pt.to_csv(os.path.join(export_dir, f"whiff_by_pitch_{a_season}_{b_season}.csv"), index=False)
    print(f"\nWrote whiff_pitcher_{a_season}_{b_season}.csv and whiff_by_pitch_{a_season}_{b_season}.csv in {export_dir}")
    return pp, pt


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