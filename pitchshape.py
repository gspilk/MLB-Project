"""Mariners starters, 2025 vs 2026: did the pitches change, or did hitters adjust?

For each pitcher and pitch type: speed, spin, movement, extension, release point and where it was thrown,
next to the whiff rate. The last table lists every pitch whose whiff rate fell 4+ points and what changed with it.

    python pitch_shape.py                  # 2025 vs 2026
    python pitch_shape.py --a 2024 --b 2025

First run re-downloads the pitch data for both seasons (about 3 minutes).
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pitch_count_study as pc
import whiff_by_pitch as wb

pd.set_option("display.width", 220)
MIN_PITCHES = 150
WHIFF_DROP = 4.0                       # points of whiff% that count as "lost swing-and-miss"
# "something changed" thresholds
T = {"mph": 1.0, "spin": 100.0, "ivb": 1.0, "hb": 1.0, "ext": 0.2, "rel_z": 0.2, "rel_x": 0.2, "heart_pct": 4.0, "pz": 0.1, "px": 0.1}
LABEL = {"mph": "velocity", "spin": "spin", "ivb": "vertical break", "hb": "horizontal break", "ext": "extension",
         "rel_z": "release height", "rel_x": "release side", "heart_pct": "pitches down the middle", "pz": "height of location",
         "px": "side-to-side location"}
UNIT = {"mph": "mph", "spin": "rpm", "ivb": "in", "hb": "in", "ext": "ft", "rel_z": "ft", "rel_x": "ft", "heart_pct": "pts", "pz": "ft", "px": "ft"}
SHAPE, RELEASE, LOCATION = ["mph", "spin", "ivb", "hb"], ["ext", "rel_z", "rel_x"], ["heart_pct", "pz", "px"]
DEC = {"mph": 1, "spin": 0, "ivb": 1, "hb": 1, "ext": 2, "rel_z": 2, "rel_x": 2, "heart_pct": 1, "pz": 2, "px": 2}


def is_heart(d):
    top = pd.to_numeric(d["sz_top"], errors="coerce").fillna(3.4)
    bot = pd.to_numeric(d["sz_bot"], errors="coerce").fillna(1.6)
    px, pz = pd.to_numeric(d["px"], errors="coerce"), pd.to_numeric(d["pz"], errors="coerce")
    ok = px.notna() & pz.notna()
    inner = 0.708 * (2 / 3)                                       # inner two-thirds of the 17-inch plate, in feet
    pad = (top - bot) / 6                                         # inner two-thirds of the zone's height
    h = (px.abs() <= inner) & (pz >= bot + pad) & (pz <= top - pad)
    return h, ok


def profile(pitches):
    """One row per starter and pitch type with shape, release, location and whiff numbers."""
    d = pitches[pitches["is_starter"].astype(bool)].dropna(subset=["pitch_type"]).copy()
    for c in ("speed", "spin", "ivb", "hb", "ext", "px", "pz", "rel_x", "rel_z"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["heart"], d["loc_ok"] = is_heart(d)
    rows = []
    for (p, t), g in d.groupby(["pitcher", "pitch_type"]):
        w = wb._metrics(g.dropna(subset=["call"])) if g["call"].notna().any() else {}
        loc = g[g["loc_ok"]]
        rows.append({"pitcher": p, "pitch_type": t, "pitch": wb.NAMES.get(t, t), "n": len(g),
                     "mph": g["speed"].mean(), "spin": g["spin"].mean(), "ivb": g["ivb"].mean(), "hb": g["hb"].mean(),
                     "ext": g["ext"].mean(), "rel_z": g["rel_z"].mean(), "rel_x": g["rel_x"].mean(),
                     "px": loc["px"].mean(), "pz": loc["pz"].mean(),
                     "heart_pct": 100 * loc["heart"].mean() if len(loc) else np.nan,
                     "whiff_pct": w.get("whiff_pct", np.nan), "csw_pct": w.get("csw_pct", np.nan)})
    return pd.DataFrame(rows)


def compare(a, b, la, lb):
    m = a.merge(b, on=["pitcher", "pitch_type", "pitch"], how="inner", suffixes=(f"_{la}", f"_{lb}"))
    m = m[(m[f"n_{la}"] >= MIN_PITCHES) & (m[f"n_{lb}"] >= MIN_PITCHES)].copy()
    for c in list(T) + ["whiff_pct", "csw_pct"]:
        m[f"{c}_chg"] = m[f"{c}_{lb}"] - m[f"{c}_{la}"]
    return m.reset_index(drop=True)


def what_changed(row):
    """Plain list of the changes that cleared their thresholds, grouped as shape / release / location."""
    out = {"shape": [], "release": [], "location": []}
    for grp, cols in (("shape", SHAPE), ("release", RELEASE), ("location", LOCATION)):
        for c in cols:
            v = row.get(f"{c}_chg")
            if v == v and abs(v) >= T[c]:
                out[grp].append(f"{LABEL[c]} {v:+.{DEC[c]}f} {UNIT[c]}")
    return out


def verdict(row):
    ch = what_changed(row)
    bits = []
    if ch["shape"]: bits.append("SHAPE: " + ", ".join(ch["shape"]))
    if ch["release"]: bits.append("RELEASE: " + ", ".join(ch["release"]))
    if ch["location"]: bits.append("LOCATION: " + ", ".join(ch["location"]))
    return "; ".join(bits) if bits else "nothing measurable changed -- hitters adjusted or it is noise"


def run(a_season, b_season, export_dir=pc.EXPORTS, get=pc._get, pause=pc.PAUSE):
    la, lb = str(a_season), str(b_season)
    pit = {}
    for s in (a_season, b_season):
        print(f"--- {s} ---")
        _, pit[s] = pc.collect(s, export_dir, get, pause=pause)
        if pit[s].empty or pd.to_numeric(pit[s]["spin"], errors="coerce").isna().all():
            print("  No spin/movement found in the feed for this season -- cannot compare shapes.")
            return None
    m = compare(profile(pit[a_season]), profile(pit[b_season]), la, lb)
    if m.empty:
        print("No pitch type was thrown 150+ times in both seasons.")
        return None
    print(f"\n== Every pitch, {a_season} -> {b_season} (change shown; velocity mph, spin rpm, breaks in inches, release/extension/location in feet) ==")
    cols = ["pitch", f"n_{la}", f"n_{lb}", "mph_chg", "spin_chg", "ivb_chg", "hb_chg", "ext_chg", "rel_z_chg", "rel_x_chg",
            "heart_pct_chg", "pz_chg", "px_chg", f"whiff_pct_{la}", f"whiff_pct_{lb}", "whiff_pct_chg"]
    for p, g in m.groupby("pitcher"):
        print(f"\n{p}")
        print(g[cols].round(2).to_string(index=False))
    lost = m[m["whiff_pct_chg"] <= -WHIFF_DROP].sort_values("whiff_pct_chg")
    print(f"\n== Pitches that lost {WHIFF_DROP:.0f}+ points of whiff rate, and what changed with them ==")
    if lost.empty:
        print("  none")
    for _, r in lost.iterrows():
        print(f"\n{r['pitcher']} {r['pitch']}: whiff {r[f'whiff_pct_{la}']:.1f}% -> {r[f'whiff_pct_{lb}']:.1f}% ({r['whiff_pct_chg']:+.1f})")
        print(f"   {verdict(r)}")
    gained = m[m["whiff_pct_chg"] >= WHIFF_DROP].sort_values("whiff_pct_chg", ascending=False)
    print(f"\n== Pitches that gained {WHIFF_DROP:.0f}+ points of whiff rate, for comparison ==")
    for _, r in gained.iterrows():
        print(f"\n{r['pitcher']} {r['pitch']}: whiff {r[f'whiff_pct_{la}']:.1f}% -> {r[f'whiff_pct_{lb}']:.1f}% ({r['whiff_pct_chg']:+.1f})")
        print(f"   {verdict(r)}")
    print("\nHow to read it: SHAPE = the pitch itself moves or is thrown differently; RELEASE = arm slot or extension moved;"
          "\nLOCATION = he is throwing it to different spots. 'Nothing measurable changed' points to hitters adjusting (or noise)."
          "\nWhiff rates rest on a modest number of swings per pitch, so treat anything near the 4-point line as a hint.")
    os.makedirs(export_dir, exist_ok=True)
    m.to_csv(os.path.join(export_dir, f"pitch_shape_{a_season}_{b_season}.csv"), index=False)
    print(f"\nWrote pitch_shape_{a_season}_{b_season}.csv in {export_dir}")
    return m


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