"""
project_2027.py
Multi-year, recency-weighted 2027 projections with shrinkage toward the league
average, plus a backtest against two naive baselines.

WHY: every score in the offseason report uses ONE season (2026). One season is
noisy -- Ty France and Jake Bauers both look like career years, and a 280-PA
sample like Grichuk's can swing a lot. This blends up to the last three seasons
(most recent counts most) and pulls small samples toward the league average.

METHOD (Marcel-style, deliberately simple and explainable)
  weight per season: 3 (latest), 2, 1
  effective sample  = weight * (PA for hitters, IP for pitchers)
  projection = (sum(effective_sample * stat) + K * league_avg)
               / (sum(effective_sample) + K)
  K is the "ballast": how much league-average the player is pulled toward.
  It is chosen by the backtest, not guessed (see the grid printed below).
  Hitters are projected on OPS, pitchers on ERA (same stats the report scores).
  No age adjustment yet -- listed as a limitation, not hidden.

BACKTEST: for target years 2025 and 2026, project that year using ONLY the
prior seasons, then compare to what really happened. Three predictors:
  league average, "naive" (last season repeated), and this projection.
  Reported as mean absolute error (MAE); lower is better.

Usage:
    python project_2027.py --check     # print what each season's tables look like
    python project_2027.py             # backtest + 2027 projections + CSV export
    python project_2027.py --no-backtest
"""
import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports")
RECENCY_WEIGHTS = [3, 2, 1]            # latest season first
DEFAULT_K = {"batting": 600, "pitching": 150}   # used only if no backtest result
K_GRID = {"batting": [200, 400, 600, 900, 1400, 2000, 3000, 5000],
          "pitching": [100, 250, 400, 600, 900, 1400]}
STAT = {"batting": ("OPS", "PA"), "pitching": ("ERA", "IP")}
MIN_TARGET = {"batting": 150, "pitching": 30}   # needs this much in the year being predicted
MIN_PRIOR = {"batting": 100, "pitching": 20}    # ...and this much in the year before
BACKTEST_TARGETS = [2025, 2026]
FOCUS_GROUPS = ["Shortstops", "Third Basemen", "Right Fielders",
                "Right-Handed Relievers", "Left-Handed Relievers"]


# ── data loading (the ONE place that touches mlb_api) ─────────────────────────
def load_season(season: int, kind: str) -> pd.DataFrame:
    """Player table for one season. kind = 'batting' or 'pitching'.
    If your mlb_api column names differ, this is the only function to adjust."""
    import mlb_api
    return mlb_api.get_player_tables(season)[kind]


_cache = {}
def _season(season: int, kind: str):
    key = (season, kind)
    if key not in _cache:
        try:
            _cache[key] = _clean(load_season(season, kind), kind)
        except Exception as e:  # a missing/failed season should not kill the run
            print(f"  [warn] {kind} {season} unavailable ({type(e).__name__}: {e})")
            _cache[key] = None
    return _cache[key]


def _ip_to_float(x):
    """Baseball innings: '65.1' means 65 and 1/3. Leave true decimals alone."""
    s = str(x).strip()
    m = re.fullmatch(r"(\d+)\.([012])", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 3
    return pd.to_numeric(s, errors="coerce")


def norm_key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii)\b", " ", s)
    return " ".join(s.split())


def _clean(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    stat, samp = STAT[kind]
    d = df.copy()
    d["name"] = d["Name"].astype(str).str.split(" \\(").str[0].str.strip()
    d["key"] = d["name"].map(norm_key)
    d[samp] = d[samp].map(_ip_to_float) if kind == "pitching" else pd.to_numeric(d[samp], errors="coerce")
    d[stat] = pd.to_numeric(d[stat], errors="coerce")
    d = d.dropna(subset=[stat, samp])
    d = d[d[samp] > 0]
    # a player split across teams can appear on several rows: combine, sample-weighted
    d["_w"] = d[stat] * d[samp]
    g = d.groupby("key").agg(name=("name", "first"), _w=("_w", "sum"), s=(samp, "sum"))
    g[stat] = g["_w"] / g["s"]
    g = g.rename(columns={"s": samp}).drop(columns="_w").reset_index()
    return g[["key", "name", stat, samp]]


# ── projection ────────────────────────────────────────────────────────────────
def project(last_season: int, kind: str, k: float, weights=RECENCY_WEIGHTS) -> pd.DataFrame:
    """Project the season AFTER last_season using last_season and the years before it."""
    stat, samp = STAT[kind]
    parts = []
    for i, w in enumerate(weights):
        t = _season(last_season - i, kind)
        if t is None:
            continue
        t = t.copy()
        t["eff"] = w * t[samp]
        t["wnum"] = t["eff"] * t[stat]
        t["yr"] = last_season - i
        parts.append(t)
    if not parts or parts[0]["yr"].iloc[0] != last_season:
        return pd.DataFrame()
    latest = parts[0]
    lg = (latest[stat] * latest[samp]).sum() / latest[samp].sum()
    allp = pd.concat(parts, ignore_index=True)
    g = allp.groupby("key").agg(name=("name", "first"), eff=("eff", "sum"),
                                wnum=("wnum", "sum"), seasons=("yr", "nunique")).reset_index()
    g["proj"] = (g["wnum"] + k * lg) / (g["eff"] + k)
    last = latest.set_index("key")
    g["last_stat"] = g["key"].map(last[stat])
    g["last_sample"] = g["key"].map(last[samp]).fillna(0)
    g["league_avg"] = lg
    return g[["key", "name", "proj", "last_stat", "last_sample", "seasons", "league_avg"]]


# ── backtest ──────────────────────────────────────────────────────────────────
def backtest(target: int, kind: str, k: float):
    stat, samp = STAT[kind]
    actual = _season(target, kind)
    proj = project(target - 1, kind, k)
    if actual is None or proj.empty:
        return None
    m = proj.merge(actual[["key", stat, samp]].rename(columns={stat: "act", samp: "act_s"}), on="key")
    m = m[(m["act_s"] >= MIN_TARGET[kind]) & (m["last_sample"] >= MIN_PRIOR[kind])
          & m["last_stat"].notna()]
    if len(m) < 20:
        return None
    err = lambda col: float((m[col] - m["act"]).abs().mean())
    m = m.assign(lg=m["league_avg"])
    return {"n": len(m), "league": err("lg"), "naive": err("last_stat"), "proj": err("proj"), "frame": m}


def run_backtest(kind: str):
    stat, _ = STAT[kind]
    fmt = "{:.3f}" if kind == "batting" else "{:.2f}"
    print(f"\n{'=' * 78}\nBACKTEST -- {kind} ({stat}); MAE vs what actually happened (lower = better)\n{'=' * 78}")
    best_k, best_mae = None, None
    for k in K_GRID[kind]:
        rows = [backtest(t, kind, k) for t in BACKTEST_TARGETS]
        rows = [r for r in rows if r]
        if not rows:
            continue
        mae = float(np.mean([r["proj"] for r in rows]))
        detail = "  ".join(f"{t}: proj {fmt.format(r['proj'])} (naive {fmt.format(r['naive'])}, "
                           f"league {fmt.format(r['league'])}, n={r['n']})"
                           for t, r in zip(BACKTEST_TARGETS, rows))
        print(f"  K={k:<5} avg proj MAE {fmt.format(mae)}   {detail}")
        if best_mae is None or mae < best_mae:
            best_k, best_mae = k, mae
    if best_k is None:
        print("  Not enough data to backtest -- using default K.")
        return DEFAULT_K[kind]
    print(f"  -> best K = {best_k}")
    # segment by how much last-season sample there was (small samples are where shrinkage should help)
    cut = 250 if kind == "batting" else 50
    unit = "PA" if kind == "batting" else "IP"
    segs = {}
    for t in BACKTEST_TARGETS:
        r = backtest(t, kind, best_k)
        if r:
            f = r["frame"]
            for label, sub in ((f"last season < {cut} {unit}", f[f["last_sample"] < cut]),
                               (f"last season >= {cut} {unit}", f[f["last_sample"] >= cut])):
                if len(sub):
                    segs.setdefault(label, []).append(sub)
    print(f"\n  By last-season sample size (best K):")
    for label, subs in segs.items():
        s = pd.concat(subs)
        e = lambda col: float((s[col] - s["act"]).abs().mean())
        print(f"    {label:<24} n={len(s):<4} naive {fmt.format(e('last_stat'))}   "
              f"proj {fmt.format(e('proj'))}   league {fmt.format(e('league_avg'))}")
    return best_k


# ── 2027 output ───────────────────────────────────────────────────────────────
def build_2027(last_season: int, ks: dict) -> pd.DataFrame:
    frames = []
    for kind in ("batting", "pitching"):
        p = project(last_season, kind, ks[kind])
        if p.empty:
            print(f"  [warn] no {kind} projections built")
            continue
        p["kind"] = kind
        p["stat"] = STAT[kind][0]
        frames.append(p)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def print_focus(proj: pd.DataFrame, last_season: int):
    path = os.path.join(EXPORT_DIR, "free_agent_scored.csv")
    if proj.empty or not os.path.exists(path):
        return
    sc = pd.read_csv(path)
    sc = sc[sc["recommendation_score"].notna()].copy()
    sc["key"] = sc["name"].map(norm_key)
    print(f"\n{'=' * 90}\nSHOPPING LIST WITH {last_season + 1} PROJECTIONS (hitters OPS, pitchers ERA)\n{'=' * 90}")
    for g in FOCUS_GROUPS:
        sub = sc[sc["position_group"] == g].sort_values("recommendation_score", ascending=False).head(6)
        print(f"\n  {g}")
        if sub.empty:
            print("    (none)")
            continue
        kind = "pitching" if "Relievers" in g or g == "Starting Pitchers" else "batting"
        fmt = "{:.2f}" if kind == "pitching" else "{:.3f}"
        lbl = "ERA" if kind == "pitching" else "OPS"
        print(f"    {'Name':<20} {'Score':>6} {last_season} {lbl:<4} {'Proj':>7} {'Seasons':>8}")
        pk = proj[proj["kind"] == kind].set_index("key")
        for _, r in sub.iterrows():
            if r["key"] in pk.index:
                p = pk.loc[r["key"]]
                print(f"    {r['name']:<20} {r['recommendation_score']:>6.2f} "
                      f"{fmt.format(p['last_stat']):>9} {fmt.format(p['proj']):>7} {int(p['seasons']):>8}")
            else:
                print(f"    {r['name']:<20} {r['recommendation_score']:>6.2f} {'n/a':>9} {'n/a':>7} {'-':>8}")
    print("=" * 90)
    print("  Proj = recency-weighted (3/2/1) blend pulled toward league average. A projection")
    print("  well below last season's number means last season looked like a career year / small sample.")
    print("=" * 90)


def export_for_app(proj: pd.DataFrame, last_season: int):
    """Writes free_agent_projection_<year>.csv (name, proj_OPS_<year>) in the exact shape
    story_app.py already looks for, using the free agents' own names from free_agent_scored.csv."""
    path = os.path.join(EXPORT_DIR, "free_agent_scored.csv")
    if proj.empty or not os.path.exists(path):
        return
    sc = pd.read_csv(path, usecols=["name"]).drop_duplicates("name")
    sc["key"] = sc["name"].map(norm_key)
    hit = proj[proj["kind"] == "batting"][["key", "proj"]]
    col = f"proj_OPS_{last_season + 1}"
    out = sc.merge(hit, on="key", how="inner").rename(columns={"proj": col})[["name", col]]
    out[col] = out[col].round(3)
    dest = os.path.join(EXPORT_DIR, f"free_agent_projection_{last_season + 1}.csv")
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} hitter projections for the app to {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-season", type=int, default=2026, help="most recent completed season")
    ap.add_argument("--check", action="store_true", help="just show what each season's table looks like")
    ap.add_argument("--no-backtest", action="store_true")
    a = ap.parse_args()

    if a.check:
        for kind in ("batting", "pitching"):
            for yr in range(a.last_season - 4, a.last_season + 1):
                try:
                    t = load_season(yr, kind)
                    c = _clean(t, kind)
                    print(f"{kind:<9} {yr}: {len(t)} raw rows -> {len(c)} players; columns: {list(t.columns)[:14]}")
                except Exception as e:
                    print(f"{kind:<9} {yr}: FAILED -- {type(e).__name__}: {e}")
        return

    ks = dict(DEFAULT_K)
    if not a.no_backtest:
        for kind in ("batting", "pitching"):
            ks[kind] = run_backtest(kind)
    print(f"\nUsing K: hitters {ks['batting']}, pitchers {ks['pitching']}")

    proj = build_2027(a.last_season, ks)
    if proj.empty:
        print("No projections produced.")
        return
    os.makedirs(EXPORT_DIR, exist_ok=True)
    out = os.path.join(EXPORT_DIR, f"projections_{a.last_season + 1}.csv")
    proj.drop(columns="key").to_csv(out, index=False)
    print(f"Wrote {len(proj)} projections to {out}")
    export_for_app(proj, a.last_season)
    print_focus(proj, a.last_season)


if __name__ == "__main__":
    main()