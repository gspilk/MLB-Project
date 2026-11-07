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
  Age adjustment (hitters): the next-season OPS change for a player's age band, taken from the R
  aging curves (data/exports/aging_by_age_band.csv), times a scale chosen by the backtest. If the
  backtest says it does not help, the scale is 0 and nothing changes. Pitchers: not adjusted.

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
AGE_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # how much of one year's aging change to apply (backtested)
AGE_MIN_GAIN = 0.0005                    # keep the age adjustment only if it cuts the error by at least this
AGING_FILE = "aging_by_age_band.csv"
AGE_BAND_CUTS = [-np.inf, 26, 29, 32, 35, np.inf]
AGE_BAND_NAMES = ["26 and under", "27-29", "30-32", "33-35", "36+"]
AGE_MIN_N = 20
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
    if "player_id" in d.columns:
        d["pid"] = pd.to_numeric(d["player_id"], errors="coerce")
    d[samp] = d[samp].map(_ip_to_float) if kind == "pitching" else pd.to_numeric(d[samp], errors="coerce")
    d[stat] = pd.to_numeric(d[stat], errors="coerce")
    d = d.dropna(subset=[stat, samp])
    d = d[d[samp] > 0]
    # a player split across teams can appear on several rows: combine, sample-weighted
    d["_w"] = d[stat] * d[samp]
    agg = {"name": ("name", "first"), "_w": ("_w", "sum"), "s": (samp, "sum")}
    if "pid" in d.columns:
        agg["pid"] = ("pid", "first")
    g = d.groupby("key").agg(**agg)
    g[stat] = g["_w"] / g["s"]
    g = g.rename(columns={"s": samp}).drop(columns="_w").reset_index()
    cols = ["key", "name", stat, samp] + (["pid"] if "pid" in g.columns else [])
    return g[cols]


# ── age adjustment (hitters) ──────────────────────────────────────────────────
_people_cache = {}
_aging = None


def _age_on(birth_date, season):
    """Age on June 30 of the season (same convention as mlb_api.season_age)."""
    try:
        y, m, d = map(int, str(birth_date)[:10].split("-"))
    except ValueError:
        return np.nan
    return season - y - (1 if (m, d) > (6, 30) else 0)


def _load_people(ids):
    need = sorted({int(i) for i in ids if pd.notna(i) and int(i) not in _people_cache})
    if not need:
        return
    import mlb_api
    df = mlb_api.get_people(need)
    for r in df.itertuples():
        _people_cache[int(r.player_id)] = (r.birth_date, r.primary_pos)


def _aging_table():
    global _aging
    if _aging is None:
        path = os.path.join(EXPORT_DIR, AGING_FILE)
        _aging = {}
        if os.path.exists(path):
            t = pd.read_csv(path)
            for _, r in t.iterrows():
                if pd.notna(r.get("d_ops")) and r.get("n", 0) >= AGE_MIN_N:
                    _aging[(r["group"], r["band"])] = float(r["d_ops"])
    return _aging


def age_delta(pid, season):
    """Average next-season OPS change for a hitter of this age in `season` (0 if age or band is unknown)."""
    tab = _aging_table()
    if pd.isna(pid) or int(pid) not in _people_cache or not tab:
        return 0.0
    birth, pos = _people_cache[int(pid)]
    age = _age_on(birth, season)
    if pd.isna(age):
        return 0.0
    band = pd.cut([age], bins=AGE_BAND_CUTS, labels=AGE_BAND_NAMES)[0]
    grp = "1B / 3B / SS" if str(pos) in ("1B", "3B", "SS") else "All other hitters"
    return tab.get((grp, band), 0.0)


def prepare_ages(seasons):
    """Look up birth dates for every hitter in the given seasons. Returns True if ages are usable."""
    try:
        ids = []
        for yr in seasons:
            t = _season(yr, "batting")
            if t is not None and "pid" in t.columns:
                ids += list(t["pid"].dropna())
        if not ids or not _aging_table():
            return False
        _load_people(ids)
        return bool(_people_cache)
    except Exception as e:
        print(f"  [warn] ages unavailable ({type(e).__name__}: {e}); no age adjustment")
        return False


# ── projection ────────────────────────────────────────────────────────────────
def project(last_season: int, kind: str, k: float, weights=RECENCY_WEIGHTS, age_scale: float = 0.0) -> pd.DataFrame:
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
    g["age_adj"] = 0.0
    if kind == "batting" and age_scale and "pid" in allp.columns:
        pid = allp.dropna(subset=["pid"]).drop_duplicates("key").set_index("key")["pid"]
        g["age_adj"] = [age_scale * age_delta(pid.get(k_), last_season + 1) for k_ in g["key"]]
        g["proj"] = g["proj"] + g["age_adj"]
    last = latest.set_index("key")
    g["last_stat"] = g["key"].map(last[stat])
    g["last_sample"] = g["key"].map(last[samp]).fillna(0)
    g["league_avg"] = lg
    return g[["key", "name", "proj", "last_stat", "last_sample", "seasons", "league_avg", "age_adj"]]


# ── backtest ──────────────────────────────────────────────────────────────────
def backtest(target: int, kind: str, k: float, age_scale: float = 0.0):
    stat, samp = STAT[kind]
    actual = _season(target, kind)
    proj = project(target - 1, kind, k, age_scale=age_scale)
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


def run_age_backtest(k: float) -> float:
    """Does adding the aging curve to the hitter projection cut the error? Returns the scale to use (0 = none)."""
    print(f"\n{'=' * 78}\nBACKTEST -- age adjustment for hitters (K={k}); MAE of OPS (lower = better)\n{'=' * 78}")
    seasons = sorted({t - i for t in BACKTEST_TARGETS for i in range(1, 4)} | set(BACKTEST_TARGETS))
    if not prepare_ages(seasons):
        print("  Ages or the aging table are not available, so no age adjustment is used.")
        print(f"  (needs data/exports/{AGING_FILE} and the MLB people lookup)")
        return 0.0
    best_s, best_mae, base_mae = 0.0, None, None
    for sc in AGE_SCALES:
        rows = [backtest(t, "batting", k, sc) for t in BACKTEST_TARGETS]
        rows = [r for r in rows if r]
        if not rows:
            continue
        mae = float(np.mean([r["proj"] for r in rows]))
        detail = "  ".join(f"{t}: {r['proj']:.4f}" for t, r in zip(BACKTEST_TARGETS, rows))
        print(f"  age scale {sc:<4} avg MAE {mae:.4f}   {detail}")
        if sc == 0.0:
            base_mae = mae
        if best_mae is None or mae < best_mae:
            best_s, best_mae = sc, mae
    if base_mae is None or best_mae is None:
        return 0.0
    gain = base_mae - best_mae
    if best_s > 0 and gain >= AGE_MIN_GAIN:
        print(f"  -> age adjustment helps: scale {best_s} cuts error {gain:.4f} OPS ({gain / base_mae:.1%}). Using it.")
        print("  (caveat: the aging curves were built from the same seasons, so this is a slightly friendly test)")
        return best_s
    print(f"  -> age adjustment does not clearly help (best gain {gain:.4f}). Not used.")
    return 0.0


# ── 2027 output ───────────────────────────────────────────────────────────────
def build_2027(last_season: int, ks: dict, age_scale: float = 0.0) -> pd.DataFrame:
    frames = []
    for kind in ("batting", "pitching"):
        p = project(last_season, kind, ks[kind], age_scale=age_scale if kind == "batting" else 0.0)
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
    age_scale = 0.0
    if not a.no_backtest:
        for kind in ("batting", "pitching"):
            ks[kind] = run_backtest(kind)
        age_scale = run_age_backtest(ks["batting"])
        if age_scale and not prepare_ages([a.last_season - i for i in range(0, 3)]):
            age_scale = 0.0
    print(f"\nUsing K: hitters {ks['batting']}, pitchers {ks['pitching']}; age scale {age_scale}")

    proj = build_2027(a.last_season, ks, age_scale)
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