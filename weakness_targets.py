"""
weakness_targets.py
Shows what the Mariners are weak at, then finds MLB players who are strong at
exactly those things -- using the public MLB Stats API, so it needs NO
free-agent list.

HOW IT WORKS
  1. Reads data/exports/diagnosed_needs.csv and keeps the rows marked WEAKNESS.
  2. Pulls every MLB hitter's and pitcher's season stats (one call each).
  3. For each weak metric, ranks players by that stat, keeps only those who are
     clearly BETTER than the Mariners' own team number, and enforces a minimum
     sample (hitters: 300 PA; relievers: 30 IP; starters: 100 IP).
  4. Writes data/exports/weakness_targets.csv with one row per (metric, player),
     plus weakness_targets_summary.csv: players who beat the Mariners on the
     MOST weak metrics, ranked by fit_score = sum over weak stats of
     (severity x how many league standard deviations better than the Mariners).
     So a big margin on a severe weakness counts for more than barely beating
     several; ties on "metrics beaten" no longer hide who is actually best.

READ THIS BEFORE TRUSTING IT
  - These are CANDIDATES, not free agents: any non-Mariners player appears,
    whatever his contract. Use it to decide what kind of player to look for;
    cross-check availability yourself.
  - Fielding (fielding_Rtot) uses the project's own loader
    (battingpitching_scraper.get_fielding -> Statcast fielding runs). Players
    need 600+ innings, and the list is the best gloves by runs above average.
    It is a separate list and is NOT mixed into the fit score. If the loader
    can't be imported, fielding is skipped with a message and the rest runs.
  - The comparison is player vs. a TEAM number. A hitter's .290 average beats
    the Mariners' .232 team average, but a team number includes weak bench
    hitters, so treat the gap as a rough guide, not a projection.
  - Not yet run against the live API (written from the documented endpoint).
    If the response shape differs, the script says which part failed.

Usage:
    python weakness_targets.py
    python weakness_targets.py --season 2026 --top 50
"""
import argparse
import os
import unicodedata

import pandas as pd
import requests

BASE = "https://statsapi.mlb.com/api/v1"
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports")
SEA = "Seattle Mariners"
MIN_PA, MIN_IP_RELIEF, MIN_IP_START, MIN_INN_FIELD = 300, 30, 100, 600

# metric -> (pool, API field, higher_is_better)
METRICS = {
    "BA": ("hit", "avg", True), "OBP": ("hit", "obp", True), "SLG": ("hit", "slg", True),
    "OPS": ("hit", "ops", True), "ISO": ("hit", "iso", True),
    "BB_pct": ("hit", "bb_pct", True), "K_pct": ("hit", "k_pct", False),
    "bullpen_ERA": ("rel", "era", False), "bullpen_WHIP": ("rel", "whip", False),
    "bullpen_K9": ("rel", "k9", True), "bullpen_BB9": ("rel", "bb9", False),
    "bullpen_HR9": ("rel", "hr9", False),
    "rotation_ERA": ("sp", "era", False), "rotation_WHIP": ("sp", "whip", False),
    "rotation_K9": ("sp", "k9", True), "rotation_BB9": ("sp", "bb9", False),
    "rotation_HR9": ("sp", "hr9", False),
    # fielding runs (Rtot) come from the project's own loader, not the Stats API call below
    "fielding_Rtot": ("fld", "rtot", True),
}
OUT_TARGET_COLS = ["metric", "rank", "name", "Team", "position", "bats", "sample", "value", "mariners", "better_by"]
FLOORS = {"hit": MIN_PA, "rel": MIN_IP_RELIEF, "sp": MIN_IP_START, "fld": MIN_INN_FIELD}


def _norm(name) -> str:
    """'Julio Rodríguez' and 'julio rodriguez' match; accents/punctuation/Jr. ignored."""
    if not isinstance(name, str):
        return ""
    t = unicodedata.normalize("NFKD", name)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = "".join(c for c in t.lower() if c.isalnum() or c == " ")
    for suf in (" jr", " sr", " ii", " iii"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return " ".join(t.split())


def _hand_map(export_dir) -> dict:
    """{normalized name: L/R/S} from player_handedness.csv (written by mlb_stats_api.py), if present.
    Two players sharing a name are left out rather than guessed."""
    path = os.path.join(export_dir, "player_handedness.csv")
    if not os.path.exists(path):
        return {}
    h = pd.read_csv(path)
    if "name" not in h.columns or "bats" not in h.columns:
        return {}
    h["k"] = h["name"].map(_norm)
    h = h[~h["k"].duplicated(keep=False)].dropna(subset=["bats"])
    return dict(zip(h["k"], h["bats"]))


def _get_json(path, params=None, fetch=None):
    if fetch is not None:
        return fetch(path, params or {})
    url = f"{BASE}/{path.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"MLB Stats API request failed: {url} {params or ''} -> {e}") from e


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ip(x):
    """'123.1' innings means 123 and 1/3."""
    try:
        whole, _, frac = str(x).partition(".")
        return int(whole) + (int(frac or 0) / 3.0)
    except ValueError:
        return float("nan")


def get_league_stats(group: str, season: int, fetch=None) -> pd.DataFrame:
    data = _get_json("stats", {"stats": "season", "group": group, "season": season, "sportIds": 1,
                               "playerPool": "ALL", "limit": 3000}, fetch)
    blocks = data.get("stats")
    if not blocks or "splits" not in blocks[0]:
        raise RuntimeError(f"No stats.splits in the {group} response (keys: {list(data)}) -- "
                           f"the endpoint shape may have changed.")
    rows = []
    for s in blocks[0]["splits"]:
        st, pl, tm = s.get("stat", {}), s.get("player", {}), s.get("team", {})
        rows.append({"player_id": pl.get("id"), "name": pl.get("fullName"), "Team": tm.get("name"), "stat": st})
    if not rows:
        raise RuntimeError(f"The API returned no {group} players for {season}.")
    return pd.DataFrame(rows)


def hitters_table(raw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in raw.iterrows():
        st = r["stat"]
        pa = _num(st.get("plateAppearances"))
        if not pa or pa != pa:
            continue
        avg, slg = _num(st.get("avg")), _num(st.get("slg"))
        out.append({"name": r["name"], "Team": r["Team"], "sample": pa, "pool": "hit",
                    "avg": avg, "obp": _num(st.get("obp")), "slg": slg, "ops": _num(st.get("ops")),
                    "iso": slg - avg, "bb_pct": _num(st.get("baseOnBalls")) / pa,
                    "k_pct": _num(st.get("strikeOuts")) / pa})
    return pd.DataFrame(out)


def pitchers_table(raw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in raw.iterrows():
        st = r["stat"]
        ip = _ip(st.get("inningsPitched"))
        if not ip or ip != ip:
            continue
        gs = _num(st.get("gamesStarted"))
        pool = "sp" if gs >= 15 else ("rel" if gs <= 2 else None)
        if pool is None:
            continue
        out.append({"name": r["name"], "Team": r["Team"], "sample": ip, "pool": pool,
                    "era": _num(st.get("era")), "whip": _num(st.get("whip")),
                    "k9": _num(st.get("strikeoutsPer9Inn")), "bb9": _num(st.get("walksPer9Inn")),
                    "hr9": _num(st.get("homeRunsPer9"))})
    return pd.DataFrame(out)


def fielding_table(raw) -> pd.DataFrame:
    """One row per player from the project's fielding loader (Name, Tm/Team, Pos, Inn, Rtot).
    Rtot and innings are summed across positions; the position with the most innings is kept."""
    if isinstance(raw, dict):
        raw = raw.get("all_players")
    if raw is None or len(raw) == 0 or "Rtot" not in raw.columns or "Name" not in raw.columns:
        raise RuntimeError("fielding data has no Name/Rtot columns")
    d = raw.copy()
    team_col = "Team" if "Team" in d.columns else ("Tm" if "Tm" in d.columns else None)
    d["Rtot"] = pd.to_numeric(d["Rtot"], errors="coerce")
    d["Inn"] = d["Inn"].map(_ip).fillna(0) if "Inn" in d.columns else 0.0   # '1197.1' = 1197 1/3
    d = d.dropna(subset=["Rtot"])
    rows = []
    for name, g in d.groupby("Name"):
        top = g.sort_values("Inn", ascending=False).iloc[0]
        rows.append({"name": name, "Team": top[team_col] if team_col else "", "sample": _ip_notation(g["Inn"].sum()),
                     "pool": "fld", "rtot": float(g["Rtot"].sum()),
                     "position": top["Pos"] if "Pos" in g.columns else ""})
    return pd.DataFrame(rows)


def _ip_notation(innings) -> float:
    """True innings (thirds) back to baseball notation: 1324.333 -> 1324.1."""
    thirds = int(round(float(innings) * 3))
    return thirds // 3 + (thirds % 3) / 10.0


def _is_mariner(team) -> bool:
    return str(team) in (SEA, "SEA", "Seattle", "Mariners")


def targets_for_metric(metric: str, team_value: float, pool_df: pd.DataFrame, top: int) -> pd.DataFrame:
    pool, field, higher = METRICS[metric]
    d = pool_df[(pool_df["pool"] == pool) & (~pool_df["Team"].map(_is_mariner))].copy()
    floor = FLOORS[pool]
    if pool == "fld":
        # Rtot is already "runs above average" (average = 0), so the Mariners TOTAL is not a bar to clear.
        d = d[d["sample"] >= floor].dropna(subset=[field])
        d["value"] = d[field].round(1)
        d["mariners"] = team_value
        d["better_by"] = d["value"]
        for c in ("bats",):
            if c not in d.columns:
                d[c] = ""
        d = d[d["value"] > 0].sort_values("value", ascending=False).head(top)
        d.insert(0, "metric", metric)
        d.insert(1, "rank", range(1, len(d) + 1))
        return d[OUT_TARGET_COLS]
    d = d[(d["sample"] >= floor)].dropna(subset=[field])
    v = d[field]
    # team value for rate stats may be stored as a percent (e.g. 21.4) while the API gives a fraction
    tv = team_value
    if metric in ("BB_pct", "K_pct") and tv > 1:
        v = v * 100
    d["value"] = v
    d["mariners"] = tv
    d["better_by"] = (d["value"] - tv) if higher else (tv - d["value"])
    d = d[d["better_by"] > 0].sort_values("better_by", ascending=False).head(top)
    d["value"], d["better_by"] = d["value"].round(3), d["better_by"].round(3)
    d.insert(0, "metric", metric)
    d.insert(1, "rank", range(1, len(d) + 1))
    for c in ("position", "bats"):
        if c not in d.columns:
            d[c] = ""
    return d[OUT_TARGET_COLS]


def run(season=2026, top=50, fetch=None, export_dir=None, get_fielding=None) -> pd.DataFrame:
    export_dir = export_dir or EXPORT_DIR
    needs_path = os.path.join(export_dir, "diagnosed_needs.csv")
    if not os.path.exists(needs_path):
        raise SystemExit(f"{needs_path} not found -- run the analysis (main.py / offseason_report.py) first.")
    needs = pd.read_csv(needs_path)
    weak = needs[needs["Verdict"] == "WEAKNESS"].sort_values("Severity_Score", ascending=False)
    print(f"{len(weak)} weaknesses: {', '.join(weak['Metric'])}")

    skipped = [m for m in weak["Metric"] if m not in METRICS]
    use = weak[weak["Metric"].isin(METRICS)]
    for m in skipped:
        print(f"  skipping {m}: no player-level source wired up for it")

    frames = []
    pools = []
    if (use["Metric"].map(lambda m: METRICS[m][0]) == "hit").any():
        pools.append(hitters_table(get_league_stats("hitting", season, fetch)))
    if (use["Metric"].map(lambda m: METRICS[m][0]).isin(["rel", "sp"])).any():
        pools.append(pitchers_table(get_league_stats("pitching", season, fetch)))
    fld_tbl = None
    try:
        if get_fielding is None:
            from battingpitching_scraper import get_fielding   # the project's own fielding loader
        fld_tbl = fielding_table(get_fielding(season))
    except Exception as e:      # missing module, changed columns, no data: skip fielding, keep the rest
        if (use["Metric"] == "fielding_Rtot").any():
            print(f"  skipping fielding_Rtot: could not load player fielding data ({type(e).__name__}: {e})")
        use = use[use["Metric"] != "fielding_Rtot"]
    if fld_tbl is not None and (use["Metric"] == "fielding_Rtot").any():
        pools.append(fld_tbl)
    pool_df = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    if not pool_df.empty:
        # position = where he played the most innings; bats = from player_handedness.csv. Blank when unknown.
        pos_map = ({_norm(n): p for n, p in zip(fld_tbl["name"], fld_tbl["position"])}
                   if fld_tbl is not None else {})
        hand = _hand_map(export_dir)
        pool_df["k"] = pool_df["name"].map(_norm)
        if "position" not in pool_df.columns:
            pool_df["position"] = ""
        pool_df["position"] = [p if isinstance(p, str) and p else pos_map.get(k, "")
                               for p, k in zip(pool_df["position"], pool_df["k"])]
        pool_df["bats"] = pool_df["k"].map(hand).fillna("")
        pool_df = pool_df.drop(columns=["k"])
    for _, r in use.iterrows():
        frames.append(targets_for_metric(r["Metric"], float(r["Team_Value"]), pool_df, top))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=OUT_TARGET_COLS)

    # who fixes the most weak metrics at once (within hitters / relievers / starters, via the metric's pool)
    all_hits = []
    for _, r in use.iterrows():
        pool, field, higher = METRICS[r["Metric"]]
        if pool == "fld":
            continue      # fielding runs are shown as their own list, not mixed into the fit score
        d = pool_df[(pool_df["pool"] == pool) & (~pool_df["Team"].map(_is_mariner))]
        floor = FLOORS[pool]
        d = d[d["sample"] >= floor].dropna(subset=[field])
        tv = float(r["Team_Value"])
        v = d[field] * 100 if (r["Metric"] in ("BB_pct", "K_pct") and tv > 1) else d[field]
        sd = float(v.std()) or 1.0
        gain = ((v - tv) if higher else (tv - v)) / sd      # how many league std-devs better than the Mariners
        beat = d[gain > 0]
        for idx, p in beat.iterrows():
            all_hits.append({"name": p["name"], "Team": p["Team"], "pool": pool, "metric": r["Metric"],
                             "severity": float(r["Severity_Score"]),
                             "position": p.get("position", ""), "bats": p.get("bats", ""),
                             "fit": float(r["Severity_Score"]) * float(gain.loc[idx])})
    if all_hits:
        h = pd.DataFrame(all_hits)
        summ = (h.groupby(["name", "Team", "pool"], as_index=False)
                  .agg(weak_metrics_beaten=("metric", "count"), severity_covered=("severity", "sum"),
                       fit_score=("fit", "sum"), position=("position", "first"), bats=("bats", "first"),
                       metrics=("metric", lambda s: ", ".join(s))))
        summ["fit_score"] = summ["fit_score"].round(2)
        summ = summ.sort_values(["fit_score", "weak_metrics_beaten"], ascending=False)
    else:
        summ = pd.DataFrame(columns=["name", "Team", "pool", "weak_metrics_beaten", "severity_covered", "fit_score", "position", "bats", "metrics"])

    os.makedirs(export_dir, exist_ok=True)
    p1 = os.path.join(export_dir, "weakness_targets.csv")
    p2 = os.path.join(export_dir, "weakness_targets_summary.csv")
    out.to_csv(p1, index=False)
    summ.to_csv(p2, index=False)
    print(f"Wrote {len(out)} rows -> {p1}\nWrote {len(summ)} players -> {p2}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find players who are strong where the Mariners are weak.")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args()
    run(a.season, a.top)