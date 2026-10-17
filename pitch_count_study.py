"""
pitch_count_study.py -- when does each Mariners starter start to fade?

Pulls play-by-play for every completed Mariners regular-season game from the
MLB Stats API (the same source the rest of the project uses), keeps one row per
plate appearance a Mariners pitcher faced, and measures how opponents hit
against each starter as his pitch count and the lineup's trip through the order
go up.

  1. First run: one schedule request plus one play-by-play request per game
     (~160), paused 0.25s apart, saved to data/exports/pitch_count_pas.csv.
  2. Later runs only fetch games that are not in that file yet.
  3. Summary -> data/exports/pitch_count_summary.csv, pitch_count_by_inning.csv
     and a printed report (by pitch count, by inning with runs, by trip through the order).

Read the output with care:
  * A manager pulls a starter when he looks tired or is losing the plate
    appearance battle, so the late-count rows come from starters who were still
    going well. That flatters the late buckets.
  * Each starter has ~30 starts, so a bucket has only a few dozen batters.
    Treat a "fade point" as a range, not a cutoff.

Usage (from the project folder):
    python pitch_count_study.py                # fetch new games, then report
    python pitch_count_study.py --refresh      # re-download every game
    python pitch_count_study.py --season 2026 --min-pa 60
"""
import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.join(ROOT, "data", "exports")
PA_FILE = "pitch_count_pas.csv"
SUMMARY_FILE = "pitch_count_summary.csv"
INNING_FILE = "pitch_count_by_inning.csv"
PITCH_FILE = "pitch_velocity_pitches.csv"
START_FILE = "pitch_velocity_by_start.csv"

BASE = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "mlb-project (personal portfolio project)"}
SEA_ID = 136
PAUSE = 0.25                    # seconds between requests

# pitch count BEFORE the plate appearance starts
BUCKETS = [(0, 25, "1-25"), (25, 50, "26-50"), (50, 75, "51-75"),
           (75, 100, "76-100"), (100, 10_000, "101+")]
MIN_INNING_PA = 30             # an inning with fewer batters faced is shown but never judged
MIN_BUCKET_PA = 15              # smaller buckets are shown but never used for a verdict
FADE_MARGIN = 0.100             # a bucket this much worse (OPS) than his average = fade
HARD_HIT_MPH = 95.0
FASTBALLS = {"FF", "FA", "FT", "SI"}     # four-seam, two-seam, sinker (velocity comparisons)
VELO_DROP_MPH = 1.0                      # a start/bucket this far below the first 25 pitches = "velocity down"
MIN_BUCKET_FB = 20                       # fewer fastballs in a bucket and it is shown but never judged
MIN_MONTH_FB = 30
PITCH_COLS = ["game_pk", "date", "inning", "pitcher_id", "pitcher", "is_starter",
              "pitch_no", "pitch_type", "speed", "pa_pitch", "strikes", "call", "zone",
              "spin", "ivb", "hb", "ext", "px", "pz", "rel_x", "rel_z", "sz_top", "sz_bot"]

HITS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
WALKS = {"walk", "intent_walk"}
NOT_AB = WALKS | {"hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf",
                  "sac_fly_double_play", "sac_bunt_double_play"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}
SAC_FLIES = {"sac_fly", "sac_fly_double_play"}

PA_COLS = ["game_pk", "date", "inning", "pitcher_id", "pitcher", "batter_id", "batter",
           "is_starter", "pitches_before", "pitches_in_pa", "tto", "event",
           "hit_bases", "is_ab", "is_walk", "is_hbp", "is_sac_fly", "is_k", "launch_speed", "runs"]


# ---------------------------------------------------------------- fetching
def _get(path, params=None):
    import requests
    r = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def list_final_games(season, get=_get, team=SEA_ID):
    """[(game_pk, date, mariners_home_bool)] for completed regular-season games."""
    j = get("schedule", {"sportId": 1, "teamId": team, "season": season, "gameType": "R"})
    out = []
    for day in j.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            if g.get("gameType", "R") != "R":
                continue
            home_id = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
            out.append((g["gamePk"], day.get("date", ""), home_id == team))
    return out


# ----------------------------------------------------------------- parsing
def parse_game(pbp, game_pk, date, sea_home):
    """One row per plate appearance faced by a Mariners pitcher."""
    plays = pbp.get("allPlays") or pbp.get("liveData", {}).get("plays", {}).get("allPlays", [])
    sea_pitching_half = "top" if sea_home else "bottom"
    thrown = {}          # pitcher_id -> pitches so far
    faced = {}           # (pitcher_id, batter_id) -> times faced
    first_pitcher = None
    opp_key = "awayScore" if sea_home else "homeScore"   # the batting (opposing) team's score
    opp_prev = 0
    rows = []
    for p in plays:
        about = p.get("about", {})
        if about.get("halfInning") != sea_pitching_half:
            continue
        res = p.get("result", {})
        event = res.get("eventType")
        if res.get("type") != "atBat" or not event or not about.get("isComplete", True):
            continue
        m = p.get("matchup", {})
        pid = m.get("pitcher", {}).get("id")
        bid = m.get("batter", {}).get("id")
        if pid is None or bid is None:
            continue
        events = p.get("playEvents", [])
        n_pitches = sum(1 for e in events if e.get("isPitch"))
        speed = None
        for e in reversed(events):
            hd = e.get("hitData") or {}
            if hd.get("launchSpeed") is not None:
                speed = hd["launchSpeed"]
                break
        if first_pitcher is None:
            first_pitcher = pid
        score = res.get(opp_key)
        runs = 0
        if score is not None:
            runs = max(0, int(score) - opp_prev)       # runs that scored on this play
            opp_prev = int(score)
        before = thrown.get(pid, 0)
        thrown[pid] = before + n_pitches
        faced[(pid, bid)] = faced.get((pid, bid), 0) + 1
        rows.append({
            "game_pk": game_pk, "date": date, "inning": about.get("inning"),
            "pitcher_id": pid, "pitcher": m.get("pitcher", {}).get("fullName", ""),
            "batter_id": bid, "batter": m.get("batter", {}).get("fullName", ""),
            "is_starter": pid == first_pitcher,
            "pitches_before": before, "pitches_in_pa": n_pitches,
            "tto": faced[(pid, bid)], "event": event,
            "hit_bases": HITS.get(event, 0),
            "is_ab": event not in NOT_AB,
            "is_walk": event in WALKS, "is_hbp": event == "hit_by_pitch",
            "is_sac_fly": event in SAC_FLIES, "is_k": event in STRIKEOUTS,
            "launch_speed": speed, "runs": runs,
        })
    return pd.DataFrame(rows, columns=PA_COLS)


def parse_pitches(pbp, game_pk, date, sea_home):
    """One row per pitch thrown by a Mariners pitcher (speed in mph, NaN if missing)."""
    plays = pbp.get("allPlays") or pbp.get("liveData", {}).get("plays", {}).get("allPlays", [])
    half = "top" if sea_home else "bottom"
    thrown, first_pitcher, rows = {}, None, []
    for p in plays:
        about = p.get("about", {})
        res = p.get("result", {})
        if about.get("halfInning") != half:
            continue
        if res.get("type") != "atBat" or not res.get("eventType") or not about.get("isComplete", True):
            continue
        m = p.get("matchup", {})
        pid = m.get("pitcher", {}).get("id")
        if pid is None:
            continue
        if first_pitcher is None:
            first_pitcher = pid
        in_pa, strikes = 0, 0
        for e in p.get("playEvents", []):
            if not e.get("isPitch"):
                continue
            thrown[pid] = thrown.get(pid, 0) + 1
            in_pa += 1
            det = e.get("details", {})
            pdat = e.get("pitchData") or {}
            brk, co = pdat.get("breaks") or {}, pdat.get("coordinates") or {}
            rows.append({
                "game_pk": game_pk, "date": date, "inning": about.get("inning"),
                "pitcher_id": pid, "pitcher": m.get("pitcher", {}).get("fullName", ""),
                "is_starter": pid == first_pitcher, "pitch_no": thrown[pid],
                "pitch_type": (det.get("type") or {}).get("code"),
                "speed": pdat.get("startSpeed"),
                "pa_pitch": in_pa, "strikes": strikes,
                "call": det.get("code"), "zone": pdat.get("zone"),
                "spin": brk.get("spinRate"), "ivb": brk.get("breakVerticalInduced"), "hb": brk.get("breakHorizontal"),
                "ext": pdat.get("extension"), "px": co.get("pX"), "pz": co.get("pZ"),
                "rel_x": co.get("x0"), "rel_z": co.get("z0"),
                "sz_top": pdat.get("strikeZoneTop"), "sz_bot": pdat.get("strikeZoneBottom"),
            })
            strikes = (e.get("count") or {}).get("strikes", strikes)
    return pd.DataFrame(rows, columns=PITCH_COLS)


def fname(base, season):
    """pitch_count_pas.csv -> pitch_count_pas_2026.csv (one cache per season)."""
    root, ext = os.path.splitext(base)
    return f"{root}_{season}{ext}"


def _migrate_legacy(export_dir, season):
    """Before seasons were in the file names, every run was 2026: rename those files once."""
    if season != 2026:
        return
    for base in (PA_FILE, PITCH_FILE, SUMMARY_FILE, INNING_FILE, START_FILE):
        old, new = os.path.join(export_dir, base), os.path.join(export_dir, fname(base, season))
        if os.path.exists(old) and not os.path.exists(new):
            os.replace(old, new)


def collect(season, export_dir=EXPORTS, get=_get, refresh=False, pause=PAUSE, verbose=True):
    """Fetch games not yet cached (plate appearances AND pitches), return (all PAs, all pitches)."""
    _migrate_legacy(export_dir, season)
    ppath = os.path.join(export_dir, fname(PA_FILE, season))
    vpath = os.path.join(export_dir, fname(PITCH_FILE, season))
    have = pd.DataFrame(columns=PA_COLS)
    havep = pd.DataFrame(columns=PITCH_COLS)
    if os.path.exists(ppath) and not refresh:
        have = pd.read_csv(ppath)
        if "runs" not in have.columns:                  # older cache: no runs column, refetch
            print("  cache is from an older version (no runs column) -- re-downloading all games")
            have = pd.DataFrame(columns=PA_COLS)
    if os.path.exists(vpath) and not refresh:
        havep = pd.read_csv(vpath)
        if not set(PITCH_COLS) <= set(havep.columns):   # older pitch cache: missing newer columns, refetch pitches
            print("  pitch cache is from an older version (missing pitch details) -- re-downloading pitch data")
            havep = pd.DataFrame(columns=PITCH_COLS)
    done_pa = set(have["game_pk"].unique()) if len(have) else set()
    done_pitch = set(havep["game_pk"].unique()) if len(havep) else set()
    games = [g for g in list_final_games(season, get) if g[0] not in done_pa or g[0] not in done_pitch]
    if verbose:
        print(f"{len(done_pa & done_pitch)} games cached, {len(games)} to fetch")
    new, newp = [], []
    for i, (pk, date, home) in enumerate(games, 1):
        try:
            pbp = get(f"game/{pk}/playByPlay")
        except Exception as e:                         # keep what we have; rerun resumes
            print(f"  [warn] game {pk} failed ({e}); stopping here -- rerun to resume")
            break
        if pk not in done_pa:
            new.append(parse_game(pbp, pk, date, home))
        if pk not in done_pitch:
            newp.append(parse_pitches(pbp, pk, date, home))
        if verbose and i % 20 == 0:
            print(f"  fetched {i}/{len(games)}")
        time.sleep(pause)
    frames = [f for f in ([have] + new) if len(f)]
    allpa = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PA_COLS)
    framesp = [f for f in ([havep] + newp) if len(f)]
    allp = pd.concat(framesp, ignore_index=True) if framesp else pd.DataFrame(columns=PITCH_COLS)
    os.makedirs(export_dir, exist_ok=True)
    allpa.to_csv(ppath, index=False)
    allp.to_csv(vpath, index=False)
    return allpa, allp


# ---------------------------------------------------------------- analysis
def bucket_of(pitches_before):
    for lo, hi, label in BUCKETS:
        if lo <= pitches_before < hi:
            return label
    return BUCKETS[-1][2]


def _rates(g):
    """OPS-against, K%, BB%, hard-hit% for a group of plate appearances."""
    pa = len(g)
    if pa == 0:
        return {"pa": 0, "ops": float("nan"), "k_pct": float("nan"),
                "bb_pct": float("nan"), "hard_hit_pct": float("nan")}
    ab = g["is_ab"].sum()
    h = (g["hit_bases"] > 0).sum()
    tb = g["hit_bases"].sum()
    bb, hbp, sf = g["is_walk"].sum(), g["is_hbp"].sum(), g["is_sac_fly"].sum()
    obp_den = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_den if obp_den else float("nan")
    slg = tb / ab if ab else float("nan")
    ls = g["launch_speed"].dropna()
    return {"pa": pa, "ops": obp + slg,
            "k_pct": 100 * g["is_k"].sum() / pa, "bb_pct": 100 * bb / pa,
            "hard_hit_pct": 100 * (ls >= HARD_HIT_MPH).mean() if len(ls) else float("nan")}


def summarize(pas, min_pa=60, only_starters=True):
    """Per-pitcher and pooled rows: one per (pitcher, bucket) plus 'ALL' pitch counts."""
    d = pas.copy()
    if only_starters:
        d = d[d["is_starter"].astype(bool)]
    d["bucket"] = d["pitches_before"].map(bucket_of)
    labels = [b[2] for b in BUCKETS]
    out = []
    totals = d.groupby("pitcher").size()
    keep = [p for p, n in totals.items() if n >= min_pa]
    groups = [(p, d[d["pitcher"] == p]) for p in sorted(keep)] + [("ALL STARTERS", d)]
    for name, g in groups:
        overall = _rates(g)
        out.append({"pitcher": name, "bucket": "ALL", **overall, "verdict": ""})
        fade = None
        for label in labels:
            r = _rates(g[g["bucket"] == label])
            verdict = ""
            if r["pa"] >= MIN_BUCKET_PA and pd.notna(r["ops"]) and pd.notna(overall["ops"]):
                if r["ops"] >= overall["ops"] + FADE_MARGIN:
                    verdict = "worse"
                    if fade is None:
                        fade = label
            elif r["pa"] > 0 and r["pa"] < MIN_BUCKET_PA:
                verdict = "small sample"
            out.append({"pitcher": name, "bucket": label, **r, "verdict": verdict})
        out[-len(labels) - 1]["fade_bucket"] = fade or ""
    cols = ["pitcher", "bucket", "pa", "ops", "k_pct", "bb_pct", "hard_hit_pct", "verdict"]
    df = pd.DataFrame(out)
    fades = df.set_index("pitcher")["fade_bucket"].dropna().to_dict() if "fade_bucket" in df else {}
    return df[cols].round(3), fades


def by_inning(pas, only_starters=True, min_pa=60):
    """Per starter (and pooled): results by the game inning the batter came up in.
    innings = how many games he pitched in that inning (a pitcher pulled mid-inning
    still counts it), ra9 = runs allowed per 9 innings counting only that inning."""
    d = pas.copy()
    if only_starters:
        d = d[d["is_starter"].astype(bool)]
    d["inn"] = d["inning"].clip(upper=7).astype(int)
    totals = d.groupby("pitcher").size()
    keep = [p for p, n in totals.items() if n >= min_pa]
    groups = [(p, d[d["pitcher"] == p]) for p in sorted(keep)] + [("ALL STARTERS", d)]
    rows, fades = [], {}
    for name, g in groups:
        overall = _rates(g)["ops"]
        fade = None
        for inn in sorted(g["inn"].unique()):
            x = g[g["inn"] == inn]
            r = _rates(x)
            n_inn = x["game_pk"].nunique() if name != "ALL STARTERS" else len(x[["game_pk", "pitcher_id"]].drop_duplicates())
            ra9 = 9 * x["runs"].sum() / n_inn if n_inn else float("nan")
            verdict = ""
            if r["pa"] < MIN_INNING_PA:
                verdict = "small sample"
            elif pd.notna(r["ops"]) and pd.notna(overall) and r["ops"] >= overall + FADE_MARGIN:
                verdict = "worse"
                if fade is None and inn >= 4:
                    fade = int(inn)
            rows.append({"pitcher": name, "inning": "7+" if inn == 7 else int(inn), "innings": n_inn,
                         "pa": r["pa"], "ops": r["ops"], "k_pct": r["k_pct"], "bb_pct": r["bb_pct"],
                         "runs": int(x["runs"].sum()), "ra9": ra9, "verdict": verdict})
        fades[name] = fade
    return pd.DataFrame(rows).round(3), fades


def by_times_through(pas, only_starters=True):
    d = pas.copy()
    if only_starters:
        d = d[d["is_starter"].astype(bool)]
    d["trip"] = d["tto"].clip(upper=3).map({1: "1st time", 2: "2nd time", 3: "3rd+ time"})
    rows = [{"trip": k, **_rates(g)} for k, g in d.groupby("trip")]
    return pd.DataFrame(rows).round(3)


def pitches_per_start(pas):
    d = pas[pas["is_starter"].astype(bool)].copy()
    d["end"] = d["pitches_before"] + d["pitches_in_pa"]
    per = d.groupby(["pitcher", "game_pk"])["end"].max().reset_index()
    return (per.groupby("pitcher")["end"].agg(starts="count", avg_pitches="mean", max_pitches="max")
            .round(1).sort_values("starts", ascending=False))


def _fb(pitches):
    f = pitches[pitches["pitch_type"].isin(FASTBALLS)].copy()
    f["speed"] = pd.to_numeric(f["speed"], errors="coerce")
    return f.dropna(subset=["speed"])


def velocity_by_start(pitches):
    """One row per starter outing: fastball velo overall, first 25 pitches, last 25 pitches, change."""
    st = pitches[pitches["is_starter"].astype(bool)]
    fb = _fb(st)
    rows = []
    for (pid, pk), g in fb.groupby(["pitcher_id", "game_pk"]):
        allg = st[(st["pitcher_id"] == pid) & (st["game_pk"] == pk)]
        n = int(allg["pitch_no"].max())
        early = g[g["pitch_no"] <= 25]["speed"]
        late = g[g["pitch_no"] > n - 25]["speed"] if n > 50 else pd.Series(dtype=float)
        rows.append({
            "pitcher": g["pitcher"].iloc[0], "date": g["date"].iloc[0], "game_pk": pk,
            "pitches": n, "fb": len(g), "fb_avg": round(g["speed"].mean(), 1),
            "fb_max": round(g["speed"].max(), 1),
            "first25": round(early.mean(), 1) if len(early) >= 8 else float("nan"),
            "last25": round(late.mean(), 1) if len(late) >= 8 else float("nan"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["in_start_change"] = (out["last25"] - out["first25"]).round(1)
    out = out.sort_values(["pitcher", "date"]).reset_index(drop=True)
    base = out.groupby("pitcher")["fb_avg"].transform("median")
    out["vs_median"] = (out["fb_avg"] - base).round(1)
    # next-outing runs allowed in the starter's own game, joined later in report
    return out


def velocity_buckets(pitches, pas, only_starters=True):
    """Fastball velo by pitch-count bucket per starter, and in-start drop vs first 25 pitches."""
    st = pitches[pitches["is_starter"].astype(bool)] if only_starters else pitches
    fb = _fb(st)
    rows = []
    for pitcher, g in fb.groupby("pitcher"):
        base = g[g["pitch_no"] <= 25]["speed"]
        r = {"pitcher": pitcher}
        for lo, hi, label in BUCKETS:
            b = g[(g["pitch_no"] > lo) & (g["pitch_no"] <= hi)]["speed"]
            r[label] = round(b.mean(), 1) if len(b) >= MIN_BUCKET_FB else float("nan")
        r["fade"] = ""
        if len(base) >= MIN_BUCKET_FB:
            b0 = base.mean()
            for lo, hi, label in BUCKETS[1:]:
                v = r[label]
                if v == v and b0 - v >= VELO_DROP_MPH:
                    r["fade"] = f"down {b0 - v:.1f} mph by {label}"
                    break
        rows.append(r)
    return pd.DataFrame(rows)


def velocity_vs_results(starts, pas):
    """Does a start with lower-than-usual velocity go worse? Joins per-start velo to runs / OPS allowed."""
    st = pas[pas["is_starter"].astype(bool)]
    res = []
    for (pk, pid), g in st.groupby(["game_pk", "pitcher_id"]):
        ab, h = g["is_ab"].sum(), g["hit_bases"].gt(0).sum()
        tb = g["hit_bases"].sum()
        obp_d = ab + g["is_walk"].sum() + g["is_hbp"].sum() + g["is_sac_fly"].sum()
        obp = (h + g["is_walk"].sum() + g["is_hbp"].sum()) / obp_d if obp_d else 0
        res.append({"game_pk": pk, "pitcher_id": pid, "runs": g["runs"].sum() if "runs" in g else 0,
                    "pa": len(g), "ops": obp + (tb / ab if ab else 0)})
    r = pd.DataFrame(res)
    if r.empty or starts.empty:
        return pd.DataFrame()
    ids = pas[["pitcher_id", "pitcher"]].drop_duplicates()
    r = r.merge(ids, on="pitcher_id")
    # per-start runs = runs scored while he faced those batters (sum of per-PA runs)
    m = starts.merge(r[["game_pk", "pitcher", "runs", "pa", "ops"]], on=["game_pk", "pitcher"])
    m["velo_state"] = pd.cut(m["vs_median"], [-99, -VELO_DROP_MPH, VELO_DROP_MPH, 99],
                             labels=["down", "normal", "up"], right=False)
    # normalise runs by batters faced so short outings are comparable
    m["runs_per_27"] = (m["runs"] / m["pa"] * 27).round(2)
    return m


def monthly_velocity(pitches):
    st = _fb(pitches[pitches["is_starter"].astype(bool)]).copy()
    st["month"] = st["date"].astype(str).str[:7]
    t = st.groupby(["pitcher", "month"])["speed"].agg(fb="count", avg="mean").reset_index()
    t["avg"] = t["avg"].round(1)
    t = t[t["fb"] >= MIN_MONTH_FB]
    return t.pivot(index="pitcher", columns="month", values="avg")


def velocity_report(pitches, pas, export_dir=None, season=2026):
    if pitches is None or pitches.empty or _fb(pitches).empty:
        print("\nNo pitch speeds found in the feed -- skipping the velocity section.")
        return None
    starts = velocity_by_start(pitches)
    print("\n== Fastball velocity by month (starters, mph, months with %d+ fastballs) ==" % MIN_MONTH_FB)
    print(monthly_velocity(pitches).to_string())
    print("\n== Fastball velocity by pitch count (first 25 pitches is his own baseline) ==")
    vb = velocity_buckets(pitches, pas)
    print(vb.to_string(index=False))
    print(f"\n== Every start: fastball avg, first 25 vs last 25 pitches (vs_median = against his own typical start) ==")
    for pitcher, g in starts.groupby("pitcher"):
        print(f"\n{pitcher}")
        print(g[["date", "pitches", "fb_avg", "fb_max", "first25", "last25", "in_start_change", "vs_median"]]
              .to_string(index=False))
    m = velocity_vs_results(starts, pas)
    if not m.empty:
        print(f"\n== Do low-velocity starts go worse? (down = {VELO_DROP_MPH} mph or more below his own median start) ==")
        t = (m.groupby("velo_state", observed=True)
               .agg(starts=("game_pk", "count"), pa=("pa", "sum"), runs=("runs", "sum"))
               .reset_index())
        t["runs_per_27_pa"] = (t["runs"] / t["pa"] * 27).round(2)
        print(t.to_string(index=False))
        print("  (a lower-velocity start can also be a bad start for other reasons -- this is a pattern, not proof of cause)")
        # within-start: when velo fell >=1 mph from first 25 to last 25, how did the late innings go?
        drop = m[m["in_start_change"] <= -VELO_DROP_MPH]
        print(f"  starts where velo fell {VELO_DROP_MPH}+ mph from the first to the last 25 pitches: {len(drop)} of {m['in_start_change'].notna().sum()}")
    if export_dir:
        starts.to_csv(os.path.join(export_dir, fname(START_FILE, season)), index=False)
    return starts


def report(pas, min_pa=60):
    if pas.empty:
        print("No plate appearances found.")
        return None
    summ, fades = summarize(pas, min_pa)
    print(f"\n{pas['game_pk'].nunique()} games, {len(pas)} plate appearances by Mariners pitchers")
    print("\n== Pitches thrown per start ==")
    print(pitches_per_start(pas).to_string())
    inn, inn_fades = by_inning(pas, min_pa=min_pa)
    print("\n== Results by inning (starters with %d+ PA; ra9 = runs allowed per 9 innings in that inning) ==" % min_pa)
    print(inn.to_string(index=False))
    print(f"\n== First inning from 4 on where a starter is {FADE_MARGIN:.3f} OPS worse than his own average "
          f"({MIN_INNING_PA}+ batters) ==")
    for p, f in inn_fades.items():
        print(f"  {p:<24} {('inning ' + str(f)) if f else 'none'}")
    print("\n== All starters: results by times through the order ==")
    print(by_times_through(pas).to_string(index=False))
    print(f"\n== Opponent OPS by pitch count (starters with {min_pa}+ PA) ==")
    print(summ.to_string(index=False))
    print("\n== Where each starter looks worse than his own average "
          f"(OPS +{FADE_MARGIN:.3f}, {MIN_BUCKET_PA}+ PA in the bucket) ==")
    for p, f in fades.items():
        print(f"  {p:<24} {('from ' + f + ' pitches') if f else 'no clear fade point'}")
    return summ, inn


def run(season=2026, export_dir=EXPORTS, get=_get, refresh=False, min_pa=60, pause=PAUSE):
    pas, pitches = collect(season, export_dir, get, refresh, pause)
    res = report(pas, min_pa)
    velocity_report(pitches, pas, export_dir, season)
    if res is not None:
        summ, inn = res
        summ.to_csv(os.path.join(export_dir, fname(SUMMARY_FILE, season)), index=False)
        inn.to_csv(os.path.join(export_dir, fname(INNING_FILE, season)), index=False)
        print(f"\nWrote {fname(SUMMARY_FILE, season)}, {fname(INNING_FILE, season)} and {fname(START_FILE, season)} in {export_dir}")
    return pas, pitches


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true", help="re-download every game")
    ap.add_argument("--min-pa", type=int, default=60)
    a = ap.parse_args()
    try:
        run(a.season, refresh=a.refresh, min_pa=a.min_pa)
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}")
        sys.exit(1)