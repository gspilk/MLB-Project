"""
schedule_scraper.py
Builds the Mariners schedule + results from the official MLB Stats API
(statsapi.mlb.com) instead of scraping the old source.

Produces three outputs:
  1. completed  — all games played so far (results, runs, W/L, streak)
  2. next7      — next 7 upcoming games as a checklist
  3. remaining  — all upcoming games AFTER those next 7 (NOT the full
                 remaining schedule by itself -- see get_schedule()'s
                 docstring below for why callers need both pieces)

Columns match the old source table: Gm#, Date, Tm, H/A, Opp
(3-letter code), W/L, R, RA, Inn, W-L, Streak, D/N, plus date, home_away,
result, played, checked. Not available from MLB's schedule data (left
empty): Rank, GB, Win/Loss/Save pitchers, Attendance, cLI.

Auto-refreshes at midnight daily via Windows Task Scheduler.

Usage:
    from mariners_schedule_scraper import get_schedule
    completed, next7, remaining = get_schedule(2026)
"""

import os
import time
import datetime
import pandas as pd

import mlb_api

# ── config ────────────────────────────────────────────────────────────────────
TEAM       = "SEA"
TEAM_ID    = mlb_api.SEA_TEAM_ID
SEASON     = 2026
CACHE_DIR  = os.path.join(os.path.dirname(__file__), "data", "cache")
# refresh once per day — Task Scheduler runs this at midnight
CACHE_TTL_HOURS = 24


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours > CACHE_TTL_HOURS


# ── build the schedule table ─────────────────────────────────────────────────
def _build_schedule(season: int, force_refresh: bool = False) -> pd.DataFrame:
    games = mlb_api.get_season_games(season, force_refresh=force_refresh)
    if games.empty:
        print("  [warn] no games returned from the MLB Stats API")
        return pd.DataFrame()

    abbr = mlb_api.get_teams(season).set_index("team_id")["abbr"].to_dict()
    mine = games[(games["home_id"] == TEAM_ID) | (games["away_id"] == TEAM_ID)].copy()
    mine = mine.sort_values(["date", "game_time"]).reset_index(drop=True)

    rows = []
    wins = losses = 0
    streak = 0
    for i, g in enumerate(mine.itertuples(), 1):
        home = g.home_id == TEAM_ID
        opp_id = g.away_id if home else g.home_id
        row = {
            "Gm#": i,
            "Date": f"{pd.Timestamp(g.date):%A, %b} {pd.Timestamp(g.date).day}",
            "Tm": TEAM,
            "H/A": "" if home else "@",
            "Opp": abbr.get(opp_id, str(opp_id)),
            "W/L": None, "R": None, "RA": None, "Inn": None,
            "W-L": None, "Rank": None, "GB": None,
            "Win": None, "Loss": None, "Save": None,
            "D/N": "D" if g.day_night == "day" else ("N" if g.day_night == "night" else None),
            "Attendance": None, "Streak": None,
            "date": pd.Timestamp(g.date),
            "home_away": "home" if home else "away",
        }
        if g.played:
            r = g.home_runs if home else g.away_runs
            ra = g.away_runs if home else g.home_runs
            won = r > ra
            wins, losses = wins + won, losses + (not won)
            streak = (streak + 1 if streak >= 0 else 1) if won else (streak - 1 if streak <= 0 else -1)
            row.update({
                "W/L": "W" if won else "L",
                "R": r, "RA": ra,
                "Inn": int(g.innings) if g.innings and g.innings > 9 else None,
                "W-L": f"{wins}-{losses}",
                "Streak": ("+" * streak) if streak > 0 else ("-" * -streak),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["R", "RA", "Rank", "GB"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["result"] = df["W/L"].apply(
        lambda x: "W" if str(x).startswith("W") else ("L" if str(x).startswith("L") else None))
    df["played"] = df["R"].notna()
    df["checked"] = False   # UI will toggle this

    print(f"  [ok]   schedule: {len(df)} total games")
    return df


# ── split into completed / next7 / remaining ─────────────────────────────────
def _split_schedule(df: pd.DataFrame):
    today     = pd.Timestamp(datetime.date.today())
    completed = df[df["played"]].copy()
    upcoming  = df[~df["played"]].copy()

    # sort upcoming by date
    upcoming  = upcoming.sort_values("date").reset_index(drop=True)
    next7     = upcoming.head(7).copy()
    remaining = upcoming.iloc[7:].copy()

    print(f"  [ok]   completed: {len(completed)} games")
    print(f"  [ok]   next 7:    {len(next7)} games")
    print(f"  [ok]   remaining: {len(remaining)} games")
    return completed, next7, remaining


# ── display helpers ───────────────────────────────────────────────────────────
DISPLAY_COLS_COMPLETED = [
    "Gm#", "date", "home_away", "Opp", "result", "R", "RA", "W-L", "Streak"
]

DISPLAY_COLS_UPCOMING = [
    "Gm#", "date", "home_away", "Opp", "checked"
]


def print_next7(next7: pd.DataFrame):
    """Prints next 7 games as a checklist."""
    print(f"\n{'#':<4} {'date':<14} {'H/A':<6} {'opponent':<25} {'done?'}")
    print("─" * 58)
    for _, row in next7.iterrows():
        date_str = row["date"].strftime("%a %b %d") if pd.notna(row["date"]) else "TBD"
        checked  = "☑" if row["checked"] else "☐"
        ha       = row.get("home_away", "")
        opp      = row.get("Opp", "")
        gm       = row.get("Gm#", "")
        print(f"{gm:<4} {date_str:<14} {ha:<6} {opp:<25} {checked}")


def print_completed_summary(completed: pd.DataFrame):
    """Last 10 completed games."""
    last10 = completed.tail(10)
    cols   = [c for c in DISPLAY_COLS_COMPLETED if c in last10.columns]
    print(last10[cols].to_string(index=False))


def get_remaining_by_opponent(remaining: pd.DataFrame) -> pd.DataFrame:
    """How many games left vs each opponent — useful for SOS analysis.

    CAVEAT: pass in the combined next7+remaining schedule if you want
    a true count -- `remaining` alone excludes the next 7 games (see
    get_schedule()'s docstring). Not currently called anywhere in the
    real pipeline, but if you wire this in later, don't repeat the
    same undercounting bug fixed elsewhere in the project on 2026-08-26.
    """
    if remaining.empty:
        return pd.DataFrame()
    return (
        remaining.groupby("Opp")
        .size()
        .reset_index(name="games_remaining")
        .sort_values("games_remaining", ascending=False)
    )


def get_home_away_remaining(remaining: pd.DataFrame) -> dict:
    """Split of remaining games home vs away."""
    if remaining.empty:
        return {}
    counts = remaining["home_away"].value_counts().to_dict()
    return {
        "home":  counts.get("home", 0),
        "away":  counts.get("away", 0),
        "total": len(remaining),
    }


# ── main public function ──────────────────────────────────────────────────────
def get_schedule(season: int = 2026, force_refresh: bool = False):
    """
    Returns:
        completed   DataFrame — all games played so far
        next7       DataFrame — next 7 upcoming games (checklist)
        remaining   DataFrame — upcoming games AFTER those next 7

    IMPORTANT for callers: `remaining` alone is NOT the full remaining
    schedule -- it deliberately excludes the next 7 games (see
    _split_schedule()). This caused a real bug elsewhere in the project
    (simulator.py/monte_carlo.py were using `remaining` alone for
    schedule-difficulty and Monte Carlo calculations, silently missing
    a quarter of the real remaining games) before it got fixed on
    2026-08-26. If you need the TRUE full remaining schedule, combine
    next7 + remaining yourself, e.g.:
        import pandas as pd
        full_remaining = pd.concat([next7, remaining], ignore_index=True)

    Refreshes daily at midnight when run via Task Scheduler.
    Pass force_refresh=True to bypass cache.
    """
    completed_cache = _cache_path(f"schedule_completed_{season}")
    next7_cache     = _cache_path(f"schedule_next7_{season}")
    remaining_cache = _cache_path(f"schedule_remaining_{season}")

    all_cached = all(
        os.path.exists(p)
        for p in [completed_cache, next7_cache, remaining_cache]
    )

    if not force_refresh and all_cached and not _is_stale(completed_cache):
        print("[cache] loading schedule from disk")
        return (
            pd.read_parquet(completed_cache),
            pd.read_parquet(next7_cache),
            pd.read_parquet(remaining_cache),
        )

    print(f"[fetch] schedule {season} (MLB Stats API) ...")
    df        = _build_schedule(season, force_refresh)
    completed, next7, remaining = _split_schedule(df)

    # cache all three
    completed.to_parquet(completed_cache, index=False)
    next7.to_parquet(next7_cache,         index=False)
    remaining.to_parquet(remaining_cache, index=False)

    return completed, next7, remaining


# ── windows task scheduler setup instructions ─────────────────────────────────
SCHEDULER_INSTRUCTIONS = """
To run this automatically at midnight every day on Windows:

1. Open Task Scheduler (search in Start Menu)
2. Click 'Create Basic Task'
3. Name it: MLB Schedule Refresh
4. Trigger: Daily at 12:00 AM
5. Action: Start a program
   Program: C:/Users/geoff/AppData/Local/Microsoft/WindowsApps/python3.11.exe
   Arguments: "c:/Users/geoff/OneDrive/Desktop/MLB PROJECT/mariners_schedule_scraper.py"
6. Finish

The script will run at midnight, re-fetch the schedule, update the cache,
and the next time you run main.py it will load fresh data automatically.
"""


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    completed, next7, remaining = get_schedule(2026, force_refresh=True)

    print("\n── last 10 completed games ──")
    print_completed_summary(completed)

    print("\n── next 7 games (checklist) ──")
    print_next7(next7)

    print(f"\n── remaining schedule ({len(remaining)} games) ──")
    by_opp = get_remaining_by_opponent(remaining)
    print(by_opp.to_string(index=False) if not by_opp.empty else "  no data")

    print("\n── home/away split (remaining) ──")
    ha = get_home_away_remaining(remaining)
    print(f"  home:  {ha.get('home', 0)}")
    print(f"  away:  {ha.get('away', 0)}")
    print(f"  total: {ha.get('total', 0)}")

    print("\n── win/loss record so far ──")
    if not completed.empty and "result" in completed.columns:
        w = (completed["result"] == "W").sum()
        l = (completed["result"] == "L").sum()
        print(f"  W: {w}  L: {l}  ({w/(w+l):.3f})")

    print(SCHEDULER_INSTRUCTIONS)