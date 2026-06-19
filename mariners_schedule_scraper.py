"""
schedule_scraper.py
Scrapes the Mariners schedule + scores page from baseball-reference.com:
  https://www.baseball-reference.com/teams/SEA/2026-schedule-scores.shtml

Produces three outputs:
  1. completed  — all games played so far (results, runs, W/L, streak)
  2. next7      — next 7 upcoming games as a checklist
  3. remaining  — all 104 remaining games after the next 7

Auto-refreshes at midnight daily via Windows Task Scheduler.

Usage:
    from schedule_scraper import get_schedule
    completed, next7, remaining = get_schedule(2026)
"""

import os
import io
import time
import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ── config ────────────────────────────────────────────────────────────────────
TEAM       = "SEA"
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


# ── http session ──────────────────────────────────────────────────────────────
def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.baseball-reference.com/",
    })
    return session


def _read_html(tag) -> pd.DataFrame:
    return pd.read_html(io.StringIO(str(tag)))[0]


# ── fetch page ────────────────────────────────────────────────────────────────
def _fetch_page(season: int) -> BeautifulSoup:
    url = (f"https://www.baseball-reference.com/teams/"
           f"{TEAM}/{season}-schedule-scores.shtml")
    print(f"  [GET] {url}")
    session = _make_session()
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    time.sleep(3)  # respect bbref rate limit
    return BeautifulSoup(resp.text, "html.parser")


# ── find table (live html or comment) ────────────────────────────────────────
def _find_table(soup: BeautifulSoup, table_id: str):
    tag = soup.find("table", {"id": table_id})
    if tag:
        return tag
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id not in comment:
            continue
        c_soup = BeautifulSoup(comment, "html.parser")
        tag = c_soup.find("table", {"id": table_id})
        if tag:
            return tag
    return None


# ── parse + clean the schedule table ─────────────────────────────────────────
def _parse_schedule(soup: BeautifulSoup) -> pd.DataFrame:
    """
    bbref schedule table id is 'team_schedule' on team pages.
    Columns include: Gm#, Date, Tm, H/A, Opp, W/L, R, RA, Inn,
                     W-L, Rank, GB, Win, Loss, Save, Time, D/N, Attendance, Streak, cLI
    """
    tag = _find_table(soup, "team_schedule")
    if tag is None:
        print("  [warn] team_schedule table not found")
        return pd.DataFrame()

    df = _read_html(tag)

    # drop repeated header rows bbref injects
    df = df[df["Gm#"].apply(
        lambda x: str(x).isdigit()
    )].copy()
    df["Gm#"] = df["Gm#"].astype(int)

    # normalize date — bbref format: "Thursday, Apr 2"
    # strip day-of-week, add season year
    def _parse_date(raw):
        try:
            raw = str(raw).split(",")[-1].strip()  # "Apr 2"
            raw = raw.replace(" (1)", "").replace(" (2)", "").strip()
            return pd.to_datetime(f"{raw} {SEASON}", format="%b %d %Y")
        except Exception:
            return pd.NaT

    df["date"] = df["Date"].apply(_parse_date)

    # home/away flag — bbref uses "@" for away games in the H/A col
    home_away_col = next(
        (c for c in df.columns if c in ("Unnamed: 4", "H/A", "")), None
    )
    if home_away_col:
        df["home_away"] = df[home_away_col].apply(
            lambda x: "away" if str(x).strip() == "@" else "home"
        )
    else:
        df["home_away"] = "home"

    # numeric columns
    for col in ["R", "RA", "Rank", "GB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # result flag — played games have W or L (or W-wo, L-wo etc)
    if "W/L" in df.columns:
        df["result"] = df["W/L"].apply(
            lambda x: "W" if str(x).startswith("W")
            else ("L" if str(x).startswith("L") else None)
        )
    else:
        df["result"] = None

    # played = has a run total
    df["played"] = df["R"].notna()

    # checklist status for upcoming games
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
    """How many games left vs each opponent — useful for SOS analysis."""
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
        remaining   DataFrame — remaining 104 games after next 7

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

    print(f"[fetch] schedule {season} ...")
    soup      = _fetch_page(season)
    df        = _parse_schedule(soup)
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
   Arguments: "c:/Users/geoff/OneDrive/Desktop/MLB PROJECT/schedule_scraper.py"
6. Finish

The script will run at midnight, re-fetch the page, update the cache,
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