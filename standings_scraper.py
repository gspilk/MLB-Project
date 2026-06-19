"""
standings_scraper.py
Scrapes both standings tables from baseball-reference.com:
  1. Division standings  (AL/NL x East/Central/West)
  2. Expanded standings  (all 30 teams, luck, SOS, splits, recent form)

bbref uses duplicate IDs: standings_E / standings_C / standings_W appear
twice on the page — first occurrence = AL, second = NL.

Usage:
    from standings_scraper import get_standings
    division, expanded = get_standings(2026)
"""

# # other scrapers can do:
#from standings_scraper import get_standings, get_mariners_context, get_all_teams

import os
import io
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ── cache config ──────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
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
    """Wraps pd.read_html to avoid FutureWarning on literal strings."""
    return pd.read_html(io.StringIO(str(tag)))[0]


# ── fetch raw html ────────────────────────────────────────────────────────────
def _fetch_page(season: int) -> BeautifulSoup:
    url = f"https://www.baseball-reference.com/leagues/majors/{season}-standings.shtml"
    print(f"  [GET] {url}")
    session = _make_session()
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    time.sleep(3)  # respect bbref rate limit — do not remove
    return BeautifulSoup(resp.text, "html.parser")


# ── table 1: division standings ───────────────────────────────────────────────
DIVISION_ORDER = [
    "AL_East", "AL_Central", "AL_West",
    "NL_East", "NL_Central", "NL_West",
]

SUFFIX_MAP = {
    "E": ["AL_East",    "NL_East"],
    "C": ["AL_Central", "NL_Central"],
    "W": ["AL_West",    "NL_West"],
}


def _parse_division_standings(soup: BeautifulSoup) -> dict:
    tables = {}
    seen = {"E": 0, "C": 0, "W": 0}

    for tag in soup.find_all("table"):
        tid = tag.get("id", "")
        for suffix in ("E", "C", "W"):
            if tid == f"standings_{suffix}":
                idx      = seen[suffix]
                division = SUFFIX_MAP[suffix][idx]
                seen[suffix] += 1

                df = _read_html(tag)
                df = df[df["Tm"].notna() & (df["Tm"] != "Tm")]
                df["division"] = division
                df["league"]   = division.split("_")[0]
                tables[division] = df.reset_index(drop=True)
                print(f"  [ok]   {division}: {len(df)} teams")

    for div in DIVISION_ORDER:
        if div not in tables:
            print(f"  [warn] missing: {div}")

    return tables


# ── table 2: expanded standings (hidden in HTML comment) ─────────────────────
def _parse_expanded_standings(soup: BeautifulSoup):
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    for comment in comments:
        if "expanded_standings_overall" not in comment:
            continue
        c_soup = BeautifulSoup(comment, "html.parser")
        tag = c_soup.find("table", {"id": "expanded_standings_overall"})
        if tag is None:
            continue

        df = _read_html(tag)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(c) for c in col).strip("_")
                for col in df.columns
            ]

        rk_col = "Rk" if "Rk" in df.columns else df.columns[0]
        df = df[pd.to_numeric(df[rk_col], errors="coerce").notna()]
        df[rk_col] = df[rk_col].astype(int)
        df = df.reset_index(drop=True)
        print(f"  [ok]   expanded standings: {len(df)} teams, {len(df.columns)} cols")
        return df

    print("  [warn] expanded_standings_overall not found in comments")
    return None


# ── public helpers ────────────────────────────────────────────────────────────
def get_al_west(division_tables: dict) -> pd.DataFrame:
    return division_tables.get("AL_West", pd.DataFrame())


def get_nl_west(division_tables: dict) -> pd.DataFrame:
    return division_tables.get("NL_West", pd.DataFrame())


def get_team_row(expanded_df, team: str = "Seattle Mariners"):
    if expanded_df is None or expanded_df.empty:
        return None
    matches = expanded_df[expanded_df["Tm"] == team]
    if matches.empty:
        print(f"  [warn] '{team}' not found in expanded standings")
        return None
    return matches.iloc[0]


def get_all_teams(division_tables: dict) -> pd.DataFrame:
    """All 30 teams across all 6 divisions, sorted by W-L%."""
    frames = [division_tables[d] for d in DIVISION_ORDER if d in division_tables]
    if not frames:
        return pd.DataFrame()
    all_teams = pd.concat(frames, ignore_index=True)
    for col in ["W-L%", "W", "L", "GB"]:
        all_teams[col] = pd.to_numeric(all_teams[col], errors="coerce")
    return all_teams.sort_values("W-L%", ascending=False).reset_index(drop=True)


def get_playoff_picture(division_tables: dict) -> pd.DataFrame:
    """All AL teams sorted by W-L% — wild card race view."""
    al_divs = ["AL_East", "AL_Central", "AL_West"]
    frames  = [division_tables[d] for d in al_divs if d in division_tables]
    if not frames:
        return pd.DataFrame()
    al = pd.concat(frames, ignore_index=True)
    al["W-L%"] = pd.to_numeric(al["W-L%"], errors="coerce")
    return al.sort_values("W-L%", ascending=False).reset_index(drop=True)


def get_mariners_context(division_tables: dict, expanded_df) -> dict:
    """
    Returns a dict with:
      mlb_rank     — Mariners rank among all 30 teams
      div_rank     — rank in AL West
      wc_gap       — W-L% gap vs 3rd wild card (negative = behind)
      wc_in_reach  — True if within ~6 games
      expanded_row — full row from expanded standings
    """
    all_teams = get_all_teams(division_tables)
    mlb_rank  = all_teams[all_teams["Tm"] == "Seattle Mariners"].index
    mlb_rank  = int(mlb_rank[0]) + 1 if len(mlb_rank) else None

    al_west  = get_al_west(division_tables).reset_index(drop=True)
    div_rank = al_west[al_west["Tm"] == "Seattle Mariners"].index
    div_rank = int(div_rank[0]) + 1 if len(div_rank) else None

    al_pic    = get_playoff_picture(division_tables)
    wc_cutoff = al_pic.iloc[5]["W-L%"] if len(al_pic) >= 6 else None
    sea_row   = al_pic[al_pic["Tm"] == "Seattle Mariners"]
    sea_pct   = float(sea_row["W-L%"].values[0]) if not sea_row.empty else None
    wc_gap    = round(sea_pct - wc_cutoff, 3) if sea_pct and wc_cutoff else None

    return {
        "mlb_rank":     mlb_rank,
        "div_rank":     div_rank,
        "wc_gap":       wc_gap,
        "wc_in_reach":  wc_gap is not None and wc_gap > -0.060,
        "expanded_row": get_team_row(expanded_df, "Seattle Mariners"),
    }


def get_category_leaders(expanded_df, team: str = "Seattle Mariners") -> pd.DataFrame:
    """
    For every numeric column in the expanded standings, show:
      - the MLB leader (team + value)
      - the Mariners value
      - Mariners rank out of 30
      - gap between Mariners and the leader

    Higher = better for most stats (R, W-L%, pythWL wins, SRS, Luck,
    last10/20/30 wins). Lower = better for RA.
    Returns a tidy DataFrame sorted by Mariners rank (worst first)
    so the biggest problem areas surface at the top.
    """
    if expanded_df is None or expanded_df.empty:
        print("  [warn] no expanded data for category leaders")
        return pd.DataFrame()

    # columns where LOWER is better (want to be near bottom of leaderboard)
    lower_is_better = {"RA", "ExInn_L", "1Run_L"}

    # numeric columns only, skip Rk
    num_cols = [
        c for c in expanded_df.columns
        if c not in ("Rk", "Tm", "Strk", "pythWL",
                     "vEast", "vCent", "vWest", "Inter",
                     "Home", "Road", "vRHP", "vLHP",
                     "≥.500", "<.500")
        and pd.to_numeric(expanded_df[c], errors="coerce").notna().any()
    ]

    sea = expanded_df[expanded_df["Tm"] == team]
    if sea.empty:
        print(f"  [warn] '{team}' not found")
        return pd.DataFrame()

    rows = []
    for col in num_cols:
        series = pd.to_numeric(expanded_df[col], errors="coerce")
        sea_val = pd.to_numeric(sea[col].values[0], errors="coerce")
        if pd.isna(sea_val):
            continue

        ascending = col in lower_is_better
        ranked    = series.rank(ascending=ascending, method="min")
        sea_rank  = int(ranked[sea.index[0]])

        if ascending:
            # lower is better → leader has smallest value
            leader_idx = series.idxmin()
        else:
            leader_idx = series.idxmax()

        leader_tm  = expanded_df.loc[leader_idx, "Tm"]
        leader_val = series[leader_idx]
        gap        = round(sea_val - leader_val, 3)

        rows.append({
            "category":    col,
            "leader_team": leader_tm,
            "leader_val":  leader_val,
            "sea_val":     sea_val,
            "sea_rank":    sea_rank,
            "gap":         gap,          # negative = behind leader
        })

    df = pd.DataFrame(rows)
    # sort worst rank first so problem areas are at top
    df = df.sort_values("sea_rank", ascending=False).reset_index(drop=True)
    return df


def print_mariners_vs_leaders(expanded_df, team: str = "Seattle Mariners"):
    """Pretty-prints the category leaders comparison."""
    df = get_category_leaders(expanded_df, team)
    if df.empty:
        return

    print(f"\n{'category':<10} {'leader':<22} {'leader val':>10} "
          f"{'SEA val':>9} {'SEA rank':>9} {'gap':>8}")
    print("─" * 75)
    for _, row in df.iterrows():
        rank_str = f"{int(row['sea_rank'])}/30"
        print(f"{row['category']:<10} {row['leader_team']:<22} "
              f"{row['leader_val']:>10.3f} {row['sea_val']:>9.3f} "
              f"{rank_str:>9} {row['gap']:>8.3f}")


# ── main public function ──────────────────────────────────────────────────────
def get_standings(season: int = 2026, force_refresh: bool = False):
    """
    Returns:
        division_tables  dict of 6 DataFrames keyed by division name
        expanded_df      single DataFrame, all 30 teams + advanced splits
    """
    expanded_cache = _cache_path(f"standings_expanded_{season}")
    division_cache = _cache_path(f"standings_division_{season}")

    if (not force_refresh
            and not _is_stale(expanded_cache)
            and not _is_stale(division_cache)):
        print("[cache] loading standings from disk")
        expanded_df     = pd.read_parquet(expanded_cache)
        combined        = pd.read_parquet(division_cache)
        division_tables = {
            d: grp.reset_index(drop=True)
            for d, grp in combined.groupby("division")
        }
        return division_tables, expanded_df

    print(f"[fetch] standings {season} ...")
    soup = _fetch_page(season)

    division_tables = _parse_division_standings(soup)
    expanded_df     = _parse_expanded_standings(soup)

    if division_tables:
        pd.concat(
            division_tables.values(), ignore_index=True
        ).to_parquet(division_cache, index=False)
    if expanded_df is not None:
        expanded_df.to_parquet(expanded_cache, index=False)

    return division_tables, expanded_df


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    division, expanded = get_standings(2026, force_refresh=True)

    print("\n── all 30 teams (sorted by W-L%) ──")
    all30 = get_all_teams(division)
    print(all30[["Tm","W","L","W-L%","division","league"]].to_string(index=False)
          if not all30.empty else "  no data")

    print("\n── AL West ──")
    al_west = get_al_west(division)
    print(al_west[["Tm","W","L","W-L%","GB"]].to_string(index=False)
          if not al_west.empty else "  no data")

    print("\n── Mariners context ──")
    ctx = get_mariners_context(division, expanded)
    print(f"  MLB rank:    {ctx['mlb_rank']} of 30")
    print(f"  Division:    {ctx['div_rank']} of 5 in AL West")
    print(f"  WC gap:      {ctx['wc_gap']} (negative = behind wild card)")
    print(f"  WC in reach: {ctx['wc_in_reach']}")
    if ctx["expanded_row"] is not None:
        print("\n── Mariners expanded row ──")
        print(ctx["expanded_row"].to_string())
    else:
        print("  expanded row: not available")

    print("\n── Mariners vs MLB leaders (worst categories first) ──")
    print_mariners_vs_leaders(expanded)