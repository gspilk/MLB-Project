"""
batting_pitching_scraper.py
Scrapes MLB batting, pitching, AND fielding leaders from
baseball-reference.com for both AL and NL leagues.

Pages:
  https://www.baseball-reference.com/leagues/AL/2026-standard-batting.shtml
  https://www.baseball-reference.com/leagues/NL/2026-standard-batting.shtml
  https://www.baseball-reference.com/leagues/AL/2026-standard-pitching.shtml
  https://www.baseball-reference.com/leagues/NL/2026-standard-pitching.shtml
  https://www.baseball-reference.com/leagues/AL/2026-standard-fielding.shtml
  https://www.baseball-reference.com/leagues/NL/2026-standard-fielding.shtml

Each page has two tables (batting/pitching in live HTML; fielding, per
seattle_scraper.py's team-level equivalent, IN COMMENTS -- _find_table()
below already checks both, so this works either way without needing to
know in advance which one bbref uses for this specific page):
  teams_standard_{stat}    ← team totals
  players_standard_{stat}  ← individual stats

FIELDING ADDED 2026-09-28: same URL/table-id pattern as batting/pitching
(bbref uses one consistent "standard-{stat}" naming scheme across all
three), added to close a real gap flagged in
free_agent_recommendations.py -- external free agents had zero defensive
data anywhere in this pipeline, only internal Mariners players did (via
seattle_scraper.py's TEAM-level fielding table). This is the LEAGUE-wide
version of that same table, matched into the free agent pool the same
way batting/pitching stats already are.

NEEDS A REAL TEST RUN: get_batting()/get_pitching() are both proven
working in real runs. get_fielding() reuses the exact same
_fetch_page/_parse_page/_clean plumbing, pointed at "fielding" as the
stat_type -- the team-level fielding table (seattle_scraper.py) is
confirmed to have Name/Pos/Inn/Rtot columns and to live in HTML
comments, and this project's _find_table() already handles comment-
wrapped tables automatically, so this SHOULD work the same way at league
scale. But the league-wide version of this specific page has never
actually been fetched by this project before, so the real column set
(Rtot vs. Rdrs, whether "Pos" comes back as one combined multi-position
string like "65/HD" the same way it does at team level, whether there's
a "teams_standard_fielding" totals table at all) needs confirming
against a real run, the same honest caveat every other new scraper in
this project has carried before its first live test.

Usage:
    from batting_pitching_scraper import get_batting, get_pitching, get_fielding
    bat = get_batting(2026)
    pit = get_pitching(2026)
    fld = get_fielding(2026)
"""

import os
import io
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ── config ────────────────────────────────────────────────────────────────────
LEAGUES         = ["AL", "NL"]
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
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
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer":         "https://www.baseball-reference.com/",
    })
    return session


def _read_html(tag) -> pd.DataFrame:
    return pd.read_html(io.StringIO(str(tag)))[0]


# ── fetch page ────────────────────────────────────────────────────────────────
def _fetch_page(league: str, stat_type: str, season: int) -> BeautifulSoup:
    url = (f"https://www.baseball-reference.com/leagues/"
           f"{league}/{season}-standard-{stat_type}.shtml")
    print(f"  [GET] {url}")
    session = _make_session()
    resp    = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)  # respect bbref rate limit
    return BeautifulSoup(resp.text, "html.parser")


# ── find table ────────────────────────────────────────────────────────────────
def _find_table(soup: BeautifulSoup, table_id: str):
    tag = soup.find("table", {"id": table_id})
    if tag:
        return tag
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id not in comment:
            continue
        c_soup = BeautifulSoup(comment, "html.parser")
        tag    = c_soup.find("table", {"id": table_id})
        if tag:
            return tag
    return None


# ── flatten grouped/MultiIndex headers ────────────────────────────────────────
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUG FIX 2026-09-28: the league-wide FIELDING page (unlike batting/
    pitching, both flat single-level headers) has a real, GROUPED header
    -- columns organized under "Standard" / "Total Zone" / "DRS" /
    "Range Factor" / "Baserunners" section labels -- confirmed directly
    via a live debug run (columns came back as literal tuples like
    ('Total Zone', 'Rtot') instead of plain 'Rtot'). This is the exact
    same real bug seattle_scraper.py already hit and fixed for the
    TEAM-level fielding table -- ported here unchanged, since it's the
    same underlying bbref page template at a different scope.

    Without this, _clean() below can't find a "Player" column (it's
    really a tuple like ('Unnamed: 1_level_0', 'Player')), silently
    returns the dataframe completely untouched (no cleaning, no
    numeric conversion, no league/stat_type tags), and Rtot/Rdrs never
    become plain, matchable column names -- confirmed via a live run
    where 0 of 279 free agents got a real fielding match even though
    the page fetched and parsed "successfully" (887/883 rows).

    Only affects fielding -- batting/pitching columns aren't a
    MultiIndex, so this returns immediately for them, unchanged.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for top, bottom in df.columns:
        new_cols.append(str(bottom))  # bbref's real fielding stat names
                                       # (Rtot, Player, PO, Fld%, etc.)
                                       # are already unique on their own,
                                       # no group-prefix needed
    df = df.copy()
    df.columns = new_cols
    return df


# ── clean dataframe ───────────────────────────────────────────────────────────
def _clean(df: pd.DataFrame, league: str, stat_type: str) -> pd.DataFrame:
    df = _flatten_columns(df)
    # auto detect name column
    name_col = next(
        (c for c in ["Name", "Player", "Pitcher", "Tm"]
         if c in df.columns), None
    )
    if name_col is None:
        return df

    # drop repeated header rows and totals
    df = df[df[name_col].notna()].copy()
    df = df[~df[name_col].isin([
        name_col, "Name", "Player", "Pitcher",
        "Totals", "Team Totals", "League Average"
    ])]

    # strip bbref asterisks and hashes
    df[name_col] = (df[name_col]
                    .str.replace(r"[*#]", "", regex=True)
                    .str.strip())

    # rename to Name for consistency
    if name_col != "Name":
        df = df.rename(columns={name_col: "Name"})

    # preserve Tm column before numeric conversion
    tm_backup = None
    if "Tm" in df.columns:
        tm_backup = df["Tm"].copy()

    # tag league and stat type
    df["league"]    = league
    df["stat_type"] = stat_type

    # numeric conversion — skip text columns. "Pos"/"OnActv"/"IL" join the
    # skip set for fielding the same reason seattle_scraper.py's _clean()
    # needed them skipped at the team level: a combined position string
    # ("65/HD") or an active-roster marker would otherwise get silently
    # force-converted to NaN. "Team" (not "Tm") joins the set 2026-09-28:
    # the league-wide fielding page's team column flattens (via
    # _flatten_columns) to "Team", not "Tm" -- confirmed in a real debug
    # run where the raw MultiIndex column was
    # ('Unnamed: 3_level_0', 'Team'). Without this, bbref's team
    # abbreviations there ("SEA", "TBR", ...) would get silently
    # force-converted to NaN by pd.to_numeric below, same failure mode
    # "Tm" was already guarded against.
    skip = {"Name", "Tm", "Team", "Lg", "Pos", "league", "stat_type", "OnActv", "IL"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # restore Tm if it got wiped
    if tm_backup is not None:
        df["Tm"] = tm_backup

    return df.reset_index(drop=True)


# ── parse one page ────────────────────────────────────────────────────────────
def _parse_page(league: str, stat_type: str,
                season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (teams_df, players_df) for one league + stat type combo.
    """
    soup       = _fetch_page(league, stat_type, season)
    team_id    = f"teams_standard_{stat_type}"
    player_id  = f"players_standard_{stat_type}"

    # teams table
    tag = _find_table(soup, team_id)
    if tag:
        teams_df = _clean(_read_html(tag), league, stat_type)
        print(f"  [ok]   {league} {stat_type} teams:   {len(teams_df)} teams")
    else:
        print(f"  [warn] {team_id} not found")
        teams_df = pd.DataFrame()

    # players table
    tag = _find_table(soup, player_id)
    if tag:
        players_df = _clean(_read_html(tag), league, stat_type)
        print(f"  [ok]   {league} {stat_type} players: {len(players_df)} players")
    else:
        print(f"  [warn] {player_id} not found")
        players_df = pd.DataFrame()

    return teams_df, players_df


# ── analysis helpers ──────────────────────────────────────────────────────────
def get_leaders(df: pd.DataFrame, stat: str,
                top: int = 10, ascending: bool = False,
                league: str = None) -> pd.DataFrame:
    """Top N players by any stat, optionally filtered by league."""
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    if league:
        df = df[df["league"] == league]
    cols = [c for c in ["Name", "Tm", "league", stat]
            if c in df.columns]
    return (df[cols]
            .dropna(subset=[stat])
            .sort_values(stat, ascending=ascending)
            .head(top)
            .reset_index(drop=True))


def get_mariners_rank(df: pd.DataFrame, stat: str,
                      ascending: bool = False) -> dict:
    """
    Where do the Mariners rank among all players in a given stat.
    Returns rank, value, and league average.
    """
    if df.empty or stat not in df.columns:
        return {}

    col = pd.to_numeric(df[stat], errors="coerce")
    lg_avg = col.mean()

    sea_mask = df["Tm"].str.contains("SEA", na=False)
    if not sea_mask.any():
        return {}

    sea_vals = col[sea_mask]
    ranked   = col.rank(ascending=ascending, method="min")

    results = []
    for idx in sea_vals.index:
        results.append({
            "Name":    df.loc[idx, "Name"],
            "value":   sea_vals[idx],
            "rank":    int(ranked[idx]),
            "total":   len(col.dropna()),
            "lg_avg":  round(lg_avg, 3),
            "gap":     round(sea_vals[idx] - lg_avg, 3),
        })
    return results


def get_team_rankings(teams_df: pd.DataFrame,
                      team: str = "Seattle Mariners") -> pd.DataFrame:
    """
    Rank all teams by every numeric stat.
    Shows where SEA stands vs all AL/NL teams.
    """
    if teams_df.empty:
        return pd.DataFrame()

    # find the team name column
    name_col = next((c for c in ["Name", "Tm", "Team", "Franchise"]
                      if c in teams_df.columns), None)
    if not name_col:
        print(f"  [warn] no name column in teams. columns: {list(teams_df.columns)}")
        return pd.DataFrame()

    numeric_cols = [c for c in teams_df.columns
                    if c not in {"Name", "Tm", "Lg", "league", "stat_type"}
                    and pd.to_numeric(teams_df[c], errors="coerce").notna().any()]

    sea = teams_df[teams_df[name_col].str.contains("SEA|Seattle", na=False)]
    if sea.empty:
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        series  = pd.to_numeric(teams_df[col], errors="coerce")
        sea_val = pd.to_numeric(sea[col].values[0], errors="coerce")
        if pd.isna(sea_val):
            continue
        lg_avg  = series.mean()
        ranked  = series.rank(ascending=False, method="min")
        sea_rank = int(ranked[sea.index[0]])
        rows.append({
            "stat":     col,
            "sea_val":  sea_val,
            "lg_avg":   round(lg_avg, 3),
            "rank":     sea_rank,
            "total":    len(series.dropna()),
            "gap":      round(sea_val - lg_avg, 3),
        })

    return (pd.DataFrame(rows)
            .sort_values("rank", ascending=False)
            .reset_index(drop=True))


def compare_mariners_to_league(players_df: pd.DataFrame,
                               teams_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each SEA player, show how they rank vs all players.
    Auto-detects the team column name.
    """
    if players_df.empty:
        return pd.DataFrame()

    # find team column — bbref uses Tm, Team, or similar
    tm_col = next((c for c in players_df.columns
                   if c.lower() in ("tm", "team", "franchise")), None)
    if tm_col is None:
        print(f"  [warn] no team column found. columns: {list(players_df.columns)}")
        return pd.DataFrame()

    sea_players = players_df[
        players_df[tm_col].astype(str).str.contains("SEA", na=False)
    ].copy()

    print(f"  [ok]   found {len(sea_players)} SEA players")
    return sea_players.reset_index(drop=True)


# ── main public functions ─────────────────────────────────────────────────────
def get_batting(season: int = 2026,
                force_refresh: bool = False) -> dict:
    """
    Returns dict:
        al_teams      AL team batting totals
        al_players    AL individual batting stats
        nl_teams      NL team batting totals
        nl_players    NL individual batting stats
        all_players   AL + NL combined, sorted by OPS
        all_teams     AL + NL combined, sorted by R
    """
    cache_keys  = ["al_teams", "al_players", "nl_teams",
                   "nl_players", "all_players", "all_teams"]
    cache_paths = {k: _cache_path(f"batting_{k}_{season}") for k in cache_keys}

    all_cached = all(os.path.exists(p) for p in cache_paths.values())
    if not force_refresh and all_cached and not _is_stale(cache_paths["al_players"]):
        print("[cache] loading batting from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] batting leaders {season} ...")
    al_teams, al_players = _parse_page("AL", "batting", season)
    time.sleep(2)
    nl_teams, nl_players = _parse_page("NL", "batting", season)

    # combine
    all_players = pd.concat([al_players, nl_players],
                            ignore_index=True)
    all_teams   = pd.concat([al_teams, nl_teams],
                            ignore_index=True)

    # sort combined
    if "OPS" in all_players.columns:
        all_players = all_players.sort_values("OPS", ascending=False)
    if "R" in all_teams.columns:
        all_teams = all_teams.sort_values("R", ascending=False)

    result = {
        "al_teams":   al_teams,
        "al_players": al_players,
        "nl_teams":   nl_teams,
        "nl_players": nl_players,
        "all_players": all_players.reset_index(drop=True),
        "all_teams":   all_teams.reset_index(drop=True),
    }

    for k, df in result.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)

    return result


def get_pitching(season: int = 2026,
                 force_refresh: bool = False) -> dict:
    """
    Returns dict:
        al_teams      AL team pitching totals
        al_players    AL individual pitching stats
        nl_teams      NL team pitching totals
        nl_players    NL individual pitching stats
        all_players   AL + NL combined, sorted by ERA
        all_teams     AL + NL combined, sorted by ERA
    """
    cache_keys  = ["al_teams", "al_players", "nl_teams",
                   "nl_players", "all_players", "all_teams"]
    cache_paths = {k: _cache_path(f"pitching_{k}_{season}") for k in cache_keys}

    all_cached = all(os.path.exists(p) for p in cache_paths.values())
    if not force_refresh and all_cached and not _is_stale(cache_paths["al_players"]):
        print("[cache] loading pitching from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] pitching leaders {season} ...")
    al_teams, al_players = _parse_page("AL", "pitching", season)
    time.sleep(2)
    nl_teams, nl_players = _parse_page("NL", "pitching", season)

    all_players = pd.concat([al_players, nl_players],
                            ignore_index=True)
    all_teams   = pd.concat([al_teams, nl_teams],
                            ignore_index=True)

    if "ERA" in all_players.columns:
        all_players = all_players.sort_values("ERA", ascending=True)
    if "ERA" in all_teams.columns:
        all_teams = all_teams.sort_values("ERA", ascending=True)

    result = {
        "al_teams":    al_teams,
        "al_players":  al_players,
        "nl_teams":    nl_teams,
        "nl_players":  nl_players,
        "all_players": all_players.reset_index(drop=True),
        "all_teams":   all_teams.reset_index(drop=True),
    }

    for k, df in result.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)

    return result


def get_fielding(season: int = 2026,
                 force_refresh: bool = False) -> dict:
    """
    Returns dict:
        al_teams      AL team fielding totals (if bbref exposes this page
                      -- may come back empty; not essential, the player-
                      level table is what matters here)
        al_players    AL individual fielding stats
        nl_teams      NL team fielding totals
        nl_players    NL individual fielding stats
        all_players   AL + NL combined, sorted by Rtot (or Rdrs if that's
                      what this particular season's page uses instead --
                      bbref has used both naming conventions across
                      different years) descending, best defenders first
        all_teams     AL + NL combined

    See this module's docstring for the "needs a real test run" caveat --
    reuses the exact same, already-proven _fetch_page/_parse_page/_clean
    plumbing as get_batting()/get_pitching(), just pointed at "fielding".
    """
    cache_keys  = ["al_teams", "al_players", "nl_teams",
                   "nl_players", "all_players", "all_teams"]
    cache_paths = {k: _cache_path(f"fielding_{k}_{season}") for k in cache_keys}

    all_cached = all(os.path.exists(p) for p in cache_paths.values())
    if not force_refresh and all_cached and not _is_stale(cache_paths["al_players"]):
        print("[cache] loading fielding from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] fielding leaders {season} ...")
    al_teams, al_players = _parse_page("AL", "fielding", season)
    time.sleep(2)
    nl_teams, nl_players = _parse_page("NL", "fielding", season)

    all_players = pd.concat([al_players, nl_players], ignore_index=True)
    all_teams   = pd.concat([al_teams, nl_teams], ignore_index=True)

    sort_col = "Rtot" if "Rtot" in all_players.columns else (
               "Rdrs" if "Rdrs" in all_players.columns else None)
    if sort_col:
        all_players = all_players.sort_values(sort_col, ascending=False)

    result = {
        "al_teams":    al_teams,
        "al_players":  al_players,
        "nl_teams":    nl_teams,
        "nl_players":  nl_players,
        "all_players": all_players.reset_index(drop=True),
        "all_teams":   all_teams.reset_index(drop=True),
    }

    for k, df in result.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)

    return result


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── batting ──────────────────────────────────────────────────────────────
    bat = get_batting(2026, force_refresh=True)  # always fresh

    print("\n── AL team batting (sorted by R) ──")
    al_t = bat["al_teams"]
    if not al_t.empty:
        cols = [c for c in ["Name","G","R","H","HR","RBI","BA","OBP","SLG","OPS"]
                if c in al_t.columns]
        print(al_t[cols].to_string(index=False))

    print("\n── AL batting leaders OPS (top 15) ──")
    print(get_leaders(bat["al_players"], "OPS", top=15, league="AL")
          .to_string(index=False))

    print("\n── AL batting leaders HR (top 15) ──")
    print(get_leaders(bat["al_players"], "HR", top=15, league="AL")
          .to_string(index=False))

    print("\n── SEA players vs AL ──")
    sea_bat = compare_mariners_to_league(bat["al_players"], al_t)
    if not sea_bat.empty:
        cols = [c for c in ["Name","PA","BA","OBP","SLG","OPS","HR","RBI"]
                if c in sea_bat.columns]
        print(sea_bat[cols].to_string(index=False))

    # ── pitching ─────────────────────────────────────────────────────────────
    pit = get_pitching(2026, force_refresh=True)  # always fresh

    print("\n── AL team pitching (sorted by ERA) ──")
    al_pt = pit["al_teams"]
    if not al_pt.empty:
        cols = [c for c in ["Name","ERA","WHIP","SO","BB","HR","FIP"]
                if c in al_pt.columns]
        print(al_pt[cols].to_string(index=False))

    print("\n── AL ERA leaders starters (top 15) ──")
    starters = pit["al_players"]
    if not starters.empty:
        gs_col = "GS" if "GS" in starters.columns else None
        if gs_col:
            starters = starters.copy()
            starters[gs_col] = pd.to_numeric(starters[gs_col],
                                             errors="coerce").fillna(0)
            sp = starters[starters[gs_col] >= 5]
        else:
            sp = starters
        cols = [c for c in ["Name","Tm","GS","IP","ERA","WHIP","SO","BB","FIP"]
                if c in sp.columns]
        print(sp[cols].sort_values("ERA").head(15).to_string(index=False))

    print("\n── AL saves leaders (top 10) ──")
    print(get_leaders(pit["al_players"], "SV", top=10, league="AL")
          .to_string(index=False))

    # ── fielding ─────────────────────────────────────────────────────────────
    fld = get_fielding(2026, force_refresh=True)  # always fresh

    print("\n── AL fielding leaders by Rtot (top 15) ──")
    sort_col = "Rtot" if "Rtot" in fld["al_players"].columns else "Rdrs"
    print(get_leaders(fld["al_players"], sort_col, top=15, league="AL", ascending=False)
          .to_string(index=False))