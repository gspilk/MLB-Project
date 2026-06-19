"""
seattle_scraper.py
Scrapes Seattle Mariners player stats from baseball-reference.com
https://www.baseball-reference.com/teams/SEA/2026.shtml

Tables:
  LIVE HTML:
    players_standard_batting    → batting stats per player
    players_standard_pitching   → pitching stats per player
  IN COMMENTS:
    players_standard_fielding   → fielding stats per player
    players_value_batting       → WAR + value batting
    players_value_pitching      → WAR + value pitching
    the40man                    → 40-man roster

Usage:
    from seattle_scraper import get_seattle_stats
    stats = get_seattle_stats(2026)
"""

import os
import io
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ── config ────────────────────────────────────────────────────────────────────
TEAM            = "SEA"
SEASON          = 2026
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
def _fetch_page(season: int) -> BeautifulSoup:
    url = f"https://www.baseball-reference.com/teams/{TEAM}/{season}.shtml"
    print(f"  [GET] {url}")
    session = _make_session()
    resp    = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = 'utf-8'          
    time.sleep(3)
    return BeautifulSoup(resp.text, "html.parser")


# ── find table in live HTML or comments ──────────────────────────────────────
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


# ── clean dataframe ───────────────────────────────────────────────────────────
def _clean(df: pd.DataFrame, name_col: str = None) -> pd.DataFrame:
    # auto-detect name column — bbref uses Name or Name (link text)
    if name_col is None:
        for candidate in ["Name", "Player", "Pitcher", "Batter"]:
            if candidate in df.columns:
                name_col = candidate
                break
    if name_col is None or name_col not in df.columns:
        return df
    # drop repeated header rows and totals
    df = df[df[name_col].notna()].copy()
    df = df[~df[name_col].isin([name_col, "Name", "Player", "Totals",
                                 "Team Totals", "Tm"])]
    # strip bbref asterisks / hashes from names
    df[name_col] = (df[name_col]
                    .str.replace(r"[*#]", "", regex=True)
                    .str.strip())
    # rename to Name for consistency
    if name_col != "Name":
        df = df.rename(columns={name_col: "Name"})
    # numeric conversion on everything except text cols
    skip = {"Name", "Pos", "Tm", "Lg"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


# ── parse each table ──────────────────────────────────────────────────────────
def _parse(soup, table_id: str, label: str) -> pd.DataFrame:
    tag = _find_table(soup, table_id)
    if tag is None:
        print(f"  [warn] {table_id} not found")
        return pd.DataFrame()
    df = _clean(_read_html(tag))
    print(f"  [ok]   {label}: {len(df)} players, {len(df.columns)} cols")
    return df


# ── analysis helpers ──────────────────────────────────────────────────────────
def get_rotation(pitching_df: pd.DataFrame) -> pd.DataFrame:
    """Starting pitchers — GS >= 1, sorted by GS."""
    if pitching_df.empty or "GS" not in pitching_df.columns:
        return pd.DataFrame()
    df = pitching_df.copy()
    df["GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
    return (df[df["GS"] >= 1]
            .sort_values("GS", ascending=False)
            .reset_index(drop=True))


def get_bullpen(pitching_df: pd.DataFrame) -> pd.DataFrame:
    """Relief pitchers — GS = 0."""
    if pitching_df.empty or "GS" not in pitching_df.columns:
        return pd.DataFrame()
    df = pitching_df.copy()
    df["GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
    df["G"]  = pd.to_numeric(df["G"],  errors="coerce").fillna(0)
    return (df[df["GS"] == 0]
            .sort_values("G", ascending=False)
            .reset_index(drop=True))


def get_struggling_batters(batting_df: pd.DataFrame,
                           avg_threshold: float = 0.200,
                           min_pa: int = 50) -> pd.DataFrame:
    """Batters hitting below threshold with min PA."""
    if batting_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Name","PA","BA","OBP","SLG","OPS","HR","RBI"]
            if c in batting_df.columns]
    df   = batting_df[cols].dropna(subset=["BA","PA"])
    df   = df[df["PA"] >= min_pa]
    return (df[df["BA"] <= avg_threshold]
            .sort_values("BA")
            .reset_index(drop=True))


def get_struggling_pitchers(pitching_df: pd.DataFrame,
                            era_threshold: float = 5.00,
                            min_ip: float = 5.0) -> pd.DataFrame:
    """Pitchers with ERA above threshold and min IP."""
    if pitching_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Name","G","GS","IP","ERA","WHIP","SO","BB"]
            if c in pitching_df.columns]
    df   = pitching_df[cols].dropna(subset=["ERA","IP"])
    df   = df[df["IP"] >= min_ip]
    return (df[df["ERA"] >= era_threshold]
            .sort_values("ERA", ascending=False)
            .reset_index(drop=True))


def get_war_leaders(value_df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Top players by WAR."""
    if value_df.empty:
        return pd.DataFrame()
    war_col = next((c for c in value_df.columns
                    if "WAR" in str(c) and "162" not in str(c)), None)
    if not war_col:
        return pd.DataFrame()
    return (value_df[["Name", war_col]]
            .dropna()
            .sort_values(war_col, ascending=False)
            .head(top)
            .reset_index(drop=True))


def get_leaders(df: pd.DataFrame, stat: str,
                top: int = 5, ascending: bool = False) -> pd.DataFrame:
    """Top N players by any stat."""
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    return (df[["Name", stat]]
            .dropna()
            .sort_values(stat, ascending=ascending)
            .head(top)
            .reset_index(drop=True))


def get_player(df: pd.DataFrame, name: str) -> pd.Series:
    """Pull single player by partial name match."""
    if df.empty or "Name" not in df.columns:
        return pd.Series()
    matches = df[df["Name"].str.contains(name, case=False, na=False)]
    return matches.iloc[0] if not matches.empty else pd.Series()


# ── main public function ──────────────────────────────────────────────────────
def get_seattle_stats(season: int = 2026,
                      force_refresh: bool = False) -> dict:
    """
    Returns dict:
        batting          standard batting per player
        pitching         standard pitching per player
        fielding         fielding stats per player
        value_batting    WAR + value per batter
        value_pitching   WAR + value per pitcher
        roster           40-man roster
    """
    keys        = ["batting","pitching","fielding",
                   "value_batting","value_pitching","roster"]
    cache_paths = {k: _cache_path(f"sea_{k}_{season}") for k in keys}

    all_cached = all(os.path.exists(p) for p in cache_paths.values())
    if not force_refresh and all_cached and not _is_stale(cache_paths["batting"]):
        print("[cache] loading seattle stats from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] seattle stats {season} from bbref ...")
    soup = _fetch_page(season)

    stats = {
        "batting":        _parse(soup, "players_standard_batting",  "batting"),
        "pitching":       _parse(soup, "players_standard_pitching", "pitching"),
        "fielding":       _parse(soup, "players_standard_fielding", "fielding"),
        "value_batting":  _parse(soup, "players_value_batting",     "value batting"),
        "value_pitching": _parse(soup, "players_value_pitching",    "value pitching"),
        "roster":         _parse(soup, "the40man",                  "40-man roster"),
    }

    for k, df in stats.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)

    return stats


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stats = get_seattle_stats(2026, force_refresh=True)

    # ── batting ──
    bat = stats["batting"]
    print("\n── all batters (sorted by OPS) ──")
    if not bat.empty:
        cols = [c for c in ["Name","PA","BA","OBP","SLG","OPS","HR","RBI","SB"]
                if c in bat.columns]
        df   = bat[cols].copy()
        for c in ["OPS","PA","BA","OBP","SLG","HR","RBI","SB"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["OPS"]) if "OPS" in df.columns else df
        print(df.sort_values("OPS", ascending=False).to_string(index=False))

    # ── rotation ──
    pit = stats["pitching"]
    print("\n── rotation (GS >= 1) ──")
    rot = get_rotation(pit)
    if not rot.empty:
        cols = [c for c in ["Name","GS","IP","ERA","WHIP","SO","BB","W","L"]
                if c in rot.columns]
        print(rot[cols].to_string(index=False))

    # ── bullpen ──
    print("\n── bullpen ──")
    bp = get_bullpen(pit)
    if not bp.empty:
        cols = [c for c in ["Name","G","IP","ERA","WHIP","SO","BB","SV"]
                if c in bp.columns]
        print(bp[cols].to_string(index=False))

    # ── WAR leaders ──
    print("\n── WAR leaders (batting) ──")
    war = get_war_leaders(stats["value_batting"])
    print(war.to_string(index=False) if not war.empty else "  no data")

    print("\n── WAR leaders (pitching) ──")
    war = get_war_leaders(stats["value_pitching"])
    print(war.to_string(index=False) if not war.empty else "  no data")

    # ── struggling ──
    print("\n── struggling batters (BA <= .200, min 50 PA) ──")
    s = get_struggling_batters(bat)
    print(s.to_string(index=False) if not s.empty else "  none")

    print("\n── struggling pitchers (ERA >= 5.00, min 5 IP) ──")
    sp = get_struggling_pitchers(pit)
    print(sp.to_string(index=False) if not sp.empty else "  none")
