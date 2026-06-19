"""
mlb_overview_scraper.py
Scrapes the MLB overview page from baseball-reference.com:
  https://www.baseball-reference.com/leagues/majors/2026.shtml

Four tables:
  LIVE HTML:
    teams_standard_batting    → all 30 team batting totals
  IN COMMENTS:
    teams_standard_pitching   → all 30 team pitching totals
    teams_standard_fielding   → all 30 team fielding stats
    team_output               → WAR by position for all 30 teams
"""

import os
import io
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24

POSITIONS = [
    "Total", "All P", "SP", "RP", "Non-P",
    "C", "1B", "2B", "3B", "SS",
    "LF", "CF", "RF", "OF (All)", "DH", "PH"
]


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_stale(path):
    if not os.path.exists(path):
        return True
    return (time.time() - os.path.getmtime(path)) / 3600 > CACHE_TTL_HOURS


def _make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.baseball-reference.com/",
    })
    return s


def _read_html(tag):
    return pd.read_html(io.StringIO(str(tag)))[0]


def _fetch_page(season):
    url = f"https://www.baseball-reference.com/leagues/majors/{season}.shtml"
    print(f"  [GET] {url}")
    s    = _make_session()
    resp = s.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)
    return BeautifulSoup(resp.text, "html.parser")


def _find_table(soup, table_id):
    tag = soup.find("table", {"id": table_id})
    if tag:
        return tag
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id not in comment:
            continue
        c = BeautifulSoup(comment, "html.parser")
        tag = c.find("table", {"id": table_id})
        if tag:
            return tag
    return None


def _clean(df):
    tm_col = next((c for c in ["Tm","Team","Name"] if c in df.columns), None)
    if tm_col is None:
        return df
    tm_vals = df[tm_col].copy()
    df = df[df[tm_col].notna()].copy()
    df = df[~df[tm_col].isin([
        tm_col, "Tm", "Team", "League Average", "Totals", "Average", ""
    ])]
    df[tm_col] = df[tm_col].str.replace(r"[*#]", "", regex=True).str.strip()
    if tm_col != "Tm":
        df = df.rename(columns={tm_col: "Tm"})
    skip = {"Tm", "Team", "Lg"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


# ── WAR by position parser ────────────────────────────────────────────────────
def _parse_war_positions(soup):
    """
    The team_output table has cells like 'SEA3.2' or 'LAD12.7'
    mixing team abbreviation and WAR value in each cell.
    We parse it row by row to extract team + WAR per position.
    """
    tag = _find_table(soup, "team_output")
    if tag is None:
        print("  [warn] team_output not found")
        return pd.DataFrame()

    rows = []
    trs  = tag.find_all("tr")

    for tr in trs:
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue

        # first cell is rank number
        first = tds[0].get_text(strip=True)
        if not first.isdigit():
            continue

        rank = int(first)

        # second cell is "TeamName WAR" like "Los Angeles Dodgers12.7"
        # or just the team name with WAR as separate td
        # bbref format: each td contains "ABBR value" e.g. "SEA3.2"
        row = {"rank": rank}

        # collect all td text values after rank
        cell_texts = [td.get_text(strip=True) for td in tds[1:]]

        # parse each cell — format is like "SEA3.2" or "LAD-0.4"
        for i, text in enumerate(cell_texts):
            if i >= len(POSITIONS):
                break
            pos = POSITIONS[i]
            # extract team abbr (letters) and value (number)
            match = re.match(r"([A-Z]{2,3})([-\d.]+)", text)
            if match:
                row[f"{pos}_team"] = match.group(1)
                row[f"{pos}_war"]  = float(match.group(2))
            else:
                row[f"{pos}_team"] = None
                row[f"{pos}_war"]  = None

        rows.append(row)

    if not rows:
        # fallback — return raw table
        df = _read_html(tag)
        print(f"  [ok]   WAR by position (raw): {len(df)} rows")
        return df

    df = pd.DataFrame(rows)
    print(f"  [ok]   WAR by position: {len(df)} rows, {len(df.columns)} cols")
    return df


def _parse_batting(soup):
    tag = _find_table(soup, "teams_standard_batting")
    if tag is None:
        print("  [warn] teams_standard_batting not found")
        return pd.DataFrame()
    df = _clean(_read_html(tag))
    print(f"  [ok]   batting: {len(df)} teams, {len(df.columns)} cols")
    return df


def _parse_pitching(soup):
    tag = _find_table(soup, "teams_standard_pitching")
    if tag is None:
        print("  [warn] teams_standard_pitching not found")
        return pd.DataFrame()
    df = _clean(_read_html(tag))
    print(f"  [ok]   pitching: {len(df)} teams, {len(df.columns)} cols")
    return df


def _parse_fielding(soup):
    tag = _find_table(soup, "teams_standard_fielding")
    if tag is None:
        print("  [warn] teams_standard_fielding not found")
        return pd.DataFrame()
    df = _clean(_read_html(tag))
    print(f"  [ok]   fielding: {len(df)} teams, {len(df.columns)} cols")
    return df


# ── WAR analysis helpers ──────────────────────────────────────────────────────
def get_war_by_position(war_df, team_abbr="SEA"):
    """
    Pull WAR for every position for a specific team.
    Returns a clean summary showing position, WAR, and rank.
    """
    if war_df.empty:
        return pd.DataFrame()

    rows = []
    for pos in POSITIONS:
        war_col  = f"{pos}_war"
        team_col = f"{pos}_team"
        if war_col not in war_df.columns:
            continue

        # find the row where this position's team = our team
        mask = war_df[team_col].astype(str).str.upper() == team_abbr.upper()
        if not mask.any():
            continue

        row_idx = war_df[mask].index[0]
        war_val = war_df.loc[row_idx, war_col]
        rank    = int(war_df[mask]["rank"].values[0])

        # rank among all 30 teams for this position
        all_wars = war_df[war_col].dropna().sort_values(ascending=False)
        pos_rank = int((all_wars > war_val).sum()) + 1

        rows.append({
            "position": pos,
            "WAR":      war_val,
            "rank":     pos_rank,
            "total":    len(all_wars),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("rank", ascending=False).reset_index(drop=True)
    return df


def get_position_leaders(war_df, position="1B", top=5):
    """Top N teams by WAR at a specific position."""
    war_col  = f"{position}_war"
    team_col = f"{position}_team"
    if war_col not in war_df.columns:
        return pd.DataFrame()
    result = war_df[[team_col, war_col]].copy()
    result = result.dropna(subset=[war_col])
    result = result.sort_values(war_col, ascending=False).head(top)
    result.columns = ["team", "WAR"]
    result["rank"] = range(1, len(result)+1)
    return result.reset_index(drop=True)


def get_team_row(df, team="Seattle"):
    if df.empty:
        return pd.Series()
    tm_col = next((c for c in ["Tm","Team","Name"] if c in df.columns), None)
    if tm_col is None:
        return pd.Series()
    matches = df[df[tm_col].str.contains(team, case=False, na=False)]
    return matches.iloc[0] if not matches.empty else pd.Series()


def rank_teams(df, stat, ascending=False):
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    tm_col = next((c for c in ["Tm","Team","Name"] if c in df.columns), "Tm")
    result = df[[tm_col, stat]].copy()
    result[stat] = pd.to_numeric(result[stat], errors="coerce")
    result = result.dropna(subset=[stat])
    result = result.sort_values(stat, ascending=ascending).reset_index(drop=True)
    result["rank"]   = result.index + 1
    result["is_SEA"] = result[tm_col].str.contains("Seattle", na=False)
    return result


def get_sea_rankings(df):
    if df.empty:
        return pd.DataFrame()
    tm_col = next((c for c in ["Tm","Team","Name"] if c in df.columns), None)
    if tm_col is None:
        return pd.DataFrame()
    sea = df[df[tm_col].str.contains("Seattle", case=False, na=False)]
    if sea.empty:
        return pd.DataFrame()
    rows = []
    num_cols = [c for c in df.columns
                if c not in {tm_col,"Tm","Team","Lg","Name"}
                and pd.to_numeric(df[c], errors="coerce").notna().any()]
    for col in num_cols:
        series  = pd.to_numeric(df[col], errors="coerce")
        sea_val = pd.to_numeric(sea[col].values[0], errors="coerce")
        if pd.isna(sea_val):
            continue
        lg_avg   = series.mean()
        ranked   = series.rank(ascending=False, method="min")
        sea_rank = int(ranked[sea.index[0]])
        rows.append({
            "stat":    col,
            "sea_val": round(sea_val, 3),
            "lg_avg":  round(lg_avg, 3),
            "rank":    sea_rank,
            "total":   int(series.notna().sum()),
            "gap":     round(sea_val - lg_avg, 3),
        })
    return (pd.DataFrame(rows)
            .sort_values("rank", ascending=False)
            .reset_index(drop=True))


# ── main public function ──────────────────────────────────────────────────────
def get_mlb_overview(season=2026, force_refresh=False):
    keys        = ["batting","pitching","fielding","war_positions"]
    cache_paths = {k: _cache_path(f"mlb_{k}_{season}") for k in keys}
    all_cached  = all(os.path.exists(p) for p in cache_paths.values())

    if not force_refresh and all_cached and not _is_stale(cache_paths["batting"]):
        print("[cache] loading MLB overview from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] MLB overview {season} ...")
    soup = _fetch_page(season)
    data = {
        "batting":       _parse_batting(soup),
        "pitching":      _parse_pitching(soup),
        "fielding":      _parse_fielding(soup),
        "war_positions": _parse_war_positions(soup),
    }
    for k, df in data.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)
    return data


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = get_mlb_overview(2026, force_refresh=True)

    # ── batting ──
    bat = data["batting"]
    print("\n── all 30 teams batting (sorted by R) ──")
    if not bat.empty:
        cols = [c for c in ["Tm","R","HR","BA","OBP","SLG","OPS","OPS+"]
                if c in bat.columns]
        b = bat[cols].copy()
        b["R"] = pd.to_numeric(b["R"], errors="coerce")
        print(b.sort_values("R", ascending=False).to_string(index=False))

    print("\n── SEA batting rankings (worst first) ──")
    print(get_sea_rankings(bat).to_string(index=False)
          if not bat.empty else "  no data")

    # ── pitching ──
    pit = data["pitching"]
    print("\n── all 30 teams pitching (sorted by ERA) ──")
    if not pit.empty:
        cols = [c for c in ["Tm","ERA","WHIP","SO","BB","FIP","ERA+","SO/W"]
                if c in pit.columns]
        p = pit[cols].copy()
        p["ERA"] = pd.to_numeric(p["ERA"], errors="coerce")
        print(p.sort_values("ERA").to_string(index=False))

    # ── fielding ──
    fld = data["fielding"]
    print("\n── all 30 teams fielding (sorted by Rtot) ──")
    if not fld.empty:
        cols = [c for c in ["Tm","DefEff","Fld%","Rtot","Rdrs","E","DP"]
                if c in fld.columns]
        f = fld[cols].copy()
        f["Rtot"] = pd.to_numeric(f["Rtot"], errors="coerce")
        print(f.sort_values("Rtot", ascending=False).to_string(index=False))

    # ── WAR by position ──
    war = data["war_positions"]
    print("\n── SEA WAR by every position (worst first) ──")
    sea_war = get_war_by_position(war, "SEA")
    if not sea_war.empty:
        print(sea_war.to_string(index=False))
    else:
        print("  parsing fallback — raw table:")
        print(war.head(10).to_string(index=False))

    print("\n── 1B leaders (SEA's biggest hole) ──")
    print(get_position_leaders(war, "1B").to_string(index=False))

    print("\n── SP leaders ──")
    print(get_position_leaders(war, "SP").to_string(index=False))

    print("\n── RP leaders ──")
    print(get_position_leaders(war, "RP").to_string(index=False))

    print("\n── LF leaders ──")
    print(get_position_leaders(war, "LF").to_string(index=False))