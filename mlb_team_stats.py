"""
mlb_overview_scraper.py
MLB-wide team overview from the official MLB Stats API, replacing
the old source's league overview page.

Four tables (same names and columns as before):
    batting         all 30 team batting totals (Tm = full team name, OPS+)
    pitching        all 30 team pitching totals (ERA+, FIP, R = runs allowed)
    fielding        all 30 team fielding totals (DefEff, Fld%, Rtot/Rdrs =
                    Statcast fielding runs)
    war_positions   WAR by position: row N lists the team ranked Nth at each
                    position ("{pos}_team" / "{pos}_war"), the same layout
                    the old source's team_output table had

WAR is this project's own calculation (mlb_api.py). A player's batting WAR
is split across the positions he played by innings (DH by games); pitcher
WAR goes to SP or RP by role.
"""

import os
import time
import numpy as np
import pandas as pd

import mlb_api

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


# ── team tables ───────────────────────────────────────────────────────────────
def _team_tables(season, force_refresh=False):
    tables = mlb_api.get_player_tables(season, force_refresh=force_refresh)
    lc = tables["context"]
    games = mlb_api.games_played_by_team(season, force_refresh=force_refresh)

    bat = mlb_api.team_totals(tables["batting"], "hitting", season, games)
    bat = mlb_api.add_batting_rates(bat, lc)
    bat["R/G"] = (bat["R"] / bat["G"]).round(2)
    bat = mlb_api.shape(bat, ["Tm", "Lg", "G", "R/G", "PA", "AB", "R", "H", "2B", "3B", "HR",
                              "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS", "OPS+",
                              "TB", "GIDP", "HBP", "SH", "SF", "IBB", "WAR"])

    pit_players = tables["pitching"]
    pit = mlb_api.team_totals(pit_players, "pitching", season, games)
    pit = mlb_api.add_pitching_rates(pit, lc)
    pit["RA/G"] = (pit["R"] / pit["G"]).round(2)
    pit["RA9"] = (9 * pit["R"] / (pit["_outs"] / 3)).round(2)
    pit_out = mlb_api.shape(pit, ["Tm", "Lg", "RA/G", "W", "L", "W-L%", "ERA", "G", "GS", "CG",
                                  "SHO", "SV", "IP", "H", "R", "ER", "HR", "BB", "IBB", "SO",
                                  "HBP", "BK", "WP", "BF", "ERA+", "FIP", "WHIP", "H9", "HR9",
                                  "BB9", "SO9", "SO/W", "RA9", "WAR"])

    fld = mlb_api.team_totals(tables["fielding"], "fielding", season, games)
    fld = mlb_api.add_fielding_rates(fld)
    # defensive efficiency: share of balls in play turned into outs
    bip = pit["BF"] - pit["SO"] - pit["BB"] - pit["HBP"] - pit["HR"]
    eff = 1 - (pit["H"] - pit["HR"]) / bip.replace(0, np.nan)
    fld["DefEff"] = fld["team_id"].map(dict(zip(pit["team_id"], eff.round(3))))
    fld_out = mlb_api.shape(fld, ["Tm", "Lg", "G", "Inn", "Ch", "PO", "A", "E", "DP", "Fld%",
                                  "DefEff", "Rtot", "Rdrs"])
    for name, df in (("batting", bat), ("pitching", pit_out), ("fielding", fld_out)):
        print(f"  [ok]   {name}: {len(df)} teams, {len(df.columns)} cols")
    return bat, pit_out, fld_out


# ── WAR by position ───────────────────────────────────────────────────────────
OUTFIELD = {"LF", "CF", "RF"}


def _war_by_position(season, force_refresh=False):
    """
    Splits every player's WAR across the positions he actually played for
    each team, then ranks the 30 teams at each position. Returned in the same
    rank-per-row layout as the old source's team_output table.
    """
    tables = mlb_api.get_player_tables(season, force_refresh=force_refresh)
    bat = tables["batting"]; bat = bat[~bat["_total"]]
    fld = tables["fielding"]; fld = fld[~fld["_total"] & (fld["Pos"] != "P")]
    pit = tables["pitching"]; pit = pit[~pit["_total"]]
    abbr = mlb_api.get_teams(season).set_index("team_id")["abbr"]

    totals = {}   # (team_id, pos) -> WAR

    def add(team, pos, war):
        totals[(team, pos)] = totals.get((team, pos), 0.0) + war

    for r in bat.itertuples():
        war = r.WAR if pd.notna(r.WAR) else 0.0
        rows = fld[(fld["player_id"] == r.player_id) & (fld["team_id"] == r.team_id)]
        field_games = rows["G"].sum()
        dh_games = max(0.0, (r.G or 0) - field_games)
        weights = {p: g for p, g in zip(rows["Pos"], rows["G"]) if g > 0}
        if dh_games > 0:
            weights["DH"] = dh_games
        if not weights:
            weights = {"PH": 1.0}
        total_w = sum(weights.values())
        for pos, w in weights.items():
            add(r.team_id, pos, war * w / total_w)
    for r in pit.itertuples():
        war = r.WAR if pd.notna(r.WAR) else 0.0
        role = "SP" if (r.GS or 0) >= 0.5 * max(r.G or 0, 1) else "RP"
        add(r.team_id, role, war)

    team_ids = sorted({t for t, _ in totals})
    per_team = pd.DataFrame(index=team_ids)
    for pos in ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "PH", "SP", "RP"]:
        per_team[pos] = [totals.get((t, pos), 0.0) for t in team_ids]
    per_team["All P"] = per_team["SP"] + per_team["RP"]
    per_team["OF (All)"] = per_team["LF"] + per_team["CF"] + per_team["RF"]
    per_team["Non-P"] = per_team[["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "PH"]].sum(axis=1)
    per_team["Total"] = per_team["All P"] + per_team["Non-P"]

    out = pd.DataFrame({"rank": range(1, len(per_team) + 1)})
    for pos in POSITIONS:
        ranked = per_team[pos].sort_values(ascending=False)
        out[f"{pos}_team"] = [abbr.get(t, str(t)) for t in ranked.index]
        out[f"{pos}_war"] = ranked.round(1).values
    print(f"  [ok]   WAR by position: {len(out)} rows, {len(out.columns)} cols")
    return out


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

    print(f"[fetch] MLB overview {season} (MLB Stats API) ...")
    bat, pit, fld = _team_tables(season, force_refresh)
    data = {
        "batting":       bat,
        "pitching":      pit,
        "fielding":      fld,
        "war_positions": _war_by_position(season, force_refresh),
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
        print("  no WAR-by-position data")

    print("\n── 1B leaders (SEA's biggest hole) ──")
    print(get_position_leaders(war, "1B").to_string(index=False))

    print("\n── SP leaders ──")
    print(get_position_leaders(war, "SP").to_string(index=False))

    print("\n── RP leaders ──")
    print(get_position_leaders(war, "RP").to_string(index=False))

    print("\n── LF leaders ──")
    print(get_position_leaders(war, "LF").to_string(index=False))