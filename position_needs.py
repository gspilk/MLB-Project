"""
position_needs.py
Where the Mariners lose wins, position by position, using ONLY what each player did
while he was a Mariner.

Why this exists: the older "incumbent" logic took each player's whole-season row, which
for a traded player holds his time with other teams too, and picked the starter by plate
appearances alone. That produced starters like a DH with 27 plate appearances.

Method (same split used for team WAR in mlb_team_stats.py):
  1. Every player-team row is cut by playing time (mlb_api.split_by_team), so a traded
     player's WAR only counts for the team he was with.
  2. A hitter's WAR and plate appearances are shared across the positions he played in
     proportion to games at each (games not in the field count as DH).
  3. For each position, the Mariners' WAR is compared with the other 29 teams: league
     average, a good team (75th percentile) and the best. The gap is in wins.
  4. The Mariners' starter at a position is the player with the most plate appearances
     there (at least MIN_STARTER_PA), with his Mariners-only OPS.

Outputs:  data/exports/position_needs.csv  and  data/exports/position_contributors.csv

Usage:    python position_needs.py
"""
import os

import numpy as np
import pandas as pd

import mlb_api

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports")
MIN_STARTER_PA = 50     # allocated plate appearances needed to be called a starter
MIN_DEPTH_PA = 25
HIT_POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
GROUP_NAME = {
    "C": "Catchers", "1B": "First Basemen", "2B": "Second Basemen", "3B": "Third Basemen",
    "SS": "Shortstops", "LF": "Left Fielders", "CF": "Center Fielders", "RF": "Right Fielders",
    "DH": "Designated Hitters", "SP": "Starting Pitchers", "RP": "Relievers",
}


def _ops(g: pd.DataFrame) -> float:
    ab, h, bb, hbp, sf, tb = (g[c].sum() for c in ["AB", "H", "BB", "HBP", "SF", "TB"])
    pa_den = ab + bb + hbp + sf
    if ab <= 0 or pa_den <= 0:
        return np.nan
    return round((h + bb + hbp) / pa_den + tb / ab, 3)


def allocate(bat: pd.DataFrame, fld: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """One row per team x player x position with allocated PA and WAR (pitchers: SP or RP)."""
    rows = []
    fld_by = {k: g for k, g in fld.groupby(["player_id", "team_id"])}
    for r in bat.itertuples():
        war = r.WAR if pd.notna(r.WAR) else 0.0
        pa = r.PA if pd.notna(r.PA) else 0.0
        g = fld_by.get((r.player_id, r.team_id))
        field_games = g["G"].sum() if g is not None else 0.0
        weights = {}
        if g is not None:
            weights = {p: w for p, w in zip(g["Pos"], g["G"]) if w > 0}
        dh = max(0.0, (r.G if pd.notna(r.G) else 0.0) - field_games)
        if dh > 0:
            weights["DH"] = weights.get("DH", 0.0) + dh
        if not weights:
            weights = {"PH": 1.0}
        tot = sum(weights.values())
        for pos, w in weights.items():
            rows.append({"team_id": r.team_id, "player_id": r.player_id, "Name": r.Name, "pos": pos,
                         "PA": pa * w / tot, "WAR": war * w / tot, "kind": "hit"})
    for r in pit.itertuples():
        war = r.WAR if pd.notna(r.WAR) else 0.0
        gs, gp = (r.GS if pd.notna(r.GS) else 0.0), (r.G if pd.notna(r.G) else 0.0)
        role = "SP" if gs >= 0.5 * max(gp, 1) else "RP"
        rows.append({"team_id": r.team_id, "player_id": r.player_id, "Name": r.Name, "pos": role,
                     "PA": np.nan, "WAR": war, "kind": "pit"})
    return pd.DataFrame(rows)


def position_table(alloc: pd.DataFrame, bat: pd.DataFrame, team_id: int, abbr: dict = None) -> tuple:
    """(position table for team_id, contributor rows for team_id)."""
    per = alloc.groupby(["team_id", "pos"])["WAR"].sum().unstack(fill_value=0.0)
    ops_by = {pid: _ops(g) for pid, g in bat[bat["team_id"] == team_id].groupby("player_id")}
    mine = alloc[alloc["team_id"] == team_id].copy()
    mine["OPS"] = mine["player_id"].map(ops_by)
    out = []
    for pos in HIT_POS + ["SP", "RP"]:
        if pos not in per.columns:
            continue
        col = per[pos]
        sea = float(col.get(team_id, 0.0))
        others = col.drop(index=team_id, errors="ignore")
        avg, p75, best = float(col.mean()), float(col.quantile(0.75)), float(col.max())
        rank = int((col > sea).sum()) + 1
        rows = mine[mine["pos"] == pos]
        if pos in HIT_POS:
            rows = rows.sort_values("PA", ascending=False)
            top = rows[rows["PA"] >= MIN_STARTER_PA]
            depth_rows = rows[(rows["PA"] >= MIN_DEPTH_PA)]
        else:
            rows = rows.sort_values("WAR", ascending=False)
            top = rows.head(3)
            depth_rows = rows.iloc[0:0]
        st = top.iloc[0] if not top.empty else None
        if pos in HIT_POS:
            starter = st["Name"] if st is not None else "no clear starter"
        else:
            starter = ", ".join(top["Name"]) if not top.empty else ""
        depth = [n for n in depth_rows["Name"] if st is None or n != st["Name"]][:4]
        out.append({
            "position": pos, "position_group": GROUP_NAME[pos],
            "sea_war": round(sea, 1), "league_avg_war": round(avg, 1),
            "good_team_war": round(p75, 1), "best_war": round(best, 1),
            "sea_rank": rank, "of_teams": int(col.size),
            "wins_below_avg": round(avg - sea, 1), "wins_below_good_team": round(p75 - sea, 1),
            "starter": starter,
            "starter_pa_with_sea": round(float(st["PA"])) if st is not None and pos in HIT_POS else np.nan,
            "starter_ops_with_sea": float(st["OPS"]) if st is not None and pos in HIT_POS and pd.notna(st["OPS"]) else np.nan,
            "starter_war_with_sea": round(float(st["WAR"]), 1) if st is not None else np.nan,
            "depth": ", ".join(depth),
        })
    tbl = pd.DataFrame(out).sort_values("wins_below_avg", ascending=False).reset_index(drop=True)
    # label from the gap in wins (WAR) to the league average at the position
    tbl["need"] = np.where(tbl["wins_below_avg"] >= 1.5, "Biggest need",
                   np.where(tbl["wins_below_avg"] > 0.25, "Below average", "Fine"))
    contrib = mine[["Name", "pos", "PA", "WAR", "OPS"]].sort_values(["pos", "PA"], ascending=[True, False])
    contrib = contrib.round({"PA": 0, "WAR": 2})
    return tbl, contrib.reset_index(drop=True)


def seattle_incumbents(tbl: pd.DataFrame) -> dict:
    """{position_group: {"name", "OPS", "PA"}} for hitters, in the shape free_agent_recs expects."""
    inc = {}
    for r in tbl[tbl["position"].isin(HIT_POS)].itertuples():
        if pd.isna(r.starter_pa_with_sea) or r.starter == "no clear starter":
            continue
        inc[r.position_group] = {"name": r.starter, "OPS": r.starter_ops_with_sea,
                                 "PA": float(r.starter_pa_with_sea)}
    return inc


def build(season: int = 2026, team_id: int = mlb_api.SEA_TEAM_ID, force_refresh: bool = False):
    """(table, contributors). Raises if the per-team split data can't be loaded."""
    tables = mlb_api.get_player_tables(season, force_refresh=force_refresh)
    splits = mlb_api.fetch_team_splits(season, "hitting", force_refresh=force_refresh)
    if splits.empty:
        raise RuntimeError("per-team batting splits unavailable")
    bat = mlb_api.split_by_team(tables["batting"], "hitting", season, ["WAR"], "PA",
                                force_refresh=force_refresh)
    fld = mlb_api.split_by_team(tables["fielding"], "fielding", season, ["Rfield"], "_inn_outs",
                                keys=("player_id", "Pos"), force_refresh=force_refresh)
    fld = fld[fld["Pos"] != "P"]
    pit = mlb_api.split_by_team(tables["pitching"], "pitching", season, ["WAR"], "_outs",
                                force_refresh=force_refresh)
    return position_table(allocate(bat, fld, pit), bat, team_id)


def run(season: int = 2026, export_dir: str = None, force_refresh: bool = False, quiet: bool = False):
    export_dir = export_dir or EXPORT_DIR
    tbl, contrib = build(season, force_refresh=force_refresh)
    os.makedirs(export_dir, exist_ok=True)
    tbl.to_csv(os.path.join(export_dir, "position_needs.csv"), index=False)
    contrib.to_csv(os.path.join(export_dir, "position_contributors.csv"), index=False)
    if not quiet:
        show = tbl[["position", "sea_war", "league_avg_war", "sea_rank", "wins_below_avg", "starter",
                    "starter_pa_with_sea", "starter_ops_with_sea", "need"]]
        print("\nWhere the Mariners lose wins, by position (Mariners-only playing time):")
        print(show.to_string(index=False))
        print("  wins_below_avg = league-average WAR at the position minus the Mariners' WAR there.")
    return tbl, contrib


if __name__ == "__main__":
    run()