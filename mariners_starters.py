"""
mariners_starters.py
Builds data/exports/mariners_starters.csv: who starts at each position for the
Mariners right now, straight from the Mariners' own stats -- no free-agent list
needed. The Streamlit depth chart reads this for its "Now" column.

HOW STARTERS ARE PICKED (hitters)
  * Candidates are players on the Mariners 40-man roster (so someone traded
    away mid-season is not a starter).
  * Players on the injured list are skipped as starters (they still show up
    as "Roster depth", with their IL note).
  * For C, 1B, 2B, SS, 3B, LF, CF, RF: the player with the most innings at
    that position this season. Each player starts at ONE position only: the
    biggest innings totals are assigned first, so a player who split time is
    placed where he played most.
  * Designated Hitters: the healthy roster player whose primary position is DH
    (no fielding innings) with the most plate appearances.
  * Stat shown = his 2026 OPS.

PITCHING
  Same convention the rest of the project uses: a staff average, not one name.
  Starting Pitchers = ERA of everyone with a start; relievers (right- and
  left-handed rows both) = ERA of everyone with no starts. Weighted by
  innings pitched.

LIMITS
  * "Most innings" is a season-long view. A player who took over a spot late
    in the year can lose to one who played it earlier. Check the names.
  * After the season, injury statuses may be stale or reset.

Usage:
    python mariners_starters.py
"""
import os

import numpy as np
import pandas as pd

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports")

POS_GROUP = {"C": "Catchers", "1B": "First Basemen", "2B": "Second Basemen", "SS": "Shortstops",
             "3B": "Third Basemen", "LF": "Left Fielders", "CF": "Center Fielders",
             "RF": "Right Fielders", "DH": "Designated Hitters"}
FIELD_POS = ["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF"]
OUT_COLS = ["position_group", "player", "player_id", "role", "stat_name", "stat", "sample"]


def _inn(x) -> float:
    """Baseball innings notation -> real innings: 123.1 means 123 and 1/3."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    whole = int(x)
    return whole + int(round((x - whole) * 10)) / 3.0


def _ids(series) -> set:
    return {int(v) for v in pd.to_numeric(series, errors="coerce").dropna()}


def build(stats: dict) -> pd.DataFrame:
    bat, fld, pit = stats.get("batting"), stats.get("fielding"), stats.get("pitching")
    roster = stats.get("roster", pd.DataFrame())
    if bat is None or fld is None or bat.empty or fld.empty:
        raise RuntimeError("Mariners batting/fielding tables are empty -- nothing to build starters from.")

    on_roster = _ids(roster["player_id"]) if not roster.empty and "player_id" in roster.columns else None
    il_ids = (_ids(roster.loc[roster["IL"].notna(), "player_id"])
              if not roster.empty and "IL" in roster.columns else set())

    b = bat.copy()
    for c in ("PA", "OPS"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")
    label = b["Name"].to_dict()      # batting Name carries the "(60-day IL)" note

    f = fld[fld["Pos"].isin(FIELD_POS)].copy()
    f["inn"] = f["Inn"].map(_inn)
    f = f.dropna(subset=["player_id", "inn"])
    f["player_id"] = f["player_id"].astype(int)
    if on_roster is not None:
        f = f[f["player_id"].isin(on_roster)]
    g = f.groupby(["player_id", "Pos"], as_index=False)["inn"].sum()
    names_f = f.drop_duplicates("player_id").set_index("player_id")["Name"].to_dict()

    def name_of(pid):
        return label.get(pid) or names_f.get(pid) or str(pid)

    def stat_of(pid):
        if pid in b.index:
            return float(b.loc[pid, "OPS"]), float(b.loc[pid, "PA"])
        return float("nan"), float("nan")

    rows, taken, filled = [], set(), set()
    for _, r in g.sort_values("inn", ascending=False).iterrows():
        pid, pos = int(r["player_id"]), r["Pos"]
        if pid in il_ids or pid in taken or pos in filled:
            continue
        ops, pa = stat_of(pid)
        rows.append({"position_group": POS_GROUP[pos], "player": name_of(pid), "player_id": pid,
                     "role": "Current starter", "stat_name": "OPS", "stat": ops, "sample": pa})
        taken.add(pid)
        filled.add(pos)

    # designated hitter: healthy roster player whose primary position is DH, most PA
    dh = b[(b["Pos"] == "DH")] if "Pos" in b.columns else b.iloc[0:0]
    dh = dh[~dh.index.isin(il_ids | taken)]
    if on_roster is not None:
        dh = dh[dh.index.isin(on_roster)]
    if not dh.empty:
        pid = int(dh["PA"].idxmax())
        ops, pa = stat_of(pid)
        rows.append({"position_group": POS_GROUP["DH"], "player": name_of(pid), "player_id": pid,
                     "role": "Current starter", "stat_name": "OPS", "stat": ops, "sample": pa})
        taken.add(pid)

    # roster depth: everyone else who played the position (injured players included, with their note)
    starters_at = {r["position_group"]: r["player_id"] for r in rows}
    for _, r in g.sort_values("inn", ascending=False).iterrows():
        pid, grp = int(r["player_id"]), POS_GROUP[r["Pos"]]
        if starters_at.get(grp) == pid or r["inn"] <= 0:
            continue
        ops, pa = stat_of(pid)
        rows.append({"position_group": grp, "player": name_of(pid), "player_id": pid,
                     "role": "Roster depth", "stat_name": "OPS", "stat": ops, "sample": pa})

    # pitching staff averages (innings-weighted ERA)
    if pit is not None and not pit.empty:
        p = pit.copy()
        p["ip"] = p["IP"].map(_inn)
        for c in ("ER", "GS"):
            p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
        p = p.dropna(subset=["ip"])
        for groups, mask, nm in ((["Starting Pitchers"], p["GS"] >= 1, "Real rotation average"),
                                 (["Right-Handed Relievers", "Left-Handed Relievers"], p["GS"] == 0,
                                  "Real bullpen average")):
            sub = p[mask]
            ip = float(sub["ip"].sum())
            if ip > 0:
                era = round(9 * float(sub["ER"].sum()) / ip, 2)
                for grp in groups:
                    rows.append({"position_group": grp, "player": nm, "player_id": np.nan,
                                 "role": "Staff average", "stat_name": "ERA", "stat": era, "sample": round(ip, 1)})

    return pd.DataFrame(rows, columns=OUT_COLS)


def run(stats=None, season=2026, export_dir=None) -> pd.DataFrame:
    if stats is None:
        from mariners_stats import get_seattle_stats      # the project's own Mariners loader
        stats = get_seattle_stats(season)
    df = build(stats)
    export_dir = export_dir or EXPORT_DIR
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, "mariners_starters.csv")
    df.to_csv(path, index=False)
    starters = df[df["role"] == "Current starter"]
    print(f"Wrote {len(df)} rows -> {path}")
    for _, r in starters.iterrows():
        print(f"  {r['position_group']:<14} {r['player']}  (OPS {r['stat']:.3f}, {r['sample']:.0f} PA)")
    missing = [g for g in POS_GROUP.values() if g not in set(starters["position_group"])]
    if missing:
        print(f"  no starter found at: {', '.join(missing)}")
    return df


if __name__ == "__main__":
    run()