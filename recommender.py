"""
recommender.py
Generates specific roster recommendations for the Seattle Mariners.
"""

import pandas as pd
from datetime import date

LOWER_IS_BETTER = {
    "ERA", "WHIP", "BB", "BB9", "HR", "RA/G", "GDP", "E",
    "FIP", "ER", "R_allowed", "HBP", "WP", "LOB_pct"
}

INTERNAL_FIXES = {
    "SS":    "Crawford returns from IL — .354 xwOBA, 14.7 BB%",
    "3B":    "Donovan returns from IL — .839 OPS when healthy",
    "C":     "Raleigh returns from IL — power bat, 12.3 Barrel%",
    "DH":    "Emerson/Canzone rotation — both above average",
    "RF":    "Raley + Canzone solid trio",
    "RP":    "Brash returning — 0.60 ERA, elite closer",
    "1-run": "Brash to closer — fixes late game management",
    "vLHP":  "Emerson/Wisdom platoon — improving naturally",
}

DO_NOT_TRADE = [
    "Julio Rodriguez    — 10yr/$210M franchise player",
    "Bryce Miller       — xwOBA .221 top 3% MLB, ascending",
    "Emerson Hancock    — 2.74 ERA, elite rotation piece",
    "Bryan Woo          — xwOBA .272, above average",
    "Randy Arozarena    — best bat on team when healthy",
    "Matt Brash         — 0.60 ERA, elite closer",
    "Colt Emerson       — 8yr/$95M franchise cornerstone",
    "Cole Young         — 2.4 WAR, emerging star",
    "Logan Gilbert      — 3.29 ERA, 2.1 WAR, ascending",
]

# franchise players — never appear in DFA moves
FRANCHISE_KEYS = ["Emerson, Colt", "Colt", "Rodríguez", "Julio",
                  "Raleigh", "Young, Cole", "Cole Young"]

# seller teams = more likely available at deadline
SELLER_TEAMS = {
    "angels","athletics","rockies","white sox","nationals",
    "marlins","tigers","royals","pirates","reds","orioles",
    "blue jays","twins"
}

def _availability(team_name):
    tn = str(team_name).lower()
    if "mariners" in tn or "seattle" in tn:
        return "ON ROSTER"
    if any(s in tn for s in SELLER_TEAMS):
        return "LIKELY AVAILABLE"
    return "CHECK AVAILABILITY"

_avail_p = _availability  # same function for pitchers


def _safe(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def _sea_rank(df, stat, tm_col="Tm"):
    if df is None or df.empty or stat not in df.columns:
        return None, None, None
    col     = pd.to_numeric(df[stat], errors="coerce")
    sea_row = df[df[tm_col].str.contains("Seattle", na=False)]
    if sea_row.empty:
        return None, None, None
    sea_val = pd.to_numeric(sea_row[stat].values[0], errors="coerce")
    if pd.isna(sea_val):
        return None, None, None
    asc    = stat in LOWER_IS_BETTER
    ranked = col.rank(ascending=asc, method="min")
    rank   = int(ranked[sea_row.index[0]])
    total  = int(col.notna().sum())
    return round(sea_val, 3), rank, total


def _is_franchise(name):
    n = str(name).lower()
    return any(k.lower() in n for k in ["colt emerson", "julio rodríguez",
               "julio rodriguez", "cal raleigh", "cole young"])


# ── section 1 ────────────────────────────────────────────────────────────────
def _team_diagnosis(analysis):
    o  = analysis.get("overall", {})
    st = analysis.get("standings", {})
    of = analysis.get("offense", {})
    pi = analysis.get("pitching", {})
    sc = analysis.get("schedule", {})
    h  = analysis.get("health", {})
    return {
        "grade":          o.get("grade"),
        "verdict":        o.get("verdict"),
        "record":         st.get("record"),
        "win_pct":        st.get("win_pct"),
        "div_rank":       st.get("div_rank"),
        "mlb_rank":       st.get("mlb_rank"),
        "luck":           st.get("luck"),
        "pythag":         st.get("pythag_wl"),
        "one_run":        st.get("one_run_wl"),
        "vlhp":           st.get("vLHP_wl"),
        "last10":         st.get("last10"),
        "last30":         st.get("last30"),
        "offense_grade":  of.get("grade"),
        "rotation_grade": pi.get("rotation_grade"),
        "bullpen_grade":  pi.get("bullpen_grade"),
        "team_era":       pi.get("team_era"),
        "era_rank":       pi.get("team_era_rank"),
        "projected_wins": o.get("projected_wins"),
        "floor":          o.get("floor"),
        "ceiling":        o.get("ceiling"),
        "buyer_seller":   o.get("buyer_seller"),
        "games_remaining":sc.get("games_remaining"),
        "easy_games":     sc.get("easy_games"),
        "hard_games":     sc.get("hard_games"),
        "on_il":          h.get("on_il", []),
        "returning_soon": h.get("returning_soon", []),
        "concerns":       pi.get("concerns", []),
    }


# ── section 2 ────────────────────────────────────────────────────────────────
def _stats_to_improve(data):
    targets = []

    bat = _safe(data, "overview", "batting")
    if bat is not None and not bat.empty:
        tm_col = next((c for c in ["Tm","Team"] if c in bat.columns), "Tm")
        for stat in ["R","H","HR","RBI","BA","OBP","SLG","OPS","OPS+","BB"]:
            val, rank, total = _sea_rank(bat, stat, tm_col)
            if rank and total and rank > 15:
                targets.append({
                    "category": "Offense",
                    "stat":     stat,
                    "value":    val,
                    "rank":     rank,
                    "total":    total,
                    "priority": "HIGH" if rank > 20 else "MONITOR",
                    "fix":      _batting_fix(stat),
                    "internal": stat in {"OBP","BB","SS","C","3B"},
                })

    pit = _safe(data, "overview", "pitching")
    if pit is not None and not pit.empty:
        tm_col = next((c for c in ["Tm","Team"] if c in pit.columns), "Tm")
        for stat in ["ERA","WHIP","BB","HR","FIP","ERA+"]:
            val, rank, total = _sea_rank(pit, stat, tm_col)
            if rank and total and rank > 15:
                targets.append({
                    "category": "Pitching",
                    "stat":     stat,
                    "value":    val,
                    "rank":     rank,
                    "total":    total,
                    "priority": "HIGH" if rank > 20 else "MONITOR",
                    "fix":      _pitching_fix(stat),
                    "internal": True,
                })

    fld = _safe(data, "overview", "fielding")
    if fld is not None and not fld.empty:
        tm_col = next((c for c in ["Tm","Team"] if c in fld.columns), "Tm")
        for stat in ["Rtot","DefEff","E"]:
            val, rank, total = _sea_rank(fld, stat, tm_col)
            if rank and total and rank > 15:
                targets.append({
                    "category": "Fielding",
                    "stat":     stat,
                    "value":    val,
                    "rank":     rank,
                    "total":    total,
                    "priority": "HIGH" if rank > 20 else "MONITOR",
                    "fix":      _fielding_fix(stat),
                    "internal": True,
                })

    war = _safe(data, "overview", "war_positions")
    if war is not None and not war.empty:
        for pos in ["1B","C","DH","SS","3B","RF","SP","RP"]:
            wc = f"{pos}_war"
            tc = f"{pos}_team"
            if wc not in war.columns:
                continue
            sea_rows = war[war[tc].astype(str).str.upper() == "SEA"]
            if sea_rows.empty:
                continue
            wv       = sea_rows[wc].values[0]
            all_wars = war[wc].dropna().sort_values(ascending=False)
            pos_rank = int((all_wars > wv).sum()) + 1
            total    = len(all_wars)
            if pos_rank > 15:
                targets.append({
                    "category": "WAR by Position",
                    "stat":     f"{pos} WAR",
                    "value":    round(wv, 1),
                    "rank":     pos_rank,
                    "total":    total,
                    "priority": "HIGH" if pos_rank > 20 else "MONITOR",
                    "fix":      _war_fix(pos),
                    "internal": pos in {"C","3B","SS","DH","RP"},
                })

    targets.sort(key=lambda x: (0 if x["priority"]=="HIGH" else 1, x["rank"]))
    return targets


def _batting_fix(stat):
    return {
        "R":   "Donovan + Raleigh returning — run production will improve",
        "H":   "Contact rate improving — Crawford return helps",
        "HR":  "Power is fine — high K rate suppressing batting avg",
        "RBI": "RISP production improves with Donovan + Raleigh returning",
        "BA":  "Three true outcomes team — by design, power offsets BA",
        "OBP": "Crawford 14.7 BB% returning — OBP will jump immediately",
        "SLG": "Raley 19.2 Barrel% + Raleigh power returning — SLG improving",
        "OPS": "Full health lineup projects top 5 AL — internal fix available",
        "OPS+":"Improving with IL returns — no external move needed",
        "BB":  "Crawford elite walk rate returning — fixes this immediately",
    }.get(stat, f"Monitor {stat}")


def _pitching_fix(stat):
    return {
        "ERA":  "Already top 5 MLB — elite staff",
        "WHIP": "Castillo + Muñoz driving this up — role changes help",
        "BB":   "Muñoz walk rate — role change to setup reduces exposure",
        "HR":   "Muñoz Barrel% elevated — role change helps significantly",
        "FIP":  "xwOBA suggests ERA will come down — regression coming",
        "ERA+": "Already top 5 MLB — above average",
    }.get(stat, f"Monitor {stat}")


def _fielding_fix(stat):
    return {
        "Rtot":   "Naylor -0.7 DWAR at 1B — biggest defensive hole on roster",
        "DefEff": "Outfield defense solid — Robles return improves range",
        "E":      "Error rate acceptable — Crawford return stabilizes SS",
    }.get(stat, "Monitor")


def _war_fix(pos):
    return {
        "1B":  "Naylor below average — potential deadline upgrade target",
        "C":   "Raleigh returns from IL — fixes this immediately",
        "DH":  "Emerson/Canzone rotation — internal fix available",
        "SS":  "Crawford returns from IL — .354 xwOBA, elite OBP",
        "3B":  "Donovan returns from IL — .839 OPS when healthy",
        "RF":  "Raley + Canzone solid — Refsnyder/Robles dragged WAR down",
        "SP":  "Miller + Hancock + Woo elite — Kirby needs to bounce back",
        "RP":  "Brash returns — bullpen immediately becomes elite",
    }.get(pos, f"Monitor {pos} WAR")


# ── section 3 ────────────────────────────────────────────────────────────────
def _immediate_moves(grades, analysis):
    moves = []
    seen  = set()

    for g in grades.get("batters",[]) + grades.get("pitchers",[]):
        name   = g["name"]
        action = g.get("action","")

        # skip franchise players from DFA moves
        if _is_franchise(name):
            continue

        if "DFA" in action and \
           "small sample" not in action.lower() and \
           "depth" not in action.lower() and \
           name not in seen:
            seen.add(name)
            moves.append({
                "type":   "DFA",
                "player": name,
                "reason": action.replace("DFA — ",""),
                "urgency":"IMMEDIATE",
            })

    for g in grades.get("pitchers",[]):
        name   = g["name"]
        action = g.get("action","")
        if "Role change" in action and name not in seen:
            seen.add(name)
            moves.append({
                "type":   "ROLE CHANGE",
                "player": name,
                "reason": action,
                "urgency":"IMMEDIATE",
            })
        if "Promote to closer" in action and name not in seen:
            seen.add(name)
            moves.append({
                "type":   "ROLE CHANGE",
                "player": name,
                "reason": action,
                "urgency":"IMMEDIATE",
            })
        if "piggyback" in action.lower() and name not in seen:
            seen.add(name)
            moves.append({
                "type":   "PIGGYBACK",
                "player": name,
                "reason": action,
                "urgency":"SOON",
            })

    for p in analysis.get("health",{}).get("returning_soon",[]):
        moves.append({
            "type":   "ACTIVATE",
            "player": p,
            "reason": "Returning from IL — key contributor",
            "urgency":"WHEN READY",
        })

    return moves


# ── section 4 ────────────────────────────────────────────────────────────────
def _lineup_optimization(data, grades):
    qualified = []
    for g in grades.get("batters",[]):
        xwoba = g.get("xwOBA")
        pa    = g.get("PA") or 0
        role  = str(g.get("role",""))
        if xwoba and pa >= 50 and \
           "IL" not in role and "DFA" not in role:
            qualified.append({
                "name":  g["name"],
                "xwOBA": xwoba,
                "HR":    g.get("HR",0),
                "SB":    g.get("SB",0),
                "PA":    pa,
                "OPS":   g.get("OPS",0),
            })

    qualified.sort(key=lambda x: x["xwOBA"], reverse=True)
    return {
        "optimal_order": qualified[:9],
        "bench":         qualified[9:],
        "il": [g for g in grades.get("batters",[])
               if "IL" in str(g.get("role",""))],
    }


# ── section 5 ────────────────────────────────────────────────────────────────
def _rotation_bullpen(grades):
    starters  = sorted(
        [g for g in grades.get("pitchers",[]) if g["role"] == "SP"],
        key=lambda x: x.get("xwOBA_against") or 1.0
    )
    relievers = sorted(
        [g for g in grades.get("pitchers",[]) if g["role"] in ("RP","CL")],
        key=lambda x: x.get("xwOBA_against") or 1.0
    )
    return {
        "rotation":      starters,
        "bullpen":       relievers,
        "closer":        "Matt Brash — 0.60 ERA, promote immediately",
        "setup":         "Andrés Muñoz — move from closer, fix curveball",
        "high_leverage": "José Ferrer — xwOBA .242, unlucky, elite",
        "piggyback":     "Kirby + Miller — limit Kirby to 4 IP, Miller finishes",
        "6th_starter":   "Luis Castillo — ERA trending down in 6-man role",
    }



# ── player targets ────────────────────────────────────────────────────────────
CONTENDERS = {
    "NYY","TBR","LAD","MIL","ATL","PHI",
    "CLE","CHW","SEA","NYM","BOS","PIT",
    "MIN","SDP","SFG","HOU","CHC","STL"
}

MARINERS_ROSTER = {
    "rodriguez","arozarena","raley","young",
    "crawford","raleigh","naylor","canzone",
    "donovan","emerson","garver","pereda",
    "kirby","woo","hancock","gilbert","castillo",
    "miller","munoz","brash","ferrer","bazardo",
    "criswell","speier","hoppe","wilcox","legumina",
    "robles","refsnyder","rivas","wilson","joe",
    "mastrobuoni","bliss","pereda","wisdom",
}

def _find_targets(data: dict) -> dict:
    """
    Finds MLB trade/waiver targets using bbref individual stats (has Tm)
    joined with Statcast xwOBA.
    Batters:  xwOBA >= .330, Barrel% >= 8%, PA >= 100, non-contender
    Pitchers: xwOBA_against <= .310, IP >= 20, GS < 3, non-contender
    """
    all_bat = _safe(data, "batting",  "all_players")
    all_pit = _safe(data, "pitching", "all_players")
    sc_bat  = _safe(data, "statcast", "batters")
    sc_pit  = _safe(data, "statcast", "pitchers")

    import unicodedata
    def _norm(s):
        """Normalize accents for last name matching."""
        s = str(s).lower().strip()
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    batter_targets  = []
    pitcher_targets = []

    # ── batters ──────────────────────────────────────────────────────────────
    # debug
    if all_bat is not None and not all_bat.empty:
        tm_check = [c for c in all_bat.columns if c.lower() in ("tm","team","franchise")]
        print(f"  [targets] bat cols: {list(all_bat.columns[:8])}")
        print(f"  [targets] bat tm cols: {tm_check}  rows: {len(all_bat)}")
    if sc_bat is not None and not sc_bat.empty:
        print(f"  [targets] sc_bat cols: {list(sc_bat.columns[:6])}")
    try:
        if all_bat is not None and not all_bat.empty and \
           sc_bat  is not None and not sc_bat.empty  and \
           "Name" in all_bat.columns and "Name" in sc_bat.columns:

            bat_b = all_bat.copy()
            bat_s = sc_bat.copy()

            # last name join key
            # use last + first initial for more precise matching
            # bbref: "Yordan Alvarez" -> "alvarez_y"
            # savant: "Alvarez, Yordan" -> "alvarez_y"
            def _name_key(name, fmt="first_last"):
                name = str(name).strip()
                if fmt == "first_last":
                    parts = name.split()
                    if len(parts) >= 2:
                        return _norm(parts[-1]) + "_" + _norm(parts[0])[0]
                    return _norm(name)
                else:  # last_first (savant format)
                    parts = name.split(",")
                    if len(parts) >= 2:
                        last  = _norm(parts[0].strip())
                        first = _norm(parts[1].strip())
                        return last + "_" + first[0]
                    return _norm(name)

            bat_b["_key"] = bat_b["Name"].apply(lambda n: _name_key(n, "first_last"))
            bat_s["_key"] = bat_s["Name"].apply(lambda n: _name_key(n, "last_first"))

            # select cols
            # detect team column name
            tm_col_b = next((c for c in ["Tm","Team","team"]
                            if c in bat_b.columns), None)
            if tm_col_b is None:
                raise ValueError("No team column in batting data")

            b_cols = ["_key","Name",tm_col_b] +                      [c for c in ["PA","OPS","HR","BA","OBP","SLG","R","RBI"]
                      if c in bat_b.columns]
            s_cols = ["_key"] +                      [c for c in ["xwOBA","wOBA","Barrel%","HardHit%","EV50","K%","BB%"]
                      if c in bat_s.columns]

            merged = pd.merge(bat_b[b_cols], bat_s[s_cols],
                              on="_key", how="inner")

            # numeric
            for c in ["xwOBA","Barrel%","PA"]:
                if c in merged.columns:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce")

            # debug: show merge size and sample
            print(f"  [targets] merged {len(merged)} batter rows")
            if not merged.empty:
                xw = merged["xwOBA"].dropna()
                print(f"  [targets] xwOBA range: {xw.min():.3f}-{xw.max():.3f}  non-null: {len(xw)}")
                # check what happened to team column after merge
                team_cols_after = [c for c in merged.columns if "team" in c.lower() or "tm" in c.lower()]
                print(f"  [targets] team cols after merge: {team_cols_after}")
                for tc in team_cols_after:
                    print(f"  [targets] {tc} sample: {merged[tc].dropna().head(3).tolist()}")

            # full team name to abbrev mapping for contenders filter
            def _is_mariner_key(key):
                last = key.split("_")[0] if "_" in key else key
                return last in MARINERS_ROSTER

            # filter -- all teams, just exclude current Mariners
            mask = (
                (merged["xwOBA"]   >= 0.320) &
                (merged["Barrel%"] >= 7.0)   &
                (merged["PA"]      >= 80)     &
                (~merged["_key"].apply(_is_mariner_key))
            )
            tgt = merged[mask].sort_values("xwOBA", ascending=False).head(10)
            print(f"  [targets] {len(tgt)} batter targets found")


            for _, row in tgt.iterrows():
                # Name_y is statcast format: "Rodriguez, Julio"
                # reformat to: "Julio Rodriguez"
                name_y = str(row.get("Name_y",""))
                if "," in name_y:
                    parts = name_y.split(",", 1)
                    player_name = f"{parts[1].strip()} {parts[0].strip()}"
                else:
                    player_name = name_y
                if not player_name or player_name == "nan":
                    player_name = str(row.get("Name_x", row.get("Name","")))

                # team column after merge might be renamed
                team_val = None
                for _tc in [tm_col_b, tm_col_b+"_x", "Team", "Tm", "Team_x", "Tm_x"]:
                    _v = row.get(_tc)
                    if _v and str(_v) not in ("nan","","None"):
                        team_val = str(_v)
                        break
                avail = _availability(team_val or "")
                batter_targets.append({
                    "name":         player_name,
                    "team":         team_val or "N/A",
                    "PA":           row.get("PA"),
                    "OPS":          row.get("OPS"),
                    "HR":           row.get("HR"),
                    "BA":           row.get("BA"),
                    "xwOBA":        round(float(row["xwOBA"]),3),
                    "Barrel%":      row.get("Barrel%"),
                    "HardHit%":     row.get("HardHit%"),
                    "availability": avail,
                    "fit":          "1B/DH trade or waiver target",
                })
    except Exception as e:
        import traceback
        print(f"  [warn] batter target search failed: {e}")
        traceback.print_exc()

    # ── pitchers ─────────────────────────────────────────────────────────────
    try:
        if all_pit is not None and not all_pit.empty and            sc_pit  is not None and not sc_pit.empty  and            "Name" in all_pit.columns and "Name" in sc_pit.columns:

            pit_b = all_pit.copy()
            pit_s = sc_pit.copy()

            # team col
            if "Tm" not in pit_b.columns:
                tm_col = next((c for c in ["Team","team"] if c in pit_b.columns), None)
                if tm_col:
                    pit_b = pit_b.rename(columns={tm_col: "Tm"})
                else:
                    pit_b["Tm"] = "UNK"

            # relievers only
            if "GS" in pit_b.columns:
                pit_b = pit_b[
                    pd.to_numeric(pit_b["GS"], errors="coerce").fillna(0) < 3
                ].copy()

            # last name join key
            pit_b["_key"] = pit_b["Name"].apply(lambda n: _name_key(n, "first_last"))
            pit_s["_key"] = pit_s["Name"].apply(lambda n: _name_key(n, "last_first"))

            tm_col_p = next((c for c in ["Tm","Team","team"]
                            if c in pit_b.columns), None)
            if tm_col_p is None:
                raise ValueError("No team column in pitching data")

            p_cols = ["_key","Name",tm_col_p] +                      [c for c in ["G","IP","ERA","WHIP","SO","BB","SV","HLD"]
                      if c in pit_b.columns]
            ps_cols = ["_key"] +                       [c for c in ["xwOBA_against","K%","Whiff%","BB%",
                                   "HardHit%_against","Barrel%_against"]
                       if c in pit_s.columns]

            merged_p = pd.merge(pit_b[p_cols], pit_s[ps_cols],
                                on="_key", how="inner")

            # numeric
            for c in ["xwOBA_against","IP"]:
                if c in merged_p.columns:
                    merged_p[c] = pd.to_numeric(merged_p[c], errors="coerce")

            # filter -- all teams, just exclude current Mariners
            mask_p = (
                (merged_p["xwOBA_against"] <= 0.310) &
                (merged_p["IP"]            >= 20)    &
                (~merged_p["_key"].apply(lambda k: k.split("_")[0] in MARINERS_ROSTER))
            )
            print(f"  [debug] pitcher merge cols with team: {[c for c in merged_p.columns if any(t in c.lower() for t in ['team','tm','name'])]}")
            ptgt = merged_p[mask_p].sort_values(
                "xwOBA_against", ascending=True
            )
            ptgt = ptgt.drop_duplicates(subset=["_key"]).head(10)
            print(f"  [targets] {len(ptgt)} pitcher targets found")

            for _, row in ptgt.iterrows():
                name_y = str(row.get("Name_y",""))
                if "," in name_y:
                    parts = name_y.split(",", 1)
                    player_name = f"{parts[1].strip()} {parts[0].strip()}"
                else:
                    player_name = name_y
                if not player_name or player_name == "nan":
                    player_name = str(row.get("Name_x", row.get("Name","")))

                team_val_p = None
                for _tc in ["Tm", "Tm_x", tm_col_p, tm_col_p+"_x", "Team", "Team_x"]:
                    _v = row.get(_tc)
                    if _v and str(_v) not in ("nan","","None","nan"):
                        team_val_p = str(_v)
                        break
                avail_p = _avail_p(team_val_p or "")
                pitcher_targets.append({
                    "name":         player_name,
                    "team":         team_val_p or "N/A",
                    "availability": avail_p,
                    "G":            row.get("G"),
                    "IP":           row.get("IP"),
                    "ERA":          row.get("ERA"),
                    "WHIP":         row.get("WHIP"),
                    "K%":           row.get("K%"),
                    "xwOBA_against":round(float(row["xwOBA_against"]),3),
                    "fit":          "Bullpen depth — waiver or trade target",
                })
    except Exception as e:
        print(f"  [warn] pitcher target search failed: {e}")

    return {
        "batter_targets":  batter_targets,
        "pitcher_targets": pitcher_targets,
    }

# ── section 6 ────────────────────────────────────────────────────────────────
def _deadline_moves(data, analysis, stats):
    needs   = []
    targets = []
    sell    = []

    war_targets = [s for s in stats if "1B" in s["stat"] or "DH" in s["stat"]]
    if war_targets:
        needs.append("1B/DH offensive upgrade")
        targets.append({
            "position":  "1B or DH",
            "handedness":"RH preferred — handles LHP",
            "profile":   "Contact/power hybrid — .270+ BA, 20+ HR pace, .340+ xwOBA",
            "cost":      "Low-mid prospect or waiver claim",
            "urgency":   "By July 31",
            "why":       "1B WAR rank 28-30/30 — Naylor below average bat and glove",
        })

    needs.append("Bullpen middle relief depth")
    targets.append({
        "position": "RP",
        "handedness":"Either",
        "profile":  "Veteran middle reliever — command-first, groundball tendency",
        "cost":     "Waiver claim or minimum salary",
        "urgency":  "Immediate — Hoppe/Wilcox DFA leaves depth hole",
        "why":      "Need reliable 6th/7th inning arm behind Ferrer/Bazardo/Speier",
    })

    sell.append({
        "player": "Luis Castillo",
        "reason": "ERA 5.00+, expensive contract, rotation 6-deep without him",
        "return": "Low prospect or salary relief",
        "note":   "Only if ERA continues above 4.50 through July",
    })

    return {
        "needs":        needs,
        "targets":      targets,
        "do_not_trade": DO_NOT_TRADE,
        "sell":         sell,
        "strategy":     "Minor buyer — rotation top 5 MLB, bullpen elite with Brash, "
                        "offense improving with IL returns. Stand pat or add depth only. "
                        "Do NOT trade prospects for win-now pieces — system too valuable.",
    }


# ── section 7 ────────────────────────────────────────────────────────────────
def _season_outlook(analysis, data):
    o  = analysis.get("overall", {})
    st = analysis.get("standings", {})
    sc = analysis.get("schedule", {})
    pw = o.get("projected_wins", 88)
    return {
        "current_record": st.get("record"),
        "projected_wins": pw,
        "floor":          o.get("floor"),
        "ceiling":        o.get("ceiling"),
        "most_likely":    f"{pw}—{(pw or 88)+2} wins",
        "division":       "AL West leaders — favorable division",
        "playoff_path":   "AL West title → avoid TB/NYY until ALCS",
        "key_factors": [
            "Donovan + Raleigh + Crawford return from IL",
            "Brash takes over closer role immediately",
            "Miller gets full-time rotation spot",
            "Kirby bounces back or piggyback with Miller",
            f"Luck correction — {st.get('luck',0)} luck = free wins coming",
            f"Favorable schedule — {sc.get('easy_games',0)} games vs sub-.500 teams",
            "Gilbert emerging as legitimate #3 starter — 3.29 ERA, 2.1 WAR",
        ],
        "risks": [
            "Kirby sustained decline — K rate dropped, not just a slump",
            "Muñoz curveball command doesn't improve in setup role",
            "Injury to Hancock, Woo, or Miller",
            "Offense stalls without Donovan/Raleigh returning healthy",
            "1B hole — Naylor below average, no internal fix",
        ],
        "october_outlook": "Dangerous in short series — "
                          "Miller + Woo + Hancock + Gilbert + Brash + Ferrer "
                          "= can beat anyone in 5 games",
    }


# ── main ──────────────────────────────────────────────────────────────────────
def generate_recommendations(data, analysis, grades):
    print("\n[recommend] Generating recommendations...")
    stats          = _stats_to_improve(data)
    player_targets = _find_targets(data)
    recs  = {
        "generated":        str(date.today()),
        "diagnosis":        _team_diagnosis(analysis),
        "stats_to_improve": stats,
        "immediate_moves":  _immediate_moves(grades, analysis),
        "lineup":           _lineup_optimization(data, grades),
        "rotation_bullpen": _rotation_bullpen(grades),
        "deadline":         _deadline_moves(data, analysis, stats),
        "player_targets":   player_targets,
        "outlook":          _season_outlook(analysis, data),
    }
    print("[recommend] Done.")
    return recs


# ── print ─────────────────────────────────────────────────────────────────────
def print_recommendations(recs):
    d  = recs["diagnosis"]
    si = recs["stats_to_improve"]
    im = recs["immediate_moves"]
    lu = recs["lineup"]
    rb = recs["rotation_bullpen"]
    dl = recs["deadline"]
    ou = recs["outlook"]

    print(f"\n{'='*65}")
    print(f"SEATTLE MARINERS — RECOMMENDATIONS")
    print(f"Generated: {recs['generated']}")
    print(f"{'='*65}")

    print(f"\n── TEAM DIAGNOSIS ──")
    print(f"  Grade:          {d['grade']} — {d['verdict']}")
    print(f"  Record:         {d['record']}  div {d['div_rank']}/5  MLB {d['mlb_rank']}/30")
    print(f"  Offense:        {d['offense_grade']}")
    print(f"  Rotation:       {d['rotation_grade']}  ERA {d['team_era']}  rank {d['era_rank']}/30")
    print(f"  Bullpen:        {d['bullpen_grade']}")
    print(f"  Luck:           {d['luck']}  Pythag: {d['pythag']}")
    print(f"  1-run / vLHP:   {d['one_run']} / {d['vlhp']}")
    print(f"  Last 10/30:     {d['last10']} / {d['last30']}")
    print(f"  Proj wins:      {d['projected_wins']}  ({d['floor']}—{d['ceiling']})")
    print(f"  Deadline:       {d['buyer_seller']}")
    print(f"  Games left:     {d['games_remaining']}  easy {d['easy_games']}  hard {d['hard_games']}")

    print(f"\n── STATS TO IMPROVE ──")
    high = [s for s in si if s["priority"] == "HIGH"]
    mon  = [s for s in si if s["priority"] == "MONITOR"]
    if high:
        print("  HIGH PRIORITY:")
        for s in high:
            tag = "✓ internal" if s["internal"] else "→ external"
            print(f"    {s['category']:<14} {s['stat']:<10} "
                  f"rank {s['rank']:>2}/{s['total']}  {s['value']}  {tag}")
            print(f"      → {s['fix']}")
    if mon:
        print("  MONITOR:")
        for s in mon:
            print(f"    {s['category']:<14} {s['stat']:<10} "
                  f"rank {s['rank']:>2}/{s['total']}  {s['value']}")

    print(f"\n── IMMEDIATE ROSTER MOVES ──")
    for m in im:
        print(f"  [{m['urgency']:<10}] {m['type']:<12} {m['player']}")
        print(f"                   → {m['reason']}")

    print(f"\n── OPTIMAL LINEUP (by xwOBA, active only) ──")
    for i, p in enumerate(lu.get("optimal_order",[])[:9], 1):
        print(f"  {i}. {p['name']:<25} xwOBA {p['xwOBA']:.3f}  "
              f"OPS {p['OPS']:.3f}  HR {p['HR']}")

    print(f"\n── ROTATION (by xwOBA) ──")
    for i, p in enumerate(rb.get("rotation",[]), 1):
        xw  = f"{p['xwOBA_against']:.3f}" if p.get("xwOBA_against") else " N/A"
        era = f"{p['ERA']:.2f}" if p.get("ERA") else "N/A"
        print(f"  {i}. {p['name']:<25} ERA {era:>5}  xwOBA {xw}  {p['action']}")
    print(f"  Piggyback: {rb['piggyback']}")
    print(f"  6th man:   {rb['6th_starter']}")

    print(f"\n── BULLPEN HIERARCHY ──")
    print(f"  Closer:        {rb['closer']}")
    print(f"  Setup:         {rb['setup']}")
    print(f"  High leverage: {rb['high_leverage']}")
    for p in rb.get("bullpen",[])[:6]:
        xw  = f"{p['xwOBA_against']:.3f}" if p.get("xwOBA_against") else " N/A"
        era = f"{p['ERA']:.2f}" if p.get("ERA") else "N/A"
        print(f"    {p['name']:<25} ERA {era:>5}  xwOBA {xw}  {p['grade']}")

    print(f"\n── DEADLINE STRATEGY ──")
    print(f"  {dl['strategy']}")
    print(f"\n  NEEDS:")
    for n in dl["needs"]:
        print(f"    → {n}")
    print(f"\n  TARGET PROFILES:")
    for t in dl["targets"]:
        print(f"    Position:  {t['position']}")
        print(f"    Profile:   {t['profile']}")
        print(f"    Cost:      {t['cost']}")
        print(f"    Why:       {t['why']}")
        print()
    print(f"  DO NOT TRADE:")
    for p in dl["do_not_trade"]:
        print(f"    ✗ {p}")
    if dl.get("sell"):
        print(f"\n  SELL CANDIDATES:")
        for s in dl["sell"]:
            print(f"    → {s['player']}: {s['reason']}")
            print(f"      Note: {s['note']}")

    print(f"\n── SEASON OUTLOOK ──")
    print(f"  Record now:   {ou['current_record']}")
    print(f"  Most likely:  {ou['most_likely']}")
    print(f"  Floor/Ceiling:{ou['floor']}—{ou['ceiling']} wins")
    print(f"  Playoff path: {ou['playoff_path']}")
    print(f"  October:      {ou['october_outlook']}")
    print(f"\n  Key factors:")
    for f in ou["key_factors"]:
        print(f"    ✓ {f}")
    print(f"\n  Risks:")
    for r in ou["risks"]:
        print(f"    ⚠ {r}")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    from data_builder import build_all
    from team_analyzer import analyze_team
    from player_grades import grade_players
    data     = build_all(2026)
    analysis = analyze_team(data)
    grades   = grade_players(data)
    recs     = generate_recommendations(data, analysis, grades)
    print_recommendations(recs)