"""
team_analyzer.py
Diagnoses the Seattle Mariners' strengths, weaknesses,
and generates actionable flags for the recommender.

Reads from data_builder output — never imports scrapers directly.

Usage:
    from data_builder import build_all
    from team_analyzer import analyze_team
    data = build_all(2026)
    analysis = analyze_team(data)
    print(analysis["summary"])
"""

import pandas as pd
from datetime import date

# ── thresholds ────────────────────────────────────────────────────────────────
# batting
XWOBA_ELITE       = 0.370
XWOBA_ABOVE       = 0.340
XWOBA_AVERAGE     = 0.310
XWOBA_BELOW       = 0.280

# pitching (lower is better)
XWOBA_P_ELITE     = 0.260
XWOBA_P_ABOVE     = 0.290
XWOBA_P_AVERAGE   = 0.315
XWOBA_P_BELOW     = 0.340

# team ERA
ERA_ELITE         = 3.50
ERA_ABOVE         = 4.00
ERA_AVERAGE       = 4.20

# WAR
WAR_ELITE         = 2.0
WAR_ABOVE         = 1.0
WAR_AVERAGE       = 0.0

# luck
LUCK_THRESHOLD    = 0.020

# 1-run record
ONE_RUN_AVG       = 0.500


# ── helpers ───────────────────────────────────────────────────────────────────
def _safe_get(d: dict, *keys, default=None):
    """Safely navigate nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def _grade(value, thresholds: list, labels: list) -> str:
    """Generic grader — thresholds in descending order."""
    for threshold, label in zip(thresholds, labels):
        if value >= threshold:
            return label
    return labels[-1]


def _grade_asc(value, thresholds: list, labels: list) -> str:
    """Generic grader for ascending metrics (lower = better)."""
    for threshold, label in zip(thresholds, labels):
        if value <= threshold:
            return label
    return labels[-1]


# ── team offense analysis ─────────────────────────────────────────────────────
def analyze_offense(data: dict) -> dict:
    """
    Analyzes SEA team offense vs MLB averages.
    Uses team batting from overview + Statcast from statcast_scraper.
    """
    result = {
        "grade":        "Unknown",
        "rank_ops":     None,
        "rank_hr":      None,
        "rank_r":       None,
        "avg_xwoba":    None,
        "flags":        [],
        "strengths":    [],
        "weaknesses":   [],
    }

    # team batting from overview
    overview_bat = _safe_get(data, "overview", "batting")
    if overview_bat is not None and not overview_bat.empty:
        tm_col = next((c for c in ["Tm","Team","Name"]
                       if c in overview_bat.columns), None)
        if tm_col:
            sea = overview_bat[
                overview_bat[tm_col].str.contains("Seattle", na=False)
            ]
            if not sea.empty:
                # rank in key stats (higher = better rank)
                for stat, key in [("OPS","rank_ops"),
                                   ("HR","rank_hr"),
                                   ("R","rank_r")]:
                    if stat in overview_bat.columns:
                        col = pd.to_numeric(overview_bat[stat],
                                            errors="coerce")
                        sea_val = pd.to_numeric(sea[stat].values[0],
                                                errors="coerce")
                        if not pd.isna(sea_val):
                            rank = int(col.rank(ascending=False,
                                               method="min")[sea.index[0]])
                            result[key] = rank

    # Statcast xwOBA for SEA batters
    sea_bat = _safe_get(data, "statcast", "sea_batters")
    if sea_bat is not None and not sea_bat.empty:
        if "xwOBA" in sea_bat.columns:
            qualified = sea_bat[
                pd.to_numeric(sea_bat.get("PA", 0),
                              errors="coerce") >= 50
            ]
            if not qualified.empty:
                avg_xwoba = qualified["xwOBA"].mean()
                result["avg_xwoba"] = round(avg_xwoba, 3)

                if avg_xwoba >= XWOBA_ELITE:
                    result["grade"] = "Elite"
                elif avg_xwoba >= XWOBA_ABOVE:
                    result["grade"] = "Above Average"
                elif avg_xwoba >= XWOBA_AVERAGE:
                    result["grade"] = "Average"
                else:
                    result["grade"] = "Below Average"

    # strengths
    if result["rank_hr"] and result["rank_hr"] <= 5:
        result["strengths"].append(f"Power: top {result['rank_hr']} MLB in HR")
    if result["rank_ops"] and result["rank_ops"] <= 5:
        result["strengths"].append(f"OPS: top {result['rank_ops']} MLB")

    # weaknesses
    if result["rank_r"] and result["rank_r"] > 15:
        result["weaknesses"].append(f"Run scoring: rank {result['rank_r']}/30")

    # luck from statcast
    bat_luck = _safe_get(data, "statcast", "bat_luck")
    if bat_luck is not None and not bat_luck.empty:
        lucky   = bat_luck[bat_luck["verdict"] == "LUCKY"]
        unlucky = bat_luck[bat_luck["verdict"] == "UNLUCKY"]
        if len(unlucky) >= 3:
            result["flags"].append(
                f"{len(unlucky)} batters unlucky — offense will improve"
            )
        if len(lucky) >= 2:
            result["flags"].append(
                f"{len(lucky)} batters lucky — minor regression expected"
            )

    return result


# ── team pitching analysis ────────────────────────────────────────────────────
def analyze_pitching(data: dict) -> dict:
    """
    Analyzes SEA rotation and bullpen vs MLB averages.
    """
    result = {
        "rotation_grade":  "Unknown",
        "bullpen_grade":   "Unknown",
        "team_era":        None,
        "team_era_rank":   None,
        "flags":           [],
        "strengths":       [],
        "weaknesses":      [],
        "concerns":        [],
    }

    # team ERA from overview
    overview_pit = _safe_get(data, "overview", "pitching")
    if overview_pit is not None and not overview_pit.empty:
        tm_col = next((c for c in ["Tm","Team","Name"]
                       if c in overview_pit.columns), None)
        if tm_col:
            sea = overview_pit[
                overview_pit[tm_col].str.contains("Seattle", na=False)
            ]
            if not sea.empty and "ERA" in overview_pit.columns:
                col     = pd.to_numeric(overview_pit["ERA"], errors="coerce")
                sea_era = pd.to_numeric(sea["ERA"].values[0], errors="coerce")
                if not pd.isna(sea_era):
                    result["team_era"]      = sea_era
                    result["team_era_rank"] = int(
                        col.rank(ascending=True, method="min")[sea.index[0]]
                    )

    # grade team ERA
    era = result["team_era"]
    if era:
        if era <= ERA_ELITE:
            result["rotation_grade"] = "Elite"
            result["strengths"].append(
                f"Team ERA {era} — top {result['team_era_rank']}/30 MLB"
            )
        elif era <= ERA_ABOVE:
            result["rotation_grade"] = "Above Average"
        else:
            result["rotation_grade"] = "Average"

    # individual pitcher Statcast
    sea_pit = _safe_get(data, "statcast", "sea_pitchers")
    if sea_pit is not None and not sea_pit.empty:
        xwoba_col = "xwOBA_against"
        if xwoba_col in sea_pit.columns:
            # rotation starters
            seattle_bat = _safe_get(data, "seattle", "pitching")
            if seattle_bat is not None and not seattle_bat.empty:
                starters = seattle_bat[
                    pd.to_numeric(seattle_bat.get("GS", 0),
                                  errors="coerce") >= 3
                ]["Name"].tolist() if "Name" in seattle_bat.columns else []

                rot_pit = sea_pit[
                    sea_pit["Name"].isin(starters)
                ] if starters else sea_pit

                if not rot_pit.empty:
                    avg_rot_xwoba = rot_pit[xwoba_col].mean()
                    if avg_rot_xwoba <= XWOBA_P_ELITE:
                        result["rotation_grade"] = "Elite"
                    elif avg_rot_xwoba <= XWOBA_P_ABOVE:
                        result["rotation_grade"] = "Above Average"

            # bullpen
            bp_keywords = ["Mu", "Brash", "Ferrer", "Bazardo", "Criswell"]
            bp = sea_pit[sea_pit["Name"].str.contains(
                "|".join(bp_keywords), case=False, na=False
            )]
            if not bp.empty:
                avg_bp_xwoba = bp[xwoba_col].mean()
                if avg_bp_xwoba <= XWOBA_P_ELITE:
                    result["bullpen_grade"] = "Elite"
                    result["strengths"].append(
                        f"Bullpen xwOBA {avg_bp_xwoba:.3f} — elite"
                    )
                elif avg_bp_xwoba <= XWOBA_P_ABOVE:
                    result["bullpen_grade"] = "Above Average"

    # pitcher luck
    pit_luck = _safe_get(data, "statcast", "pit_luck")
    if pit_luck is not None and not pit_luck.empty:
        lucky   = pit_luck[pit_luck["verdict"].str.contains(
            "LUCKY.*worse", na=False)]
        unlucky = pit_luck[pit_luck["verdict"].str.contains(
            "UNLUCKY", na=False)]
        if len(unlucky) >= 2:
            result["flags"].append(
                f"{len(unlucky)} pitchers unlucky — ERA will drop"
            )
        if len(lucky) >= 2:
            result["flags"].append(
                f"{len(lucky)} pitchers lucky — ERA will rise slightly"
            )

    # auto-detect struggling pitchers from bbref stats
    seattle_pit = _safe_get(data, "seattle", "pitching")
    if seattle_pit is not None and not seattle_pit.empty:
        for _, row in seattle_pit.iterrows():
            name = str(row.get("Name",""))
            era  = pd.to_numeric(row.get("ERA"), errors="coerce")
            whip = pd.to_numeric(row.get("WHIP"), errors="coerce")
            gs   = pd.to_numeric(row.get("GS",0), errors="coerce") or 0
            g    = pd.to_numeric(row.get("G",0), errors="coerce") or 0
            ip   = pd.to_numeric(row.get("IP",0), errors="coerce") or 0
            so   = pd.to_numeric(row.get("SO",0), errors="coerce") or 0
            bb   = pd.to_numeric(row.get("BB",0), errors="coerce") or 0

            if pd.isna(era) or pd.isna(whip):
                continue

            # starter concerns
            if gs >= 3 and ip > 0:
                k9 = (so / ip * 9) if ip > 0 else 0
                kbb = (so / bb) if bb > 0 else 99
                if era > 5.00:
                    result["concerns"].append(
                        f"{name}: ERA {era:.2f} — below average starter"
                        f" — {ip:.1f} IP, {k9:.1f} K/9"
                    )
                elif era > 4.00 and whip > 1.40:
                    result["concerns"].append(
                        f"{name}: ERA {era:.2f} WHIP {whip:.3f}"
                        f" — trending in wrong direction"
                    )
                elif k9 < 6.0 and gs >= 5:
                    result["concerns"].append(
                        f"{name}: K/9 {k9:.1f} — low strikeout rate"
                        f" for starter, hitters making contact"
                    )

            # reliever concerns
            if gs == 0 and g >= 10 and ip > 0:
                k9 = (so / ip * 9) if ip > 0 else 0
                if era > 5.00 and ip >= 8:
                    result["concerns"].append(
                        f"{name}: ERA {era:.2f} reliever"
                        f" — {ip:.1f} IP, {g} appearances"
                    )
                elif bb / ip * 9 > 4.5 and ip >= 8:
                    result["concerns"].append(
                        f"{name}: BB/9 {bb/ip*9:.1f} — walk rate elevated"
                    )

    return result


# ── standings analysis ────────────────────────────────────────────────────────
def analyze_standings(data: dict) -> dict:
    """
    Analyzes SEA position in standings, luck, and schedule.
    """
    result = {
        "record":       None,
        "win_pct":      None,
        "div_rank":     None,
        "mlb_rank":     None,
        "wc_gap":       None,
        "wc_in_reach":  None,
        "luck":         None,
        "pythag_wl":    None,
        "one_run_wl":   None,
        "last10":       None,
        "last30":       None,
        "vLHP_wl":      None,
        "flags":        [],
        "strengths":    [],
        "weaknesses":   [],
    }

    ctx = _safe_get(data, "standings", "mariners")
    if ctx:
        result["div_rank"]    = ctx.get("div_rank")
        result["mlb_rank"]    = ctx.get("mlb_rank")
        result["wc_gap"]      = ctx.get("wc_gap")
        result["wc_in_reach"] = ctx.get("wc_in_reach")

        row = ctx.get("expanded_row")
        if row is not None and not row.empty:
            result["luck"]      = row.get("Luck")
            result["pythag_wl"] = row.get("pythWL")
            result["one_run_wl"]= row.get("1Run")
            result["last10"]    = row.get("last10")
            result["last30"]    = row.get("last30")
            result["vLHP_wl"]   = row.get("vLHP")

            # W-L from row
            w = row.get("W")
            l = row.get("L")
            if w and l:
                try:
                    result["record"]  = f"{int(float(w))}-{int(float(l))}"
                    result["win_pct"] = round(int(float(w))/(int(float(w))+int(float(l))), 3)
                except Exception:
                    pass

    # fallback: get record from division standings
    if not result["record"]:
        div = _safe_get(data, "standings", "division")
        if div:
            for div_name, div_df in div.items():
                if div_df is None or div_df.empty:
                    continue
                tm_col = next((c for c in ["Tm","Team","Name"]
                               if c in div_df.columns), None)
                if not tm_col:
                    continue
                sea = div_df[div_df[tm_col].str.contains("Seattle", na=False)]
                if not sea.empty:
                    w = sea["W"].values[0] if "W" in sea.columns else None
                    l = sea["L"].values[0] if "L" in sea.columns else None
                    if w and l:
                        try:
                            result["record"]  = f"{int(float(w))}-{int(float(l))}"
                            result["win_pct"] = round(int(float(w))/(int(float(w))+int(float(l))), 3)
                        except Exception:
                            pass
                    break

    # flags
    luck = result["luck"]
    if luck and float(luck) < -2:
        result["flags"].append(
            f"Luck {luck} — team winning fewer games than deserved"
            f" — regression to mean = free wins coming"
        )

    one_run = result["one_run_wl"]
    if one_run:
        try:
            w, l = map(int, str(one_run).split("-"))
            pct  = w / (w + l)
            if pct < 0.450:
                result["weaknesses"].append(
                    f"1-run record {one_run} ({pct:.3f})"
                    f" — below average, bullpen/luck issue"
                )
        except Exception:
            pass

    vlhp = result["vLHP_wl"]
    if vlhp:
        try:
            w, l = map(int, str(vlhp).split("-"))
            pct  = w / (w + l)
            if pct < 0.450:
                result["weaknesses"].append(
                    f"vs LHP record {vlhp} ({pct:.3f})"
                    f" — platoon weakness"
                )
        except Exception:
            pass

    if result["div_rank"] and result["div_rank"] == 1:
        result["strengths"].append("AL West division leaders")

    return result


# ── schedule analysis ─────────────────────────────────────────────────────────
def analyze_schedule(data: dict) -> dict:
    """
    Analyzes remaining schedule difficulty and upcoming series.
    """
    result = {
        "games_remaining": None,
        "next7":           [],
        "easy_games":      0,
        "hard_games":      0,
        "flags":           [],
    }

    remaining = _safe_get(data, "schedule", "remaining")
    if remaining is not None and not remaining.empty:
        result["games_remaining"] = len(remaining)

        # opponent breakdown
        bad_teams  = ["LAA","HOU","COL","MIA","DET","KC","BAL","BOS","TOR"]
        good_teams = ["TBR","NYY","LAD","MIL","CHC","ATL"]

        if "Opp" in remaining.columns:
            for _, row in remaining.iterrows():
                opp = str(row.get("Opp",""))
                if any(t in opp for t in bad_teams):
                    result["easy_games"] += 1
                elif any(t in opp for t in good_teams):
                    result["hard_games"] += 1

        if result["easy_games"] > 40:
            result["flags"].append(
                f"{result['easy_games']} games vs sub-.500 teams remaining"
                f" — favorable schedule"
            )

    next7 = _safe_get(data, "schedule", "next7")
    if next7 is not None and not next7.empty:
        for _, row in next7.iterrows():
            result["next7"].append({
                "game":     row.get("Gm#"),
                "date":     str(row.get("date",""))[:10],
                "home_away":row.get("home_away",""),
                "opp":      row.get("Opp",""),
                "checked":  row.get("checked", False),
            })

    return result


# ── roster health analysis ────────────────────────────────────────────────────
def analyze_roster_health(data: dict) -> dict:
    """
    Identifies IL players, returning players, and roster holes.
    """
    result = {
        "on_il":          [],
        "returning_soon": [],
        "dfa_candidates": [],
        "flags":          [],
    }

    batting = _safe_get(data, "seattle", "batting")
    if batting is not None and not batting.empty:
        if "Name" in batting.columns:
            for _, row in batting.iterrows():
                name = str(row.get("Name",""))
                if "IL" in name:
                    result["on_il"].append(name)
                    if any(p in name for p in
                           ["Donovan","Raleigh","Robles","Speier"]):
                        result["returning_soon"].append(name)

    # known DFA candidates from Statcast
    # exclude franchise players and IL players
    EXCLUDE = ["Emerson", "Rodríguez", "Julio", "Raleigh",
               "Young, Cole", "Crawford", "Donovan", "Robles",
               "Wilson"]
    sea_bat = _safe_get(data, "statcast", "sea_batters")
    if sea_bat is not None and not sea_bat.empty:
        if "xwOBA" in sea_bat.columns and "PA" in sea_bat.columns:
            for _, row in sea_bat.iterrows():
                xwoba = pd.to_numeric(row.get("xwOBA"), errors="coerce")
                pa    = pd.to_numeric(row.get("PA"), errors="coerce")
                name  = str(row.get("Name",""))
                if any(e.lower() in name.lower() for e in EXCLUDE):
                    continue
                if not pd.isna(xwoba) and not pd.isna(pa):
                    if xwoba < XWOBA_BELOW and pa >= 50:
                        result["dfa_candidates"].append({
                            "name":  name,
                            "xwOBA": xwoba,
                            "PA":    int(pa),
                        })

    if result["on_il"]:
        result["flags"].append(
            f"{len(result['on_il'])} players on IL"
        )
    if result["returning_soon"]:
        result["flags"].append(
            f"Key returners: {', '.join(result['returning_soon'])}"
        )
    if result["dfa_candidates"]:
        names = [p["name"] for p in result["dfa_candidates"]]
        result["flags"].append(
            f"DFA candidates: {', '.join(names)}"
        )

    return result


# ── overall team grade ────────────────────────────────────────────────────────
def overall_grade(offense: dict, pitching: dict,
                  standings: dict) -> dict:
    """
    Combines all analysis into an overall team grade and verdict.
    """
    grades = {
        "Elite":         4,
        "Above Average": 3,
        "Average":       2,
        "Below Average": 1,
        "Unknown":       2,
    }

    off_score = grades.get(offense.get("grade","Unknown"), 2)
    rot_score = grades.get(pitching.get("rotation_grade","Unknown"), 2)
    bp_score  = grades.get(pitching.get("bullpen_grade","Unknown"), 2)

    total = (off_score + rot_score + bp_score) / 3

    if total >= 3.5:
        grade   = "A"
        verdict = "World Series contender"
    elif total >= 3.0:
        grade   = "A-"
        verdict = "Legitimate playoff threat"
    elif total >= 2.5:
        grade   = "B+"
        verdict = "Playoff team, first round danger"
    elif total >= 2.0:
        grade   = "B"
        verdict = "Wild card contender"
    else:
        grade   = "C"
        verdict = "Bubble team"

    # override deadline if pitching is elite regardless of record
    # elite pitching + injuries = buy not sell

    # win projection -- blend current pace with last30 pace
    record   = standings.get("record","0-0")
    last30   = standings.get("last30","")
    try:
        w, l = map(int, record.split("-"))
        games_played    = w + l
        games_remaining = 162 - games_played
        current_pct     = w / games_played if games_played > 0 else 0.500

        # get last30 pace if available
        last30_pct = current_pct
        if last30 and "-" in str(last30):
            try:
                lw, ll  = map(int, str(last30).split("-"))
                last30_pct = lw / (lw + ll)
            except Exception:
                pass

        # blend 40% current pace + 60% last30 pace
        blended_pct  = (current_pct * 0.40) + (last30_pct * 0.60)

        # grade adjustment
        pace_adj = {"A": 0.020, "A-": 0.015, "B+": 0.010,
                    "B": 0.000, "C": -0.020}
        projected_pct  = min(0.650, blended_pct + pace_adj.get(grade, 0))
        projected_wins = w + int(games_remaining * projected_pct)
    except Exception:
        projected_wins = 88

    # deadline logic:
    # elite rotation (rank <=6) = never sell, always at least minor buyer
    # injuries driving bad record = buy not sell
    era_rank = pitching.get("team_era_rank", 15)
    luck     = float(standings.get("luck", 0) or 0)

    if era_rank and era_rank <= 6:
        deadline = "Minor buyer — elite pitching, add depth"
    elif era_rank and era_rank <= 10 and luck < -1.0:
        deadline = "Minor buyer — unlucky, improvement coming"
    elif total >= 2.5:
        deadline = "Minor buyer"
    elif total >= 2.0 and luck < -1.5:
        deadline = "Stand pat — injuries + bad luck, not a seller"
    else:
        deadline = "Seller"

    return {
        "grade":          grade,
        "verdict":        verdict,
        "offense_grade":  offense.get("grade"),
        "rotation_grade": pitching.get("rotation_grade"),
        "bullpen_grade":  pitching.get("bullpen_grade"),
        "projected_wins": projected_wins,
        "floor":          projected_wins - 3,
        "ceiling":        projected_wins + 4,
        "buyer_seller":   deadline,
    }


# ── main analyze function ─────────────────────────────────────────────────────
def analyze_team(data: dict) -> dict:
    """
    Full team analysis. Returns complete diagnosis dict.

    Usage:
        from data_builder import build_all
        from team_analyzer import analyze_team
        data    = build_all(2026)
        analysis = analyze_team(data)
    """
    print("\n[analyze] Running team analysis...")

    offense   = analyze_offense(data)
    pitching  = analyze_pitching(data)
    standings = analyze_standings(data)
    schedule  = analyze_schedule(data)
    health    = analyze_roster_health(data)
    overall   = overall_grade(offense, pitching, standings)

    analysis = {
        "overall":   overall,
        "offense":   offense,
        "pitching":  pitching,
        "standings": standings,
        "schedule":  schedule,
        "health":    health,
        "generated": str(date.today()),
    }

    print("[analyze] Done.")
    return analysis


# ── pretty print ──────────────────────────────────────────────────────────────
def print_analysis(analysis: dict):
    """Pretty prints the full team analysis."""
    o  = analysis["overall"]
    st = analysis["standings"]
    of = analysis["offense"]
    pi = analysis["pitching"]
    sc = analysis["schedule"]
    h  = analysis["health"]

    print(f"\n{'='*60}")
    print(f"SEATTLE MARINERS — MIDSEASON ANALYSIS")
    print(f"Generated: {analysis['generated']}")
    print(f"{'='*60}")

    print(f"\n── OVERALL GRADE: {o['grade']} ──")
    print(f"  Verdict:        {o['verdict']}")
    print(f"  Offense:        {o['offense_grade']}")
    print(f"  Rotation:       {o['rotation_grade']}")
    print(f"  Bullpen:        {o['bullpen_grade']}")
    print(f"  Projected wins: {o['projected_wins']}")
    print(f"  Range:          {o['floor']}—{o['ceiling']} wins")
    print(f"  Deadline:       {o['buyer_seller']}")

    print(f"\n── STANDINGS ──")
    print(f"  Record:         {st.get('record')}")
    print(f"  Win %:          {st.get('win_pct')}")
    print(f"  Division rank:  {st.get('div_rank')} of 5")
    print(f"  MLB rank:       {st.get('mlb_rank')} of 30")
    print(f"  WC gap:         {st.get('wc_gap')}")
    print(f"  Luck:           {st.get('luck')}")
    print(f"  Pythag W-L:     {st.get('pythag_wl')}")
    print(f"  1-run record:   {st.get('one_run_wl')}")
    print(f"  vs LHP:         {st.get('vLHP_wl')}")
    print(f"  Last 10:        {st.get('last10')}")
    print(f"  Last 30:        {st.get('last30')}")

    print(f"\n── OFFENSE ──")
    print(f"  Grade:          {of.get('grade')}")
    print(f"  Avg xwOBA:      {of.get('avg_xwoba')}")
    print(f"  OPS rank:       {of.get('rank_ops')}/30")
    print(f"  HR rank:        {of.get('rank_hr')}/30")
    if of.get("strengths"):
        for s in of["strengths"]:
            print(f"  ✓ {s}")
    if of.get("weaknesses"):
        for w in of["weaknesses"]:
            print(f"  ✗ {w}")
    if of.get("flags"):
        for f in of["flags"]:
            print(f"  → {f}")

    print(f"\n── PITCHING ──")
    print(f"  Rotation grade: {pi.get('rotation_grade')}")
    print(f"  Bullpen grade:  {pi.get('bullpen_grade')}")
    print(f"  Team ERA:       {pi.get('team_era')}")
    print(f"  ERA rank:       {pi.get('team_era_rank')}/30")
    if pi.get("strengths"):
        for s in pi["strengths"]:
            print(f"  ✓ {s}")
    if pi.get("concerns"):
        for c in pi["concerns"]:
            print(f"  ⚠ {c}")
    if pi.get("flags"):
        for f in pi["flags"]:
            print(f"  → {f}")

    print(f"\n── SCHEDULE ──")
    print(f"  Games remaining: {sc.get('games_remaining')}")
    print(f"  Easy games:      {sc.get('easy_games')}")
    print(f"  Hard games:      {sc.get('hard_games')}")
    print(f"\n  Next 7 games:")
    for g in sc.get("next7", []):
        status = "☑" if g["checked"] else "☐"
        print(f"    {status} {g['date']}  {g['home_away']:4}  {g['opp']}")

    print(f"\n── ROSTER HEALTH ──")
    if h.get("on_il"):
        print(f"  On IL:          {', '.join(h['on_il'])}")
    if h.get("returning_soon"):
        print(f"  Returning soon: {', '.join(h['returning_soon'])}")
    if h.get("dfa_candidates"):
        for p in h["dfa_candidates"]:
            print(f"  DFA candidate:  {p['name']} "
                  f"(xwOBA {p['xwOBA']:.3f}, {p['PA']} PA)")

    print(f"\n{'='*60}\n")


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_builder import build_all
    data     = build_all(2026)
    analysis = analyze_team(data)
    print_analysis(analysis)