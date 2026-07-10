"""
player_grader.py
Grades EVERY Seattle Mariners player using:
  - bbref batting/pitching/fielding as the primary source
  - Statcast xwOBA + Barrel% + HardHit% as supplement where available
  - Age + contract context
  - Luck analysis
  - Role context

Usage:
    from data_builder import build_all
    from player_grader import grade_players, print_grades
    data   = build_all(2026)
    grades = grade_players(data)
    print_grades(grades)
"""

import pandas as pd

# ── thresholds ────────────────────────────────────────────────────────────────
# batter OPS (primary when xwOBA not available)
OPS_ELITE   = 0.850
OPS_ABOVE   = 0.750
OPS_AVG     = 0.680
OPS_BELOW   = 0.600

# batter xwOBA (preferred when available)
XWOBA_ELITE = 0.370
XWOBA_ABOVE = 0.340
XWOBA_AVG   = 0.310
XWOBA_BELOW = 0.280

# pitcher ERA
ERA_ELITE   = 3.00
ERA_ABOVE   = 3.75
ERA_AVG     = 4.50
ERA_BELOW   = 5.25

# pitcher xwOBA against (lower = better)
XWOBA_P_ELITE = 0.260
XWOBA_P_ABOVE = 0.290
XWOBA_P_AVG   = 0.315
XWOBA_P_BELOW = 0.340

# minimum PA/IP for full grade
MIN_PA = 30
MIN_IP = 5.0

# young player threshold
YOUNG_AGE = 23

# ── roster context ────────────────────────────────────────────────────────────
FRANCHISE_PLAYERS = {
    "Colt":       {"contract": "8yr/$95M",  "age": 20,
                   "note": "franchise cornerstone — never DFA"},
    "Rodríguez":  {"contract": "10yr/$210M","age": 25,
                   "note": "franchise player"},
    "Raleigh":    {"contract": "extension", "age": 28,
                   "note": "franchise catcher"},
    "Young":      {"contract": "pre-arb",   "age": 22,
                   "note": "emerging star"},
}

DFA_CANDIDATES = {
    "Refsnyder":  "no role with Robles/Canzone/Raley healthy",
    "Rivas":      "no role with Donovan/Emerson returning",
    "Wilcox":     "ERA 5.40, .408 xwOBA, replacement level",
    "Hoppe":      "ERA 6.46, replacement level reliever",
}

# players who should be optioned to AAA not DFA
AAA_CANDIDATES = {
    "Rucker":   "6.23 ERA, command issues — needs AAA time",
    "Simpson":  "9.00 ERA, not ready for MLB",
    "Gonzalez": "4.70 ERA, high WHIP — needs refinement",
}

# no hardcoded grades — trust scrapers only
# if bbref data is stale, clear cache and re-run
HARDCODED_GRADES = {}

IL_PLAYERS = {
    "Raleigh":    "10-day IL — power expected to return",
    "Crawford":   "10-day IL — elite walk rate, critical return",
    "Donovan":    "10-day IL — .839 OPS when healthy",
    "Wilson":     "10-day IL — bench depth",
}

# roster notes are generated from data — no hardcoded text
ROSTER_NOTES = {}

def _generate_batter_note(pa, ops, xwoba, woba, barrel, hardhit, hr, war):
    """Generate data-driven note for a batter."""
    notes = []
    if xwoba and woba:
        gap = round(woba - xwoba, 3)
        if gap > 0.030:
            notes.append(f"LUCKY +{gap:.3f} — regression expected")
        elif gap < -0.030:
            notes.append(f"UNLUCKY {gap:.3f} — improvement expected")
    if barrel and barrel >= 15:
        notes.append(f"Elite power: {barrel:.1f}% Barrel")
    if barrel and barrel >= 10:
        notes.append(f"Above avg power: {barrel:.1f}% Barrel")
    if hardhit and hardhit >= 50:
        notes.append(f"Elite contact: {hardhit:.1f}% HardHit")
    if war and war >= 2.0:
        notes.append(f"Elite WAR: {war:.1f}")
    if hr and hr >= 15:
        notes.append(f"{hr} HR — power producing")
    return " | ".join(notes)

def _generate_pitcher_note(era, xwoba, k9, bb9, whiff, war, ip):
    """Generate data-driven note for a pitcher."""
    notes = []
    if xwoba and era and ip and ip >= 8:
        # ERA vs xwOBA gap
        era_expected = (xwoba - 0.200) * 20  # rough conversion
        if era - era_expected > 1.0:
            notes.append(f"ERA {era:.2f} misleading — xwOBA {xwoba:.3f} better")
        elif era_expected - era > 1.0:
            notes.append(f"ERA {era:.2f} may rise — xwOBA {xwoba:.3f} concerning")
    if k9 and k9 >= 10.0 and ip and ip >= 8:
        notes.append(f"Elite K/9: {k9:.1f}")
    elif k9 and k9 < 6.0 and ip and ip >= 20:
        notes.append(f"Low K/9: {k9:.1f} — contact pitcher")
    if bb9 and bb9 >= 4.5 and ip and ip >= 8:
        notes.append(f"High BB/9: {bb9:.1f} — command concern")
    if whiff and whiff >= 35:
        notes.append(f"Elite Whiff%: {whiff:.1f}%")
    if war and war >= 1.5:
        notes.append(f"High WAR: {war:.1f}")
    return " | ".join(notes)


# ── grade helpers ─────────────────────────────────────────────────────────────
def _is_franchise(name: str) -> bool:
    name_clean = name.lower().replace(","," ")
    name_parts = name_clean.split()
    for k in FRANCHISE_PLAYERS:
        # exact word match only
        if k.lower() in name_parts:
            return True
    return False

def _is_dfa(name: str) -> bool:
    return any(k.lower() in name.lower() for k in DFA_CANDIDATES)

def _is_il(name: str) -> bool:
    return "IL" in name or any(k.lower() in name.lower()
                               for k in IL_PLAYERS)

def _get_note(name: str) -> str:
    # notes are generated from data now
    return ""

def _get_franchise_note(name: str) -> str:
    name_parts = name.lower().replace(","," ").split()
    for k, v in FRANCHISE_PLAYERS.items():
        if k.lower() in name_parts:
            return f"{v['contract']} — {v['note']}"
    return ""

def _get_dfa_reason(name: str) -> str:
    for k, v in DFA_CANDIDATES.items():
        if k.lower() in name.lower():
            return v
    return ""

def _get_il_note(name: str) -> str:
    # check IL in name first
    if "IL" in name:
        for k, v in IL_PLAYERS.items():
            if k.lower() in name.lower():
                return v
        return "on IL"
    for k, v in IL_PLAYERS.items():
        if k.lower() in name.lower():
            return v
    return ""


def _grade_ops(ops: float) -> str:
    if pd.isna(ops): return "Unknown"
    if ops >= OPS_ELITE:   return "Elite"
    if ops >= OPS_ABOVE:   return "Above Average"
    if ops >= OPS_AVG:     return "Average"
    if ops >= OPS_BELOW:   return "Below Average"
    return "DFA"

def _grade_xwoba(xwoba: float) -> str:
    if pd.isna(xwoba): return None
    if xwoba >= XWOBA_ELITE:  return "Elite"
    if xwoba >= XWOBA_ABOVE:  return "Above Average"
    if xwoba >= XWOBA_AVG:    return "Average"
    if xwoba >= XWOBA_BELOW:  return "Below Average"
    return "DFA"

def _grade_era(era: float, xwoba: float = None,
              war: float = None, ip: float = 0) -> str:
    """
    Blend ERA + xwOBA + WAR for pitcher grade.
    Requires minimum IP to avoid small sample grades.
    """
    # insufficient sample
    if ip < 5:
        return "Small sample"
    # for very small samples weight ERA heavily
    if ip < 15 and xwoba is None:
        if era is None: return "Small sample"
        if era <= 3.00: return "Above Average"
        if era <= 4.50: return "Average"
        return "Below Average"

    # score each metric 1-5
    scores = []

    # ERA score
    if era is not None and not pd.isna(era):
        if era <= 2.50:   scores.append(5)
        elif era <= 3.25: scores.append(4)
        elif era <= 3.75: scores.append(3)
        elif era <= 4.50: scores.append(2)
        elif era <= 5.25: scores.append(1)
        else:             scores.append(0)

    # xwOBA score (lower = better)
    # league avg pitcher xwOBA ~.315, adjust thresholds accordingly
    if xwoba is not None and not pd.isna(xwoba):
        if xwoba <= 0.260:   scores.append(5)
        elif xwoba <= 0.285: scores.append(4)
        elif xwoba <= 0.315: scores.append(3)
        elif xwoba <= 0.345: scores.append(2)
        elif xwoba <= 0.370: scores.append(1)
        else:                scores.append(0)

    # WAR score
    if war is not None and not pd.isna(war):
        if war >= 2.0:    scores.append(5)
        elif war >= 1.0:  scores.append(4)
        elif war >= 0.3:  scores.append(3)
        elif war >= 0.0:  scores.append(2)
        else:             scores.append(1)

    if not scores:
        return "Unknown"

    avg = sum(scores) / len(scores)

    if avg >= 4.0: return "Elite"
    if avg >= 3.0: return "Above Average"
    if avg >= 2.0: return "Average"
    if avg >= 1.0: return "Below Average"
    return "DFA"

def _batter_action(name: str, grade: str, luck: float,
                   pa: float, war: float,
                   is_injured: bool = False) -> str:
    if _is_franchise(name):
        return f"Keep — {_get_franchise_note(name)}"
    if _is_dfa(name):
        return f"DFA — {_get_dfa_reason(name)}"
    if _is_il(name) or is_injured:
        return f"IL — evaluate when healthy"
    if pa and pa < 20:
        return "Depth piece — insufficient sample"
    if pa and pa < 30:
        return "Monitor — small sample, limited role"
    if grade == "Elite":
        return "Keep — core piece"
    if grade == "Above Average":
        return "Keep — valuable contributor"
    if grade == "Average":
        if luck and luck < -0.030:
            return "Keep — unlucky, improvement coming"
        return "Keep — serviceable"
    if grade == "Below Average":
        if luck and luck < -0.030:
            return "Monitor — unlucky, give more time"
        if pa and pa < 60:
            return "Monitor — small sample"
        return "Consider replacement"
    if grade == "DFA":
        if pa and pa < 30:
            return "Depth piece — small sample, not enough to judge"
        return "DFA — below replacement level"
    return "Monitor"

def _pitcher_action(name: str, grade: str,
                    luck: float, era: float, role: str) -> str:
    if _is_dfa(name):
        return f"DFA — {_get_dfa_reason(name)}"
    for k, v in AAA_CANDIDATES.items():
        if k.lower() in name.lower():
            return f"Option to AAA — {v}"
    # specific role overrides
    if "Brash" in name:
        return "Promote to closer — 0.60 ERA, elite"
    if "Muñoz" in name or "Munoz" in name:
        return "Role change — move to setup, Brash to close"
    if "Miller" in name and role == "SP":
        return "Give more starts — xwOBA top 3% MLB"
    if "Castillo" in name:
        return "Keep in 6-man — ERA trending down"
    if "Kirby" in name:
        return "Monitor — K rate declined, piggyback with Miller"
    if grade == "Elite":
        return "Keep — core piece"
    if grade == "Above Average":
        return "Keep — valuable"
    if grade == "Average":
        if luck and luck > 0.030:
            return "Keep — unlucky, better than ERA shows"
        return "Keep — serviceable"
    if grade == "Below Average":
        if luck and luck > 0.030:
            return "Monitor — unlucky, give time"
        return "Consider replacement"
    if grade == "DFA":
        return "DFA — below replacement level"
    return "Monitor"


# ── main grader ───────────────────────────────────────────────────────────────
def grade_players(data: dict) -> dict:
    """
    Grades ALL Mariners players from bbref roster.
    Supplements with Statcast where available.
    """
    print("\n[grade] Grading ALL Mariners players...")

    batter_grades  = []
    pitcher_grades = []

    # ── get data sources ──
    bat_bbref  = data.get("seattle", {}).get("batting",  pd.DataFrame())
    pit_bbref  = data.get("seattle", {}).get("pitching", pd.DataFrame())
    val_bat    = data.get("seattle", {}).get("value_batting",  pd.DataFrame())
    val_pit    = data.get("seattle", {}).get("value_pitching", pd.DataFrame())
    sea_sc_bat = data.get("statcast", {}).get("sea_batters",  pd.DataFrame())
    sea_sc_pit = data.get("statcast", {}).get("sea_pitchers", pd.DataFrame())
    bat_luck   = data.get("statcast", {}).get("bat_luck", pd.DataFrame())
    pit_luck   = data.get("statcast", {}).get("pit_luck", pd.DataFrame())
    roster_df  = data.get("seattle", {}).get("roster", pd.DataFrame())

    # ── grade every batter from bbref ──
    if not bat_bbref.empty and "Name" in bat_bbref.columns:
        for _, row in bat_bbref.iterrows():
            name = str(row.get("Name","")).strip()
            if not name or name in ("Name","Tm",""):
                continue

            # core stats from bbref
            pa  = pd.to_numeric(row.get("PA",  0), errors="coerce") or 0
            ops = pd.to_numeric(row.get("OPS", None), errors="coerce")
            ba  = pd.to_numeric(row.get("BA",  None), errors="coerce")
            obp = pd.to_numeric(row.get("OBP", None), errors="coerce")
            slg = pd.to_numeric(row.get("SLG", None), errors="coerce")
            hr  = pd.to_numeric(row.get("HR",  0), errors="coerce") or 0
            rbi = pd.to_numeric(row.get("RBI", 0), errors="coerce") or 0
            sb  = pd.to_numeric(row.get("SB",  0), errors="coerce") or 0

            # WAR from value table
            war = None
            if not val_bat.empty and "Name" in val_bat.columns:
                name_parts = name.replace("(","").replace(")","").replace("40-man","").replace("10-day IL","").replace("15-day IL","").strip().split()
                last = name_parts[-1].strip() if name_parts else name.split(",")[0].strip()
                m = val_bat[val_bat["Name"].str.contains(
                    last, case=False, na=False, regex=False)]
                if not m.empty and "WAR" in m.columns:
                    war = pd.to_numeric(m["WAR"].values[0],
                                        errors="coerce")

            # Statcast supplement
            # bbref format: "Julio Rodríguez" → last name is last word
            # statcast format: "Rodríguez, Julio" → last name before comma
            xwoba = None
            woba  = None
            barrel = None
            hardhit = None
            if not sea_sc_bat.empty and "Name" in sea_sc_bat.columns:
                # try last word of bbref name as last name
                name_parts = name.replace("(","").replace(")","").replace("40-man","").replace("10-day IL","").replace("15-day IL","").strip().split()
                last = name_parts[-1].strip() if name_parts else name.split(",")[0].strip()
                m = sea_sc_bat[sea_sc_bat["Name"].str.contains(
                    last, case=False, na=False, regex=False)]
                if not m.empty:
                    xwoba   = pd.to_numeric(
                        m["xwOBA"].values[0], errors="coerce")
                    woba    = pd.to_numeric(
                        m["wOBA"].values[0],  errors="coerce")
                    barrel  = pd.to_numeric(
                        m.get("Barrel%", pd.Series([None])).values[0],
                        errors="coerce")
                    hardhit = pd.to_numeric(
                        m.get("HardHit%", pd.Series([None])).values[0],
                        errors="coerce")

            # luck
            luck = round(float(woba) - float(xwoba), 3) \
                   if woba and xwoba and \
                   not pd.isna(woba) and not pd.isna(xwoba) else None

            # grade — prefer xwOBA, fall back to OPS
            xg = _grade_xwoba(xwoba)
            grade = xg if xg else _grade_ops(ops)

            # franchise / DFA / IL overrides
            role = "IL" if _is_il(name) else \
                   ("Starter" if pa >= 100 else "Bench")

            action = _batter_action(name, grade, luck, pa, war)
            note   = _generate_batter_note(pa, ops, xwoba, woba,
                                           barrel, hardhit, hr, None)

            # age adjustment for young players
            if grade in ("Below Average","DFA") and _is_franchise(name):
                grade = "Below Average — developing"

            batter_grades.append({
                "name":    name,
                "grade":   grade,
                "role":    role,
                "action":  action,
                "note":    note,
                "PA":      int(pa),
                "OPS":     round(ops, 3)    if ops    else None,
                "BA":      round(ba,  3)    if ba     else None,
                "HR":      int(hr),
                "RBI":     int(rbi),
                "SB":      int(sb),
                "xwOBA":   round(xwoba, 3) if xwoba  else None,
                "wOBA":    round(woba,  3) if woba   else None,
                "luck":    luck,
                "Barrel%": round(barrel,1) if barrel else None,
                "HardHit%":round(hardhit,1)if hardhit else None,
                "WAR":     round(war, 1)   if war    else None,
            })

    # ── grade every pitcher from bbref ──
    if not pit_bbref.empty and "Name" in pit_bbref.columns:
        for _, row in pit_bbref.iterrows():
            name = str(row.get("Name","")).strip()
            if not name or name in ("Name","Tm",""):
                continue

            # core stats
            era  = pd.to_numeric(row.get("ERA",  None), errors="coerce")
            whip = pd.to_numeric(row.get("WHIP", None), errors="coerce")
            ip   = pd.to_numeric(row.get("IP",   0),    errors="coerce") or 0
            so   = pd.to_numeric(row.get("SO",   0),    errors="coerce") or 0
            bb   = pd.to_numeric(row.get("BB",   0),    errors="coerce") or 0
            hr   = pd.to_numeric(row.get("HR",   0),    errors="coerce") or 0
            gs   = pd.to_numeric(row.get("GS",   0),    errors="coerce") or 0
            g    = pd.to_numeric(row.get("G",    0),    errors="coerce") or 0
            sv   = pd.to_numeric(row.get("SV",   0),    errors="coerce") or 0
            w    = pd.to_numeric(row.get("W",    0),    errors="coerce") or 0
            l    = pd.to_numeric(row.get("L",    0),    errors="coerce") or 0

            # derived stats
            k9   = round(so / ip * 9, 1) if ip > 0 else None
            bb9  = round(bb / ip * 9, 1) if ip > 0 else None
            kbb  = round(so / bb, 2)     if bb > 0 else None

            # WAR
            war = None
            if not val_pit.empty and "Name" in val_pit.columns:
                name_parts = name.replace("(","").replace(")","").replace("40-man","").replace("10-day IL","").replace("15-day IL","").strip().split()
                last = name_parts[-1].strip() if name_parts else name.split(",")[0].strip()
                m = val_pit[val_pit["Name"].str.contains(
                    last, case=False, na=False, regex=False)]
                if not m.empty and "WAR" in m.columns:
                    war = pd.to_numeric(m["WAR"].values[0],
                                        errors="coerce")

            # Statcast supplement
            xwoba_against = None
            woba_against  = None
            whiff = None
            barrel_against = None
            if not sea_sc_pit.empty and "Name" in sea_sc_pit.columns:
                name_parts = name.replace("(","").replace(")","").replace("40-man","").replace("10-day IL","").replace("15-day IL","").strip().split()
                last = name_parts[-1].strip() if name_parts else name.split(",")[0].strip()
                m = sea_sc_pit[sea_sc_pit["Name"].str.contains(
                    last, case=False, na=False, regex=False)]
                if not m.empty:
                    xwoba_against  = pd.to_numeric(
                        m["xwOBA_against"].values[0],  errors="coerce")
                    woba_against   = pd.to_numeric(
                        m["wOBA_against"].values[0],   errors="coerce")
                    whiff = pd.to_numeric(
                        m.get("Whiff%", pd.Series([None])).values[0],
                        errors="coerce")
                    barrel_against = pd.to_numeric(
                        m.get("Barrel%_against",
                              pd.Series([None])).values[0],
                        errors="coerce")

            # luck for pitchers
            luck = round(float(woba_against) - float(xwoba_against), 3) \
                   if woba_against and xwoba_against and \
                   not pd.isna(woba_against) and \
                   not pd.isna(xwoba_against) else None

            # role
            role = "SP" if gs >= 3 else ("CL" if sv >= 3 else "RP")

            # grade — blend ERA + xwOBA + WAR
            grade = _grade_era(era, xwoba_against, war, ip)

            action = _pitcher_action(name, grade, luck, era, role)
            note   = _generate_pitcher_note(era, xwoba_against, k9,
                                            bb9, whiff, war, ip)

            pitcher_grades.append({
                "name":           name,
                "grade":          grade,
                "role":           role,
                "action":         action,
                "note":           note,
                "G":              int(g),
                "GS":             int(gs),
                "IP":             round(ip, 1),
                "ERA":            round(era, 2)  if era   else None,
                "WHIP":           round(whip, 3) if whip  else None,
                "K/9":            k9,
                "BB/9":           bb9,
                "K/BB":           kbb,
                "W":              int(w),
                "L":              int(l),
                "SV":             int(sv),
                "xwOBA_against":  round(xwoba_against, 3)
                                  if xwoba_against else None,
                "wOBA_against":   round(woba_against, 3)
                                  if woba_against  else None,
                "luck":           luck,
                "Whiff%":         round(whiff, 1)          if whiff  else None,
                "Barrel%_against":round(barrel_against, 1)
                                  if barrel_against else None,
                "WAR":            round(war, 1) if war else None,
            })

    # ── summary ──
    all_grades  = [g["grade"].split("—")[0].strip()
                   for g in batter_grades + pitcher_grades]
    dfa_list    = [g["name"] for g in batter_grades + pitcher_grades
                   if "DFA" in g.get("action","")
                   and "small sample" not in g.get("action","").lower()
                   and "depth piece" not in g.get("action","").lower()
                   and not any(fp.lower() in g["name"].lower()
                               for fp in FRANCHISE_PLAYERS)]
    elite_list  = [g["name"] for g in batter_grades + pitcher_grades
                   if g["grade"] == "Elite"]
    concern_list= [g["name"] for g in batter_grades + pitcher_grades
                   if ("Monitor" in g.get("action","") or
                      "Role change" in g.get("action",""))
                   and "Small sample" not in g.get("grade","")
                   and (g.get("IP",0) or 0) >= 8
                      or (g.get("PA",0) or 0) >= 30
                      and ("Monitor" in g.get("action","") or
                           "Role change" in g.get("action",""))
                   ]

    summary = {
        "total_batters":  len(batter_grades),
        "total_pitchers": len(pitcher_grades),
        "elite":          elite_list,
        "dfa":            dfa_list,
        "concerns":       concern_list,
        "grade_dist":     {g: all_grades.count(g) for g in set(all_grades)},
    }

    print(f"[grade] {len(batter_grades)} batters, "
          f"{len(pitcher_grades)} pitchers graded.")
    return {"batters": batter_grades, "pitchers": pitcher_grades,
            "summary": summary}


# ── pretty print ──────────────────────────────────────────────────────────────
def print_grades(grades: dict):
    ORDER = ["Elite","Above Average","Average",
             "Below Average","Below Average — developing",
             "DFA","Unknown"]

    def sort_key(g):
        base = g["grade"].split("—")[0].strip()
        return ORDER.index(base) if base in ORDER else 99

    print(f"\n{'='*80}")
    print("SEATTLE MARINERS — FULL ROSTER GRADES")
    print(f"{'='*80}")

    # ── batters ──
    print(f"\n── BATTERS ({len(grades['batters'])}) ──")
    print(f"{'Name':<28} {'Grade':<25} {'PA':>4} {'OPS':>5} "
          f"{'xwOBA':>6} {'Luck':>6} {'HR':>3} {'WAR':>4}  Action")
    print("─" * 120)

    for g in sorted(grades["batters"], key=sort_key):
        ops   = f"{g['OPS']:.3f}"   if g["OPS"]   else "  N/A"
        xw    = f"{g['xwOBA']:.3f}" if g["xwOBA"] else "  N/A"
        luck  = f"{g['luck']:+.3f}" if g["luck"]  else "   N/A"
        war   = f"{g['WAR']:>4.1f}" if g["WAR"]   else " N/A"
        note  = f"  [{g['note']}]"  if g["note"]  else ""
        print(f"{g['name']:<28} {g['grade']:<25} {g['PA']:>4} "
              f"{ops:>5} {xw:>6} {luck:>6} {g['HR']:>3} {war:>4}  "
              f"{g['action']}{note}")

    # ── pitchers ──
    print(f"\n── PITCHERS ({len(grades['pitchers'])}) ──")
    print(f"{'Name':<28} {'Rol':<3} {'Grade':<20} {'IP':>5} "
          f"{'ERA':>5} {'WHIP':>5} {'K/9':>4} {'xwOBA':>6} "
          f"{'Luck':>6} {'WAR':>4}  Action")
    print("─" * 130)

    for g in sorted(grades["pitchers"], key=sort_key):
        era   = f"{g['ERA']:.2f}"           if g["ERA"] is not None else " 0.00"
        whip  = f"{g['WHIP']:.3f}"          if g["WHIP"]          else "  N/A"
        k9    = f"{g['K/9']:.1f}"           if g["K/9"]           else " N/A"
        xw    = f"{g['xwOBA_against']:.3f}" if g["xwOBA_against"] else "  N/A"
        luck  = f"{g['luck']:+.3f}"         if g["luck"]          else "   N/A"
        war   = f"{g['WAR']:>4.1f}"         if g["WAR"]           else " N/A"
        note  = f"  [{g['note']}]"          if g["note"]          else ""
        print(f"{g['name']:<28} {g['role']:<3} {g['grade']:<20} "
              f"{g['IP']:>5} {era:>5} {whip:>5} {k9:>4} {xw:>6} "
              f"{luck:>6} {war:>4}  {g['action']}{note}")

    # ── summary ──
    s = grades["summary"]
    print(f"\n── SUMMARY ──")
    print(f"  Total players:  {s['total_batters']} batters, "
          f"{s['total_pitchers']} pitchers")
    print(f"  Elite:          {', '.join(s['elite']) or 'none'}")
    print(f"  DFA candidates: {', '.join(s['dfa']) or 'none'}")
    print(f"  Concerns:       {', '.join(s['concerns']) or 'none'}")
    print(f"  Grade dist:     {s['grade_dist']}")
    print(f"\n{'='*80}\n")


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_builder import build_all
    data   = build_all(2026)
    grades = grade_players(data)
    print_grades(grades)