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
import re
from roster import get_roster_last_names
from name_matching import last_name_only


def _clean_name_for_lookup(name: str) -> list:
    """
    Strips ANY parenthetical annotation from a player name before
    extracting their last name for a WAR/Statcast table lookup --
    "(60-day IL)", "(15-day IL)", "(40-man)", or any future variant.

    BUG FIX 2026-09-01: the previous version hardcoded specific phrases
    to strip ("10-day IL", "15-day IL", "40-man") but never "60-day IL"
    or "7-day IL". For any player tagged with one of those two, the
    leftover word after splitting was literally "IL" (not their
    surname) -- e.g. "Matt Brash (60-day IL)" produced last="IL", and
    a substring search for "IL" in the WAR table coincidentally matched
    "Gilbert" (contains "il") first, silently giving Brash, Davila, and
    anyone else on a 60-day or 7-day IL stint someone ELSE's WAR and
    xwOBA. A real user caught this by noticing three unrelated pitchers
    (Brash, Gilbert, Davila) all showing an identical WAR value -- not
    a coincidence, all three were resolving to Gilbert's row. Using a
    regex to strip ANY "(...)" content generically fixes this for every
    current and future annotation variant, not just the ones someone
    thought to hardcode.
    """
    cleaned = re.sub(r"\([^)]*\)", "", name).strip()
    return cleaned.split() if cleaned else []

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

# flavor text ONLY -- consulted by _get_il_note() after _is_il() has
# already confirmed a player is on the IL via bbref's real annotation.
# Keep this in sync with actual current IL status (e.g. via ESPN), but a
# stale/missing entry here is low-risk now -- it just falls back to a
# generic "on IL" note instead of misidentifying someone as injured (that
# used to be possible; see _is_il()'s docstring for the bug this replaced).
# Last synced 2026-07-23 vs ESPN IL report.
IL_PLAYERS = {
    "Refsnyder":  "10-day IL — DH depth, est. return Jul 24",
    "Donovan":    "10-day IL — .839 OPS when healthy, est. return Jul 28",
    "Wilson":     "60-day IL — bench depth, est. return Jul 28",
    "Vargas":     "60-day IL — bullpen depth, est. return Aug 3",
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
    # Trust bbref's own live annotation on the name (e.g. "Rob Refsnyder
    # (10-day IL)") -- this reflects real, current status. The hardcoded
    # IL_PLAYERS dict below is used only to add flavor text once a player
    # is already confirmed IL via this check, NOT as an independent
    # trigger -- it previously matched on bare last-name fragments (e.g.
    # "Crawford") regardless of real status, which incorrectly flagged
    # J.P. Crawford as "on IL" while he was merely day-to-day.
    return "IL" in name

def _get_note(name: str) -> str:
    # notes are generated from data now
    return ""

# UNUSED as of 2026-09-01 -- _batter_action()/_pitcher_action() no longer
# call these three functions. Kept here (not deleted) only because
# FRANCHISE_PLAYERS/DFA_CANDIDATES/IL_PLAYERS themselves are still used
# elsewhere for name-matching (_is_franchise(), _is_dfa(), _is_il()).
# These specific functions pulled the flavor-text half of those dicts,
# including a hardcoded contract dollar figure for franchise players
# that was never based on real scraped data. Removed from the player-
# facing Action column for exactly that reason -- don't wire these back
# in without a real data source behind them.
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
    # only consult the flavor-text dict once bbref's own annotation has
    # already confirmed IL status -- don't match on name fragments alone
    if "IL" not in name:
        return ""
    for k, v in IL_PLAYERS.items():
        if k.lower() in name.lower():
            return v
    return "on IL"


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
    """
    SIMPLIFIED 2026-09-01: previously returned a full sentence with
    baked-in reasoning, including a hardcoded CONTRACT DOLLAR FIGURE for
    franchise players (e.g. "Keep -- 10yr/$210M -- franchise player").
    This project has never scraped or had access to real contract data
    anywhere -- that number was invented flavor text, not derived from
    anything real, sitting next to otherwise genuinely data-driven
    grades. Simplified to a plain category tag with no fabricated
    detail: KEEP, DFA, IL, MONITOR, or DEPTH.
    """
    if _is_franchise(name):
        return "KEEP"
    if _is_dfa(name):
        return "DFA"
    if _is_il(name) or is_injured:
        return "IL"
    if pa and pa < 30:
        return "DEPTH"
    if grade in ("Elite", "Above Average", "Average"):
        return "KEEP"
    if grade == "Below Average":
        return "MONITOR"
    if grade == "DFA":
        return "DFA" if not (pa and pa < 30) else "DEPTH"
    return "MONITOR"

def _pitcher_action(name: str, grade: str,
                    luck: float, era: float, role: str) -> str:
    """
    SIMPLIFIED 2026-09-01: previously included several hardcoded, name-
    matched "special case" overrides (Brash, Munoz, Miller, Castillo,
    Kirby) that returned fixed opinion text regardless of current real
    role or performance data -- e.g. Munoz always got "move to setup,
    Brash to close" even when that wasn't (or was no longer) true, since
    it was a frozen guess, not read from any live source. Removed those
    entirely; every player now gets the same plain, data-driven tag
    based on grade/DFA/AAA status: KEEP, DFA, AAA, or MONITOR.
    """
    if _is_dfa(name):
        return "DFA"
    for k in AAA_CANDIDATES:
        if k.lower() in name.lower():
            return "AAA"
    if grade in ("Elite", "Above Average", "Average"):
        return "KEEP"
    if grade == "Below Average":
        return "MONITOR"
    if grade == "DFA":
        return "DFA"
    return "MONITOR"


# ── main grader ───────────────────────────────────────────────────────────────
def compute_league_baselines(data: dict) -> dict:
    """
    Computes REAL, live, PA/BF-weighted league-average xwOBA (batters)
    and xwOBA-against (pitchers) from the same Statcast leaderboard data
    already scraped for the whole league -- not a hardcoded constant.
    Used as the shrinkage target below.
    """
    bat = data.get("statcast", {}).get("batters", pd.DataFrame())
    pit = data.get("statcast", {}).get("pitchers", pd.DataFrame())

    bat_avg = 0.310  # sane fallback only if the live computation fails
    pit_avg = 0.320

    if not bat.empty and "xwOBA" in bat.columns and "PA" in bat.columns:
        xw = pd.to_numeric(bat["xwOBA"], errors="coerce")
        pa = pd.to_numeric(bat["PA"], errors="coerce")
        valid = xw.notna() & pa.notna() & (pa > 0)
        if valid.any():
            bat_avg = round((xw[valid] * pa[valid]).sum() / pa[valid].sum(), 3)

    if not pit.empty and "xwOBA_against" in pit.columns and "PA" in pit.columns:
        xw = pd.to_numeric(pit["xwOBA_against"], errors="coerce")
        bf = pd.to_numeric(pit["PA"], errors="coerce")  # "PA" here is
                                                        # really batters
                                                        # faced, bbref's
                                                        # own column name
        valid = xw.notna() & bf.notna() & (bf > 0)
        if valid.any():
            pit_avg = round((xw[valid] * bf[valid]).sum() / bf[valid].sum(), 3)

    # BUG FIX 2026-09-01: OPS and ERA fallback grading (used whenever
    # xwOBA/xwOBA-against is missing OR discarded as contaminated -- see
    # the multi-team contamination fix above) previously used the RAW
    # value with NO shrinkage protection at all, unlike the xwOBA path.
    # This let the exact same small-sample noise problem back in through
    # a different door: Will Wilson's 1.133 OPS in 6 PA graded "Elite"
    # even after correctly discarding his contaminated Statcast data,
    # since _grade_ops() has no PA floor or regression built in. Adding
    # real, live, PA/IP-weighted OPS and ERA baselines here so the same
    # _shrink() treatment can be applied to both fallback paths, not
    # just the primary xwOBA-based one.
    ops_avg = 0.720
    era_avg = 4.30
    bat_leaders = data.get("batting", {}).get("all_players", pd.DataFrame())
    pit_leaders = data.get("pitching", {}).get("all_players", pd.DataFrame())

    if not bat_leaders.empty and "OPS" in bat_leaders.columns and "PA" in bat_leaders.columns:
        ops = pd.to_numeric(bat_leaders["OPS"], errors="coerce")
        pa = pd.to_numeric(bat_leaders["PA"], errors="coerce")
        valid = ops.notna() & pa.notna() & (pa > 0)
        if valid.any():
            ops_avg = round((ops[valid] * pa[valid]).sum() / pa[valid].sum(), 3)

    if not pit_leaders.empty and "ERA" in pit_leaders.columns and "IP" in pit_leaders.columns:
        era = pd.to_numeric(pit_leaders["ERA"], errors="coerce")
        ip = pd.to_numeric(pit_leaders["IP"], errors="coerce")
        valid = era.notna() & ip.notna() & (ip > 0)
        if valid.any():
            era_avg = round((era[valid] * ip[valid]).sum() / ip[valid].sum(), 2)

    return {"batter_xwoba": bat_avg, "pitcher_xwoba_against": pit_avg,
            "batter_ops": ops_avg, "pitcher_era": era_avg}


def _sample_confidence(n: float, thresholds: tuple) -> str:
    """
    Real, separate confidence indicator based purely on sample size --
    NOT baked into the grade or the stat itself. This is the resolution
    to a real tension found tonight: Matt Brash's 0.54 ERA over 16.2 IP
    genuinely IS an elite rate (shrinking the ERA itself to "fix" this
    was tried and reverted -- it corrupted a real, true fact about what
    he actually did). But 16.2 IP genuinely isn't enough to be
    CONFIDENT that rate reflects his true, sustained talent level
    either. Both things are true at once: describe what happened
    honestly (the grade), and separately flag how much to trust it
    continuing (this). `thresholds` is (low_cutoff, medium_cutoff) --
    different for batters (PA) vs pitchers (IP) since they stabilize at
    different sample sizes.
    """
    low, medium = thresholds
    if n is None or pd.isna(n):
        return "LOW"
    if n < low:
        return "LOW"
    if n < medium:
        return "MEDIUM"
    return "HIGH"


def _shrink(observed: float, n: float, baseline: float, k: float) -> float:
    """
    Standard shrinkage/regression-to-the-mean: blends an observed rate
    toward a league-average baseline, weighted by how much real sample
    size backs it up. At n=0 this returns the baseline; as n grows much
    larger than k, this converges to the raw observed rate; the
    resulting number will always sit BETWEEN them.

    `k` is the stabilization point -- how many PA/BF it takes for a
    rate stat to become reasonably trustworthy on its own. This project
    uses PA=100 for batter xwOBA and BF=100 for pitcher xwOBA-against,
    both standard, widely-cited sabermetric rules of thumb for wOBA-
    type rate stats (see e.g. Russell Carleton's stabilization-point
    research) -- not numbers invented for this project.

    DIRECTLY MOTIVATED by two real cases found in this project: Will
    Wilson's 2-PA sample producing a nonsensical extrapolated WAR
    figure, and Matt Brash's 16.2-IP sample producing a 743 ERA+ that's
    mathematically correct but not a trustworthy read on his true talent
    level. Shrinkage fixes this at the GRADING step -- the raw stat is
    still shown for transparency, but the grade itself is based on the
    more defensible, shrunk value.
    """
    if observed is None or pd.isna(observed) or n is None or n <= 0:
        return baseline
    return round((observed * n + baseline * k) / (n + k), 3)


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
    baselines  = compute_league_baselines(data)
    print(f"  [grade] league baselines (live, PA/BF-weighted): "
          f"batter xwOBA {baselines['batter_xwoba']}, "
          f"pitcher xwOBA against {baselines['pitcher_xwoba_against']}")

    # BUG FIX 2026-09-01: grade_players() used to pull its player list
    # straight from bbref's season-long team stat pages, which correctly
    # (and by bbref convention) keep a player's partial-season stat line
    # even after they've been traded away -- e.g. Luis Castillo still
    # showed up here with a live "MONITOR" action a full month after
    # being traded to Chicago on Aug 1, since nothing checked whether he
    # was still actually on the team. The live 40-man roster (already
    # scraped and used correctly elsewhere via roster.py) is the real
    # source of truth for "who's actually still here" -- cross-check
    # every player against it and skip anyone who's departed, rather
    # than grading and recommending actions for players who are no
    # longer on the roster to act on.
    current_roster = get_roster_last_names(data)
    if not current_roster:
        print("  [warn] live roster unavailable -- cannot filter departed "
              "players, grades may include anyone with any stats this season")
    departed = []

    def _still_on_roster(name: str) -> bool:
        if not current_roster:
            return True  # roster scrape failed -- don't silently drop
                        # everyone, fail open with a warning instead
        return last_name_only(name) in current_roster

    # ── grade every batter from bbref ──
    if not bat_bbref.empty and "Name" in bat_bbref.columns:
        for _, row in bat_bbref.iterrows():
            name = str(row.get("Name","")).strip()
            if not name or name in ("Name","Tm",""):
                continue
            if not _still_on_roster(name):
                departed.append(name)
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
                name_parts = _clean_name_for_lookup(name)
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
            stats_contaminated = False
            if not sea_sc_bat.empty and "Name" in sea_sc_bat.columns:
                # try last word of bbref name as last name
                name_parts = _clean_name_for_lookup(name)
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

                    # BUG FIX 2026-09-01: Baseball Savant's leaderboard
                    # reports a player's FULL-SEASON total across every
                    # team they played for, not team-specific splits.
                    # For anyone traded mid-season onto Seattle, this
                    # silently blends their (often better) performance
                    # with their OLD team into the same xwOBA/wOBA/
                    # Barrel%/HardHit% used to grade their time here --
                    # a real user caught this directly with Taylor Ward:
                    # Statcast showed 285 PA / .350 xwOBA, but his real
                    # Mariners-only line (bbref) is just 89 PA and a
                    # .366 OPS. The "he's been unlucky" read was
                    # actually wrong -- his good pre-trade performance
                    # was masking a genuinely poor Mariners performance,
                    # not a bug in which player got matched (the name
                    # match was correct), a bug in which TIME PERIOD the
                    # matched numbers cover. Detected by comparing
                    # Statcast PA against bbref's team-specific PA: if
                    # Statcast's PA is meaningfully larger, the numbers
                    # aren't scoped to this team and shouldn't be
                    # trusted for grading -- fall back to OPS-based
                    # grading instead, same fallback path already used
                    # when Statcast data is missing entirely.
                    sc_pa = pd.to_numeric(m.get("PA", pd.Series([None])).values[0],
                                         errors="coerce")
                    if sc_pa and pa and sc_pa > pa * 1.5:
                        print(f"  [warn] {name}: Statcast PA ({int(sc_pa)}) far "
                              f"exceeds Seattle PA ({int(pa)}) -- likely blends "
                              f"pre-trade stats with another team. Discarding "
                              f"Statcast fields, grading on OPS instead.")
                        xwoba = woba = barrel = hardhit = None
                        stats_contaminated = True

            # luck
            luck = round(float(woba) - float(xwoba), 3) \
                   if woba and xwoba and \
                   not pd.isna(woba) and not pd.isna(xwoba) else None

            # grade — prefer xwOBA, fall back to OPS. Uses the SHRUNK
            # xwOBA (see _shrink()) rather than the raw value -- at low
            # PA this pulls a noisy small-sample rate toward the real
            # league average instead of grading off it directly. Has
            # near-zero effect once PA is already large; raw xwOBA is
            # still shown in the output for transparency.
            #
            # BUG FIX 2026-09-01: when stats_contaminated is True (see
            # above), xwoba is deliberately None -- but _shrink(None,...)
            # returns the league BASELINE, not None, since that's the
            # correct behavior for a player who genuinely has no
            # Statcast data at all (want a defined grade, not "Unknown").
            # That meant _grade_xwoba() still produced a real grade off
            # the league-average baseline even for a contaminated player,
            # so the intended "fall back to OPS" branch below never
            # actually triggered -- confirmed directly on Taylor Ward,
            # who kept grading "Average" (from the .320 league baseline)
            # instead of correctly grading on his real .362 OPS. Skip the
            # xwOBA path entirely when contaminated, rather than relying
            # on it happening to return something falsy.
            if stats_contaminated:
                xwoba_shrunk = None
                xg = None
            else:
                xwoba_shrunk = _shrink(xwoba, pa, baselines["batter_xwoba"], k=100)
                xg = _grade_xwoba(xwoba_shrunk)
            # BUG FIX 2026-09-01: OPS fallback now shrunk too (see
            # compute_league_baselines() note above) -- k=150 rather
            # than the 100 used for xwOBA, since OPS is a composite of
            # multiple component rates (BA+OBP+SLG) and is genuinely
            # noisier at a given PA than a single rate stat like xwOBA.
            ops_shrunk = _shrink(ops, pa, baselines["batter_ops"], k=150)
            grade = xg if xg else _grade_ops(ops_shrunk)

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
                "xwOBA_shrunk": xwoba_shrunk,
                "confidence": _sample_confidence(pa, thresholds=(50, 150)),
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
            if not _still_on_roster(name):
                departed.append(name)
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
            bf   = pd.to_numeric(row.get("BF",   0),    errors="coerce") or 0
            w    = pd.to_numeric(row.get("W",    0),    errors="coerce") or 0
            l    = pd.to_numeric(row.get("L",    0),    errors="coerce") or 0

            # derived stats
            k9   = round(so / ip * 9, 1) if ip > 0 else None
            bb9  = round(bb / ip * 9, 1) if ip > 0 else None
            kbb  = round(so / bb, 2)     if bb > 0 else None

            # WAR
            war = None
            if not val_pit.empty and "Name" in val_pit.columns:
                name_parts = _clean_name_for_lookup(name)
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
            stats_contaminated = False
            if not sea_sc_pit.empty and "Name" in sea_sc_pit.columns:
                name_parts = _clean_name_for_lookup(name)
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

                    # BUG FIX 2026-09-01: same multi-team contamination
                    # issue as the batter side above -- Baseball Savant's
                    # per-player search returns SEASON-TOTAL stats across
                    # every team a player appeared for, not a split for
                    # just their time with Seattle. Confirmed on
                    # Dominguez directly: Statcast showed 97 total
                    # batters faced (Chicago White Sox + Seattle
                    # combined), while his real Mariners-only line is
                    # much smaller. Same detection and fallback as
                    # batters: if Statcast's own reported PA (batters
                    # faced) is much larger than bbref's real,
                    # team-specific BF, don't trust it for grading.
                    sc_bf = pd.to_numeric(m.get("PA", pd.Series([None])).values[0],
                                         errors="coerce")
                    if sc_bf and bf and sc_bf > bf * 1.5:
                        print(f"  [warn] {name}: Statcast BF ({int(sc_bf)}) far "
                              f"exceeds Seattle BF ({int(bf)}) -- likely blends "
                              f"stats with another team. Discarding Statcast "
                              f"fields, grading on ERA instead.")
                        xwoba_against = woba_against = whiff = barrel_against = None
                        stats_contaminated = True

            # luck for pitchers
            luck = round(float(woba_against) - float(xwoba_against), 3) \
                   if woba_against and xwoba_against and \
                   not pd.isna(woba_against) and \
                   not pd.isna(xwoba_against) else None

            # role
            role = "SP" if gs >= 3 else ("CL" if sv >= 3 else "RP")

            # grade — blend ERA + xwOBA + WAR. Uses SHRUNK xwOBA against
            # (see _shrink()) rather than the raw value -- same reasoning
            # as the batter side above, using batters-faced (BF) as the
            # sample size to match how the league baseline itself was
            # BF-weighted. Directly motivated by Matt Brash's 16.2-IP,
            # 743-ERA+ sample from earlier this session -- mathematically
            # real, not a trustworthy read on true talent at that sample
            # size. Raw xwOBA against still shown in output for transparency.
            #
            # BUG FIX 2026-09-01: same contamination-bypass fix as the
            # batter side -- when stats_contaminated is True, skip the
            # shrinkage/xwOBA path entirely rather than letting
            # _shrink(None,...) quietly return the league baseline and
            # produce a real (wrong) grade anyway.
            if stats_contaminated:
                xwoba_against_shrunk = None
            else:
                xwoba_against_shrunk = _shrink(xwoba_against, bf,
                                               baselines["pitcher_xwoba_against"], k=100)
            # REVERTED 2026-09-11: shrinking ERA the same way as OPS
            # directly broke a real case we'd already worked through
            # tonight -- Matt Brash's 0.54 ERA over 16.2 IP got pulled
            # all the way to a shrunk 3.22, dropping his grade from
            # Elite to Above Average. That's a real overcorrection, not
            # a fix: 16.2 IP is a meaningful fraction of a full relief
            # season (many closers only throw 50-60 IP total), not the
            # same kind of near-meaningless sample as Wilson's 6 PA --
            # and there was never an actual demonstrated problem with
            # raw ERA the way Wilson's 1.133 OPS proved one for OPS
            # (Hoby Milner's 2.1 IP was already caught by _grade_era's
            # existing ip<5 "Small sample" floor). Reverting to raw ERA;
            # the xwOBA-against and OPS shrinkage fixes both stay, since
            # both had a real, demonstrated failure case behind them.
            grade = _grade_era(era, xwoba_against_shrunk, war, ip)

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
                "xwOBA_against_shrunk": xwoba_against_shrunk,
                "confidence": _sample_confidence(ip, thresholds=(20, 50)),
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

    # BUG FIX 2026-09-01: these three lists were still checking for
    # action text from BEFORE the KEEP/DFA/MONITOR/AAA/IL/DEPTH tag
    # simplification earlier tonight -- "Monitor" (mixed case, no
    # longer exists -- the real tag is "MONITOR") and "Role change"
    # (removed entirely, was one of the hardcoded overrides taken out).
    # Since Python's `in` is case-sensitive, "Monitor" in "MONITOR" is
    # False, so concern_list was silently almost always empty ("Concerns:
    # none" even with several real MONITOR-tagged players visible in the
    # same report) -- a real user caught this directly. Also
    # deduplicating by name here: a player who appears as both a batter
    # AND a mop-up pitcher (e.g. Leo Rivas) could legitimately hit the
    # DFA condition in both of their entries, showing up twice in what
    # should be a simple name list.
    seen_dfa = set()
    dfa_list = []
    for g in batter_grades + pitcher_grades:
        if (g.get("action","") == "DFA"
                and not any(fp.lower() in g["name"].lower() for fp in FRANCHISE_PLAYERS)
                and g["name"] not in seen_dfa):
            dfa_list.append(g["name"])
            seen_dfa.add(g["name"])

    seen_elite = set()
    elite_list = []
    for g in batter_grades + pitcher_grades:
        if g["grade"] == "Elite" and g["name"] not in seen_elite:
            elite_list.append(g["name"])
            seen_elite.add(g["name"])

    seen_concern = set()
    concern_list = []
    for g in batter_grades + pitcher_grades:
        action = g.get("action", "")
        is_concern = action == "MONITOR"
        big_enough_sample = (g.get("IP", 0) or 0) >= 8 or (g.get("PA", 0) or 0) >= 30
        if is_concern and big_enough_sample and g["name"] not in seen_concern:
            concern_list.append(g["name"])
            seen_concern.add(g["name"])

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
    if departed:
        print(f"  [grade] excluded {len(departed)} player(s) no longer on "
              f"the live roster (traded/DFA'd/released): {', '.join(departed)}")

    # SEPARATE, NEW as of 2026-09-01: some 40-man roster players have
    # never actually appeared in a game this season (never called up,
    # or a recent minor-league addition) -- bbref's season stat
    # leaderboards only include players with real accumulated PA/IP, so
    # these players never got a row to grade at all. They're neither
    # "departed" (they're still on the roster) nor gradeable (no stats
    # exist yet). Rather than silently vanishing from the report, list
    # them plainly. Not classified as batter/pitcher since the roster
    # scrape has no clean position field -- guessing that would risk
    # exactly the kind of wrong-category mistake this project has spent
    # a lot of effort fixing elsewhere.
    graded_last_names = {last_name_only(p["name"]) for p in batter_grades + pitcher_grades}
    departed_last_names = {last_name_only(n) for n in departed}
    no_action_players = []
    if not roster_df.empty and "Name" in roster_df.columns:
        for _, row in roster_df.iterrows():
            rname = str(row.get("Name", "")).strip()
            if not rname:
                continue
            rlast = last_name_only(rname)
            if rlast not in graded_last_names and rlast not in departed_last_names:
                no_action_players.append(rname)

    if no_action_players:
        print(f"  [grade] {len(no_action_players)} roster player(s) with no "
              f"MLB action yet this season: {', '.join(no_action_players)}")

    return {"batters": batter_grades, "pitchers": pitcher_grades,
            "no_mlb_action": no_action_players,
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
          f"{'xwOBA':>6} {'Luck':>6} {'HR':>3} {'WAR':>4} {'Conf':>4}  Action")
    print("─" * 120)

    for g in sorted(grades["batters"], key=sort_key):
        ops   = f"{g['OPS']:.3f}"   if g["OPS"]   else "  N/A"
        xw    = f"{g['xwOBA']:.3f}" if g["xwOBA"] else "  N/A"
        luck  = f"{g['luck']:+.3f}" if g["luck"]  else "   N/A"
        war   = f"{g['WAR']:>4.1f}" if g["WAR"]   else " N/A"
        # note field intentionally not appended here anymore -- every
        # number it restated (HR, WAR, xwOBA gap) is already visible in
        # its own column on this same row, so the extra prose was pure
        # redundancy on top of what the user wanted as a plain tag
        print(f"{g['name']:<28} {g['grade']:<25} {g['PA']:>4} "
              f"{ops:>5} {xw:>6} {luck:>6} {g['HR']:>3} {war:>4} "
              f"{g.get('confidence','?'):>4}  {g['action']}")

    # ── pitchers ──
    print(f"\n── PITCHERS ({len(grades['pitchers'])}) ──")
    print(f"{'Name':<28} {'Rol':<3} {'Grade':<20} {'IP':>5} "
          f"{'ERA':>5} {'WHIP':>5} {'K/9':>4} {'xwOBA':>6} "
          f"{'Luck':>6} {'WAR':>4} {'Conf':>4}  Action")
    print("─" * 130)

    for g in sorted(grades["pitchers"], key=sort_key):
        era   = f"{g['ERA']:.2f}"           if g["ERA"] is not None else " 0.00"
        whip  = f"{g['WHIP']:.3f}"          if g["WHIP"]          else "  N/A"
        k9    = f"{g['K/9']:.1f}"           if g["K/9"]           else " N/A"
        xw    = f"{g['xwOBA_against']:.3f}" if g["xwOBA_against"] else "  N/A"
        luck  = f"{g['luck']:+.3f}"         if g["luck"]          else "   N/A"
        war   = f"{g['WAR']:>4.1f}"         if g["WAR"]           else " N/A"
        print(f"{g['name']:<28} {g['role']:<3} {g['grade']:<20} "
              f"{g['IP']:>5} {era:>5} {whip:>5} {k9:>4} {xw:>6} "
              f"{luck:>6} {war:>4} {g.get('confidence','?'):>4}  {g['action']}")

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