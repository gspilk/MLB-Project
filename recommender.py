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
    # Emerson Hancock removed 2026-07-28 -- per current reports he's not
    # untouchable anymore. (Also worth noting: this list's stat text is
    # hand-typed and can go stale like this one had -- "2.74 ERA" no
    # longer matched his actual live ERA even before this removal. If you
    # keep entries here, it's worth periodically checking the number
    # against the real report output, not just the player's status.)
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

# team-name-to-abbreviation map -- standings data uses full names
# ("Houston Astros"), but the leaderboard scrapers' Team column (when
# populated) uses 3-letter codes ("HOU"). Needed so live seller detection
# can match against whichever format shows up.
TEAM_ABBR = {
    "arizona diamondbacks": "ari", "atlanta braves": "atl",
    "baltimore orioles": "bal", "boston red sox": "bos",
    "chicago cubs": "chc", "chicago white sox": "chw",
    "cincinnati reds": "cin", "cleveland guardians": "cle",
    "colorado rockies": "col", "detroit tigers": "det",
    "houston astros": "hou", "kansas city royals": "kcr",
    "los angeles angels": "laa", "los angeles dodgers": "lad",
    "miami marlins": "mia", "milwaukee brewers": "mil",
    "minnesota twins": "min", "new york mets": "nym",
    "new york yankees": "nyy", "athletics": "ath",
    "philadelphia phillies": "phi", "pittsburgh pirates": "pit",
    "san diego padres": "sdp", "san francisco giants": "sfg",
    "seattle mariners": "sea", "st. louis cardinals": "stl",
    "tampa bay rays": "tbr", "texas rangers": "tex",
    "toronto blue jays": "tor", "washington nationals": "was",
}

# Front-office intent doesn't always match record -- a slumping big-market
# contender (e.g. Houston mid-season) isn't a real seller even if sub-.500,
# and a mediocre-record team can still be an obvious seller. Update these
# by hand as situations become clear; this is the same maintenance model
# as NOT_AVAILABLE_NAMES, just for teams instead of players.
SELLER_OVERRIDE_EXCLUDE = {"houston astros"}
SELLER_OVERRIDE_INCLUDE = set()


def _compute_seller_teams(data: dict) -> set:
    """
    Determines likely-seller teams from LIVE standings instead of a
    hardcoded, season-to-season guess -- replaces the old static
    SELLER_TEAMS set. Refreshes automatically: standings_scraper.py's own
    24-hour cache TTL means every main.py run on a new day re-pulls
    current records, so this needs no separate "daily update" mechanism.

    Flags the bottom third of MLB by win% as likely sellers, then applies
    the manual override lists above for cases the record alone gets
    wrong (see SELLER_OVERRIDE_EXCLUDE/INCLUDE docstring above).

    Returns a set of lowercased full team names AND their abbreviations,
    so _availability() can match against either format.
    """
    all_teams = _safe(data, "standings", "all_teams")
    if all_teams is None or all_teams.empty:
        print("  [warn] no live standings -- seller detection unavailable, "
              "availability will show CHECK AVAILABILITY for everyone")
        return set()

    df = all_teams.copy()
    df["W-L%"] = pd.to_numeric(df["W-L%"], errors="coerce")
    df = df.dropna(subset=["W-L%"]).sort_values("W-L%")

    n_sellers = max(1, len(df) // 3)  # bottom third of MLB
    sellers = {str(t).lower() for t in df.head(n_sellers)["Tm"]}

    sellers -= {t.lower() for t in SELLER_OVERRIDE_EXCLUDE}
    sellers |= {t.lower() for t in SELLER_OVERRIDE_INCLUDE}

    # add matching abbreviations so a "HOU"-style Team value also matches
    abbrs = {TEAM_ABBR[t] for t in sellers if t in TEAM_ABBR}
    return sellers | abbrs


def _availability(team_name, seller_teams=None):
    tn = str(team_name).lower()
    if "mariners" in tn or "seattle" in tn:
        return "ON ROSTER"
    seller_teams = seller_teams if seller_teams is not None else set()
    if any(s == tn or s in tn for s in seller_teams):
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
def _position_has_il_player(data, pos_code):
    """
    Checks whether the player(s) at a given batting position are
    CURRENTLY on IL, using bbref's own live annotation on the Name
    field (e.g. "Brendan Donovan (7-day IL)") -- same trusted, live
    source used throughout this project. Replaces a hardcoded, frozen
    position set that used to assume a fixed list of positions
    (C/3B/SS/DH/RP) always had an IL return pending, regardless of
    whether that was still true. Returns (bool, player_name_or_None).
    """
    bat = data.get("seattle", {}).get("batting")
    if bat is None or bat.empty or "Pos" not in bat.columns:
        return False, None
    matches = bat[bat["Pos"] == pos_code]
    for _, row in matches.iterrows():
        name = str(row.get("Name", ""))
        if "IL" in name:
            return True, name
    return False, None


def _pitching_role_has_il_player(data, role_prefix):
    """Same idea as _position_has_il_player() but for SP/RP, using the
    pitching table's GS count to distinguish starters from relievers
    since there's no direct Pos column for pitching role."""
    pit = data.get("seattle", {}).get("pitching")
    if pit is None or pit.empty:
        return False, None
    for _, row in pit.iterrows():
        name = str(row.get("Name", ""))
        if "IL" not in name:
            continue
        gs = pd.to_numeric(row.get("GS", 0), errors="coerce") or 0
        is_sp = gs >= 3
        if (role_prefix == "SP" and is_sp) or (role_prefix == "RP" and not is_sp):
            return True, name
    return False, None


def _stats_to_improve(data):
    targets = []

    bat = _safe(data, "overview", "batting")
    if bat is not None and not bat.empty:
        tm_col = next((c for c in ["Tm","Team"] if c in bat.columns), "Tm")
        for stat in ["R","H","HR","RBI","BA","OBP","SLG","OPS","OPS+","BB"]:
            val, rank, total = _sea_rank(bat, stat, tm_col)
            if rank and total and rank > 15:
                is_internal = stat in {"OBP","BB","SS","C","3B"}
                targets.append({
                    "category": "Offense",
                    "stat":     stat,
                    "value":    val,
                    "rank":     rank,
                    "total":    total,
                    "priority": "HIGH" if rank > 20 else "MONITOR",
                    "fix":      _batting_fix(stat, is_internal),
                    "internal": is_internal,
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
                    "fix":      _pitching_fix(stat, True),
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
                    "fix":      _fielding_fix(stat, True),
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
                if pos in ("1B","C","DH","SS","3B","RF"):
                    is_internal, il_name = _position_has_il_player(data, pos)
                elif pos in ("SP","RP"):
                    is_internal, il_name = _pitching_role_has_il_player(data, pos)
                else:
                    is_internal, il_name = False, None

                if is_internal and il_name:
                    fix_text = f"INTERNAL — awaiting return: {il_name}"
                elif is_internal:
                    fix_text = "INTERNAL"
                else:
                    fix_text = "EXTERNAL — no IL return pending"

                targets.append({
                    "category": "WAR by Position",
                    "stat":     f"{pos} WAR",
                    "value":    round(wv, 1),
                    "rank":     pos_rank,
                    "total":    total,
                    "priority": "HIGH" if pos_rank > 20 else "MONITOR",
                    "fix":      fix_text,
                    "internal": is_internal,
                })

    targets.sort(key=lambda x: (0 if x["priority"]=="HIGH" else 1, x["rank"]))
    return targets


def _batting_fix(stat, internal):
    """
    SIMPLIFIED 2026-09-01: previously returned a full hardcoded sentence
    per stat (e.g. "Crawford 14.7 BB% returning -- OBP will jump
    immediately"), frozen from whenever this was written. Several had
    already gone stale/wrong -- Crawford and Raleigh are both confirmed
    ACTIVE (not on IL) as of this report, yet the old text for OBP/BB/C/
    SS/3B all assumed they were still out. Simplified to match the
    existing internal/external tag already computed above, rather than
    a sentence that can silently drift out of sync with real IL status.
    """
    return "INTERNAL" if internal else "EXTERNAL"


def _pitching_fix(stat, internal):
    return "INTERNAL" if internal else "EXTERNAL"


def _fielding_fix(stat, internal):
    return "INTERNAL" if internal else "EXTERNAL"


def _war_fix(pos, internal):
    return "INTERNAL" if internal else "EXTERNAL"


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

        # BUG FIX 2026-09-01: this was still checking substring patterns
        # from BEFORE the KEEP/DFA/MONITOR/AAA/IL/DEPTH tag simplification
        # earlier tonight -- "small sample"/"depth" substring exclusions
        # (dead now that DFA is an exact tag, never a compound sentence),
        # and action.replace("DFA -- ", "") (silently a no-op now, since
        # that text pattern doesn't exist in the tag anymore -- the
        # "reason" ended up just being the literal word "DFA"). Fixed to
        # use an exact tag match and build a real reason from actual
        # available stats instead of parsing text that no longer carries
        # that information.
        if action == "DFA" and name not in seen:
            seen.add(name)
            stat_bit = (f"{g['WAR']} WAR" if g.get("WAR") is not None else
                       f"{g.get('grade','')} grade")
            moves.append({
                "type":   "DFA",
                "player": name,
                "reason": f"Below replacement level -- {stat_bit}",
                "urgency":"IMMEDIATE",
            })

    # REMOVED 2026-09-01: this used to check for "Role change",
    # "Promote to closer", and "piggyback" substrings in the action
    # text -- all three were hardcoded, name-matched special-case
    # overrides (Munoz/Brash/Kirby specifically) that got removed
    # entirely from _pitcher_action() earlier tonight for being frozen
    # guesses rather than live data (see that function's own docstring).
    # There's no equivalent dynamic detection to replace them with --
    # role changes and rotation piggybacking aren't something derivable
    # from grade/ERA/WAR alone -- so rather than leave permanently-dead
    # checks in place, they're just gone. If real role-change detection
    # ever gets built, it needs actual role/usage data, not a text
    # pattern match on an action tag that no longer carries that info.

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

    # BUG FIX 2026-09-01: this used to return 5 fully hardcoded,
    # never-updated strings -- "Closer: Matt Brash", "Setup: Andres
    # Munoz -- move from closer", "6th man: Luis Castillo" -- written
    # once early in the project and never touched since. A real user
    # caught this directly: Castillo was traded away in August, yet
    # this was STILL recommending him as the 6th starter in a report
    # generated in September. "closer"/"setup"/"high_leverage" are
    # removed entirely -- they were purely redundant with (and by now
    # flatly contradicted by) the real, correctly-sorted `bullpen` list
    # right above, which already shows exactly who's actually performing
    # best today. "piggyback" is removed too -- which two starters
    # should share a workload isn't something derivable from xwOBA
    # ranking alone, so making up a specific pairing would be inventing
    # a recommendation the data doesn't actually support (same reasoning
    # as removing Key Factors/Risks earlier tonight).
    #
    # "6th_starter" IS cleanly derivable -- it's just whichever starter
    # ranks 6th in the same real, sorted `starters` list already being
    # returned -- so that one gets fixed to be live instead of removed.
    sixth_starter = None
    if len(starters) > 5:
        s = starters[5]
        xw = f"{s['xwOBA_against']:.3f}" if s.get("xwOBA_against") else "N/A"
        sixth_starter = f"{s['name']} — ERA {s.get('ERA','N/A')}, xwOBA {xw}"

    return {
        "rotation":      starters,
        "bullpen":       relievers,
        "6th_starter":   sixth_starter,
    }



# ── player targets ────────────────────────────────────────────────────────────
CONTENDERS = {
    "NYY","TBR","LAD","MIL","ATL","PHI",
    "CLE","CHW","SEA","NYM","BOS","PIT",
    "MIN","SDP","SFG","HOU","CHC","STL"
}

from name_matching import key_from_first_last, key_from_last_first
from roster import get_roster_last_names

def _weak_positions(stats: list) -> set:
    """
    Pulls the set of position codes the team has a REAL external need at,
    from _stats_to_improve()'s "WAR by Position" findings -- but only
    where internal=False. A position flagged weak with internal=True
    means the diagnosis itself says an IL return already fixes it (e.g.
    SS/3B/C here are tagged as fixed once Crawford/Donovan/Raleigh are
    back) -- that's not a real trade need, and including it would barely
    filter anything since the team is flagged weak at most positions
    simultaneously purely due to injuries, not talent gaps.

    Maps bbref position labels to the single-character position codes used
    in the batting leaders' "Pos" column (e.g. "2D/H"), so results can be
    matched against those codes.
    """
    POS_CODE = {
        "C": "2", "1B": "3", "2B": "4", "3B": "5", "SS": "6",
        "LF": "7", "CF": "8", "RF": "9", "DH": "D",
    }
    weak = set()
    for s in stats:
        if s.get("category") != "WAR by Position":
            continue
        if s.get("internal"):
            continue  # already covered by an IL return -- not a trade need
        label = s["stat"].replace(" WAR", "").strip()  # e.g. "1B WAR" -> "1B"
        if label in POS_CODE:
            weak.add(POS_CODE[label])
    return weak


def _pos_matches_need(pos_str: str, weak_codes: set) -> bool:
    """True if any of the team's weak-position codes appear in a player's
    bbref Pos string (e.g. weak {'3','D'} matches Pos "2D/H")."""
    if not weak_codes:
        return True  # no specific position weakness flagged -- don't over-filter
    pos_str = str(pos_str)
    return any(code in pos_str for code in weak_codes)


def _find_targets(data: dict, stats: list = None) -> dict:
    """
    Finds realistic MLB trade targets for SEA needs.
    Batters:  on sub-.500 teams, xwOBA .310-.380, position fit
    Pitchers: on sub-.500 teams, xwOBA_against <= .310, reliever

    `stats` (from _stats_to_improve()) drives batter position filtering --
    only players at a position the team is actually flagged weak at are
    surfaced, instead of any high-xwOBA hitter leaguewide regardless of
    whether the team needs that position at all.
    """
    weak_codes = _weak_positions(stats or [])
    if weak_codes:
        print(f"  [targets] filtering batters to positions the team "
              f"actually needs: {sorted(weak_codes)}")
    else:
        print("  [targets] no specific position weakness flagged -- "
              "showing all qualifying batters")

    all_bat = _safe(data, "batting",  "all_players")
    all_pit = _safe(data, "pitching", "all_players")
    sc_bat  = _safe(data, "statcast", "batters")
    sc_pit  = _safe(data, "statcast", "pitchers")

    seller_teams = _compute_seller_teams(data)
    if seller_teams:
        print(f"  [targets] {len([s for s in seller_teams if len(s) > 3])} "
              f"teams flagged as likely sellers based on live standings")

    # live roster, not a hardcoded list -- see roster.py. Updates
    # automatically after trades/call-ups instead of needing manual edits.
    MARINERS_ROSTER = get_roster_last_names(data)
    if not MARINERS_ROSTER:
        print("  [warn] live roster unavailable -- trade targets may "
              "include current Mariners players")

    def _key_b(name, fmt):
        """Join key: last+first_initial, accent normalized."""
        return key_from_first_last(name) if fmt == "fl" else key_from_last_first(name)

    batter_targets  = []
    pitcher_targets = []

    # ── batters ──────────────────────────────────────────────────────────────
    # note: bbref batting leaders page has no team column
    # we filter by position fit and xwOBA quality instead
    # user should verify team availability separately
    try:
        if all_bat is not None and not all_bat.empty and            sc_bat  is not None and not sc_bat.empty  and            "Name" in all_bat.columns and "Name" in sc_bat.columns:

            bat_b = all_bat.copy()
            bat_s = sc_bat.copy()

            bat_b["_key"] = bat_b["Name"].apply(lambda n: _key_b(n,"fl"))
            bat_s["_key"] = bat_s["Name"].apply(lambda n: _key_b(n,"lf"))

            b_cols = ["_key","Name","Pos","league"] +                      [c for c in ["PA","OPS","HR","BA","RBI"]
                      if c in bat_b.columns]
            s_cols = ["_key","Name"] +                      [c for c in ["xwOBA","wOBA","Barrel%","HardHit%","K%","BB%"]
                      if c in bat_s.columns]

            merged = pd.merge(bat_b[b_cols], bat_s[s_cols],
                              on="_key", how="inner")

            for c in ["xwOBA","Barrel%","PA"]:
                if c in merged.columns:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce")

            def _is_mariner_key(key):
                return key.split("_")[0] in MARINERS_ROSTER

            # exclude known untouchable contender players -- built from
            # real names via key_from_first_last() rather than hardcoded
            # key strings. Hardcoded keys silently stopped matching the
            # last time the key format changed (initial -> full first
            # name), which let Judge/Ohtani/Soto/Harper etc. show up as
            # "trade targets." Generating from names avoids that class of
            # bug recurring if the key format ever changes again.
            UNTOUCHABLE_NAMES = [
                "Aaron Judge", "Shohei Ohtani", "Juan Soto", "Bryce Harper",
                "Vladimir Guerrero Jr.", "Ronald Acuna Jr.", "Mike Trout",
                "Mookie Betts", "Yordan Alvarez", "Freddie Freeman",
            ]
            UNTOUCHABLE = {key_from_first_last(n) for n in UNTOUCHABLE_NAMES}

            # players who statistically fit but aren't realistic sells --
            # their team isn't rebuilding, they're extension/franchise
            # pieces, etc. This is a JUDGMENT CALL list, not something
            # derivable from stats -- update it by hand as team situations
            # change (same maintenance model as SELLER_TEAMS/IL_RETURNS).
            # Last reviewed 2026-07-28 against current team competitiveness.
            NOT_AVAILABLE_NAMES = [
                "Ben Rice", "Nick Kurtz", "Willson Contreras",
                "Jonathan Aranda", "Paul Goldschmidt", "Brandon Nimmo",
                "Munetaka Murakami", "Cody Bellinger", "Spencer Steer",
                "Kyle Schwarber", "Matt Olson", "Junior Caminero",
                "Bryan Reynolds", "Pete Alonso", "Taylor Ward",
                "Otto Lopez", "Curtis Mead", "Sal Stewart",
                "Samuel Basallo", "Dalton Rushing", "Corbin Carroll",
                "Bryce Eldridge", "Justin Foscue",
            ]
            NOT_AVAILABLE = {key_from_first_last(n) for n in NOT_AVAILABLE_NAMES}

            mask = (
                (merged["xwOBA"]   >= 0.330) &
                (merged["xwOBA"]   <= 0.420) &
                (merged["Barrel%"] >= 8.0)   &
                (merged["PA"]      >= 100)   &
                (~merged["_key"].apply(_is_mariner_key)) &
                (~merged["_key"].isin(UNTOUCHABLE)) &
                (~merged["_key"].isin(NOT_AVAILABLE)) &
                (merged["Pos"].apply(lambda p: _pos_matches_need(p, weak_codes)))
            )
            tgt = (merged[mask]
                   .sort_values("xwOBA", ascending=False)
                   .drop_duplicates(subset=["_key"])
                   .head(10))
            print(f"  [targets] {len(tgt)} batter targets found")

            for _, row in tgt.iterrows():
                name_y = str(row.get("Name_y",""))
                if "," in name_y:
                    parts = name_y.split(",",1)
                    player_name = f"{parts[1].strip()} {parts[0].strip()}"
                else:
                    player_name = name_y
                if not player_name or player_name == "nan":
                    player_name = str(row.get("Name_x",""))

                pos    = str(row.get("Pos","?"))
                league = str(row.get("league","?"))

                batter_targets.append({
                    "name":         player_name,
                    "team":         "See MLB standings",
                    "pos":          pos,
                    "league":       league,
                    "PA":           row.get("PA"),
                    "OPS":          row.get("OPS"),
                    "HR":           row.get("HR"),
                    "BA":           row.get("BA"),
                    "xwOBA":        round(float(row["xwOBA"]),3),
                    "Barrel%":      row.get("Barrel%"),
                    "HardHit%":     row.get("HardHit%"),
                    "availability": "Check team record",
                    "fit":          "1B/DH/OF trade target",
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
            pit_b["_key"] = pit_b["Name"].apply(lambda n: _key_b(n, "fl"))
            pit_s["_key"] = pit_s["Name"].apply(lambda n: _key_b(n, "lf"))

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
            # same judgment-call mechanism as NOT_AVAILABLE_NAMES for
            # batters -- pitchers confirmed locked up, no-trade clauses,
            # or already dealt to a contender this deadline season.
            # Last reviewed 2026-07-31.
            NOT_AVAILABLE_PITCHERS = [
                "Josh Hader",     # full no-trade clause, signed thru 2028
                "A.J. Minter",    # just traded Mets->Twins, Twins are buyers
            ]
            NOT_AVAILABLE_P = {key_from_first_last(n) for n in NOT_AVAILABLE_PITCHERS}

            mask_p = (
                (merged_p["xwOBA_against"] <= 0.310) &
                (merged_p["IP"]            >= 20)    &
                (~merged_p["_key"].apply(lambda k: k.split("_")[0] in MARINERS_ROSTER)) &
                (~merged_p["_key"].isin(NOT_AVAILABLE_P))
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

                team_val_p = str(row.get("_team",""))
                if not team_val_p or team_val_p in ("nan","","None"):
                    for _tc in ["Tm","_team","Team"]:
                        _v = row.get(_tc)
                        if _v and str(_v) not in ("nan","","None"):
                            team_val_p = str(_v)
                            break
                avail_p = _avail_p(team_val_p or "", seller_teams)
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

    div_rank      = st.get("div_rank")
    mlb_rank      = st.get("mlb_rank")
    wc_games_back = st.get("wc_games_back")
    wc_in_reach   = st.get("wc_in_reach")

    if div_rank is not None:
        division_text = f"{div_rank}{'st' if div_rank==1 else 'nd' if div_rank==2 else 'rd' if div_rank==3 else 'th'} place AL West"
        if mlb_rank is not None:
            division_text += f" (#{mlb_rank}/30 in MLB)"
    else:
        division_text = "Division rank unavailable"

    if wc_games_back is not None:
        if wc_games_back <= 0:
            playoff_path_text = "Currently holding a Wild Card spot"
        else:
            playoff_path_text = (f"Wild Card {'in reach' if wc_in_reach else 'a stretch'} "
                                 f"— {wc_games_back:.1f} games back")
    else:
        playoff_path_text = "Wild Card gap unavailable"

    return {
        "current_record": st.get("record"),
        "projected_wins": pw,
        "floor":          o.get("floor"),
        "ceiling":        o.get("ceiling"),
        "most_likely":    f"{pw}—{(pw or 88)+2} wins",
        "division":       division_text,
        "playoff_path":   playoff_path_text,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def generate_recommendations(data, analysis, grades):
    print("\n[recommend] Generating recommendations...")
    stats          = _stats_to_improve(data)
    player_targets = _find_targets(data, stats)
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

    # IMMEDIATE ROSTER MOVES section REMOVED 2026-09-01 -- see
    # generate_report()'s docstring note in output_report.py for why
    # (deadline-specific content gone stale/wrong a month past the
    # actual Aug 3 deadline). recs["immediate_moves"] is still computed,
    # just no longer printed here.

    print(f"\n── OPTIMAL LINEUP (by xwOBA, active only) ──")
    for i, p in enumerate(lu.get("optimal_order",[])[:9], 1):
        print(f"  {i}. {p['name']:<25} xwOBA {p['xwOBA']:.3f}  "
              f"OPS {p['OPS']:.3f}  HR {p['HR']}")

    print(f"\n── ROTATION (by xwOBA) ──")
    for i, p in enumerate(rb.get("rotation",[]), 1):
        xw  = f"{p['xwOBA_against']:.3f}" if p.get("xwOBA_against") else " N/A"
        era = f"{p['ERA']:.2f}" if p.get("ERA") else "N/A"
        print(f"  {i}. {p['name']:<25} ERA {era:>5}  xwOBA {xw}  {p['action']}")
    if rb.get("6th_starter"):
        print(f"  6th man:   {rb['6th_starter']}")

    print(f"\n── BULLPEN (by xwOBA) ──")
    for p in rb.get("bullpen",[])[:6]:
        xw  = f"{p['xwOBA_against']:.3f}" if p.get("xwOBA_against") else " N/A"
        era = f"{p['ERA']:.2f}" if p.get("ERA") else "N/A"
        print(f"    {p['name']:<25} ERA {era:>5}  xwOBA {xw}  {p['grade']}")

    # ── player targets ──
    tg = recs.get("player_targets", {})
    bt = tg.get("batter_targets", [])
    pt = tg.get("pitcher_targets", [])

    if bt:
        print(f"\n── BATTER TRADE TARGETS (xwOBA .330-.420, Barrel% 8%+) ──")
        print(f"  {'Name':<25} {'Pos':<8} {'Lg':<3} {'xwOBA':>6} "
              f"{'Barrel%':>8} {'HR':>4} {'OPS':>6}")
        print("  " + "─"*65)
        for p in bt:
            print(f"  {str(p['name']):<25} {str(p.get('pos','?')):<8} "
                  f"{str(p.get('league','?')):<3} {str(p['xwOBA']):>6} "
                  f"{str(p.get('Barrel%','N/A')):>8} "
                  f"{str(p.get('HR','N/A')):>4} {str(p.get('OPS','N/A')):>6}")

    if pt:
        print(f"\n── PITCHER TRADE TARGETS (xwOBA against ≤.310, 20+ IP) ──")
        print(f"  {'Name':<25} {'G':>3} {'IP':>5} {'ERA':>5} {'xwOBA vs':>9}")
        print("  " + "─"*55)
        for p in pt:
            era = f"{p['ERA']:.2f}" if p.get('ERA') else "N/A"
            print(f"  {str(p['name']):<25} {str(p.get('G','?')):>3} "
                  f"{str(p.get('IP','?')):>5} {era:>5} "
                  f"{str(p['xwOBA_against']):>9}")

    # DEADLINE STRATEGY section REMOVED 2026-09-01 -- same reasoning as
    # the IMMEDIATE ROSTER MOVES removal above. This section had gone
    # from stale to actively wrong: it was still listing Luis Castillo
    # as a "sell candidate" over a month after he was actually traded
    # away, plus hardcoded contract figures in DO NOT TRADE that were
    # never real scraped data. recs["deadline"] is still computed by
    # _deadline_moves(), just no longer printed here.

    print(f"\n── SEASON OUTLOOK ──")
    print(f"  Record now:   {ou['current_record']}")
    print(f"  Most likely:  {ou['most_likely']}")
    print(f"  Floor/Ceiling:{ou['floor']}—{ou['ceiling']} wins")
    print(f"  Playoff path: {ou['playoff_path']}")
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