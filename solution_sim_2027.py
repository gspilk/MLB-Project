"""
solution_sim.py  (full-roster version)
"Here's the solution" -> "what does it do to the 2027 season?"

1. Builds the 2027 roster position by position. A starter who is a FREE AGENT is an open spot.
2. Searches every combination of signings (one player per spot) for the best team at each budget.
3. Simulates the season with and without the best slate.

Run these first, in this order (each writes files this script reads):
    python offseason_report.py
    python project_2027.py
    python solution_sim.py            # or:  python solution_sim.py --budget 40

WHAT IT ASSUMES (all printed when it runs; the constants are at the top of this file)
  * Hitters are valued on projected 2027 OPS (project_2027.py) plus fielding runs (Rtot, half kept).
  * Players already on the roster and under contract cost $0 (their pay is already spent).
  * Price of a free agent = the larger of his last contract and ~$8M per projected win. Last contract
    alone makes last year's bargains (France, Grichuk) look free, which the market will not allow.
    Use --price prev to see it with last contracts only.
  * Catcher, 2B, CF and DH are frozen (you said those are fine). Left field is locked to re-signing
    Arozarena. Change either with --open "Group" and --lock "Group=Name" (or =none).
  * The rotation is five spots (the 2026 top five by starts). A starter under contract is free; a free-agent starter is a
    spot to fill; any free-agent starter can take any spot. Starters are valued on projected ERA over 170 innings.
  * The result is "what this plan does to the team", not a guarantee: projections have real error.
"""
import argparse
import os

import numpy as np
import pandas as pd

from project_2027 import load_season, norm_key, _ip_to_float

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports")
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "inputs")
SEA_ID = 136
# Official 2026 final (76-86): 664 runs scored, 722 allowed. The player tables summed to ~735 / ~667,
# which made the team look like an 89-win club, so the official totals are used instead (override with --rs / --ra).
SEA_RS_2026 = 664
SEA_RA_2026 = 722

HIT_SLOTS = ["Catchers", "First Basemen", "Second Basemen", "Third Basemen", "Shortstops",
             "Left Fielders", "Center Fielders", "Right Fielders", "Designated Hitters"]
PIT_SLOTS = ["Right-Handed Relievers", "Left-Handed Relievers"]     # one setup spot of each
OF_POOL = {"Left Fielders", "Center Fielders", "Right Fielders", "Designated Hitters"}   # depth can cover any of these
FROZEN = {"Catchers", "Second Basemen", "Center Fielders", "Designated Hitters"}
LOCKS = {"Left Fielders": "Randy Arozarena"}

# ── assumptions ──────────────────────────────────────────────────────────────
SLOT_PA = 600
DEF_KEEP = 0.5
RP_IP = 60
LEVERAGE = 1.3
RUNS_PER_ER = 1.08
SD_BAT = 0.080
SD_ERA = 1.20
SD_DEF = 4.0
PYTH_EXP = 1.83
N_SIMS = 10_000
MARKET_PER_WAR = 8.0       # $M per projected win above replacement
REPL_RUNS_HIT = 20.0       # runs a full-time hitter is worth above a replacement player
REPL_RUNS_RP = 3.5
DEFAULT_RUNS_PER_PA_PER_OPS = 0.31
# ── rotation (five starter spots) ───────────────────────────────────────────
SP_SPOTS = 5
SP_IP = 170                # innings a rotation starter is expected to throw
SD_SP_ERA = 0.80           # talent spread for a starter's true ERA (more innings than a reliever, so smaller)
REPL_SP_ADD = 0.90         # a replacement-level starter is this much worse than league ERA
UNIT = 0.5                 # budget search resolution, $M

# ── the improvement story: three scenarios ──────────────────────────────────
# rebound = how far Raleigh, Naylor and Julio move from their 2026 OPS back toward their 2025 OPS
# (0 = trust the three-year projection as is, 0.5 = halfway, 1.0 = fully back).
# trades = how many "impact bats" may be added by trade; budget = total new salary in $M.
REBOUND_PLAYERS = ["Cal Raleigh", "Josh Naylor", "Julio Rodriguez"]
SCENARIOS = [
    {"name": "Conservative", "rebound": 0.0, "trades": 0, "budget": 50.0,
     "story": "Projection only; free-agent plan"},
    {"name": "Base", "rebound": 0.5, "trades": 2, "budget": 100.0,
     "story": "Three recover halfway; 2 impact bats"},
    {"name": "Optimistic", "rebound": 1.0, "trades": 4, "budget": 150.0,
     "story": "Three fully recover; 4 impact bats"},
]
IMPACT_OPS = 0.850         # placeholder impact bat (about a top-25 hitter) when trade_targets.csv is absent
IMPACT_SALARY = 25.0       # $M per year for the placeholder
IMPACT_SPOTS = ["First Basemen", "Third Basemen", "Shortstops", "Right Fielders", "Designated Hitters"]
SP_FRAC = 0.0              # --sp-rebound: how far returning starters move from 2026 ERA back toward 2025 ERA
SP_TABLE = {}
BUDGET_BONUS = 0.0         # --trade-away: salary freed by trading a player, added to every scenario budget
PLAYOFF_BAR = 85           # wins that usually make the AL field
FRONTIER = [0, 20, 40, 60, 80]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _read(name, folder=None, required=True):
    path = os.path.join(folder or EXPORT_DIR, name)
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"Missing {path}. Run offseason_report.py and project_2027.py first.")
        return None
    return pd.read_csv(path)


def pyth(rs, ra):
    return rs ** PYTH_EXP / (rs ** PYTH_EXP + ra ** PYTH_EXP)


# ── league calibration ───────────────────────────────────────────────────────
def calibrate(rs_override=None, ra_override=None):
    bat = load_season(2026, "batting").copy()
    pit = load_season(2026, "pitching").copy()
    for c in ("PA", "OPS", "R", "team_id"):
        bat[c] = _num(bat[c])
    bat = bat.dropna(subset=["PA", "OPS", "R", "team_id"])
    bat = bat[bat["PA"] > 0]
    bat["_o"] = bat["OPS"] * bat["PA"]
    t = bat.groupby("team_id").agg(PA=("PA", "sum"), R=("R", "sum"), o=("_o", "sum"))
    t["OPS"] = t["o"] / t["PA"]
    t["rpa"] = t["R"] / t["PA"]
    slope, note = DEFAULT_RUNS_PER_PA_PER_OPS, "fallback value"
    if len(t) >= 25:
        s = float(np.polyfit(t["OPS"], t["rpa"], 1)[0])
        if 0.2 <= s <= 0.45:
            slope, note = s, f"fitted on {len(t)} teams"
        else:
            note = f"league fit gave {s:.2f}, outside the sane range, so the fallback is used"
    if SEA_ID not in t.index:
        raise SystemExit("Seattle (team_id 136) not found in the 2026 batting table.")
    rs = float(t.loc[SEA_ID, "R"])
    pit["team_id"] = _num(pit["team_id"])
    sp = pit[pit["team_id"] == SEA_ID]
    if "R" in pit.columns and _num(sp["R"]).notna().any():
        ra, ra_src = float(_num(sp["R"]).sum()), "runs allowed from the pitching table"
    elif "ER" in pit.columns:
        ra, ra_src = float(_num(sp["ER"]).sum()) * RUNS_PER_ER, "earned runs x 1.08"
    else:
        raise SystemExit("Could not find runs or earned runs in the pitching table.")
    table_rs, table_ra = rs, ra
    rs = float(rs_override if rs_override is not None else SEA_RS_2026)
    ra = float(ra_override if ra_override is not None else SEA_RA_2026)
    pit["_ip"] = pit["IP"].map(_ip_to_float) if "IP" in pit.columns else np.nan
    pit["ER"] = _num(pit["ER"]) if "ER" in pit.columns else np.nan
    ok = pit.dropna(subset=["_ip", "ER"])
    lg_era = float(ok["ER"].sum() / ok["_ip"].sum() * 9) if ok["_ip"].sum() > 0 else 4.2
    rot = []
    if {"GS", "Name", "ERA"} <= set(pit.columns):
        mine = pit[pit["team_id"] == SEA_ID].copy()
        mine["GS"], mine["ERA"] = _num(mine["GS"]), _num(mine["ERA"])
        mine = mine.dropna(subset=["GS", "ERA"]).sort_values("GS", ascending=False).head(SP_SPOTS)
        rot = [{"name": str(r["Name"]).split(" (")[0].strip(), "era": float(r["ERA"]), "gs": int(r["GS"])}
               for _, r in mine.iterrows()]
    rpw = 10.0 / (162 * (pyth(rs + 10, ra) - pyth(rs, ra)))
    return {"slope": slope, "slope_note": note, "rs": rs, "ra": ra, "ra_src": ra_src,
            "table_rs": table_rs, "table_ra": table_ra,
            "lg_ops": float(t["o"].sum() / t["PA"].sum()), "rpw": rpw,
            "lg_era": lg_era, "rotation": rot}


# ── data lookups ─────────────────────────────────────────────────────────────
class Data:
    def __init__(self):
        self.scored = _read("free_agent_scored.csv")
        self.scored["_k"] = self.scored["name"].map(norm_key)
        self.proj = _read("projections_2027.csv")
        self.proj["_k"] = self.proj["name"].map(norm_key)
        self.needs = _read("position_needs.csv")
        self.ledger = _read("fielding_ledger.csv", required=False)
        if self.ledger is not None:
            self.ledger["_k"] = self.ledger["Name"].map(norm_key)
        diag = _read("diagnosed_needs.csv", required=False)
        self.lg_bp = 4.2
        if diag is not None:
            r = diag[diag["Metric"] == "bullpen_ERA"]
            if not r.empty:
                self.lg_bp = float(r.iloc[0]["League_Avg"])
        rs = _read("resign_candidates.csv", required=False)
        self.resign = rs if rs is not None else pd.DataFrame(columns=["name", "position_group"])
        self.resign["_k"] = self.resign["name"].map(norm_key)
        mk = _read("free_agent_market.csv", INPUT_DIR, required=False)
        self.market = pd.Series(dtype=float)
        if mk is not None and {"Name", "Prev_AAV"} <= set(mk.columns):
            mk = mk.copy()
            mk["_k"] = mk["Name"].map(norm_key)
            mk["_c"] = _num(mk["Prev_AAV"]) / 1e6
            self.market = mk.dropna(subset=["_c"]).drop_duplicates("_k").set_index("_k")["_c"]
        self.skipped, self.unpriced = [], 0

    def projection(self, name, kind):
        r = self.proj[(self.proj["_k"] == norm_key(name)) & (self.proj["kind"] == kind)]
        return float(r.iloc[0]["proj"]) if not r.empty else None

    def rtot(self, name, fallback=None):
        if self.ledger is not None:
            r = self.ledger[self.ledger["_k"] == norm_key(name)]
            if not r.empty and pd.notna(r.iloc[0]["Rtot"]):
                return float(r.iloc[0]["Rtot"]), True
        if fallback is not None and pd.notna(fallback):
            return float(fallback), True
        return 0.0, False

    def prev_cost(self, name, scored_cost=None):
        if scored_cost is not None and pd.notna(scored_cost):
            return float(scored_cost)
        return float(self.market[norm_key(name)]) if norm_key(name) in self.market.index else None


def price_of(prev, war, mode):
    if mode == "prev":
        return prev
    mkt = MARKET_PER_WAR * max(war, 0.0)
    return max(prev, mkt) if prev is not None else mkt


# ── the 2027 spots ───────────────────────────────────────────────────────────
def build_slots(cal, d, price_mode):
    resign_keys = set(d.resign["_k"])
    slots = []
    for group in HIT_SLOTS + PIT_SLOTS:
        hitter = group in HIT_SLOTS
        slot = {"group": group, "hitter": hitter, "opts": [], "ref_ops": cal["lg_ops"], "ref_rtot": 0.0,
                "starter": None, "starter_status": "-", "ref_known": False}
        seen = set()

        def add(name, source, prev, rtot, rtot_known, projected, row_cost=None):
            k = norm_key(name)
            if k in seen:
                return
            if projected is None:
                d.skipped.append(name)
                return
            if hitter:
                v = cal["slope"] * SLOT_PA * (projected - cal["lg_ops"]) + DEF_KEEP * rtot
                war = (v + REPL_RUNS_HIT) / cal["rpw"]
            else:
                v = (d.lg_bp - projected) * RP_IP / 9 * RUNS_PER_ER * LEVERAGE
                war = (v + REPL_RUNS_RP) / cal["rpw"]
            free = source in ("Returns", "Internal depth", "Replacement")
            price = 0.0 if free else price_of(prev, war, price_mode)
            if price is None:
                d.unpriced += 1
                return
            seen.add(k)
            slot["opts"].append({"slot": group, "name": name, "key": k, "source": source, "proj": projected,
                                 "rtot": rtot, "rtot_known": rtot_known, "v": v, "war": war,
                                 "price": price, "prev": prev, "cu": int(round(price / UNIT)), "hitter": hitter})

        if hitter:
            nr = d.needs[d.needs["position_group"] == group]
            if not nr.empty and str(nr.iloc[0]["starter"]) not in ("nan", "no clear starter"):
                st = str(nr.iloc[0]["starter"])
                ops = _num(pd.Series([nr.iloc[0].get("starter_ops_with_sea")])).iloc[0]
                rt, known = d.rtot(st)
                slot.update({"starter": st, "ref_rtot": rt, "ref_known": True})
                if pd.notna(ops):
                    slot["ref_ops"] = float(ops)
                is_fa = norm_key(st) in resign_keys
                slot["starter_status"] = "FREE AGENT" if is_fa else "returns"
                if not is_fa:
                    add(st, "Returns", None, rt, known, d.projection(st, "batting") or (float(ops) if pd.notna(ops) else None))
            pool = OF_POOL if group in OF_POOL else {group}
            inter = d.scored[d.scored["position_group"].isin(pool) & d.scored["source"].astype(str).str.contains("Internal")]
            for _, r in inter.iterrows():
                if norm_key(r["name"]) in resign_keys:
                    continue
                rt, known = d.rtot(r["name"], r.get("Rtot"))
                add(r["name"], "Internal depth", None, rt, known, d.projection(r["name"], "batting"))
            for _, r in d.resign[d.resign["position_group"] == group].iterrows():
                rt, known = d.rtot(r["name"])
                add(r["name"], "Re-sign", d.prev_cost(r["name"]), rt, known, d.projection(r["name"], "batting"))
            fa = d.scored[(d.scored["position_group"] == group) & d.scored["source"].astype(str).str.contains("Free Agent")]
            for _, r in fa.iterrows():
                rt, known = d.rtot(r["name"], r.get("Rtot"))
                sc = r.get("Est_Cost_M") if str(r.get("Market_Matched")).lower() == "true" else None
                add(r["name"], "Free agent", d.prev_cost(r["name"], sc), rt, known, d.projection(r["name"], "batting"))
            # always a $0 fallback so a spot with no depth is never impossible
            repl_ops = cal["lg_ops"] - REPL_RUNS_HIT / (cal["slope"] * SLOT_PA)
            slot["opts"].append({"slot": group, "name": "Replacement-level player", "key": f"replacement_{group}",
                                 "source": "Replacement", "proj": repl_ops, "rtot": 0.0, "rtot_known": True,
                                 "v": -REPL_RUNS_HIT, "war": 0.0, "price": 0.0, "prev": None, "cu": 0, "hitter": True})
        else:
            slot["opts"].append({"slot": group, "name": "(no signing)", "key": None, "source": "-", "proj": None,
                                 "rtot": 0.0, "rtot_known": True, "v": 0.0, "war": 0.0, "price": 0.0,
                                 "prev": None, "cu": 0, "hitter": False})
            fa = d.scored[(d.scored["position_group"] == group) & d.scored["source"].astype(str).str.contains("Free Agent")]
            for _, r in fa.iterrows():
                sc = r.get("Est_Cost_M") if str(r.get("Market_Matched")).lower() == "true" else None
                add(r["name"], "Free agent", d.prev_cost(r["name"], sc), 0.0, True, d.projection(r["name"], "pitching"))
        if not slot["opts"]:
            raise SystemExit(f"No usable options for {group}. Check that projections_2027.csv covers the roster.")
        slots.append(slot)
    slots += build_rotation(cal, d, price_mode, resign_keys)
    return slots


def build_rotation(cal, d, price_mode, resign_keys):
    """Five starter spots. Each starts as one of the 2026 rotation (free if he is under contract, a re-sign if
    he is a free agent); any free-agent starter can take any spot, and a replacement-level arm is the fallback."""
    out = []
    pool = d.scored[d.scored["position_group"].astype(str).str.contains("Starting")
                    & d.scored["source"].astype(str).str.contains("Free Agent")]
    repl_era = cal["lg_era"] + REPL_SP_ADD
    repl_runs = REPL_SP_ADD * SP_IP / 9 * RUNS_PER_ER
    occupants = list(cal.get("rotation", []))
    for i in range(SP_SPOTS):
        occ = occupants[i] if i < len(occupants) else None
        group = f"Rotation {i + 1}"
        slot = {"group": group, "hitter": False, "sp": True, "opts": [], "ref_ops": None, "ref_rtot": 0.0,
                "ref_era": occ["era"] if occ else repl_era, "starter": occ["name"] if occ else None,
                "starter_status": "-", "ref_known": occ is not None}
        seen = set()

        def add(name, source, prev, projected):
            k = norm_key(name)
            if k in seen:
                return
            if projected is None:
                d.skipped.append(name)
                return
            v = (cal["lg_era"] - projected) * SP_IP / 9 * RUNS_PER_ER
            war = (v + repl_runs) / cal["rpw"]
            price = 0.0 if source in ("Returns", "Replacement") else price_of(prev, war, price_mode)
            if price is None:
                d.unpriced += 1
                return
            seen.add(k)
            slot["opts"].append({"slot": group, "name": name, "key": k, "source": source, "proj": projected,
                                 "rtot": 0.0, "rtot_known": True, "v": v, "war": war, "price": price, "prev": prev,
                                 "cu": int(round(price / UNIT)), "hitter": False, "sd": SD_SP_ERA})

        if occ:
            is_fa = norm_key(occ["name"]) in resign_keys
            slot["starter_status"] = "FREE AGENT" if is_fa else "returns"
            proj = d.projection(occ["name"], "pitching")
            proj = proj if proj is not None else occ["era"]
            if is_fa:
                add(occ["name"], "Re-sign", d.prev_cost(occ["name"]), proj)
            else:
                add(occ["name"], "Returns", None, proj)
        for _, r in pool.iterrows():
            sc = r.get("Est_Cost_M") if str(r.get("Market_Matched")).lower() == "true" else None
            add(r["name"], "Free agent", d.prev_cost(r["name"], sc), d.projection(r["name"], "pitching"))
        slot["opts"].append({"slot": group, "name": "Replacement-level starter", "key": f"replacement_{group}",
                             "source": "Replacement", "proj": repl_era, "rtot": 0.0, "rtot_known": True,
                             "v": -repl_runs, "war": 0.0, "price": 0.0, "prev": None, "cu": 0, "hitter": False,
                             "sd": SD_SP_ERA})
        out.append(slot)
    return out


def restrict(slot, frozen, lock):
    """Options this spot is allowed to use."""
    if lock:
        m = [o for o in slot["opts"] if o["key"] == norm_key(lock)]
        if not m:
            raise SystemExit(f"Locked player '{lock}' is not an option at {slot['group']} (no projection, price, or wrong position).")
        return m
    if frozen:
        ret = [o for o in slot["opts"] if o["source"] in ("Returns", "Trade")]
        return ret if ret else [o for o in slot["opts"] if o["cu"] == 0]
    return slot["opts"]


# ── the search ───────────────────────────────────────────────────────────────
def solve(option_lists, budget_units, max_trades=None):
    """Best total value within the budget. max_trades caps how many 'Trade' options may be used."""
    B = budget_units
    K = (max_trades + 1) if max_trades is not None else 1
    NEG = -1e18
    n = len(option_lists)
    dp = [[[NEG] * K for _ in range(B + 1)] for _ in range(n + 1)]
    for b in range(B + 1):
        dp[0][b][0] = 0.0
    choice = [[[None] * K for _ in range(B + 1)] for _ in range(n)]
    for i, opts in enumerate(option_lists):
        for b in range(B + 1):
            for k in range(K):
                best, bj = NEG, None
                for j, o in enumerate(opts):
                    t = 1 if (max_trades is not None and o["source"] == "Trade") else 0
                    if o["cu"] > b or k - t < 0:
                        continue
                    prev = dp[i][b - o["cu"]][k - t]
                    if prev > NEG / 2 and prev + o["v"] > best:
                        best, bj = prev + o["v"], j
                dp[i + 1][b][k], choice[i][b][k] = best, bj
    k_best = max(range(K), key=lambda k: dp[n][B][k])
    if dp[n][B][k_best] < NEG / 2:
        return None
    b, k, picks = B, k_best, []
    for i in reversed(range(n)):
        j = choice[i][b][k]
        o = option_lists[i][j]
        picks.append(o)
        b -= o["cu"]
        if max_trades is not None and o["source"] == "Trade":
            k -= 1
    return picks[::-1]


def best_plan(slots, budget, frozen, locks, use_locks=True, max_trades=None):
    allowed = []
    for s in slots:
        lock = locks.get(s["group"]) if use_locks else None
        allowed.append(restrict(s, s["group"] in frozen and not lock, lock))
    for _ in range(40):          # a player can only fill one spot
        picks = solve(allowed, int(round(budget / UNIT)), max_trades)
        if picks is None:
            return None
        names = {}
        dup = None
        for p in picks:
            if p["key"] and p["key"] in names:
                dup = p
                break
            if p["key"]:
                names[p["key"]] = p
        if dup is None:
            return picks
        other = names[dup["key"]]
        if dup["source"] == "Returns" and other["source"] != "Returns":
            drop = other
        elif other["source"] == "Returns" and dup["source"] != "Returns":
            drop = dup
        else:
            drop = dup if dup["v"] <= other["v"] else other
        for i, opts in enumerate(allowed):
            if opts and opts[0]["slot"] == drop["slot"] and len(opts) > 1:
                allowed[i] = [o for o in opts if o is not drop]
                break
        else:
            return picks
    return picks


# ── scoring a plan, with or without random talent noise ─────────────────────
def score_plan(slots, picks, cal, d, draws=None):
    off = dfn = pit = 0.0
    for s, o in zip(slots, picks):
        if o["key"] is None:
            continue
        if s["hitter"]:
            noise_o = draws["off"][o["key"]] if draws else 0.0
            noise_d = draws["def"][o["key"]] if draws else 0.0
            off = off + cal["slope"] * SLOT_PA * (o["proj"] - s["ref_ops"]) + noise_o
            dfn = dfn + DEF_KEEP * o["rtot"] - s["ref_rtot"] + noise_d
        elif s.get("sp"):
            noise = draws["pit"][o["key"]] if draws else 0.0
            pit = pit + (s["ref_era"] - (o["proj"] + noise)) * SP_IP / 9 * RUNS_PER_ER
        else:
            noise = draws["pit"][o["key"]] if draws else 0.0
            pit = pit + (d.lg_bp - (o["proj"] + noise)) * RP_IP / 9 * RUNS_PER_ER * LEVERAGE
    return cal["rs"] + off, cal["ra"] - dfn - pit, off, dfn, pit


def make_draws(plans, cal, rng, n):
    keys_h = {p["key"] for plan in plans for p in plan if p["key"] and p["hitter"]}
    keys_p = {p["key"]: p.get("sd", SD_ERA) for plan in plans for p in plan if p["key"] and not p["hitter"]}
    return {"off": {k: cal["slope"] * SLOT_PA * rng.normal(0, SD_BAT, n) for k in keys_h},
            "def": {k: rng.normal(0, SD_DEF, n) for k in keys_h},
            "pit": {k: rng.normal(0, sd, n) for k, sd in keys_p.items()}}


# ── scenario pieces ──────────────────────────────────────────────────────────
def rebound_table(d):
    """2026 and 2025 OPS for the rebound players (the one lookup that needs the season tables)."""
    out = {}
    seasons = {}
    for yr in (2025, 2026):
        t = load_season(yr, "batting").copy()
        t["PA"], t["OPS"] = _num(t["PA"]), _num(t["OPS"])
        t["_k"] = t["Name"].map(norm_key)
        seasons[yr] = t.dropna(subset=["PA", "OPS"]).sort_values("PA", ascending=False).drop_duplicates("_k").set_index("_k")
    for nm in REBOUND_PLAYERS:
        k = norm_key(nm)
        if k in seasons[2025].index and k in seasons[2026].index:
            out[k] = {"name": nm, "ops25": float(seasons[2025].loc[k, "OPS"]), "ops26": float(seasons[2026].loc[k, "OPS"])}
    return out


def sp_rebound_table():
    """2025 and 2026 ERA for pitchers with a real workload (40+ IP in both seasons)."""
    seasons = {}
    for yr in (2025, 2026):
        t = load_season(yr, "pitching").copy()
        t["_ip"] = t["IP"].map(_ip_to_float)
        t["ERA"] = _num(t["ERA"])
        t["_k"] = t["Name"].astype(str).str.split(" \\(").str[0].map(norm_key)
        seasons[yr] = t.dropna(subset=["_ip", "ERA"]).query("_ip >= 40").sort_values("_ip", ascending=False)\
                       .drop_duplicates("_k").set_index("_k")
    out = {}
    for k in seasons[2026].index:
        if k in seasons[2025].index:
            out[k] = {"name": str(seasons[2026].loc[k, "Name"]).split(" (")[0],
                      "era25": float(seasons[2025].loc[k, "ERA"]), "era26": float(seasons[2026].loc[k, "ERA"])}
    return out


def apply_rebound(slots, table, frac, cal):
    """Copy of the slots where each rebound player moves frac of the way from 2026 OPS back to 2025 OPS.
    Never lower than the three-year projection (a rebound assumption should not make him worse)."""
    new, log = [], []
    for s in slots:
        s2 = dict(s)
        s2["opts"] = []
        for o in s["opts"]:
            o2 = dict(o)
            r = table.get(o["key"]) if o["hitter"] else None
            if r is not None and o["source"] != "Replacement":
                target = r["ops26"] + frac * (r["ops25"] - r["ops26"])
                if frac > 0 and target > o["proj"]:
                    o2["proj"] = target
                    o2["v"] = cal["slope"] * SLOT_PA * (target - cal["lg_ops"]) + DEF_KEEP * o["rtot"]
                if o["source"] == "Returns":
                    log.append((r["name"], r["ops26"], r["ops25"], o["proj"], o2["proj"]))
            if (s.get("sp") and SP_FRAC > 0 and o["source"] == "Returns" and o["key"] in SP_TABLE):
                r = SP_TABLE[o["key"]]
                target = r["era26"] + SP_FRAC * (r["era25"] - r["era26"])
                if target < o["proj"]:
                    o2["proj"] = target
                    o2["v"] = (cal["lg_era"] - target) * SP_IP / 9 * RUNS_PER_ER
            s2["opts"].append(o2)
        new.append(s2)
    return new, log


def load_trade_targets(cal, d):
    """Impact bats you could trade for. Reads data/inputs/trade_targets.csv (Name, Pos, Salary_M, optional Proj_OPS, Rtot);
    if the file is absent, uses one placeholder impact bat at each open lineup spot."""
    rows, real = [], False
    tt = _read("trade_targets.csv", INPUT_DIR, required=False)
    if tt is not None and {"Name", "Pos", "Salary_M"} <= set(tt.columns):
        rev = {v: k for k, v in ABBR.items()}
        for _, r in tt.iterrows():
            g = str(r["Pos"]).strip()
            if g.upper() in ("SP", "STARTING PITCHERS", "STARTING PITCHER"):
                sal = _num(pd.Series([r["Salary_M"]])).iloc[0]
                era = _num(pd.Series([r.get("Proj_ERA")])).iloc[0] if "Proj_ERA" in tt.columns else np.nan
                if pd.isna(era):
                    era = d.projection(r["Name"], "pitching")
                if pd.isna(sal) or era is None or pd.isna(era):
                    d.skipped.append(str(r["Name"]))
                else:
                    rows.append({"group": "ROTATION", "name": str(r["Name"]), "era": float(era), "salary": float(sal)})
                continue
            g = g if g in HIT_SLOTS else rev.get(g.upper())
            sal = _num(pd.Series([r["Salary_M"]])).iloc[0]
            ops = _num(pd.Series([r.get("Proj_OPS")])).iloc[0] if "Proj_OPS" in tt.columns else np.nan
            if pd.isna(ops):
                ops = d.projection(r["Name"], "batting")
            rt = _num(pd.Series([r.get("Rtot")])).iloc[0] if "Rtot" in tt.columns else np.nan
            if g is None or pd.isna(sal) or ops is None or pd.isna(ops):
                d.skipped.append(str(r["Name"]))
                continue
            rows.append({"group": g, "name": str(r["Name"]), "ops": float(ops), "rtot": 0.0 if pd.isna(rt) else float(rt),
                         "salary": float(sal)})
        real = bool(rows)
    if not rows:
        for g in IMPACT_SPOTS:
            rows.append({"group": g, "name": f"Impact bat ({ABBR[g]}, placeholder)", "ops": IMPACT_OPS, "rtot": 0.0,
                         "salary": IMPACT_SALARY})
    return rows, real


def add_trade_options(slots, targets, cal):
    new = []
    for s in slots:
        s2 = dict(s)
        s2["opts"] = list(s["opts"])
        for t in targets:
            if t["group"] == "ROTATION":
                if not s.get("sp"):
                    continue
                v = (cal["lg_era"] - t["era"]) * SP_IP / 9 * RUNS_PER_ER
                s2["opts"].append({"slot": s["group"], "name": t["name"], "key": norm_key(t["name"]), "source": "Trade",
                                   "proj": t["era"], "rtot": 0.0, "rtot_known": True, "v": v,
                                   "war": (v + REPL_SP_ADD * SP_IP / 9 * RUNS_PER_ER) / cal["rpw"], "price": t["salary"],
                                   "prev": t["salary"], "cu": int(round(t["salary"] / UNIT)), "hitter": False,
                                   "sd": SD_SP_ERA})
                continue
            if t["group"] != s["group"]:
                continue
            v = cal["slope"] * SLOT_PA * (t["ops"] - cal["lg_ops"]) + DEF_KEEP * t["rtot"]
            s2["opts"].append({"slot": s["group"], "name": t["name"], "key": norm_key(t["name"]), "source": "Trade",
                               "proj": t["ops"], "rtot": t["rtot"], "rtot_known": True, "v": v,
                               "war": (v + REPL_RUNS_HIT) / cal["rpw"], "price": t["salary"], "prev": t["salary"],
                               "cu": int(round(t["salary"] / UNIT)), "hitter": True})
        new.append(s2)
    return new


def run_scenarios(slots, cal, d, frozen, locks, table, targets, real_targets, budgets_override=None):
    """Plan and simulate every scenario against the same 2026 starting point and the same luck."""
    plans = []
    for sc in SCENARIOS:
        sl, log = apply_rebound(slots, table, sc["rebound"], cal)
        if sc["trades"] > 0:
            sl = add_trade_options(sl, targets, cal)
        bud = (budgets_override.get(sc["name"], sc["budget"]) if budgets_override else sc["budget"]) + BUDGET_BONUS
        pk = best_plan(sl, bud, frozen, locks, max_trades=sc["trades"] if sc["trades"] > 0 else None)
        if pk is None:
            raise SystemExit(f"Scenario {sc['name']}: budget ${bud:.0f}M cannot cover the locked players.")
        # ladder steps: (1) nobody new but the rebound, (2) plus free agents, (3) plus trades
        sl_fa = apply_rebound(slots, table, sc["rebound"], cal)[0]
        none_pk = best_plan(sl_fa, 0, set(HIT_SLOTS + PIT_SLOTS), {}, use_locks=False)
        fa_pk = best_plan(sl_fa, bud, frozen, locks)
        plans.append({"sc": sc, "slots": sl, "picks": pk, "budget": bud, "log": log,
                      "ladder": [("start", sl_fa, none_pk), ("fa", sl_fa, fa_pk)]})
    rng = np.random.default_rng(11)
    draws = make_draws([p["picks"] for p in plans] + [st[2] for p in plans for st in p["ladder"]], cal, rng, N_SIMS)
    for p in plans:
        rs, ra, off, dfn, pit = score_plan(p["slots"], p["picks"], cal, d, draws)
        pw = pyth(rs, ra)
        p["exp"] = 162 * pw
        none_pk = p["ladder"][0][2]
        p["moves"] = []
        for i, o in enumerate(p["picks"]):
            if o["price"] > 0:
                alt = list(p["picks"])
                alt[i] = none_pk[i]
                w_alt = 162 * pyth(*score_plan(p["slots"], alt, cal, d, draws)[:2])
                p["moves"].append((o, p["exp"].mean() - w_alt.mean(), none_pk[i]))
        p["steps"] = [162 * pyth(*score_plan(sl_, pk_, cal, d, draws)[:2]) for _, sl_, pk_ in p["ladder"]]
        p["played"] = rng.binomial(162, pw)
        p["spend"] = sum(o["price"] for o in p["picks"])
        p["trades"] = [o for o in p["picks"] if o["source"] == "Trade"]
        p["adds"] = [o for o in p["picks"] if o["price"] > 0 and o["source"] != "Trade"]
    return plans


def print_scenarios(plans, table, real_targets, base_wins):
    w = 100
    print("\n" + "=" * w)
    print("THE IMPROVEMENT STORY -- what has to go right (10,000 simulated seasons each)")
    print("=" * w)
    print(f"  2026 actual: 76-86.  2027 with no new money and no rebound: {base_wins:.1f} expected wins.")
    print(f"\n  {'Scenario':<13} {'What has to go right':<40} {'Exp wins':>8} {'80% range':>10} {'P(' + str(PLAYOFF_BAR) + '+)':>7} {'P(90+)':>7} {'New $M':>7}")
    for p in plans:
        sc = p["sc"]
        pl = p["played"]
        lo, hi = np.percentile(p["exp"], 10), np.percentile(p["exp"], 90)
        print(f"  {sc['name']:<13} {sc['story']:<40} {p['exp'].mean():>8.1f} {lo:>4.0f}-{hi:<4.0f} "
              f"{(pl >= PLAYOFF_BAR).mean():>6.0%} {(pl >= 90).mean():>7.0%} {p['spend']:>7.1f}")
    print("\n  WHERE THE WINS COME FROM (expected wins, each step added in order)")
    print(f"  {'Scenario':<13} {'2026 level':>10} {'+ rebound':>10} {'+ free agents':>14} {'+ trades':>9} {'= total':>8}")
    sim_base = plans[0]["steps"][0].mean()      # no rebound, no new money, same simulated luck
    for p in plans:
        s0 = p["steps"][0].mean()
        s1 = p["steps"][1].mean()
        tot = p["exp"].mean()
        print(f"  {p['sc']['name']:<13} {sim_base:>10.1f} {s0 - sim_base:>+10.1f} {s1 - s0:>+14.1f} {tot - s1:>+9.1f} {tot:>8.1f}")
    print("\n  The rebound assumption (OPS), halfway = midway between 2026 and 2025:")
    print(f"  {'Player':<18} {'2025':>6} {'2026':>6} {'3-yr proj':>10} {'Base':>6} {'Optimistic':>11}")
    by = {p["sc"]["name"]: {r[0]: r for r in p["log"]} for p in plans}
    for k, r in table.items():
        nm = r["name"]
        cons = by["Conservative"].get(nm)
        if cons is None:
            continue
        print(f"  {nm:<18} {r['ops25']:>6.3f} {r['ops26']:>6.3f} {cons[4]:>10.3f} {by['Base'][nm][4]:>6.3f} {by['Optimistic'][nm][4]:>11.3f}")
    for p in plans:
        sc = p["sc"]
        print(f"\n  {sc['name'].upper()}  (${p['budget']:.0f}M available, {sc['trades']} trade(s) allowed, spent ${p['spend']:.1f}M)")
        for o in p["adds"]:
            print(f"      sign   {o['name']:<28} {ABBR.get(o['slot'], o['slot']):<4} ${o['price']:>5.1f}M   proj {o['proj']:.3f}" if o["hitter"]
                  else f"      sign   {o['name']:<28} {ABBR.get(o['slot'], o['slot']):<4} ${o['price']:>5.1f}M   proj ERA {o['proj']:.2f}")
        for o in p["trades"]:
            print(f"      trade  {o['name']:<28} {ABBR.get(o['slot'], o['slot']):<4} ${o['price']:>5.1f}M   proj {o['proj']:.3f}")
        if not p["adds"] and not p["trades"]:
            print("      (no new players)")
    print("\n  Notes")
    if not real_targets:
        print(f"   * The impact bats are PLACEHOLDERS ({IMPACT_OPS:.3f} OPS, ${IMPACT_SALARY:.0f}M each). Put real names in data/inputs/trade_targets.csv")
        print("     (columns: Name, Pos, Salary_M, optional Proj_OPS / Proj_ERA and Rtot; Pos 'Starting Pitchers' for arms) and the same table is rebuilt with them.")
    print("   * Trades cost salary here but NOT prospects; a real trade also costs young talent, so treat trade gains as an upper bound.")
    print("   * The rebound only moves hitting (OPS). Defense is left at its 2026 level; starters are valued on ERA alone. Whether they bounce back is the biggest bet in the table.")
    print("   * Spend is capped by each scenario's budget; unspent money is simply not used. Change them in SCENARIOS or with --scenario-budget Name=40.")


POS_NAME = {"Catchers": "catcher", "First Basemen": "first base", "Second Basemen": "second base", "Third Basemen": "third base",
            "Shortstops": "shortstop", "Left Fielders": "left field", "Center Fielders": "center field",
            "Right Fielders": "right field", "Designated Hitters": "DH"}


def export_for_app(plans, table, cal, path_dir):
    """CSVs the Streamlit app (screen 6) reads: scenarios, the problem->fix ledger, and the rebound assumptions."""
    sim = plans[0]["steps"][0].mean()
    sc_rows, led = [], []
    for p in plans:
        name = p["sc"]["name"]
        s0, s1, tot = p["steps"][0].mean(), p["steps"][1].mean(), p["exp"].mean()
        pl = p["played"]
        sc_rows.append({"scenario": name, "story": p["sc"]["story"], "expected_wins": tot,
                        "wins_p10": float(np.percentile(p["exp"], 10)), "wins_p90": float(np.percentile(p["exp"], 90)),
                        "p_85_plus": float((pl >= PLAYOFF_BAR).mean()), "p_90_plus": float((pl >= 90).mean()),
                        "new_money_M": p["spend"], "start_wins": sim, "rebound_wins": s0 - sim,
                        "free_agent_wins": s1 - s0, "trade_wins": tot - s1,
                        "adds": "; ".join(o["name"] for o in p["adds"] + p["trades"])})
        rn = ", ".join(r["name"].split()[-1] for r in table.values())
        led.append({"scenario": name, "problem": f"Offense fell off: {rn} had down years",
                    "fix": f"Rebound {p['sc']['rebound']:.0%} of the way back to 2025 (nothing to buy)",
                    "wins": s0 - sim, "cost_M": 0.0})
        for o, w, was in p["moves"]:
            g = o["slot"]
            if g in POS_NAME:
                prob = (f"Open lineup spot at {POS_NAME[g]}" if was["source"] in ("Replacement", "Internal depth")
                        and was["proj"] < cal["lg_ops"] else f"Weak spot at {POS_NAME[g]} ({was['name']} projects {was['proj']:.3f} OPS)")
            elif g.startswith("Rotation"):
                prob = ("Rotation spot open (a starter left as a free agent)" if was["source"] == "Replacement"
                        else f"Rotation: {was['name']} projects {was['proj']:.2f} ERA")
            else:
                prob = "Bullpen needs setup arms (" + ("right" if g.startswith("Right") else "left") + "-handed)"
            how = "Trade for" if o["source"] == "Trade" else ("Re-sign" if o["source"] == "Re-sign" else "Sign")
            led.append({"scenario": name, "problem": prob, "fix": f"{how} {o['name']}", "wins": w, "cost_M": o["price"]})
    pd.DataFrame(sc_rows).to_csv(os.path.join(path_dir, "scenarios.csv"), index=False)
    pd.DataFrame(led).to_csv(os.path.join(path_dir, "plan_ledger.csv"), index=False)
    pd.DataFrame([{"player": r["name"], "ops_2025": r["ops25"], "ops_2026": r["ops26"]} for r in table.values()]
                 ).to_csv(os.path.join(path_dir, "plan_rebound.csv"), index=False)


def write_plan_page(plans, table, base_wins, sim_base, cal, real_targets, path):
    """One-page 'here is what we need to do' for the front office. Self-contained HTML."""
    import html
    e = html.escape
    cons, base, opt = plans
    def bar(w):
        return f'<div class="bar"><span style="width:{max(0, min(100, (w - 60) / 45 * 100)):.0f}%"></span><i style="left:{(PLAYOFF_BAR - 60) / 45 * 100:.0f}%"></i></div>'
    rows = ""
    for p in plans:
        pl = p["played"]
        rows += (f'<tr><td><b>{e(p["sc"]["name"])}</b><br><small>{e(p["sc"]["story"])}</small></td>'
                 f'<td class="n">{p["exp"].mean():.1f}</td><td>{bar(p["exp"].mean())}</td>'
                 f'<td class="n">{(pl >= PLAYOFF_BAR).mean():.0%}</td><td class="n">{(pl >= 90).mean():.0%}</td>'
                 f'<td class="n">${p["spend"]:.0f}M</td></tr>')
    def moves(p):
        out = []
        for o in p["adds"]:
            out.append(f'{e(o["name"])} ({e(ABBR.get(o["slot"], o["slot"]))}), ${o["price"]:.0f}M')
        for o in p["trades"]:
            out.append(f'{e(o["name"])} ({e(ABBR.get(o["slot"], o["slot"]))}), ${o["price"]:.0f}M salary')
        return "".join(f"<li>{m}</li>" for m in out) or "<li>No new players</li>"
    reb = ""
    cons_log = {r[0]: r for r in cons["log"]}
    base_log = {r[0]: r for r in base["log"]}
    for k, r in table.items():
        nm = r["name"]
        if nm in cons_log:
            reb += (f'<tr><td>{e(nm)}</td><td class="n">{r["ops25"]:.3f}</td><td class="n">{r["ops26"]:.3f}</td>'
                    f'<td class="n">{base_log[nm][4]:.3f}</td></tr>')
    sim = plans[0]["steps"][0].mean()
    steps = ""
    for p in plans:
        s0, s1, tot = p["steps"][0].mean(), p["steps"][1].mean(), p["exp"].mean()
        steps += (f'<tr><td>{e(p["sc"]["name"])}</td><td class="n">{s0 - sim:+.1f}</td><td class="n">{s1 - s0:+.1f}</td>'
                  f'<td class="n">{tot - s1:+.1f}</td><td class="n"><b>{tot:.1f}</b></td></tr>')
    ledger = ""
    pos_name = {"Catchers": "catcher", "First Basemen": "first base", "Second Basemen": "second base", "Third Basemen": "third base",
                "Shortstops": "shortstop", "Left Fielders": "left field", "Center Fielders": "center field",
                "Right Fielders": "right field", "Designated Hitters": "DH"}
    reb_w = base["steps"][0].mean() - sim
    rn = ", ".join(r["name"].split()[-1] for r in table.values())
    ledger += (f'<tr><td>Offense fell off: {e(rn)} all had down years</td><td>Bet on a rebound (halfway back to 2025 OPS); '
               f'nothing to buy, but it has to happen</td><td class="n">{reb_w:+.1f}</td></tr>')
    for o, w, was in base["moves"]:
        g = o["slot"]
        if g in pos_name:
            prob = (f"Open lineup spot at {pos_name[g]} (starter left as a free agent)" if was["source"] in ("Replacement", "Internal depth")
                    and was["proj"] < cal["lg_ops"] else f"Weak spot at {pos_name[g]} ({e(was['name'])} projects {was['proj']:.3f} OPS)")
        elif g.startswith("Rotation"):
            prob = (f"Rotation spot open" if was["source"] == "Replacement"
                    else f"Rotation: {e(was['name'])} projects {was['proj']:.2f} ERA")
        else:
            prob = "Bullpen needs setup arms (" + ("right" if g.startswith("Right") else "left") + "-handed)"
        how = "Trade for" if o["source"] == "Trade" else ("Re-sign" if o["source"] == "Re-sign" else "Sign")
        ledger += f'<tr><td>{prob}</td><td>{how} {e(o["name"])} (${o["price"]:.0f}M)</td><td class="n">{w:+.1f}</td></tr>'
    ph = ("" if real_targets else
          f'<p class="warn">The impact bats here are placeholders ({IMPACT_OPS:.3f} OPS, ${IMPACT_SALARY:.0f}M). '
          f'Real trade targets go in data/inputs/trade_targets.csv.</p>')
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Mariners 2027: what we need to do</title>
<style>
body{{font:16px/1.5 system-ui,Segoe UI,sans-serif;max-width:860px;margin:32px auto;padding:0 20px;color:#1b2430}}
h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 8px;border-bottom:2px solid #005c5c;padding-bottom:4px;color:#005c5c}}
.sub{{color:#5b6672;margin:0 0 18px}} table{{border-collapse:collapse;width:100%}} td,th{{padding:7px 8px;border-bottom:1px solid #e3e7ec;text-align:left;vertical-align:middle}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#5b6672}} .n{{text-align:right;font-variant-numeric:tabular-nums}}
small{{color:#5b6672}} .bar{{position:relative;height:12px;background:#e9eef2;border-radius:6px;min-width:140px}} .bar span{{display:block;height:100%;background:#005c5c;border-radius:6px}}
.bar i{{position:absolute;top:-3px;width:2px;height:18px;background:#c4ced4}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.card{{border:1px solid #d5dbe1;border-radius:10px;padding:12px 14px}} .card h3{{margin:0 0 6px;font-size:15px}} .card ul{{margin:0;padding-left:18px;font-size:14px}}
.warn{{background:#fff6e0;border-left:4px solid #e0a100;padding:8px 12px;font-size:14px}} .big{{font-size:20px}}
ol li{{margin-bottom:6px}} @media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Mariners 2027: what we need to do</h1>
<p class="sub">From 76-86 to a playoff team. Model-based scenarios; every assumption is listed at the bottom.</p>

<h2>The problem</h2>
<p class="big">The offense fell apart, not the whole team. Seattle scored {cal["rs"]:.0f} runs in 2026 (28th in MLB) and allowed {cal["ra"]:.0f}.</p>
<ul>
<li>Team OPS fell from .762 in 2025 to .695. Raleigh, Naylor and Rodriguez account for about .051 of that .067 drop.</li>
<li>Home runs fell from 263 to 190, and team fielding was below average in both seasons.</li>
<li>Shortstop and left field are open (Crawford and Arozarena are free agents), and the bullpen needs setup arms.</li>
</ul>

<h2>Problem to solution (Base plan)</h2>
<table><tr><th>Problem</th><th>Fix</th><th class="n">Wins</th></tr>{ledger}</table>
<p class="big">Do these and the model has Seattle at <b>{base["exp"].mean():.1f} wins</b> on average, with a {(base["played"] >= PLAYOFF_BAR).mean():.0%} chance of reaching {PLAYOFF_BAR}+.</p>

<h2>What each path gets us</h2>
<table><tr><th>Scenario</th><th class="n">Exp. wins</th><th></th><th class="n">{PLAYOFF_BAR}+ wins</th><th class="n">90+ wins</th><th class="n">New money</th></tr>{rows}</table>
<small>The gray tick marks {PLAYOFF_BAR} wins, roughly the AL playoff line. Chances include game-by-game luck (10,000 simulated seasons).</small>

<h2>What each scenario signs</h2>
<div class="cards">
<div class="card"><h3>Conservative</h3><ul>{moves(cons)}</ul></div>
<div class="card"><h3>Base</h3><ul>{moves(base)}</ul></div>
<div class="card"><h3>Optimistic</h3><ul>{moves(opt)}</ul></div>
</div>
{ph}

<h2>What has to go right</h2>
<table><tr><th>Wins added by</th><th class="n">Rebound</th><th class="n">Free agents</th><th class="n">Trades</th><th class="n">Total wins</th></tr>{steps}</table>
<p>Base assumes the rebound is halfway back to 2025 (OPS):</p>
<table><tr><th>Player</th><th class="n">2025</th><th class="n">2026</th><th class="n">Base assumes</th></tr>{reb}</table>

<h2>Limits we are upfront about</h2>
<ul>
<li>Projections use three years of results with no age adjustment. They beat "same as last year" by about 19% in backtests but are still uncertain.</li>
<li>Prices are anchors (last contract or $8M per projected win), not predicted bids. Trades cost salary here but not prospects.</li>
<li>Starters are valued on projected ERA alone (170 innings each, no injury risk). The rebound moves hitting only, not defense.</li>
</ul>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# ── reporting ────────────────────────────────────────────────────────────────
ABBR = {"Catchers": "C", "First Basemen": "1B", "Second Basemen": "2B", "Third Basemen": "3B",
        "Shortstops": "SS", "Left Fielders": "LF", "Center Fielders": "CF", "Right Fielders": "RF",
        "Designated Hitters": "DH", "Right-Handed Relievers": "RHP", "Left-Handed Relievers": "LHP",
        **{f"Rotation {i}": "SP" for i in range(1, SP_SPOTS + 1)}}


def main():
    ap = argparse.ArgumentParser(description="Build the 2027 roster, find the best signings, simulate the season.")
    ap.add_argument("--budget", type=float, default=50.0, help="new money in $M (estimated prices) for the detailed plan")
    ap.add_argument("--price", choices=["market", "prev"], default="market")
    ap.add_argument("--lock", action="append", default=[], help='force a player: --lock "Left Fielders=Randy Arozarena" (=none clears)')
    ap.add_argument("--open", action="append", default=[], help='unfreeze a spot: --open "Catchers"')
    ap.add_argument("--target-wins", type=int, default=90)
    ap.add_argument("--rs", type=float, default=None, help="2026 runs scored (default: official 664)")
    ap.add_argument("--ra", type=float, default=None, help="2026 runs allowed (default: official 722)")
    ap.add_argument("--scenario-budget", action="append", default=[], help='override a scenario budget: --scenario-budget "Base=120"')
    ap.add_argument("--exclude", action="append", default=[], help='rule a player out of every plan: --exclude "Tarik Skubal"')
    ap.add_argument("--sp-rebound", type=float, default=0.0, help="0-1: returning starters move this far from 2026 ERA back to 2025 ERA")
    ap.add_argument("--trade-away", action="append", default=[], help='trade a player off the roster: --trade-away "George Kirby=9.8" (salary freed in $M)')
    ap.add_argument("--no-scenarios", action="store_true", help="skip the three-scenario improvement story")
    a = ap.parse_args()

    locks = dict(LOCKS)
    for item in a.lock:
        if "=" not in item:
            raise SystemExit(f"--lock needs 'Position group=Player', got: {item}")
        g, n = item.split("=", 1)
        if n.strip().lower() in ("none", ""):
            locks.pop(g.strip(), None)
        else:
            locks[g.strip()] = n.strip()
    frozen = set(FROZEN) - set(a.open)

    global SP_FRAC, SP_TABLE, BUDGET_BONUS
    cal = calibrate(a.rs, a.ra)
    d = Data()
    slots = build_slots(cal, d, a.price)
    SP_FRAC = max(0.0, min(1.0, a.sp_rebound))
    if SP_FRAC > 0:
        SP_TABLE = sp_rebound_table()
    away = []
    for item in a.trade_away:
        nm, _, sal = item.partition("=")
        away.append((nm.strip(), float(sal) if sal.strip() else 0.0))
        BUDGET_BONUS += away[-1][1]
    if away:
        gone_away = {norm_key(n) for n, _ in away}
        for s_ in slots:
            s_["opts"] = [o for o in s_["opts"] if o["key"] not in gone_away]
    if a.exclude:
        gone = {norm_key(n) for n in a.exclude}
        for s_ in slots:
            kept = [o for o in s_["opts"] if o["key"] not in gone]
            s_["opts"] = kept
        trade_gone = gone
    else:
        trade_gone = set()

    # do nothing: every spot uses its best $0 option (a free-agent starter is simply gone)
    base = best_plan(slots, 0, set(HIT_SLOTS + PIT_SLOTS), {}, use_locks=False)
    plan = best_plan(slots, a.budget, frozen, locks)
    if plan is None:
        need = sum(min(o["price"] for o in restrict(s, False, locks.get(s["group"]))) for s in slots)
        raise SystemExit(f"Budget ${a.budget:.0f}M cannot cover the locked players (about ${need:.0f}M). Raise --budget or change --lock.")

    rs0, ra0, *_ = score_plan(slots, base, cal, d)
    w_base = 162 * pyth(rs0, ra0)

    print("\n" + "=" * 100)
    print("THE HOLES -- who played each spot in 2026 and what is left for 2027")
    print("=" * 100)
    print(f"  {'Spot':<24} {'2026 starter':<20} {'Status':<11} {'2027 with no new money':<26} {'Proj':>6}")
    for s, b in zip(slots, base):
        if s["hitter"]:
            print(f"  {s['group']:<24} {str(s['starter'] or '-'):<20} {s['starter_status']:<11} {b['name']:<26} {b['proj']:>6.3f}")
        elif s.get("sp"):
            print(f"  {s['group']:<24} {str(s['starter'] or '-'):<20} {s['starter_status']:<11} {b['name']:<26} {b['proj']:>6.2f}")
        else:
            print(f"  {s['group']:<24} {'(setup spot)':<20} {'':<11} {'league-average reliever':<26}")
    print(f"\n  2026 Mariners: 76-86, {cal['rs']:.0f} runs scored, {cal['ra']:.0f} allowed "
          f"(Pythagorean {162 * pyth(cal['rs'], cal['ra']):.1f} wins).")
    print(f"  Projected 2027 with NO new money: {w_base:.1f} wins ({rs0:.0f} scored / {ra0:.0f} allowed).")

    print("\n" + "=" * 100)
    print("BEST SLATE AT EACH BUDGET (new money, estimated prices)")
    print("=" * 100)
    print(f"  {'Budget':>7} {'Spend':>7} {'Wins':>6} {'Gain':>6}   Signings")
    print(f"  {'none':>7} {0.0:>6.1f}M {w_base:>6.1f} {0.0:>+6.1f}   (nobody new; free-agent starters leave)")
    for bud in sorted(set(FRONTIER + [a.budget])):
        if bud <= 0:
            continue
        pk = best_plan(slots, bud, frozen, locks)
        if pk is None:
            print(f"  ${bud:>5.0f}M  -- too low to cover the locked players --")
            continue
        rs1, ra1, *_ = score_plan(slots, pk, cal, d)
        spend = sum(p["price"] for p in pk)
        names = ", ".join(f"{p['name']} ({ABBR.get(p['slot'], p['slot'])})" for p in pk if p["price"] > 0) or "(none)"
        mark = " <- detailed plan below" if abs(bud - a.budget) < 1e-9 else ""
        print(f"  ${bud:>5.0f}M {spend:>6.1f}M {162 * pyth(rs1, ra1):>6.1f} {162 * (pyth(rs1, ra1) - pyth(rs0, ra0)):>+6.1f}   {names}{mark}")

    print("\n" + "=" * 100)
    print(f"THE SOLUTION at ${a.budget:.0f}M -- one player per spot")
    print("=" * 100)
    print(f"  {'Spot':<24} {'Who':<20} {'How':<18} {'Proj':>6} {'Rtot':>5} {'Est $M':>7} {'Last $M':>8}")
    rows = []
    for s, p, b in zip(slots, plan, base):
        if p["key"] is None:
            continue
        stat = f"{p['proj']:.3f}" if p["hitter"] else f"{p['proj']:.2f}"
        rt = (f"{p['rtot']:+.0f}" if p["rtot_known"] else "?") if p["hitter"] else "-"
        prev = f"{p['prev']:.1f}" if p["prev"] is not None else ("-" if p["price"] == 0 else "n/a")
        lock_tag = " (locked)" if locks.get(s["group"]) else ""
        print(f"  {s['group']:<24} {p['name']:<20} {p['source'] + lock_tag:<18} {stat:>6} {rt:>5} {p['price']:>7.1f} {prev:>8}")
        rows.append({"spot": s["group"], "player": p["name"], "how": p["source"], "projection": p["proj"],
                     "rtot": p["rtot"], "est_price_M": p["price"], "last_contract_M": p["prev"]})
    model_priced = [p["name"] for p in plan if p["price"] > 0 and p["prev"] is None]
    if model_priced:
        print(f"  ('n/a' last contract = not in your market CSV, so the price for {', '.join(model_priced)} is the model's estimate)")
    unknown = [p["name"] for p in plan if p["hitter"] and p["key"] and not p["rtot_known"]]
    if unknown:
        print(f"  (no fielding match for {', '.join(unknown)}: treated as average defense, marked '?')")
    pd.DataFrame(rows).to_csv(os.path.join(EXPORT_DIR, "roster_plan.csv"), index=False)

    # the simulation: same random talent for the same player in both scenarios
    rng = np.random.default_rng(7)
    draws = make_draws([base, plan], cal, rng, N_SIMS)
    rs_b, ra_b, off_b, dfn_b, pit_b = score_plan(slots, base, cal, d, draws)
    rs_p, ra_p, off, dfn, pit = score_plan(slots, plan, cal, d, draws)
    p_b, p_p = pyth(rs_b, ra_b), pyth(rs_p, ra_p)
    exp_b, exp_p = 162 * p_b, 162 * p_p
    gain = exp_p - exp_b
    w_b = rng.binomial(162, p_b)
    w_p = rng.binomial(162, p_p)
    t = a.target_wins

    print("\n" + "=" * 100)
    print("THE 2027 SEASON, WITH AND WITHOUT THE SOLUTION (10,000 simulated seasons)")
    print("=" * 100)
    print(f"  No new money : {exp_b.mean():.1f} expected wins   (80% range {np.percentile(exp_b, 10):.0f} to {np.percentile(exp_b, 90):.0f})")
    print(f"  The solution : {exp_p.mean():.1f} expected wins   (80% range {np.percentile(exp_p, 10):.0f} to {np.percentile(exp_p, 90):.0f})")
    print(f"  Gain         : {gain.mean():+.1f} wins   (80% range {np.percentile(gain, 10):+.1f} to {np.percentile(gain, 90):+.1f}); "
          f"adds wins in {(gain > 0).mean():.0%} of simulations")
    print(f"  Runs behind the gain (plan minus no-new-money): offense {np.mean(off) - np.mean(off_b):+.0f}, "
          f"defense {np.mean(dfn) - np.mean(dfn_b):+.0f}, pitching {np.mean(pit) - np.mean(pit_b):+.0f}")
    print(f"  Playing the games (luck included): {w_b.mean():.1f} -> {w_p.mean():.1f} wins; "
          f"chance of {t}+ wins {(w_b >= t).mean():.0%} -> {(w_p >= t).mean():.0%}")
    spend = sum(p["price"] for p in plan)
    if gain.mean() > 0 and spend:
        print(f"  Cost: ${spend:.1f}M estimated new money  ->  ${spend / gain.mean():.1f}M per added win")
    print("=" * 100)
    print("ASSUMPTIONS (constants at the top of solution_sim.py)")
    print(f"  Runs per PA per OPS point {cal['slope']:.3f} ({cal['slope_note']}); {SLOT_PA} PA per starter; "
          f"{DEF_KEEP:.0%} of one season's Rtot kept; league OPS {cal['lg_ops']:.3f}.")
    print(f"  Reliever setup spot: {RP_IP} IP against a league bullpen ERA of {d.lg_bp:.2f}, leverage x{LEVERAGE}.")
    print(f"  Price: " + ("last contract only." if a.price == "prev" else
          f"larger of last contract and ${MARKET_PER_WAR:.0f}M per projected win (so cheap-last-year players are not free)."))
    if SP_FRAC > 0:
        print(f"  Starter rebound: returning starters {SP_FRAC:.0%} of the way back to their 2025 ERA (never worse than the projection).")
    if away:
        print("  Traded away: " + ", ".join(f"{n} (frees ${v:.1f}M)" for n, v in away) +
              ". What comes back in the trade (players, prospects) is NOT modelled; the open spot is filled from free agents, trade targets or replacement level.")
    if a.exclude:
        print(f"  Ruled out by you: {', '.join(a.exclude)}.")
    print(f"  Frozen spots (no signing): {', '.join(sorted(frozen)) or 'none'}. Locked: "
          f"{', '.join(f'{k}={v}' for k, v in locks.items()) or 'none'}. Rotation: five spots, ERA over "
          f"{SP_IP} IP each.")
    print(f"  Talent spread +/-{SD_BAT:.3f} OPS (hitters), +/-{SD_ERA:.2f} ERA (relievers). "
          f"2026 runs {cal['rs']:.0f}/{cal['ra']:.0f} are the official totals "
          f"(the player tables summed to {cal['table_rs']:.0f}/{cal['table_ra']:.0f}, so they were not used).")
    if d.unpriced:
        print(f"  {d.unpriced} free agent(s) were left out because no price could be set.")
    if d.skipped:
        print(f"  {len(set(d.skipped))} player(s) were left out for lack of a 2027 projection.")
    print("=" * 100)

    if not a.no_scenarios:
        override = {}
        for item in a.scenario_budget:
            if "=" not in item:
                raise SystemExit(f"--scenario-budget needs 'Name=Millions', got: {item}")
            k, v = item.split("=", 1)
            override[k.strip().title()] = float(v)
        table = rebound_table(d)
        if SP_FRAC > 0:
            print(f"\n  STARTER REBOUND ({SP_FRAC:.0%} of the way from 2026 ERA back to 2025 ERA)")
            print(f"  {'Starter':<20} {'2025':>6} {'2026':>6} {'3-yr proj':>10} {'Assumed':>8}")
            for s_ in slots:
                for o in s_["opts"]:
                    if s_.get("sp") and o["source"] == "Returns" and o["key"] in SP_TABLE:
                        r = SP_TABLE[o["key"]]
                        tgt = r["era26"] + SP_FRAC * (r["era25"] - r["era26"])
                        print(f"  {o['name']:<20} {r['era25']:>6.2f} {r['era26']:>6.2f} {o['proj']:>10.2f} {min(o['proj'], tgt):>8.2f}")
        missing = [n for n in REBOUND_PLAYERS if norm_key(n) not in table]
        if missing:
            print(f"\n  [warning] no 2025 and 2026 batting line found for: {', '.join(missing)}; they get no rebound.")
        targets, real = load_trade_targets(cal, d)
        targets = [t for t in targets if norm_key(t["name"]) not in trade_gone]
        plans = run_scenarios(slots, cal, d, frozen, locks, table, targets, real, override)
        print_scenarios(plans, table, real, w_base)
        export_for_app(plans, table, cal, EXPORT_DIR)
        page_path = os.path.join(EXPORT_DIR, "mariners_2027_plan.html")
        write_plan_page(plans, table, w_base, plans[0]["steps"][0].mean(), cal, real, page_path)
        print(f"\n  One-page plan written to {page_path}  (double-click it to open in your browser)")
        print("  The Streamlit app reads scenarios.csv / plan_ledger.csv: refresh it and open screen 6.")


if __name__ == "__main__":
    main()