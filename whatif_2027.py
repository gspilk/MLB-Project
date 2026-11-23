"""What-if tests on top of solution_sim_2027.py: force specific signings, compare candidates for one spot,
and measure the impact of adding one more player. Nothing here changes the main model or the app.

    python whatif_2027.py --sign "Seiya Suzuki,Edmundo Sosa,Mauricio Dubon,Jojo Romero,Bryan Abreu" --maybe "Ian Happ" ^
        --lf "Randy Arozarena,Ian Happ,Steven Kwan" --impact "Eugenio Suarez" --exclude "Tarik Skubal"

How it works: start from the 'no new money' roster (every spot at its best free option), then swap in the players
you name. A signed player only replaces the current occupant if he projects better; otherwise he is counted as
bench depth (cost, no wins). Every comparison uses the same simulated talent luck for the same player.
"""
import argparse

import numpy as np
import pandas as pd

import solution_sim_2027 as S

KNOWN_SALARY_M = {"steven kwan": 11.3}      # MLB Trade Rumors 2027 arbitration projection (final arb year)


def names(items):
    out = []
    for it in items:
        out += [x.strip() for x in str(it).split(",") if x.strip()]
    return out


def find_opt(slots, name):
    k = S.norm_key(name)
    hits = [(i, o) for i, s in enumerate(slots) for o in s["opts"] if o["key"] == k and o["source"] != "Replacement"]
    return max(hits, key=lambda h: h[1]["v"]) if hits else None


def add_trade_candidate(slots, d, cal, name, group, salary):
    ops = d.projection(name, "batting")
    if ops is None:
        return slots, False
    rt = S.league_rtot().get(S.norm_key(name), 0.0)
    t = {"group": group, "name": name, "ops": float(ops), "rtot": float(rt), "salary": float(salary)}
    return S.add_trade_options(slots, [t], cal), True


def apply(base, slots, picks_in):
    """picks_in: list of (slot_index, option). Returns (picks, cost, notes); notes = (option, what, why, slot index).
    A signing fills its spot only if he projects better than the current occupant. A starter goes into whichever
    rotation spot holds the weakest starter (the five rotation spots are interchangeable)."""
    picks = list(base)
    cost, notes, claimed = 0.0, [], set()
    for i, o in sorted(picks_in, key=lambda x: -x[1]["v"]):
        cost += o["price"]
        if slots[i].get("sp"):
            free = [j for j, s_ in enumerate(slots) if s_.get("sp") and j not in claimed]
            if free:
                i = min(free, key=lambda j: picks[j]["v"])
        occupant = picks[i]
        if i in claimed:
            notes.append((o, "bench", "same spot already filled by a better signing", i))
        elif o["v"] > occupant["v"]:
            picks[i] = o
            claimed.add(i)
            notes.append((o, "starts", occupant["name"], i))
        else:
            notes.append((o, "bench", f"does not beat {occupant['name']}", i))
    return picks, cost, notes


def wins_table(slots, cal, d, plans, seed=7):
    rng = np.random.default_rng(seed)
    draws = S.make_draws(list(plans.values()), cal, rng, S.N_SIMS)
    out = {}
    for lab, pk in plans.items():
        rs, ra = S.score_plan(slots, pk, cal, d, draws)[:2]
        out[lab] = float((162 * S.pyth(rs, ra)).mean())
    return out


def odds(slots, cal, d, plans, seed=7):
    """Expected wins plus the chance of 85+ and 90+ wins when the games are actually played (luck included)."""
    rng = np.random.default_rng(seed)
    draws = S.make_draws(list(plans.values()), cal, rng, S.N_SIMS)
    out = {}
    for lab, pk in plans.items():
        rs, ra = S.score_plan(slots, pk, cal, d, draws)[:2]
        pw = S.pyth(rs, ra)
        played = np.random.default_rng(99).binomial(162, pw)
        out[lab] = (float((162 * pw).mean()), float((played >= 85).mean()), float((played >= 90).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", action="append", default=[], help="players to sign (comma list ok)")
    ap.add_argument("--maybe", action="append", default=[], help="optional extras, tested with and without")
    ap.add_argument("--lf", action="append", default=[], help="candidates to compare for left field")
    ap.add_argument("--compare", action="append", default=[], help='e.g. "Left-Handed Relievers=JoJo Romero,Adrian Morejon"')
    ap.add_argument("--impact", action="append", default=[], help="players whose added impact to measure")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--add-rp", action="append", default=[], help='e.g. "James McArthur=Right-Handed Relievers,1.0" (a reliever acquired by trade: spot, salary $M)')
    ap.add_argument("--move", action="append", default=[], help='e.g. "Brendan Donovan=Left Fielders" (a returning player changes spots)')
    ap.add_argument("--trade-away", action="append", default=[], help='e.g. "George Kirby=9.8" (salary freed in $M)')
    ap.add_argument("--rebounds", default="0,0.5,1.0", help="hitter rebound levels to test")
    a = ap.parse_args()
    sign, maybe, lf, impact = names(a.sign), names(a.maybe), names(a.lf), names(a.impact)

    cal = S.calibrate(None, None)
    d = S.Data()
    slots0 = S.build_slots(cal, d, "market")
    slots_orig = [dict(s_, opts=list(s_["opts"])) for s_ in slots0]   # the untouched roster, for the "versus today" line
    gone = {S.norm_key(n) for n in a.exclude}
    for s_ in slots0:
        s_["opts"] = [o for o in s_["opts"] if o["key"] not in gone]
    freed, away = 0.0, []
    for item in a.trade_away:
        nm, _, sal = item.partition("=")
        away.append(nm.strip())
        freed += float(sal) if sal.strip() else 0.0
    if away:
        gone_a = {S.norm_key(n) for n in away}
        for s_ in slots0:
            s_["opts"] = [o for o in s_["opts"] if o["key"] not in gone_a]
    table = S.rebound_table(d)
    allsets = set(S.HIT_SLOTS + S.PIT_SLOTS)

    # trade targets from data/inputs/trade_targets.csv become options too (salary only, no prospect cost)
    try:
        targets, _real = S.load_trade_targets(cal, d)
        targets = [t for t in targets if S.norm_key(t["name"]) not in gone and "placeholder" not in t["name"]]
        if targets:
            slots0 = S.add_trade_options(slots0, targets, cal)
    except Exception as e:
        print(f"  (trade_targets.csv not used: {e})")
    # candidates that are trade targets (not free agents): add them to their slot
    trade_adds = []
    for nm in lf:
        if find_opt(slots0, nm) is None:
            sal = KNOWN_SALARY_M.get(S.norm_key(nm))
            if sal is not None:
                trade_adds.append((nm, "Left Fielders", sal))
    for nm, g, sal in trade_adds:
        slots0, ok = add_trade_candidate(slots0, d, cal, nm, g, sal)
        if not ok:
            print(f"  (no 2027 projection found for {nm}; left out)")

    # --add-rp "Name=Right-Handed Relievers,1.0": a reliever who is already under contract or acquired by trade. He is added
    # to that setup spot at his projected ERA; sign him with --sign to count him (his salary is charged).
    for item in a.add_rp:
        nm, _, rest = item.partition("=")
        grp, _, sal = rest.partition(",")
        grp = grp.strip()
        era = d.projection(nm.strip(), "pitching")
        tgt = [i for i, s_ in enumerate(slots0) if s_["group"] == grp and not s_["hitter"]]
        if era is None or not tgt:
            print(f"  (--add-rp skipped for '{item}': no 2027 pitching projection found, or no spot called '{grp}')")
            continue
        sal_v = float(sal) if sal.strip() else 1.0
        v = (d.lg_bp - float(era)) * S.RP_IP / 9 * S.RUNS_PER_ER * S.LEVERAGE
        o_ = {"slot": grp, "name": nm.strip(), "key": S.norm_key(nm), "source": "Trade", "proj": float(era), "rtot": 0.0,
              "rtot_known": True, "v": v, "war": (v + S.REPL_RUNS_RP) / cal["rpw"], "price": sal_v, "prev": sal_v,
              "cu": int(round(sal_v / S.UNIT)), "hitter": False}
        slots0[tgt[0]] = dict(slots0[tgt[0]], opts=slots0[tgt[0]]["opts"] + [o_])
        print(f"  Added reliever {nm.strip()} at {grp}: projected {float(era):.2f} ERA, salary ${sal_v:.1f}M.")

    # --move "Name=Spot": a returning player changes position (price 0). His old spot loses him; the new spot gains him
    # at his projected OPS with neutral defense (his old-position fielding number does not carry over).
    for item in a.move:
        nm, _, grp = item.partition("=")
        grp = grp.strip()
        h = find_opt(slots0, nm.strip())
        tgt = [i for i, s_ in enumerate(slots0) if s_["group"] == grp]
        if not h or not tgt or not h[1].get("hitter"):
            print(f"  (--move skipped for '{item}': need a returning hitter and a lineup spot such as 'Left Fielders')")
            continue
        i_old, o = h
        new_o = dict(o, slot=grp, rtot=0.0,
                     v=cal["slope"] * S.SLOT_PA * (o["proj"] - cal["lg_ops"]))
        slots0 = [dict(s_, opts=[x for x in s_["opts"] if not (j == i_old and x["key"] == o["key"])]) if j == i_old else s_
                  for j, s_ in enumerate(slots0)]
        slots0[tgt[0]] = dict(slots0[tgt[0]], opts=slots0[tgt[0]]["opts"] + [new_o])
        print(f"  Moved {o['name']} from {slots_orig[i_old]['group']} to {grp} (proj {o['proj']:.3f} OPS, defense counted as average).")

    def setup(frac):
        sl, _ = S.apply_rebound(slots0, table, frac, cal)
        no_trades = [dict(s_, opts=[o for o in s_["opts"] if o["source"] != "Trade"]) for s_ in sl]
        base = S.best_plan(no_trades, 0, allsets, {}, use_locks=False)
        return sl, base

    def resolve(sl, wanted, label):
        got, missing = [], []
        for nm in wanted:
            h = find_opt(sl, nm)
            (got if h else missing).append(h if h else nm)
        if missing:
            print(f"  Not found for {label}: {', '.join(missing)} (not in your free-agent/projection files, or no projection)")
        return got

    # ── 1. each signing on its own, at the middle rebound ───────────────────
    mid = 0.5
    sl, base = setup(mid)
    pkg_hits = resolve(sl, sign, "--sign")
    maybe_hits = resolve(sl, maybe, "--maybe")
    for _i, _o in pkg_hits + maybe_hits:
        if _o.get("hitter"):
            print(f"  {_o['name']}: projection {_o['proj']:.3f} OPS, fielding runs (Rtot) used {_o['rtot']:+.1f}, source {_o['source']}")
    plans = {"base": base}
    solo = {}
    for i, o in pkg_hits + maybe_hits:
        pk, _, nt = apply(base, sl, [(i, o)])
        plans[o["name"]] = pk
        solo[o["name"]] = (nt[0][3], o)
    w = wins_table(sl, cal, d, plans)
    print("\n" + "=" * 96)
    print("EACH SIGNING ON ITS OWN (everything else stays at the no-new-money roster; hitters recover halfway)")
    print("=" * 96)
    print(f"  No new money: {w['base']:.1f} wins\n")
    print(f"  {'Player':<20}{'Spot':<24}{'Replaces':<22}{'Proj':>6}{'Cost $M':>9}{'Wins':>7}{'$M/win':>8}")
    for nm, (i, o) in solo.items():
        occ = base[i]
        gain = w[nm] - w["base"]
        stat = f"{o['proj']:.3f}" if o["hitter"] else f"{o['proj']:.2f}"
        per = f"{o['price'] / gain:>8.1f}" if gain > 0.05 else f"{'-':>8}"
        print(f"  {nm:<20}{sl[i]['group']:<24}{occ['name']:<22}{stat:>6}{o['price']:>9.1f}{gain:>+7.2f}{per}")

    # ── 2. the package across rebound levels ────────────────────────────────
    fracs = [float(x) for x in a.rebounds.split(",")]
    print("\n" + "=" * 96)
    print("YOUR PACKAGE ACROSS REBOUND LEVELS (Raleigh, Naylor and Rodriguez recover this share of the way to 2025)")
    print("=" * 96)
    print(f"  Package: {', '.join(o['name'] for _, o in pkg_hits)}" + (f"   (+ maybe: {', '.join(o['name'] for _, o in maybe_hits)})" if maybe_hits else ""))
    if away:
        print(f"  Traded away: {', '.join(away)} (frees ${freed:.1f}M; what comes back is NOT modelled). Net new money = cost minus ${freed:.1f}M.")
    hdr = f"  {'Rebound':<9}{'No new $':>9}{'Package':>9}{'Gain':>7}{'Cost $M':>9}{'Net $M':>8}{'Best use of same $':>20}"
    if maybe_hits:
        hdr += f"{'With maybe':>12}{'Maybe adds':>12}"
    print(hdr)
    notes_shown = False
    odds_rows = []
    orig_rows = []
    for fr in fracs:
        sl_f, base_f = setup(fr)
        hits = resolve(sl_f, sign, "--sign") if False else [find_opt(sl_f, n) for n in sign]
        hits = [h for h in hits if h]
        mhits = [h for h in [find_opt(sl_f, n) for n in maybe] if h]
        pk, cost, notes = apply(base_f, sl_f, hits)
        plans = {"base": base_f, "pkg": pk}
        if mhits:
            pk2, cost2, _ = apply(base_f, sl_f, hits + mhits)
            plans["pkg2"] = pk2
        no_tr = [dict(s_, opts=[o for o in s_["opts"] if o["source"] != "Trade"]) for s_ in sl_f]
        best = S.best_plan(no_tr, cost, S.FROZEN, {}, use_locks=False)
        plans["best"] = best
        ww = wins_table(sl_f, cal, d, plans)
        od = odds(sl_f, cal, d, {k: v for k, v in plans.items() if k in ("base", "pkg", "pkg2")})
        odds_rows.append((fr, od, bool(mhits)))
        if a.move or away:
            sl_o, _ = S.apply_rebound(slots_orig, table, fr, cal)
            base_o = S.best_plan([dict(s_, opts=[o for o in s_["opts"] if o["source"] != "Trade"]) for s_ in sl_o], 0, allsets, {}, use_locks=False)
            orig_rows.append((fr, wins_table(sl_f, cal, d, {"o": base_o, "p": pk})))
        line = f"  {fr:<9.2f}{ww['base']:>9.1f}{ww['pkg']:>9.1f}{ww['pkg'] - ww['base']:>+7.1f}{cost:>9.1f}{cost - freed:>8.1f}{ww['best']:>20.1f}"
        if mhits:
            line += f"{ww['pkg2']:>12.1f}{ww['pkg2'] - ww['pkg']:>+12.1f}"
        print(line)
        if not notes_shown:
            first_notes, notes_shown = notes, True
    if orig_rows:
        print("\n  VERSUS TODAY'S ROSTER (nobody moved, nobody traded away, nothing signed)")
        print(f"  {'Rebound':<9}{'Today':>9}{'Your plan':>11}{'Net change':>12}")
        for fr, w_ in orig_rows:
            print(f"  {fr:<9.2f}{w_['o']:>9.1f}{w_['p']:>11.1f}{w_['p'] - w_['o']:>+12.1f}")
    print("\n  CHANCE OF REACHING 85 AND 90 WINS (10,000 simulated seasons, luck included)")
    print(f"  {'Rebound':<9}{'No new money':>18}{'Package':>18}" + (f"{'Package + maybe':>20}" if maybe_hits else ""))
    for fr, od, has_m in odds_rows:
        def f(k):
            m, p85, p90 = od[k]
            return f"{m:.1f} | {p85:.0%} | {p90:.0%}"
        print(f"  {fr:<9.2f}{f('base'):>18}{f('pkg'):>18}" + (f"{f('pkg2'):>20}" if has_m else ""))
    print("  Each cell: expected wins | chance of 85+ | chance of 90+")
    print("\n  How the package fills the lineup (at the middle rebound):")
    for o, what, why, _ in first_notes:
        print(f"    {o['name']:<20}{what.upper():<7} {('replaces ' + why) if what == 'starts' else why}")
    print("  'Best use of same $' = what the optimizer gets from free agents for the same money (no trades).")

    # ── 3. head-to-head comparisons for one spot (rest of the package in place) ─
    compares = {}
    if lf:
        compares["Left Fielders"] = lf
    for item in a.compare:
        g, _, rest_names = item.partition("=")
        compares.setdefault(g.strip(), []).extend(names([rest_names]))
    for group, cand_names in compares.items():
        sl, base = setup(mid)
        gi = [i for i, s_ in enumerate(sl) if s_["group"] == group]
        if not gi:
            print(f"\n  --compare: no spot called '{group}' (try 'Left-Handed Relievers', 'Right-Handed Relievers', 'Shortstops'...)")
            continue
        gi = gi[0]
        print("\n" + "=" * 96)
        print(f"{group.upper()}: candidates compared, with the rest of your signings in place (hitters recover halfway)")
        print("=" * 96)
        rest = [h for h in [find_opt(sl, n) for n in sign + maybe] if h and sl[h[0]]["group"] != group
                and not (sl[h[0]].get("sp") and sl[gi].get("sp"))]
        pk_rest, cost_rest, _ = apply(base, sl, rest)
        plans = {"__none__": pk_rest}
        cands = {}
        for nm in cand_names:
            h = next((o for o in sl[gi]["opts"] if o["key"] == S.norm_key(nm)), None)
            if h is None:
                print(f"  {nm}: not an option at {group} (no projection or price)")
                continue
            pk = list(pk_rest)
            pk[gi] = h
            plans[nm] = pk
            cands[nm] = h
        ww = wins_table(sl, cal, d, plans)
        occ = pk_rest[gi]
        hit = sl[gi]["hitter"]
        print(f"  {group} without a signing: {occ['name']}" + (f" (proj {occ['proj']:.3f})" if hit and occ['proj'] else "") + f" -> {ww['__none__']:.1f} wins\n")
        print(f"  {'Candidate':<20}{'How':<12}{'Proj':>9}{'Rtot':>6}{'Cost $M':>9}{'Wins':>8}{'vs no signing':>14}{'$M/win':>8}")
        for nm, o in cands.items():
            gain = ww[nm] - ww["__none__"]
            per = f"{o['price'] / gain:>8.1f}" if gain > 0.05 else f"{'-':>8}"
            stat = f"{o['proj']:.3f}" if hit else f"{o['proj']:.2f}"
            rt = f"{o['rtot']:+.0f}" if hit else "-"
            print(f"  {nm:<20}{o['source']:<12}{stat:>9}{rt:>6}{o['price']:>9.1f}{ww[nm]:>8.1f}{gain:>+14.2f}{per}")
        if any(o["source"] == "Trade" for o in cands.values()):
            print("  Trade candidates cost salary only (no prospect cost), so their gain is an upper bound.")
        if len(cands) > 1:
            print("  Only one of these can start, so signing two means paying for one who sits.")

    # ── 4. impact of adding a player ───────────────────────────────────────
    for nm in impact:
        print("\n" + "=" * 96)
        print(f"IMPACT OF ADDING {nm.upper()} (hitters recover halfway)")
        print("=" * 96)
        sl, base = setup(mid)
        h = find_opt(sl, nm)
        if not h:
            print(f"  Not found: {nm}")
            continue
        i, o = h
        hits = [x for x in [find_opt(sl, n) for n in sign + maybe] if x]
        pk_pkg, c_pkg, _ = apply(base, sl, hits)
        pk_pkg2, c_pkg2, n2 = apply(base, sl, hits + [h])
        pk_solo, c_solo, n_solo = apply(base, sl, [h])
        ww = wins_table(sl, cal, d, {"base": base, "solo": pk_solo, "pkg": pk_pkg, "pkg2": pk_pkg2})
        occ = base[i]
        print(f"  Spot: {sl[i]['group']}   current occupant: {occ['name']} (proj {occ['proj']:.3f})   {o['name']} proj {o['proj']:.3f}, cost ${o['price']:.1f}M")
        print(f"  On top of no new money : {ww['base']:.1f} -> {ww['solo']:.1f}  ({ww['solo'] - ww['base']:+.2f} wins)")
        print(f"  On top of your package : {ww['pkg']:.1f} -> {ww['pkg2']:.1f}  ({ww['pkg2'] - ww['pkg']:+.2f} wins)  for ${o['price']:.1f}M")
        verdict = n_solo[0][1]
        print(f"  Result: he {'would start and replace ' + n_solo[0][2] if verdict == 'starts' else 'would NOT start (' + n_solo[0][2] + ')'}.")
        print("  The model has one starter per spot and no platoon, injury or bench value, so a bench bat adds nothing here.")
    print("\n  Caveats: free-agent prices are the model's anchors; trade targets cost salary only (no prospects); one spot, one starter.")


if __name__ == "__main__":
    main()


# ───────────────────────── callable version, used by the Streamlit app (screen 4) ─────────────────────────
import functools


@functools.lru_cache(maxsize=2)
def load_model(exclude=()):
    """Builds the 2027 model once (projections, prices, calibration). Cached, so repeat slates are fast."""
    cal = S.calibrate(None, None)
    d = S.Data()
    slots0 = S.build_slots(cal, d, "market")
    gone = {S.norm_key(n) for n in exclude}
    for s_ in slots0:
        s_["opts"] = [o for o in s_["opts"] if o["key"] not in gone]
    try:
        targets, _real = S.load_trade_targets(cal, d)
        targets = [t for t in targets if S.norm_key(t["name"]) not in gone and "placeholder" not in t["name"]]
        if targets:
            slots0 = S.add_trade_options(slots0, targets, cal)
    except Exception:
        pass
    return cal, d, slots0, S.rebound_table(d)


def evaluate_slate(names_in, freed_M=0.0, away=(), exclude=("Tarik Skubal", "Drew Rasmussen"), rebounds=(0.0, 0.5, 1.0)):
    """Runs a list of signings through the 2027 model. Returns a dict of tables for the app.
    Every spot starts at its best no-new-money option; a signing starts only if he projects better than the
    current occupant (otherwise he is bench depth: cost, no wins)."""
    cal, d, base_slots, table = load_model(tuple(exclude))
    gone = {S.norm_key(n) for n in away}
    slots0 = [dict(s_, opts=[o for o in s_["opts"] if o["key"] not in gone]) for s_ in base_slots]
    allsets = set(S.HIT_SLOTS + S.PIT_SLOTS)

    def setup_level(frac):
        sl, _ = S.apply_rebound(slots0, table, frac, cal)
        no_tr = [dict(s_, opts=[o for o in s_["opts"] if o["source"] != "Trade"]) for s_ in sl]
        return sl, S.best_plan(no_tr, 0, allsets, {}, use_locks=False)

    wanted = list(dict.fromkeys(n.strip() for n in names_in if n.strip()))
    mid = 0.5 if 0.5 in rebounds else rebounds[0]
    sl, base = setup_level(mid)
    found = {n: find_opt(sl, n) for n in wanted}
    missing = [n for n, h in found.items() if h is None]
    hits = [h for h in found.values() if h]
    out = {"missing": missing, "players": [], "levels": [], "notes": [], "cost": 0.0}
    if not hits:
        return out

    plans = {"base": base}
    solo_slot = {}
    for nm, h in found.items():
        if h:
            pk, _, nt = apply(base, sl, [h])
            plans[nm] = pk
            solo_slot[nm] = nt[0][3]
    w = wins_table(sl, cal, d, plans)
    for nm, h in found.items():
        if not h:
            continue
        i, o = h
        si = solo_slot[nm]
        occ = base[si]
        gain = w[nm] - w["base"]
        out["players"].append({"player": o["name"], "spot": sl[si]["group"], "replaces": occ["name"],
                               "projection": o["proj"], "is_hitter": bool(o["hitter"]), "cost_M": o["price"],
                               "wins_alone": gain})
    for fr in rebounds:
        sl_f, base_f = setup_level(fr)
        hs = [h for h in [find_opt(sl_f, n) for n in wanted] if h]
        pk, cost, notes = apply(base_f, sl_f, hs)
        od = odds(sl_f, cal, d, {"base": base_f, "pkg": pk})
        out["levels"].append({"rebound": fr, "no_new_wins": od["base"][0], "slate_wins": od["pkg"][0],
                              "gain": od["pkg"][0] - od["base"][0], "p85": od["pkg"][1], "p90": od["pkg"][2],
                              "p85_no_new": od["base"][1]})
        out["cost"] = cost
        if abs(fr - mid) < 1e-9:
            out["notes"] = [(o["name"], what, why) for o, what, why, _ in notes]
    out["cost_net"] = out["cost"] - freed_M
    return out