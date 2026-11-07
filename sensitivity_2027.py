"""
sensitivity_2027.py
"What if we are wrong about X?"  Re-runs the Base scenario with ONE assumption changed at a time and
shows how far the expected wins and the chance of 85+ move.

Run after solution_sim_2027.py (it needs the same exports):
    python sensitivity_2027.py
    python sensitivity_2027.py --keep-arozarena        # same, but with left field locked to Arozarena
    python sensitivity_2027.py --sims 10000            # slower, smoother

The default matches the realistic plan: Skubal and Rasmussen ruled out, Arozarena not locked.
"""
import argparse

import numpy as np

import solution_sim_2027 as s
from project_2027 import norm_key


def build(price_mode="market"):
    cal = s.calibrate(None, None)
    d = s.Data()
    slots = s.build_slots(cal, d, price_mode)
    return cal, d, slots


def drop_players(slots, names):
    gone = {norm_key(n) for n in names}
    for sl in slots:
        sl["opts"] = [o for o in sl["opts"] if o["key"] not in gone]
    return gone


def haircut(slots, cal, names, delta_ops):
    """Make named hitters worse (or better) by delta_ops in every option they appear in."""
    keys = {norm_key(n) for n in names}
    for sl in slots:
        for o in sl["opts"]:
            if o["key"] in keys and o["hitter"] and o["proj"] is not None:
                o["proj"] += delta_ops
                o["v"] += cal["slope"] * s.SLOT_PA * delta_ops


def run_base(cal, d, slots, locks, frozen, gone, rebound=0.5, trades=2, budget=100.0, sp_frac=0.0,
             trade_filter=True, sims=4000, table=None, targets=None, real=True):
    """One Base run. Returns (expected wins, P(85+), P(90+), money spent)."""
    s.N_SIMS = sims
    s.SP_FRAC = sp_frac
    s.BUDGET_BONUS = 0.0
    s.SCENARIOS = [{"name": "Base", "rebound": rebound, "trades": trades, "budget": budget, "story": "sensitivity"}]
    tg = [t for t in targets if norm_key(t["name"]) not in gone] if trade_filter else []
    plans = s.run_scenarios(slots, cal, d, frozen, locks, table, tg, real, None)
    p = plans[0]
    pl = p["played"]
    return p["exp"].mean(), float((pl >= s.PLAYOFF_BAR).mean()), float((pl >= 90).mean()), p["spend"]


def main():
    ap = argparse.ArgumentParser(description="One-at-a-time sensitivity of the Base scenario.")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--keep-arozarena", action="store_true")
    ap.add_argument("--exclude", action="append", default=None, help="default: Tarik Skubal and Drew Rasmussen")
    a = ap.parse_args()
    exclude = a.exclude if a.exclude is not None else ["Tarik Skubal", "Drew Rasmussen"]
    locks = dict(s.LOCKS) if a.keep_arozarena else {}
    frozen = set(s.FROZEN)

    # keep the module constants so every variation starts from the real settings
    orig = {k: getattr(s, k) for k in ("MARKET_PER_WAR", "SD_BAT", "SD_ERA", "SD_SP_ERA", "DEF_KEEP", "SP_FRAC")}

    s.SP_TABLE = s.sp_rebound_table()

    def go(label, group, **kw):
        # every variation starts from untouched constants and a freshly built roster
        mod = kw.pop("mod", None)
        pre = kw.pop("pre", None)
        for k, v in orig.items():
            setattr(s, k, v)
        if pre:
            pre()
        c2, d2, sl2 = build(kw.pop("price", "market"))
        g2 = drop_players(sl2, exclude)
        if mod:
            mod(sl2, c2)
        tab = s.rebound_table(d2)
        tgt, re_ = s.load_trade_targets(c2, d2)
        w, p85, p90, spend = run_base(c2, d2, sl2, locks, frozen, g2, table=tab, targets=tgt, real=re_,
                                      sims=a.sims, **kw)
        return {"label": label, "group": group, "wins": w, "p85": p85, "p90": p90, "spend": spend}

    rows = [go("Base as published", "baseline")]
    base_w = rows[0]["wins"]

    def setc(**c):
        def f():
            for k, v in c.items():
                setattr(s, k, v)
        return f

    # --- the three big bets -------------------------------------------------------------
    rows += [go("Hitters do not rebound (0%)", "rebound", rebound=0.0),
             go("Hitters rebound fully (100%)", "rebound", rebound=1.0),
             go("Starters rebound fully (Woo, Gilbert back to 2025)", "starters", sp_frac=1.0),
             go("Starters rebound halfway (toward 2025 ERA)", "starters", sp_frac=0.5)]
    # --- how the money works --------------------------------------------------------------
    rows += [go("Players cost 25% more ($10M per win)", "prices", pre=setc(MARKET_PER_WAR=10.0)),
             go("Players cost 25% less ($6M per win)", "prices", pre=setc(MARKET_PER_WAR=6.0)),
             go("Budget $60M instead of $100M", "money", budget=60.0),
             go("Budget $150M instead of $100M", "money", budget=150.0),
             go("No trades possible (free agents only)", "trades", trades=0),
             go("Up to 4 trades instead of 2", "trades", trades=4)]
    # --- how much we trust the projections --------------------------------------------------
    rows += [go("Montes is worse than projected (-.040 OPS)", "unproven",
                mod=lambda sl, c: haircut(sl, c, ["Lazaro Montes"], -0.040)),
             go("Every returning hitter 0.020 OPS worse", "projections",
                mod=lambda sl, c: haircut(sl, c, [o["name"] for x in sl if x["hitter"] for o in x["opts"]
                                                 if o["source"] == "Returns"], -0.020)),
             go("Defense counts fully (all of one season's Rtot)", "defense", pre=setc(DEF_KEEP=1.0)),
             go("Defense does not count", "defense", pre=setc(DEF_KEEP=0.0)),
             go("Players are less predictable (talent spread +25%)", "noise",
                pre=setc(SD_BAT=orig["SD_BAT"] * 1.25, SD_ERA=orig["SD_ERA"] * 1.25, SD_SP_ERA=orig["SD_SP_ERA"] * 1.25)),
             go("Players are more predictable (talent spread -25%)", "noise",
                pre=setc(SD_BAT=orig["SD_BAT"] * 0.75, SD_ERA=orig["SD_ERA"] * 0.75, SD_SP_ERA=orig["SD_SP_ERA"] * 0.75))]

    # restore
    for k, v in orig.items():
        setattr(s, k, v)

    print("\n" + "=" * 104)
    print("SENSITIVITY -- Base scenario, one assumption changed at a time" +
          ("  (Arozarena locked)" if a.keep_arozarena else "  (Arozarena not locked)"))
    print("=" * 104)
    print(f"  {'What changes':<52} {'Exp wins':>9} {'vs base':>8} {'P(85+)':>7} {'P(90+)':>7} {'New $M':>7}")
    for r in rows:
        diff = "" if r["group"] == "baseline" else f"{r['wins'] - base_w:+.1f}"
        print(f"  {r['label']:<52} {r['wins']:>9.1f} {diff:>8} {r['p85']:>7.0%} {r['p90']:>7.0%} {r['spend']:>7.0f}")
    others = [r for r in rows if r["group"] != "baseline"]
    big = sorted(others, key=lambda r: abs(r["wins"] - base_w), reverse=True)[:3]
    print("\n  Biggest swings: " + "; ".join(f"{r['label']} ({r['wins'] - base_w:+.1f})" for r in big) + ".")
    print(f"  Each run uses {a.sims:,} simulated seasons, so differences under about 0.3 wins are noise.")
    print("=" * 104)


if __name__ == "__main__":
    main()