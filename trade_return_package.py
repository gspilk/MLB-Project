"""
trade_return_package.py
Tracks everyone the Mariners RECEIVED across both 2026 deadline trades:
  - Seranthony Dominguez (RHP) + Nolan Jones (OF) + Boston Smith (C/OF
    prospect) -- from the Aug 1 Luis Castillo trade with the White Sox
  - Taylor Ward (OF) -- from the Aug 3-4 trade with the Orioles
    (Hoppe + Moore + Kreiling going the other way)

Dominguez and Ward are both real, current Mariners -- pulled live from
the same bbref scrape used everywhere else in this project.

Jones and Smith are minor leaguers -- this project has no MiLB scraper
(bbref's minor league data lives on a completely different site
structure), so their numbers below are a real, sourced, DATED snapshot
from public reporting at trade time, not live-updating data. Treat them
as a point-in-time reference, not something this script re-verifies
each run the way the MLB-level stats are.

Usage:
    python trade_return_package.py
"""

from data_builder import build_all


# Real, sourced minor-league snapshot as of the trade (early Aug 2026).
# NOT live data -- there's no MiLB scraper in this project to refresh
# these automatically. Update by hand if you want a newer reference
# point; these will NOT change on their own like the MLB stats below.
JONES_SNAPSHOT = {
    "name": "Nolan Jones", "as_of": "2026-08-01 (trade date)",
    "level": "Triple-A (White Sox org pre-trade)",
    "PA": 382, "OPS": 0.807, "HR": 14, "RBI": 59, "SB": 11,
    "note": "Career MLB best was 2023 with Rockies (.931 OPS, 20 HR); "
           "posted a sub-.700 OPS in every MLB season since. Hasn't "
           "played in the majors at all in 2026 -- pure Triple-A year. "
           "Last MLB action was 2025 with Cleveland (.600 OPS).",
}
SMITH_SNAPSHOT = {
    "name": "Boston Smith", "as_of": "2026-08-01 (trade date)",
    "level": "3 levels: Low-A/High-A/Double-A (White Sox org pre-trade)",
    "games": 78, "BA": 0.291, "OBP": 0.448, "SLG": 0.589, "OPS": 1.037, "HR": 22,
    "note": "2025 6th-round pick out of Wright State. Ranked White Sox's "
           "(now Seattle's) No. 15 -> No. 14 prospect. Never played "
           "above Double-A yet -- a real prospect, not MLB-ready.",
}


def get_live_mlb_stats(data: dict) -> dict:
    bat = data["seattle"]["batting"]
    pit = data["seattle"]["pitching"]

    ward_row = bat[bat["Name"].str.contains("Ward", na=False)]
    dom_row = pit[pit["Name"].str.contains("Domínguez|Dominguez", na=False, regex=True)]

    ward = {}
    if not ward_row.empty:
        ward = {
            "PA": ward_row["PA"].values[0],
            "OPS": ward_row["OPS"].values[0],
        }
    dom = {}
    if not dom_row.empty:
        dom = {
            "IP": dom_row["IP"].values[0],
            "ERA": dom_row["ERA"].values[0],
        }
    return {"ward": ward, "dominguez": dom}


def print_report(mlb: dict):
    print("\n" + "=" * 70)
    print("TRADE RETURN PACKAGE -- everyone the Mariners received")
    print("=" * 70)

    print("\n-- MLB-level (live, current stats) --")
    ward = mlb["ward"]
    if ward:
        print(f"  Taylor Ward (from BAL):        {ward['PA']} PA, {ward['OPS']:.3f} OPS")
    dom = mlb["dominguez"]
    if dom:
        print(f"  Seranthony Dominguez (from CHW): {dom['IP']} IP, {dom['ERA']:.2f} ERA")

    print("\n-- Minor league (real, dated snapshot -- not live) --")
    j, s = JONES_SNAPSHOT, SMITH_SNAPSHOT
    print(f"  Nolan Jones (from CHW):  {j['level']}, {j['PA']} PA, "
          f"{j['OPS']:.3f} OPS, {j['HR']} HR  [as of {j['as_of']}]")
    print(f"    -> {j['note']}")
    print(f"  Boston Smith (from CHW): {s['level']}, {s['games']} G, "
          f"{s['OPS']:.3f} OPS, {s['HR']} HR  [as of {s['as_of']}]")
    print(f"    -> {s['note']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    mlb = get_live_mlb_stats(data)
    print_report(mlb)
