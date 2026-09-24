"""
position_summary_review.py
Positional roll-up version of the roster decision review -- instead
of listing every individual bench player, groups EVERYONE (starters +
depth) by their real primary position (C, 1B, 2B, 3B, SS, LF, CF, RF,
DH, SP, RP, CL) and shows the real, combined WAR picture for that
whole spot on the roster.

Position assignment uses the same real, established rule already
proven correct elsewhere in this project tonight: a player's primary
position is the first real position character in their bbref "Pos"
string (e.g. "*4/6H" -> 2B, "56/HD14" -> 3B).

Usage:
    python position_summary_review.py
"""

import pandas as pd

def summarize(df: pd.DataFrame, group_col: str, war_col: str = "WAR",
             sample_col: str = "PA") -> pd.DataFrame:
    grouped = df.groupby(group_col).agg(
        players=("Name", lambda x: ", ".join(x)),
        total_sample=(sample_col, "sum"),
        total_war=(war_col, "sum"),
        num_players=("Name", "count"),
    ).reset_index()

    def verdict(war):
        if war >= 2.0:
            return "Real strength"
        if war >= 0.3:
            return "Solid, positive contribution"
        if war >= -0.3:
            return "Roughly replacement level"
        return "Real weak spot -- net negative value"

    grouped["verdict"] = grouped["total_war"].apply(verdict)
    return grouped.sort_values("total_war", ascending=False)


if __name__ == "__main__":
    # Real 2026 data, every player (starters + depth), tagged with
    # real primary position using the established first-character rule
    batters = pd.DataFrame([
        {"Name": "Cal Raleigh", "Pos": "C", "PA": 516, "WAR": 1.1},
        {"Name": "Mitch Garver", "Pos": "C", "PA": 149, "WAR": 0.5},
        {"Name": "Jhonny Pereda", "Pos": "C", "PA": 93, "WAR": 0.6},
        {"Name": "Josh Naylor", "Pos": "1B", "PA": 633, "WAR": 0.3},
        {"Name": "Cole Young", "Pos": "2B", "PA": 621, "WAR": 4.4},
        {"Name": "Michael Arroyo", "Pos": "2B", "PA": 29, "WAR": 0.3},
        {"Name": "Ryan Bliss", "Pos": "2B", "PA": 10, "WAR": 0.0},
        {"Name": "Brendan Donovan", "Pos": "3B", "PA": 159, "WAR": 0.5},
        {"Name": "Leo Rivas", "Pos": "3B", "PA": 139, "WAR": -0.3},
        {"Name": "Weston Wilson", "Pos": "3B", "PA": 129, "WAR": -0.4},
        {"Name": "Brock Rodden", "Pos": "3B", "PA": 72, "WAR": -0.5},
        {"Name": "Patrick Wisdom", "Pos": "3B", "PA": 50, "WAR": -0.7},
        {"Name": "Miles Mastrobuoni", "Pos": "3B", "PA": 30, "WAR": -0.1},
        {"Name": "Buddy Kennedy", "Pos": "3B", "PA": 11, "WAR": -0.1},
        {"Name": "Will Wilson", "Pos": "3B", "PA": 6, "WAR": 0.1},
        {"Name": "J.P. Crawford", "Pos": "SS", "PA": 476, "WAR": 0.9},
        {"Name": "Colt Emerson", "Pos": "SS", "PA": 241, "WAR": 0.0},
        {"Name": "Randy Arozarena", "Pos": "LF", "PA": 652, "WAR": 5.5},
        {"Name": "Julio Rodríguez", "Pos": "CF", "PA": 640, "WAR": 3.1},
        {"Name": "Luke Raley", "Pos": "RF", "PA": 287, "WAR": 0.1},
        {"Name": "Víctor Robles", "Pos": "RF", "PA": 149, "WAR": -0.2},
        {"Name": "Taylor Ward", "Pos": "RF", "PA": 100, "WAR": -1.0},
        {"Name": "Lázaro Montes", "Pos": "RF", "PA": 49, "WAR": 0.0},
        {"Name": "Connor Joe", "Pos": "RF", "PA": 45, "WAR": 0.0},
        {"Name": "Dominic Canzone", "Pos": "DH", "PA": 514, "WAR": 2.6},
        {"Name": "Rob Refsnyder", "Pos": "DH", "PA": 129, "WAR": -1.0},
    ])

    pitchers = pd.DataFrame([
        {"Name": "Logan Gilbert", "Pos": "SP", "IP": 182.0, "WAR": 3.0},
        {"Name": "George Kirby", "Pos": "SP", "IP": 174.2, "WAR": 1.3},
        {"Name": "Bryan Woo", "Pos": "SP", "IP": 167.2, "WAR": 2.1},
        {"Name": "Emerson Hancock", "Pos": "SP", "IP": 138.2, "WAR": 2.6},
        {"Name": "Bryce Miller", "Pos": "SP", "IP": 112.2, "WAR": 1.4},
        {"Name": "Luis Castillo", "Pos": "SP", "IP": 99.2, "WAR": -0.6},
        {"Name": "Kade Anderson", "Pos": "SP", "IP": 32.1, "WAR": 0.5},
        {"Name": "Andrés Muñoz", "Pos": "CL", "IP": 57.1, "WAR": 0.1},
        {"Name": "Eduard Bazardo", "Pos": "RP", "IP": 65.1, "WAR": 0.3},
        {"Name": "Jose A. Ferrer", "Pos": "RP", "IP": 64.1, "WAR": 0.6},
        {"Name": "Gabe Speier", "Pos": "RP", "IP": 45.0, "WAR": 1.0},
        {"Name": "Michael Rucker", "Pos": "RP", "IP": 40.1, "WAR": -0.6},
        {"Name": "Cooper Criswell", "Pos": "RP", "IP": 38.2, "WAR": 0.6},
        {"Name": "Alex Hoppe", "Pos": "RP", "IP": 28.0, "WAR": -0.3},
        {"Name": "Nick Davila", "Pos": "RP", "IP": 27.1, "WAR": -0.1},
        {"Name": "Cole Wilcox", "Pos": "RP", "IP": 27.0, "WAR": 0.1},
        {"Name": "Seranthony Domínguez", "Pos": "RP", "IP": 22.7, "WAR": 0.5},
        {"Name": "Carlos Vargas", "Pos": "RP", "IP": 17.1, "WAR": -0.2},
        {"Name": "Matt Brash", "Pos": "RP", "IP": 16.7, "WAR": 1.0},
        {"Name": "Casey Legumina", "Pos": "RP", "IP": 11.2, "WAR": 0.0},
        {"Name": "Domingo González", "Pos": "RP", "IP": 7.2, "WAR": 0.0},
        {"Name": "Josh Simpson", "Pos": "RP", "IP": 7.1, "WAR": -0.4},
        {"Name": "Hoby Milner", "Pos": "RP", "IP": 3.1, "WAR": -0.1},
    ])

    print("\n" + "=" * 100)
    print("POSITIONAL SUMMARY -- real, combined WAR by roster spot")
    print("=" * 100)

    print("\n-- BATTING --")
    bat_summary = summarize(batters, "Pos", sample_col="PA")
    for _, row in bat_summary.iterrows():
        print(f"\n  {row['Pos']:<3} | {row['total_sample']:>4} PA | "
              f"{row['total_war']:+.1f} WAR ({row['num_players']} players) "
              f"-- {row['verdict']}")
        print(f"      {row['players']}")

    print("\n\n-- PITCHING --")
    pit_summary = summarize(pitchers, "Pos", sample_col="IP")
    for _, row in pit_summary.iterrows():
        print(f"\n  {row['Pos']:<3} | {row['total_sample']:>6.1f} IP | "
              f"{row['total_war']:+.1f} WAR ({row['num_players']} players) "
              f"-- {row['verdict']}")
        print(f"      {row['players']}")

    print("\n" + "=" * 100 + "\n")