import pandas as pd

from data_builder import build_all
from name_matching import key_from_first_last, key_from_last_first
from roster import get_roster_last_names

data    = build_all(2026)
all_bat = data.get("batting",{}).get("all_players")
sc_bat  = data.get("statcast",{}).get("batters")

all_bat["_key"] = all_bat["Name"].apply(key_from_first_last)
sc_bat["_key"]  = sc_bat["Name"].apply(key_from_last_first)

b_cols = ["_key","Name","Pos"] + \
         [c for c in ["PA","OPS","HR","BA","league"] if c in all_bat.columns]
s_cols = ["_key","Name"] + \
         [c for c in ["xwOBA","wOBA","Barrel%","HardHit%","K%","BB%"]
          if c in sc_bat.columns]

merged = pd.merge(all_bat[b_cols], sc_bat[s_cols], on="_key", how="inner")

for c in ["xwOBA","Barrel%","PA"]:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")

# live roster, not a hardcoded list -- see roster.py
MARINERS = get_roster_last_names(data)
if not MARINERS:
    print("[warn] live roster unavailable -- targets may include current Mariners")

def is_mariner(k):
    return k.split("_")[0] in MARINERS

# filter: quality threshold + min PA + not a Mariner
# no team filter since team not available
mask = (
    (merged["xwOBA"]   >= 0.330) &
    (merged["xwOBA"]   <= 0.420) &
    (merged["Barrel%"] >= 8.0)   &
    (merged["PA"]      >= 100)   &
    (~merged["_key"].apply(is_mariner))
)
tgt = merged[mask].sort_values("xwOBA", ascending=False).head(15)
print(f"FOUND {len(tgt)} BATTER TARGETS:")
print(f"  {'Name':<25} {'Pos':<8} {'League':<5} {'xwOBA':>6} {'Barrel%':>8} {'HR':>4} {'OPS':>6}")
print("  " + "-"*65)
for _, row in tgt.iterrows():
    name_y = str(row.get("Name_y",""))
    if "," in name_y:
        p = name_y.split(",",1)
        name = f"{p[1].strip()} {p[0].strip()}"
    else:
        name = name_y
    print(f"  {name:<25} {str(row.get('Pos','?')):<8} "
          f"{str(row.get('league','?')):<5} "
          f"{row['xwOBA']:>6.3f} {row['Barrel%']:>8.1f} "
          f"{str(row.get('HR','?')):>4} {str(row.get('OPS','?')):>6}")