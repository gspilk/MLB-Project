import pandas as pd
import unicodedata
import sys
sys.path.insert(0, r"c:/Users/geoff/OneDrive/Desktop/MLB PROJECT")

from data_builder import build_all

data    = build_all(2026)
all_bat = data.get("batting",{}).get("all_players")
sc_bat  = data.get("statcast",{}).get("batters")

def norm(s):
    s = str(s).lower().strip()
    return "".join(c for c in unicodedata.normalize("NFD",s)
                   if unicodedata.category(c)!="Mn")

def key_fl(name):
    p = str(name).split()
    return norm(p[-1])+"_"+norm(p[0])[0] if len(p)>=2 else norm(name)

def key_lf(name):
    p = str(name).split(",")
    return norm(p[0].strip())+"_"+norm(p[1].strip())[0] if len(p)>=2 else norm(name)

all_bat["_key"] = all_bat["Name"].apply(key_fl)
sc_bat["_key"]  = sc_bat["Name"].apply(key_lf)

b_cols = ["_key","Name","Pos"] + \
         [c for c in ["PA","OPS","HR","BA","league"] if c in all_bat.columns]
s_cols = ["_key","Name"] + \
         [c for c in ["xwOBA","wOBA","Barrel%","HardHit%","K%","BB%"]
          if c in sc_bat.columns]

merged = pd.merge(all_bat[b_cols], sc_bat[s_cols], on="_key", how="inner")

for c in ["xwOBA","Barrel%","PA"]:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")

MARINERS = {
    "rodriguez","arozarena","raley","young","crawford","raleigh",
    "naylor","canzone","donovan","emerson","garver","kirby","woo",
    "hancock","gilbert","castillo","miller","munoz","brash","ferrer",
    "bazardo","criswell","speier","pereda","wisdom","rivas","refsnyder",
    "robles","wilson","bliss","joe","mastrobuoni"
}

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