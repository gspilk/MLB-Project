from data_builder import build_all
from team_analyzer import analyze_team
from player_grades import grade_players
from recommender import generate_recommendations

data     = build_all(2026)
analysis = analyze_team(data)
grades   = grade_players(data)
recs     = generate_recommendations(data, analysis, grades)

tg = recs.get("player_targets", {})
bt = tg.get("batter_targets", [])
pt = tg.get("pitcher_targets", [])

print(f"\nBATTER TARGETS — seller teams, xwOBA .310-.390:")
print(f"  {'Name':<25} {'Team':<25} {'Avail':<20} {'xwOBA':>6} {'Barrel%':>8} {'HR':>4} {'OPS':>6}")
print("  " + "-"*95)
if bt:
    for p in bt:
        print(f"  {str(p['name']):<25} {str(p['team']):<25} "
              f"{str(p.get('availability','?')):<20} "
              f"{str(p['xwOBA']):>6} {str(p.get('Barrel%','N/A')):>8} "
              f"{str(p.get('HR','N/A')):>4} {str(p.get('OPS','N/A')):>6}")
else:
    print("  No targets found")

print(f"\nPITCHER TARGETS — seller teams, xwOBA against <=.310:")
print(f"  {'Name':<25} {'Team':<25} {'Avail':<20} {'G':>3} {'IP':>5} {'ERA':>5} {'xwOBA vs':>9}")
print("  " + "-"*95)
if pt:
    for p in pt:
        print(f"  {str(p['name']):<25} {str(p['team']):<25} "
              f"{str(p.get('availability','?')):<20} "
              f"{str(p.get('G','N/A')):>3} {str(p.get('IP','N/A')):>5} "
              f"{str(p.get('ERA','N/A')):>5} {str(p['xwOBA_against']):>9}")
else:
    print("  No targets found")