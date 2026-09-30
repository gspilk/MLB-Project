"""
One-off build script: parses the raw Spotrac free-agent paste (1B/2B/3B/SS
combined page, OF page, RP page -- pasted 2026-09-29) into a clean,
deduplicated free_agent_market.csv.

Spotrac lists a multi-position-eligible player on EVERY position page he
qualifies for (Jazz Chisholm, Jake Bauers, Adam Frazier, Miguel Andujar,
LaMonte Wade Jr., Isiah Kiner-Falefa all appear on more than one of the
three pasted pages with IDENTICAL age/team/AAV) -- deduped by normalized
name, keeping the first occurrence, since all their real financial facts
match across duplicates (only the "Pos" label sometimes differs slightly
depending on which filtered page listed them, e.g. Jazz Chisholm as "CF"
under the 1B-infield page vs. also "CF" under the OF page -- consistent
here, but the position group used for actual matching in
free_agent_recommendations.py comes from that file's own BBREF-based
position logic anyway, not from Spotrac's label, so this is just kept as
informational context).

Ryan Mountcastle appears twice on the SAME page with two different
contract "Type" values (CLUB option / UFA) -- real Spotrac quirk (his
club option was still pending/declined at scrape time) -- deduped to the
UFA line since that's his actual free agent status.
"""
import csv
import unicodedata

SRC = "/tmp/claude-0/-home-claude/701f6e76-46fe-59de-9fb2-3197a607777a/scratchpad/raw_market_paste.txt"
OUT = "/tmp/claude-0/-home-claude/701f6e76-46fe-59de-9fb2-3197a607777a/scratchpad/free_agent_market.csv"


def normalize(name):
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.strip().lower()


rows_by_key = {}
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            print(f"  [warn] skipping malformed line ({len(parts)} fields): {line!r}")
            continue
        name, pos, age, prev_team, prev_aav, contract_type, arm = parts
        name = name.replace(" QOI", "").strip()  # strip Qualifying-Offer-Issued tag
        prev_aav_num = float(prev_aav.replace("$", "").replace(",", ""))
        key = normalize(name)

        entry = {
            "Name": name,
            "Pos": pos,
            "Age": age,
            "Prev_Team": prev_team,
            "Prev_AAV": prev_aav_num,
            "Contract_Type": contract_type,
            "Arm": arm,
        }

        if key not in rows_by_key:
            rows_by_key[key] = entry
        else:
            # Ryan Mountcastle case: prefer the UFA line over a stale
            # club-option line when both exist for the same real person.
            existing = rows_by_key[key]
            if existing["Contract_Type"] != "UFA" and contract_type == "UFA":
                rows_by_key[key] = entry

with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Name", "Pos", "Age", "Prev_Team",
                                           "Prev_AAV", "Contract_Type", "Arm"])
    writer.writeheader()
    for entry in rows_by_key.values():
        writer.writerow(entry)

print(f"Parsed {len(rows_by_key)} unique free agents (from raw paste) -> {OUT}")