-- Analysis queries for the Mariners project (MySQL 8 / MariaDB).
-- Run after load_mysql.py:  USE mlb_project;  then run any query below.
-- Mariners team id = 136; Mariners abbreviation in player tables = 'SEA'.


-- 1. OFFENSE vs PITCHING: Mariners runs scored / allowed per game by month,
--    next to the rest of the league (positive diff = better than league).
WITH team_games AS (
    SELECT season, game_date, home_id AS team_id, home_runs AS rs, away_runs AS ra FROM games WHERE played = 1
    UNION ALL
    SELECT season, game_date, away_id,            away_runs,       home_runs        FROM games WHERE played = 1
)
SELECT
    season,
    MONTH(game_date)                                                    AS month,
    SUM(team_id = 136)                                                  AS games,
    ROUND(SUM(CASE WHEN team_id = 136 THEN rs END) / SUM(team_id = 136), 2)           AS sea_rs_per_g,
    ROUND(SUM(CASE WHEN team_id <> 136 THEN rs END) / SUM(team_id <> 136), 2)         AS lg_rs_per_g,
    ROUND(SUM(CASE WHEN team_id = 136 THEN ra END) / SUM(team_id = 136), 2)           AS sea_ra_per_g,
    ROUND(SUM(CASE WHEN team_id <> 136 THEN ra END) / SUM(team_id <> 136), 2)         AS lg_ra_per_g,
    ROUND(SUM(CASE WHEN team_id = 136 THEN rs END) / SUM(team_id = 136)
        - SUM(CASE WHEN team_id <> 136 THEN rs END) / SUM(team_id <> 136), 2)         AS offense_vs_lg,
    ROUND(SUM(CASE WHEN team_id <> 136 THEN ra END) / SUM(team_id <> 136)
        - SUM(CASE WHEN team_id = 136 THEN ra END) / SUM(team_id = 136), 2)           AS pitching_vs_lg
FROM team_games
GROUP BY season, MONTH(game_date)
HAVING games > 0
ORDER BY season, month;


-- 2. STARTERS: pitches per start and average fastball velocity, by season.
SELECT
    pitcher,
    season,
    COUNT(DISTINCT game_pk)                                             AS starts,
    ROUND(COUNT(*) / COUNT(DISTINCT game_pk), 1)                        AS pitches_per_start,
    ROUND(AVG(CASE WHEN pitch_type IN ('FF','FA','FT','SI') THEN speed END), 1) AS avg_fastball_mph
FROM pitches
WHERE is_starter = 1
GROUP BY pitcher, season
HAVING starts >= 10
ORDER BY pitcher, season;


-- 3. TIMES THROUGH THE ORDER: OPS against each starter, 1st / 2nd / 3rd time.
SELECT
    pitcher,
    season,
    tto                                                                  AS time_through,
    COUNT(*)                                                             AS pa,
    ROUND( (SUM(hit_bases > 0) + SUM(is_walk) + SUM(is_hbp))
         / NULLIF(SUM(is_ab) + SUM(is_walk) + SUM(is_hbp) + SUM(is_sac_fly), 0), 3)  AS obp,
    ROUND( SUM(hit_bases) / NULLIF(SUM(is_ab), 0), 3)                    AS slg,
    ROUND( (SUM(hit_bases > 0) + SUM(is_walk) + SUM(is_hbp))
         / NULLIF(SUM(is_ab) + SUM(is_walk) + SUM(is_hbp) + SUM(is_sac_fly), 0)
         + SUM(hit_bases) / NULLIF(SUM(is_ab), 0), 3)                    AS ops
FROM plate_appearances
WHERE is_starter = 1 AND tto BETWEEN 1 AND 3
GROUP BY pitcher, season, tto
HAVING pa >= 40
ORDER BY pitcher, season, tto;


-- 4. WHIFF RATE BY PITCH TYPE (whiffs / swings), per starter and season.
--    Swing calls: S, W, M, T (swinging strike types) + F, X, D, E, L, O, R (contact).
SELECT
    pitcher,
    season,
    pitch_type,
    COUNT(*)                                                             AS pitches,
    ROUND(100 * SUM(call_code IN ('S','W','M','T'))
        / NULLIF(SUM(call_code IN ('S','W','M','T','F','X','D','E','L','O','R')), 0), 1) AS whiff_pct
FROM pitches
WHERE is_starter = 1
GROUP BY pitcher, season, pitch_type
HAVING pitches >= 150
ORDER BY pitcher, pitch_type, season;


-- 5. HITTERS: Mariners batters with expected stats (batting + Statcast joined by player id).
SELECT
    b.name,
    b.pos,
    b.pa,
    b.ops,
    s.xwoba,
    s.woba,
    ROUND(s.xwoba - s.woba, 3)                                           AS woba_gap,   -- + = unlucky
    b.war
FROM batting_players b
JOIN statcast_players s
  ON s.player_id = b.player_id AND s.season = b.season AND s.player_type = 'batter'
WHERE b.season = 2026 AND b.team = 'SEA' AND b.pa >= 150
ORDER BY s.xwoba DESC;


-- 6. WEAK SPOTS: Mariners' rank in the league for each key batting stat (2026).
SELECT stat, sea_value, lg_rank, '/30' AS of_teams FROM (
    SELECT 'OPS' AS stat, ROUND(ops, 3) AS sea_value,
           RANK() OVER (ORDER BY ops DESC) AS lg_rank, team FROM (
        SELECT team, SUM(ops * pa) / SUM(pa) AS ops FROM batting_players
        WHERE season = 2026 GROUP BY team) t
) r WHERE team = 'SEA';