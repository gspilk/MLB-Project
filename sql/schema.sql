-- Schema for the Mariners analysis database (MySQL 8 / MariaDB).
-- Safe to run more than once: every table uses CREATE TABLE IF NOT EXISTS.
-- Run it from Workbench (File > Open SQL Script > run), or let load_mysql.py run it.

CREATE TABLE IF NOT EXISTS games (
    game_pk     INT PRIMARY KEY,
    season      SMALLINT NOT NULL,
    game_date   DATE NOT NULL,
    home_id     INT NOT NULL,
    away_id     INT NOT NULL,
    home_runs   SMALLINT NULL,
    away_runs   SMALLINT NULL,
    innings     SMALLINT NULL,
    day_night   VARCHAR(10) NULL,
    status      VARCHAR(40) NULL,
    played      TINYINT(1) NOT NULL,
    INDEX idx_games_season_date (season, game_date)
);

CREATE TABLE IF NOT EXISTS plate_appearances (
    pa_id          INT AUTO_INCREMENT PRIMARY KEY,
    season         SMALLINT NOT NULL,
    game_pk        INT NOT NULL,
    game_date      DATE NOT NULL,
    inning         SMALLINT,
    pitcher_id     INT NOT NULL,
    pitcher        VARCHAR(80),
    batter_id      INT NOT NULL,
    batter         VARCHAR(80),
    is_starter     TINYINT(1) NOT NULL,
    pitches_before SMALLINT,
    pitches_in_pa  SMALLINT,
    tto            SMALLINT,
    event_type     VARCHAR(60),
    hit_bases      SMALLINT,
    is_ab          TINYINT(1),
    is_walk        TINYINT(1),
    is_hbp         TINYINT(1),
    is_sac_fly     TINYINT(1),
    is_k           TINYINT(1),
    launch_speed   DOUBLE NULL,
    runs           SMALLINT,
    INDEX idx_pa_pitcher (pitcher_id, season),
    INDEX idx_pa_game (game_pk)
);

CREATE TABLE IF NOT EXISTS pitches (
    pitch_id    INT AUTO_INCREMENT PRIMARY KEY,
    season      SMALLINT NOT NULL,
    game_pk     INT NOT NULL,
    game_date   DATE NOT NULL,
    inning      SMALLINT,
    pitcher_id  INT NOT NULL,
    pitcher     VARCHAR(80),
    is_starter  TINYINT(1) NOT NULL,
    pitch_no    SMALLINT,
    pitch_type  VARCHAR(4),
    speed       DOUBLE NULL,
    pa_pitch    SMALLINT,
    strikes     SMALLINT,
    call_code   VARCHAR(4),
    zone        SMALLINT NULL,
    spin        DOUBLE NULL,
    ivb         DOUBLE NULL,
    hb          DOUBLE NULL,
    ext         DOUBLE NULL,
    px          DOUBLE NULL,
    pz          DOUBLE NULL,
    rel_x       DOUBLE NULL,
    rel_z       DOUBLE NULL,
    sz_top      DOUBLE NULL,
    sz_bot      DOUBLE NULL,
    INDEX idx_pitch_pitcher (pitcher_id, season),
    INDEX idx_pitch_game (game_pk)
);

CREATE TABLE IF NOT EXISTS batting_players (
    season     SMALLINT NOT NULL,
    player_id  INT NOT NULL,
    name       VARCHAR(80),
    age        SMALLINT NULL,
    team       VARCHAR(8),
    league     VARCHAR(4),
    pos        VARCHAR(8),
    g          SMALLINT, pa SMALLINT, ab SMALLINT, h SMALLINT,
    doubles    SMALLINT, triples SMALLINT, hr SMALLINT, rbi SMALLINT,
    sb         SMALLINT, bb SMALLINT, so SMALLINT,
    ba DOUBLE NULL, obp DOUBLE NULL, slg DOUBLE NULL, ops DOUBLE NULL,
    ops_plus   DOUBLE NULL,
    war        DOUBLE NULL,
    PRIMARY KEY (season, player_id, team)
);

CREATE TABLE IF NOT EXISTS pitching_players (
    season     SMALLINT NOT NULL,
    player_id  INT NOT NULL,
    name       VARCHAR(80),
    age        SMALLINT NULL,
    team       VARCHAR(8),
    league     VARCHAR(4),
    w SMALLINT, l SMALLINT, g SMALLINT, gs SMALLINT, sv SMALLINT,
    ip DOUBLE NULL, h SMALLINT, r SMALLINT, er SMALLINT, hr SMALLINT,
    bb SMALLINT, so SMALLINT, bf SMALLINT,
    era DOUBLE NULL, whip DOUBLE NULL, fip DOUBLE NULL, era_plus DOUBLE NULL,
    war DOUBLE NULL,
    PRIMARY KEY (season, player_id, team)
);

CREATE TABLE IF NOT EXISTS statcast_players (
    season          SMALLINT NOT NULL,
    player_type     VARCHAR(8) NOT NULL,      -- 'batter' or 'pitcher'
    player_id       INT NOT NULL,
    name            VARCHAR(80),
    pa              SMALLINT,
    k_percent       DOUBLE NULL, bb_percent DOUBLE NULL,
    woba            DOUBLE NULL, xwoba DOUBLE NULL,
    sweet_spot_percent DOUBLE NULL, barrel_batted_rate DOUBLE NULL,
    hard_hit_percent   DOUBLE NULL,
    avg_best_speed     DOUBLE NULL, avg_hyper_speed DOUBLE NULL,
    whiff_percent      DOUBLE NULL, swing_percent DOUBLE NULL,
    PRIMARY KEY (season, player_type, player_id)
);