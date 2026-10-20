# Offense vs pitching, 2025 vs 2026
#
# Does the offense or the pitching explain the Mariners' drop from 2025 to 2026?
# Reads the `games` table of the mlb_project MySQL database (see load_mysql.py),
# compares the Mariners with the rest of the league month by month, and draws
# the result.
#
# Run it from the project folder (MLB-Project), for example in RStudio:
#   setwd("C:/Users/<you>/OneDrive/Desktop/MLB-Project"); source("r/offense_vs_pitching.R")
#
# Needs: install.packages(c("DBI", "RMariaDB", "dplyr", "ggplot2", "rstudioapi"))
# The MySQL password is asked for when you run it; it is never stored in this file.
# Optional environment variables: MLB_DB_HOST, MLB_DB_PORT, MLB_DB_USER,
# MLB_DB_PASSWORD, MLB_DB_NAME (default mlb_project).

suppressPackageStartupMessages({
  library(DBI)
  library(dplyr)
  library(ggplot2)
})

SEA_ID      <- 136
SEASON_A    <- 2025
SEASON_B    <- 2026
MIN_GAMES   <- 10          # months with fewer games are drawn faded (small sample)
OUT_DIR     <- file.path("r", "figures")

# colors: validated pair (blue / orange), plus text greys
COL_A   <- "#2a78d6"
COL_B   <- "#eb6834"
INK     <- "#0b0b0b"
INK_2   <- "#52514e"
GRID    <- "#e4e3df"
BG      <- "#fcfcfb"


# ---- database -------------------------------------------------------------
connect <- function() {
  pw <- Sys.getenv("MLB_DB_PASSWORD")
  if (pw == "") {
    pw <- if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
      rstudioapi::askForPassword("MySQL password")
    } else {
      readline("MySQL password: ")
    }
  }
  dbConnect(
    RMariaDB::MariaDB(),
    host     = Sys.getenv("MLB_DB_HOST", "localhost"),
    port     = as.integer(Sys.getenv("MLB_DB_PORT", "3306")),
    user     = Sys.getenv("MLB_DB_USER", "root"),
    password = pw,
    dbname   = Sys.getenv("MLB_DB_NAME", "mlb_project"),
    bigint   = "numeric"
  )
}

# one row per season / month / (Mariners or everyone else): games, runs scored, runs allowed
fetch_team_months <- function(con) {
  dbGetQuery(con, sprintf("
    WITH tg AS (
      SELECT season, game_date, home_id AS team_id, home_runs AS rs, away_runs AS ra FROM games WHERE played = 1
      UNION ALL
      SELECT season, game_date, away_id,            away_runs,       home_runs        FROM games WHERE played = 1
    )
    SELECT season,
           MONTH(game_date)        AS month,
           (team_id = %d)          AS is_sea,
           COUNT(*)                AS games,
           SUM(rs)                 AS rs,
           SUM(ra)                 AS ra
    FROM tg
    WHERE season IN (%d, %d)
    GROUP BY season, MONTH(game_date), (team_id = %d)
  ", SEA_ID, SEASON_A, SEASON_B, SEA_ID))
}


# ---- calculations ---------------------------------------------------------
# Mariners vs the rest of the league. Positive = better than the league average.
#   offense  = runs scored per game minus league runs scored per game
#   pitching = league runs allowed per game minus Mariners runs allowed per game
#              (runs allowed include defense)
compare_to_league <- function(raw) {
  sea <- raw %>% filter(as.integer(is_sea) == 1) %>%
    transmute(season, month, games, sea_rs = rs / games, sea_ra = ra / games)
  lg <- raw %>% filter(as.integer(is_sea) == 0) %>%
    transmute(season, month, lg_rs = rs / games, lg_ra = ra / games)
  inner_join(sea, lg, by = c("season", "month")) %>%
    mutate(offense  = sea_rs - lg_rs,
           pitching = lg_ra - sea_ra) %>%
    arrange(season, month)
}

# full-season values, each month weighted by its number of games
season_summary <- function(monthly) {
  monthly %>%
    group_by(season) %>%
    summarise(offense  = sum(offense  * games) / sum(games),
              pitching = sum(pitching * games) / sum(games),
              games    = sum(games),          # last, so the weights above use each month's games
              .groups = "drop")
}


# ---- chart ----------------------------------------------------------------
# the title follows the numbers: who moved more, and in which direction
headline <- function(a, b) {
  d_off <- b$offense - a$offense
  d_pit <- b$pitching - a$pitching
  share <- abs(d_off) / (abs(d_off) + abs(d_pit))
  who <- if (share >= 0.6) "The offense explains most of"
         else if (share <= 0.4) "Pitching and defense explain most of"
         else "Offense and pitching share"
  what <- if (d_off + d_pit < 0) "drop" else "gain"
  sprintf("%s the Mariners' %d-to-%d %s", who, SEASON_A, SEASON_B, what)
}

make_chart <- function(monthly, summ) {
  month_labs <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
  long <- monthly %>%
    transmute(season = factor(season), month, games,
              Offense = offense, Pitching = pitching) %>%
    tidyr_longer()
  long$measure <- factor(long$measure, levels = c("Offense", "Pitching"),
                         labels = c("Offense: runs scored vs league",
                                    "Pitching and defense: runs allowed vs league"))
  long$small <- long$games < MIN_GAMES

  a <- summ %>% filter(season == SEASON_A)
  b <- summ %>% filter(season == SEASON_B)
  caption <- sprintf(
    paste0("Full season vs league, runs per game:  offense %+.2f (%d) to %+.2f (%d);  ",
           "pitching and defense %+.2f (%d) to %+.2f (%d).\n",
           "Faded bars = fewer than %d games that month. Runs allowed include defense. ",
           "Source: MLB Stats API."),
    a$offense, SEASON_A, b$offense, SEASON_B,
    a$pitching, SEASON_A, b$pitching, SEASON_B, MIN_GAMES)

  ggplot(long, aes(x = factor(month), y = value, fill = season, alpha = small)) +
    geom_hline(yintercept = 0, colour = INK_2, linewidth = 0.4) +
    geom_col(position = position_dodge(width = 0.78), width = 0.7) +
    facet_wrap(~ measure, ncol = 1) +
    scale_x_discrete(labels = function(m) month_labs[as.integer(m)]) +
    scale_fill_manual(values = setNames(c(COL_A, COL_B), c(SEASON_A, SEASON_B)), name = NULL) +
    scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.4), guide = "none") +
    labs(
      title = headline(a, b),
      subtitle = "Runs per game compared with the rest of the league. Above zero = better than the league average.",
      x = NULL, y = "Runs per game vs league",
      caption = caption
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.background   = element_rect(fill = BG, colour = NA),
      panel.background  = element_rect(fill = BG, colour = NA),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
      text              = element_text(colour = INK),
      plot.title        = element_text(face = "bold", size = 15, colour = INK),
      plot.subtitle     = element_text(colour = INK_2),
      plot.caption      = element_text(colour = INK_2, size = 9, hjust = 0),
      axis.text         = element_text(colour = INK_2),
      axis.title        = element_text(colour = INK_2),
      strip.text        = element_text(face = "bold", hjust = 0, colour = INK),
      legend.position   = "top",
      legend.justification = "left"
    )
}

# small helper so the chart needs only dplyr + ggplot2 (no tidyr)
tidyr_longer <- function(df) {
  bind_rows(
    df %>% transmute(season, month, games, measure = "Offense",  value = Offense),
    df %>% transmute(season, month, games, measure = "Pitching", value = Pitching)
  )
}


# ---- run ------------------------------------------------------------------
main <- function() {
  con <- connect()
  on.exit(dbDisconnect(con), add = TRUE)
  monthly <- compare_to_league(fetch_team_months(con))
  summ    <- season_summary(monthly)

  cat("\nBy month (runs per game vs league; + = better than league)\n")
  print(as.data.frame(monthly %>% mutate(across(where(is.numeric), ~ round(.x, 2)))), row.names = FALSE)
  cat("\nFull season, weighted by games\n")
  print(as.data.frame(summ %>% mutate(across(where(is.numeric), ~ round(.x, 2)))), row.names = FALSE)

  chart <- make_chart(monthly, summ)
  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  out <- file.path(OUT_DIR, sprintf("offense_vs_pitching_%d_%d.png", SEASON_A, SEASON_B))
  ggsave(out, chart, width = 9, height = 7, dpi = 200, bg = BG)
  cat("\nSaved chart:", out, "\n")
  invisible(chart)
}

if (!isTRUE(getOption("mlb.no_main"))) main()