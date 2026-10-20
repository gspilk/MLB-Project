# How do hitters change from one season to the next, by age?
#
# For the positions the Mariners need (1B, 3B, SS), compared with every other
# hitter. Uses the batting_seasons view of the mlb_project MySQL database
# (build it with build_history.py + load_mysql.py).
#
# Method ("delta method"): take every hitter who had 300+ plate appearances in
# two consecutive seasons, measure how much his production changed, and average
# the changes by the age he was in the first season. Each player is weighted by
# how many plate appearances he had in both seasons, so small samples count less.
#
# Run it from the project folder, for example in RStudio:
#   setwd("C:/Users/<you>/OneDrive/Desktop/MLB-Project"); source("r/aging_curves.R")
#
# Needs: install.packages(c("DBI", "RMariaDB", "dplyr", "ggplot2", "rstudioapi"))
# The MySQL password is asked for when you run it; it is never stored in this file.

suppressPackageStartupMessages({
  library(DBI)
  library(dplyr)
  library(ggplot2)
})

MIN_PA     <- 300                       # plate appearances needed in BOTH seasons
MIN_PAIRS  <- 20                        # age bands with fewer pairs are drawn faded
IF_SPOTS   <- c("1B", "3B", "SS")       # the positions of need
BAND_CUTS  <- c(-Inf, 26, 29, 32, 35, Inf)
BAND_NAMES <- c("26 and under", "27-29", "30-32", "33-35", "36+")
GROUPS     <- c("1B / 3B / SS", "All other hitters")
OUT_FIG    <- file.path("r", "figures", "aging_curves_infield.png")
OUT_CSV    <- file.path("data", "exports", "aging_by_age_band.csv")

COL_IF  <- "#6250d6"      # violet: the positions of need
COL_OTH <- "#008300"      # green:  everyone else
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

fetch_seasons <- function(con) {
  dbGetQuery(con, sprintf("
    SELECT season, player_id, name, age, pos, pa, ops, war
    FROM batting_seasons
    WHERE pa >= %d AND age IS NOT NULL AND ops IS NOT NULL AND war IS NOT NULL
  ", MIN_PA))
}


# ---- calculations ---------------------------------------------------------
# one row per player and pair of consecutive seasons: age in the first season
# and how OPS and WAR (per 600 PA) changed into the next
build_pairs <- function(d) {
  nxt <- d %>% transmute(player_id, season = season - 1, pa2 = pa, ops2 = ops, war2 = war)
  d %>%
    inner_join(nxt, by = c("player_id", "season")) %>%
    mutate(
      group   = factor(ifelse(pos %in% IF_SPOTS, GROUPS[1], GROUPS[2]), levels = GROUPS),
      band    = cut(age, breaks = BAND_CUTS, labels = BAND_NAMES),
      d_ops   = ops2 - ops,
      d_war   = war2 / pa2 * 600 - war / pa * 600,
      weight  = 2 / (1 / pa + 1 / pa2)      # harmonic mean of the two samples
    )
}

# average change by position group and age band, with a standard error
summarise_bands <- function(pairs) {
  pairs %>%
    group_by(group, band) %>%
    summarise(
      n        = n(),
      # standard errors first: later lines replace d_ops / d_war with their averages
      se_ops   = sd(d_ops) / sqrt(n()),
      se_war   = sd(d_war) / sqrt(n()),
      d_ops    = weighted.mean(d_ops, weight),
      d_war    = weighted.mean(d_war, weight),
      .groups  = "drop"
    )
}


# ---- chart ----------------------------------------------------------------
make_chart <- function(bands, first_season, last_season) {
  long <- bind_rows(
    bands %>% transmute(group, band, n, metric = "Change in OPS per season",
                        value = d_ops, se = se_ops),
    bands %>% transmute(group, band, n, metric = "Change in WAR per 600 plate appearances",
                        value = d_war, se = se_war)
  ) %>%
    mutate(metric = factor(metric, levels = c("Change in OPS per season",
                                               "Change in WAR per 600 plate appearances")),
           small  = n < MIN_PAIRS,
           lab_y  = ifelse(value >= 0, value + se, value - se),
           lab_v  = ifelse(value >= 0, -0.5, 1.5))

  ggplot(long, aes(x = band, y = value, fill = group, alpha = small)) +
    geom_hline(yintercept = 0, colour = INK_2, linewidth = 0.4) +
    geom_col(position = position_dodge(width = 0.78), width = 0.7) +
    geom_errorbar(aes(ymin = value - se, ymax = value + se),
                  position = position_dodge(width = 0.78), width = 0.18,
                  colour = INK_2, linewidth = 0.4) +
    geom_text(aes(y = lab_y, label = paste0("n=", n), vjust = lab_v),
              position = position_dodge(width = 0.78), size = 3, colour = INK_2) +
    facet_wrap(~ metric, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = setNames(c(COL_IF, COL_OTH), GROUPS), name = NULL) +
    scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.45), guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0.12, 0.12))) +
    labs(
      title    = "How hitters change from one season to the next, by age",
      subtitle = "Average change into the next season, by the age at the start. Below zero = decline.",
      x        = "Age at the start of the season pair",
      y        = NULL,
      caption  = sprintf(paste0(
        "Hitters with %d+ plate appearances in both seasons, %d to %d. n = player-season pairs; ",
        "error bars = one standard error;\n",
        "faded bars = fewer than %d pairs. Players who decline sharply lose playing time and drop out,\n",
        "so older bands look better than the full population. WAR is this project's own calculation. ",
        "Source: MLB Stats API."),
        MIN_PA, first_season, last_season, MIN_PAIRS)
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.background    = element_rect(fill = BG, colour = NA),
      panel.background   = element_rect(fill = BG, colour = NA),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
      text               = element_text(colour = INK),
      plot.title         = element_text(face = "bold", size = 15, colour = INK),
      plot.subtitle      = element_text(colour = INK_2),
      plot.caption       = element_text(colour = INK_2, size = 9, hjust = 0),
      axis.text          = element_text(colour = INK_2),
      axis.title         = element_text(colour = INK_2),
      strip.text         = element_text(face = "bold", hjust = 0, colour = INK),
      legend.position    = "top",
      legend.justification = "left"
    )
}


# ---- run ------------------------------------------------------------------
main <- function() {
  con <- connect()
  on.exit(dbDisconnect(con), add = TRUE)
  seasons <- fetch_seasons(con)
  pairs   <- build_pairs(seasons)
  bands   <- summarise_bands(pairs)

  cat(sprintf("\n%d player-season pairs (%d to %d), %d+ PA in both seasons\n",
              nrow(pairs), min(seasons$season), max(seasons$season), MIN_PA))
  cat("\nAverage change into the next season, by age at the start\n")
  print(as.data.frame(bands %>% mutate(across(where(is.numeric), ~ round(.x, 3)))), row.names = FALSE)

  dir.create(dirname(OUT_CSV), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(OUT_FIG), recursive = TRUE, showWarnings = FALSE)
  write.csv(bands, OUT_CSV, row.names = FALSE)
  ggsave(OUT_FIG, make_chart(bands, min(seasons$season), max(seasons$season)),
         width = 9, height = 7, dpi = 200, bg = BG)
  cat("\nSaved chart:", OUT_FIG, "\nSaved table:", OUT_CSV, "\n")
  invisible(bands)
}

if (!isTRUE(getOption("mlb.no_main"))) main()