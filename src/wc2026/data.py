"""
Data pipeline: historical WC results + synthetic recent international matches.

Recent matches are generated from a Poisson process parameterised by real
FIFA ranking points (April 2026), so the data is statistically consistent
with known team strengths while being fully reproducible.
"""

import io
import numpy as np
import pandas as pd
import requests

# ── 1. WC2026 qualified teams ─────────────────────────────────────────────────
# FIFA ranking points (approximate, April 2026)
# Confederation codes: UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC
TEAMS = [
    # UEFA (16)
    ("France",       "UEFA",     1854),
    ("Spain",        "UEFA",     1843),
    ("England",      "UEFA",     1818),
    ("Portugal",     "UEFA",     1792),
    ("Netherlands",  "UEFA",     1761),
    ("Belgium",      "UEFA",     1742),
    ("Germany",      "UEFA",     1726),
    ("Switzerland",  "UEFA",     1698),
    ("Croatia",      "UEFA",     1677),
    ("Austria",      "UEFA",     1651),
    ("Denmark",      "UEFA",     1638),
    ("Turkey",       "UEFA",     1614),
    ("Scotland",     "UEFA",     1598),
    ("Serbia",       "UEFA",     1585),
    ("Hungary",      "UEFA",     1563),
    ("Poland",       "UEFA",     1551),
    # CONMEBOL (6)
    ("Argentina",    "CONMEBOL", 1920),
    ("Brazil",       "CONMEBOL", 1838),
    ("Colombia",     "CONMEBOL", 1758),
    ("Uruguay",      "CONMEBOL", 1711),
    ("Ecuador",      "CONMEBOL", 1639),
    ("Venezuela",    "CONMEBOL", 1558),
    # CONCACAF (6)
    ("USA",          "CONCACAF", 1680),
    ("Mexico",       "CONCACAF", 1654),
    ("Canada",       "CONCACAF", 1631),
    ("Panama",       "CONCACAF", 1582),
    ("Costa Rica",   "CONCACAF", 1547),
    ("Honduras",     "CONCACAF", 1512),
    # CAF (9)
    ("Morocco",      "CAF",      1743),
    ("Senegal",      "CAF",      1693),
    ("Nigeria",      "CAF",      1647),
    ("Ivory Coast",  "CAF",      1628),
    ("Egypt",        "CAF",      1603),
    ("Ghana",        "CAF",      1578),
    ("DR Congo",     "CAF",      1551),
    ("Cameroon",     "CAF",      1527),
    ("South Africa", "CAF",      1498),
    # AFC (8)
    ("Japan",        "AFC",      1738),
    ("South Korea",  "AFC",      1692),
    ("Australia",    "AFC",      1644),
    ("Iran",         "AFC",      1617),
    ("Saudi Arabia", "AFC",      1589),
    ("Iraq",         "AFC",      1551),
    ("Uzbekistan",   "AFC",      1524),
    ("Jordan",       "AFC",      1493),
    # OFC (1)
    ("New Zealand",  "OFC",      1433),
    # Playoff representatives
    ("Algeria",      "CAF",      1631),   # extra CAF spot via playoff
    ("Bolivia",      "CONMEBOL", 1487),   # extra CONMEBOL (6th)
    ("Jamaica",      "CONCACAF", 1489),   # extra CONCACAF
]

# ── 2. Convert ranking points to Poisson attack / defence parameters ───────────
# Attack = goals scored per game, Defence = goals conceded per game.
# We anchor top team (Argentina, 1920 pts) at attack=1.90, defence=0.65
# and bottom team (~1433 pts) at attack=1.00, defence=1.55
# Linear interpolation in ranking-points space.

def _pts_to_lambda(pts: float, lo_pts=1433, hi_pts=1920,
                   lo_att=1.00, hi_att=1.90,
                   lo_def=1.55, hi_def=0.65) -> tuple[float, float]:
    frac = (pts - lo_pts) / (hi_pts - lo_pts)
    att  = lo_att + frac * (hi_att - lo_att)
    def_ = lo_def + frac * (hi_def - lo_def)
    return round(att, 4), round(def_, 4)


def build_team_table() -> pd.DataFrame:
    rows = []
    for name, conf, pts in TEAMS:
        att, def_ = _pts_to_lambda(pts)
        rows.append({"team": name, "confederation": conf,
                     "fifa_pts": pts, "true_att": att, "true_def": def_})
    return pd.DataFrame(rows).set_index("team")


# ── 3. Generate synthetic recent international results ────────────────────────
# For each team we draw n_matches ~ Poisson(μ) where μ is larger for
# established footballing nations (UEFA/CONMEBOL) and smaller for minnows.
# Goals are drawn from independent Poisson(att_home * def_away * home_adv).

HOME_ADV = 1.20   # home advantage multiplier

CONF_MATCH_MU = {   # avg qualifying-era matches per team (last 2 years ~20-35)
    "UEFA":     30,
    "CONMEBOL": 25,
    "CONCACAF": 20,
    "CAF":      16,
    "AFC":      15,
    "OFC":       9,
}


def generate_matches(team_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    teams = team_df.index.tolist()
    rows  = []

    for team in teams:
        conf     = team_df.loc[team, "confederation"]
        n_games  = int(rng.poisson(CONF_MATCH_MU[conf]))
        opponents = rng.choice([t for t in teams if t != team],
                               size=n_games, replace=True)

        for opp in opponents:
            # randomly assign home / away / neutral
            venue = rng.choice(["home", "away", "neutral"], p=[0.35, 0.35, 0.30])
            if venue == "home":
                lam_h = team_df.loc[team, "true_att"] * team_df.loc[opp,  "true_def"] * HOME_ADV
                lam_a = team_df.loc[opp,  "true_att"] * team_df.loc[team, "true_def"]
                home, away = team, opp
            elif venue == "away":
                lam_h = team_df.loc[opp,  "true_att"] * team_df.loc[team, "true_def"] * HOME_ADV
                lam_a = team_df.loc[team, "true_att"] * team_df.loc[opp,  "true_def"]
                home, away = opp, team
            else:  # neutral
                lam_h = team_df.loc[team, "true_att"] * team_df.loc[opp,  "true_def"]
                lam_a = team_df.loc[opp,  "true_att"] * team_df.loc[team, "true_def"]
                home, away = team, opp

            goals_h = int(rng.poisson(lam_h))
            goals_a = int(rng.poisson(lam_a))
            rows.append({"home_team": home, "away_team": away,
                         "home_score": goals_h, "away_score": goals_a})

    df = pd.DataFrame(rows).drop_duplicates()
    return df.reset_index(drop=True)


# ── 4. Historical WC results (1930-2022) ──────────────────────────────────────

def load_wc_history() -> pd.DataFrame:
    url = ("https://raw.githubusercontent.com/rfordatascience/tidytuesday"
           "/master/data/2022/2022-11-29/wcmatches.csv")
    r = requests.get(url, timeout=15)
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"home_team": "home_team", "away_team": "away_team",
                             "home_score": "home_score", "away_score": "away_score"})
    return df[["year", "home_team", "away_team", "home_score", "away_score"]].dropna()


# ── 5. Public interface ───────────────────────────────────────────────────────

def load_all(seed: int = 42):
    """Returns (team_df, match_df, wc_history_df)."""
    team_df   = build_team_table()
    match_df  = generate_matches(team_df, seed=seed)
    wc_hist   = load_wc_history()
    return team_df, match_df, wc_hist


if __name__ == "__main__":
    tdf, mdf, wch = load_all()
    print(f"Teams: {len(tdf)}")
    print(f"Synthetic matches: {len(mdf)}")
    print(f"WC historical matches: {len(wch)}")
    print(tdf.head())
    print(mdf.head())
