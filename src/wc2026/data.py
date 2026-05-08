"""
Data pipeline for the WC2026 Stein-Shrinkage Forecaster.

Real match data is fetched from the martj42/international_results GitHub
dataset (~48,000 international results since 1872). We filter to matches
from 2018 onwards involving WC2026-qualified teams and apply exponential
time-decay so recent form matters more than old results.

generate_matches() is kept for reproducible unit testing only.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests

# ── 1. WC2026 qualified teams ─────────────────────────────────────────────────
# FIFA ranking points (approximate, April 2026)
TEAMS = [
    # UEFA (16)
    ("France",        "UEFA",     1854),
    ("Spain",         "UEFA",     1843),
    ("England",       "UEFA",     1818),
    ("Portugal",      "UEFA",     1792),
    ("Netherlands",   "UEFA",     1761),
    ("Belgium",       "UEFA",     1742),
    ("Germany",       "UEFA",     1726),
    ("Switzerland",   "UEFA",     1698),
    ("Croatia",       "UEFA",     1677),
    ("Austria",       "UEFA",     1651),
    ("Denmark",       "UEFA",     1638),
    ("Turkey",        "UEFA",     1614),
    ("Scotland",      "UEFA",     1598),
    ("Serbia",        "UEFA",     1585),
    ("Hungary",       "UEFA",     1563),
    ("Poland",        "UEFA",     1551),
    # CONMEBOL (6)
    ("Argentina",     "CONMEBOL", 1920),
    ("Brazil",        "CONMEBOL", 1838),
    ("Colombia",      "CONMEBOL", 1758),
    ("Uruguay",       "CONMEBOL", 1711),
    ("Ecuador",       "CONMEBOL", 1639),
    ("Venezuela",     "CONMEBOL", 1558),
    # CONCACAF (6)
    ("USA",           "CONCACAF", 1680),
    ("Mexico",        "CONCACAF", 1654),
    ("Canada",        "CONCACAF", 1631),
    ("Panama",        "CONCACAF", 1582),
    ("Costa Rica",    "CONCACAF", 1547),
    ("Honduras",      "CONCACAF", 1512),
    # CAF (9)
    ("Morocco",       "CAF",      1743),
    ("Senegal",       "CAF",      1693),
    ("Nigeria",       "CAF",      1647),
    ("Ivory Coast",   "CAF",      1628),
    ("Egypt",         "CAF",      1603),
    ("Ghana",         "CAF",      1578),
    ("DR Congo",      "CAF",      1551),
    ("Cameroon",      "CAF",      1527),
    ("South Africa",  "CAF",      1498),
    # AFC (8)
    ("Japan",         "AFC",      1738),
    ("South Korea",   "AFC",      1692),
    ("Australia",     "AFC",      1644),
    ("Iran",          "AFC",      1617),
    ("Saudi Arabia",  "AFC",      1589),
    ("Iraq",          "AFC",      1551),
    ("Uzbekistan",    "AFC",      1524),
    ("Jordan",        "AFC",      1493),
    # OFC (1)
    ("New Zealand",   "OFC",      1433),
    # Playoff representatives
    ("Algeria",       "CAF",      1631),
    ("Bolivia",       "CONMEBOL", 1487),
    ("Jamaica",       "CONCACAF", 1489),
]

# ── 2. Name mapping: martj42 dataset → our standardised names ─────────────────
# The martj42 dataset uses some different country name conventions.
TEAM_NAME_MAP: dict[str, str] = {
    "United States":                  "USA",
    "Korea Republic":                 "South Korea",
    "Republic of Korea":              "South Korea",
    "Côte d'Ivoire":                  "Ivory Coast",
    "Cote d'Ivoire":                  "Ivory Coast",
    "Congo DR":                       "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "IR Iran":                        "Iran",
    "Kyrgyz Republic":                "Kyrgyzstan",
}

# Inverse: our name → all known aliases in the dataset (used for filtering)
_ALL_NAMES: set[str] = {name for name, _, _ in TEAMS} | set(TEAM_NAME_MAP.keys())


# ── 3. Build team metadata table ──────────────────────────────────────────────

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


# ── 4. Real international match data ─────────────────────────────────────────

_REAL_DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)


def load_real_matches(
    start_year: int = 2018,
    end_year: int | None = None,
    decay_rate: float = 0.003,
    wc_teams_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch real international match results and prepare them for Dixon-Coles.

    Parameters
    ----------
    start_year    : Only include matches from this year onwards (2018 = post-Russia WC).
    decay_rate    : Exponential decay per day (0.003 → half-life ≈ 231 days).
                    Matches 3 years ago get ~0.11× the weight of today's matches.
    wc_teams_only : If True, only keep matches where BOTH teams are WC2026 qualified.
                    Default False keeps all matches for WC2026 teams (more data).

    Returns
    -------
    DataFrame with columns: home_team, away_team, home_score, away_score,
                             weight, neutral, tournament, date
    """
    r = requests.get(_REAL_DATA_URL, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df["date"] = pd.to_datetime(df["date"])
    year_col = df["date"].dt.year
    mask_year = year_col >= start_year
    if end_year is not None:
        mask_year = mask_year & (year_col < end_year)
    df = df[mask_year].copy()

    # Standardise team names
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    # Keep matches involving at least one WC2026 team
    wc_teams = {name for name, _, _ in TEAMS}
    if wc_teams_only:
        mask = df["home_team"].isin(wc_teams) & df["away_team"].isin(wc_teams)
    else:
        mask = df["home_team"].isin(wc_teams) | df["away_team"].isin(wc_teams)
    df = df[mask].copy()

    # Exponential time-decay weights (more recent = higher weight)
    ref = pd.Timestamp.today().normalize()
    df["days_ago"] = df["date"].apply(lambda d: max(0, (ref - d).days))
    df["weight"]   = np.exp(-decay_rate * df["days_ago"].to_numpy())

    cols = ["home_team", "away_team", "home_score", "away_score",
            "weight", "neutral", "tournament", "date"]
    return df[cols].reset_index(drop=True)


# ── 5. Historical WC results (for backtest) ───────────────────────────────────

def load_wc_history() -> pd.DataFrame:
    url = ("https://raw.githubusercontent.com/rfordatascience/tidytuesday"
           "/master/data/2022/2022-11-29/wcmatches.csv")
    r = requests.get(url, timeout=15)
    df = pd.read_csv(io.StringIO(r.text))
    return df[["year", "home_team", "away_team",
               "home_score", "away_score"]].dropna()


# ── 6. Synthetic matches (for unit tests only) ────────────────────────────────

HOME_ADV = 1.20
CONF_MATCH_MU = {
    "UEFA": 30, "CONMEBOL": 25, "CONCACAF": 20,
    "CAF": 16,  "AFC": 15,      "OFC": 9,
}


def generate_matches(team_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible synthetic matches for unit testing."""
    rng   = np.random.default_rng(seed)
    teams = team_df.index.tolist()
    rows  = []

    for team in teams:
        conf      = team_df.loc[team, "confederation"]
        n_games   = int(rng.poisson(CONF_MATCH_MU[conf]))
        opponents = rng.choice([t for t in teams if t != team],
                               size=n_games, replace=True)
        for opp in opponents:
            venue = rng.choice(["home", "away", "neutral"], p=[0.35, 0.35, 0.30])
            if venue == "home":
                lam_h = team_df.loc[team, "true_att"] * team_df.loc[opp,  "true_def"] * HOME_ADV
                lam_a = team_df.loc[opp,  "true_att"] * team_df.loc[team, "true_def"]
                home, away = team, opp
            elif venue == "away":
                lam_h = team_df.loc[opp,  "true_att"] * team_df.loc[team, "true_def"] * HOME_ADV
                lam_a = team_df.loc[team, "true_att"] * team_df.loc[opp,  "true_def"]
                home, away = opp, team
            else:
                lam_h = team_df.loc[team, "true_att"] * team_df.loc[opp,  "true_def"]
                lam_a = team_df.loc[opp,  "true_att"] * team_df.loc[team, "true_def"]
                home, away = team, opp

            rows.append({
                "home_team":  home,
                "away_team":  away,
                "home_score": int(rng.poisson(lam_h)),
                "away_score": int(rng.poisson(lam_a)),
            })

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


# ── 7. Public interface ───────────────────────────────────────────────────────

def load_all(seed: int = 42, use_real_data: bool = True):
    """
    Returns (team_df, match_df, wc_history_df).

    When use_real_data=True (default), match_df contains real international
    results from 2018 onwards with time-decay weights.
    When use_real_data=False, synthetic data is generated for reproducible testing.
    """
    team_df = build_team_table()
    if use_real_data:
        match_df = load_real_matches()
    else:
        match_df = generate_matches(team_df, seed=seed)
    wc_hist = load_wc_history()
    return team_df, match_df, wc_hist


if __name__ == "__main__":
    tdf, mdf, wch = load_all()
    print(f"Teams:              {len(tdf)}")
    print(f"Real matches:       {len(mdf)}")
    print(f"WC historical:      {len(wch)}")
    print(f"Date range:         {mdf['date'].min().date()} → {mdf['date'].max().date()}")
    print(f"Weight range:       {mdf['weight'].min():.3f} → {mdf['weight'].max():.3f}")
    print(mdf.head())
