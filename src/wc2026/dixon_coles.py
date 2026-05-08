"""
Naive MLE estimator for team attack / defence strengths.

Model
-----
  Goals_home ~ Poisson(alpha_h * beta_a * gamma)
  Goals_away ~ Poisson(alpha_a * beta_h)

where
  alpha_i = attack strength of team i   (> 0)
  beta_i  = defence weakness of team i  (> 0, larger = weaker defence)
  gamma   = home-advantage multiplier   (> 1 typically)

We optimise log-likelihood via scipy.optimize.minimize.
Parameters are stored in log-space to enforce positivity.

This is the "naive" theta_hat^0 from the paper — one estimate per team,
independent across teams, no shrinkage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ── Helpers ───────────────────────────────────────────────────────────────────

def _neg_log_likelihood(params: np.ndarray, home: np.ndarray, away: np.ndarray,
                        home_goals: np.ndarray, away_goals: np.ndarray,
                        n_teams: int) -> float:
    log_att  = params[:n_teams]
    log_def  = params[n_teams:2 * n_teams]
    log_home = params[-1]

    lam_h = np.exp(log_att[home] + log_def[away] + log_home)
    lam_a = np.exp(log_att[away] + log_def[home])

    nll = -(poisson.logpmf(home_goals, lam_h).sum() +
            poisson.logpmf(away_goals, lam_a).sum())
    return float(nll)


# ── Main estimator ────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Fit a Poisson goals model and expose attack / defence estimates per team.

    Attributes
    ----------
    teams_       : list of team names (sorted, index matches parameter vectors)
    attack_      : pd.Series  — MLE attack parameters (alpha)
    defence_     : pd.Series  — MLE defence parameters (beta, higher = weaker)
    home_adv_    : float      — home advantage multiplier gamma
    n_matches_   : pd.Series  — number of matches observed per team
    """

    def __init__(self) -> None:
        self.teams_: list[str] = []
        self.attack_:   pd.Series | None = None
        self.defence_:  pd.Series | None = None
        self.home_adv_: float = 1.0
        self.n_matches_: pd.Series | None = None
        self._fitted = False

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        """
        Parameters
        ----------
        matches : DataFrame with columns
                  home_team, away_team, home_score, away_score
        """
        df = matches.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams_ = all_teams
        n = len(all_teams)
        t2i = {t: i for i, t in enumerate(all_teams)}

        home  = df["home_team"].map(t2i).values
        away  = df["away_team"].map(t2i).values
        hg    = df["home_score"].values
        ag    = df["away_score"].values

        # match counts per team
        counts = pd.Series(0, index=all_teams, dtype=int)
        for t in df["home_team"]:
            counts[t] += 1
        for t in df["away_team"]:
            counts[t] += 1
        self.n_matches_ = counts

        # initialise: log_att=0, log_def=0, log_home=log(1.1)
        x0 = np.zeros(2 * n + 1)
        x0[-1] = np.log(1.1)

        res = minimize(
            _neg_log_likelihood,
            x0,
            args=(home, away, hg, ag, n),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        log_att  = res.x[:n]
        log_def  = res.x[n:2 * n]
        log_home = res.x[-1]

        # normalise so mean(log_att) = 0  (identifiability)
        log_att -= log_att.mean()
        log_def -= log_def.mean()

        self.attack_   = pd.Series(np.exp(log_att), index=all_teams)
        self.defence_  = pd.Series(np.exp(log_def), index=all_teams)
        self.home_adv_ = float(np.exp(log_home))
        self._fitted   = True
        return self

    # ── Prediction helpers ────────────────────────────────────────────────────

    def goal_lambdas(self, home: str, away: str, neutral: bool = True
                     ) -> tuple[float, float]:
        """Returns (lambda_home, lambda_away) for a given fixture."""
        ha = 1.0 if neutral else self.home_adv_
        lh = self.attack_[home] * self.defence_[away] * ha
        la = self.attack_[away] * self.defence_[home]
        return lh, la

    def win_draw_loss(self, home: str, away: str, neutral: bool = True,
                      max_goals: int = 10) -> tuple[float, float, float]:
        """Returns (P(home win), P(draw), P(away win)) via score matrix."""
        lh, la = self.goal_lambdas(home, away, neutral)
        gh = np.arange(max_goals + 1)
        p_h = poisson.pmf(gh, lh)
        p_a = poisson.pmf(gh, la)
        score_mat = np.outer(p_h, p_a)   # [home_goals x away_goals]
        p_home_win = np.tril(score_mat, -1).sum()
        p_draw     = np.trace(score_mat)
        p_away_win = np.triu(score_mat, +1).sum()
        return p_home_win, p_draw, p_away_win

    def summary(self) -> pd.DataFrame:
        assert self._fitted
        df = pd.DataFrame({
            "attack":   self.attack_,
            "defence":  self.defence_,
            "n_matches": self.n_matches_,
        })
        df["strength"] = df["attack"] / df["defence"]  # composite
        return df.sort_values("strength", ascending=False)


if __name__ == "__main__":
    from data import load_all
    team_df, match_df, _ = load_all()
    model = DixonColesModel().fit(match_df)
    print(model.summary().head(10))
    print(f"\nHome advantage: {model.home_adv_:.3f}")
