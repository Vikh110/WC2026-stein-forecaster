"""
James-Stein shrinkage for team strength parameters.

Setup
-----
The Dixon-Coles MLE gives log-attack estimates log(alpha_hat_i).
By asymptotic MLE theory, Var(log(alpha_hat_i)) ≈ sigma_i^2 = 1 / n_i,
where n_i = number of matches team i has played.

The variance-scaled JS estimator (Efron-Morris, 1977) for
X_i ~ N(theta_i, sigma_i^2) is:

    theta_hat^JS_i = theta0_i + sf * (X_i - theta0_i)

    sf = max(0, 1 - (p-2) * sigma_bar^2 / ||X - theta0||^2)

where sigma_bar^2 = median(sigma_i^2) across teams in the group.

Using the median (rather than mean) makes the formula robust to
teams with extreme sample sizes.

Shrinkage target theta0: confederation mean of the log-parameters.

Teams with few matches (large sigma_i^2) get pulled harder toward
the confederation mean — exactly the Stein intuition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wc2026.dixon_coles import DixonColesModel


# ── Core JS formula (variance-scaled) ────────────────────────────────────────

def james_stein_scaled(X: np.ndarray, theta_0: np.ndarray,
                       sigma_sq: np.ndarray,
                       positive_part: bool = True) -> tuple[np.ndarray, float]:
    """
    Variance-scaled James-Stein estimator.

    Parameters
    ----------
    X          : (p,) observed log-parameter vector
    theta_0    : (p,) shrinkage target (confederation mean)
    sigma_sq   : (p,) per-team variances (approx 1/n_matches)
    positive_part : clip shrinkage factor at 0 from below

    Returns
    -------
    theta_hat  : (p,) JS-improved estimates
    sf         : scalar shrinkage factor (for diagnostics)
    """
    p = len(X)
    assert p >= 3, f"JS estimator requires p >= 3, got p={p}"

    d       = X - theta_0
    norm_sq = float(np.dot(d, d))

    if norm_sq < 1e-12:
        return X.copy(), 1.0

    sigma_bar_sq = float(np.median(sigma_sq))
    sf = 1.0 - (p - 2) * sigma_bar_sq / norm_sq

    if positive_part:
        sf = max(0.0, sf)

    return theta_0 + sf * d, sf


# ── Per-team JS: heteroscedastic shrinkage ────────────────────────────────────

def james_stein_hetero(X: np.ndarray, theta_0: np.ndarray,
                       sigma_sq: np.ndarray,
                       positive_part: bool = True) -> np.ndarray:
    """
    Heteroscedastic JS: each team gets its own shrinkage amount
    proportional to its individual variance sigma_i^2.

    This is the Efron-Morris (1977) empirical Bayes form:
      theta_hat_i = theta0_i + (1 - B * sigma_i^2) * (X_i - theta0_i)
      B = (p-2) / ||X - theta0||_sigma^{-2}^2   (sigma-weighted norm)

    Teams with MORE data (small sigma_i^2) are shrunk less.
    Teams with LESS data (large sigma_i^2) are shrunk more.
    """
    p = len(X)
    assert p >= 3

    d          = X - theta_0
    inv_sig_sq = 1.0 / (sigma_sq + 1e-12)

    # weighted sum-of-squares
    weighted_norm_sq = float(np.dot(d * inv_sig_sq, d))

    if weighted_norm_sq < 1e-12:
        return X.copy()

    B = (p - 2) / weighted_norm_sq

    sf_per_team = np.ones(p)
    for i in range(p):
        sf_i = 1.0 - B * sigma_sq[i]
        if positive_part:
            sf_i = max(0.0, sf_i)
        sf_per_team[i] = sf_i

    return theta_0 + sf_per_team * d


# ── Main estimator class ──────────────────────────────────────────────────────

class JSEstimator:
    """
    Wraps a fitted DixonColesModel and applies heteroscedastic JS shrinkage,
    returning improved attack and defence estimates.

    Shrinkage target : confederation mean of the log-parameters.
    Variance model   : sigma_i^2 = 1 / n_matches_i  (Poisson MLE asymptotics).
    """

    def __init__(self, dc_model: DixonColesModel, team_df: pd.DataFrame,
                 positive_part: bool = True) -> None:
        self.dc            = dc_model
        self.team_df       = team_df
        self.positive_part = positive_part
        self._fit()

    def _conf_for(self, team: str) -> str:
        if team in self.team_df.index:
            return self.team_df.loc[team, "confederation"]
        return "UNKNOWN"

    def _fit(self) -> None:
        teams    = self.dc.teams_
        n        = len(teams)

        log_att  = np.log(self.dc.attack_[teams].values)
        log_def  = np.log(self.dc.defence_[teams].values)
        n_match  = self.dc.n_matches_[teams].values.astype(float)
        sigma_sq = 1.0 / np.maximum(n_match, 1.0)   # Var(log MLE) ≈ 1/n

        confs = np.array([self._conf_for(t) for t in teams])

        # initialise JS estimates as copies of MLE
        js_log_att = log_att.copy()
        js_log_def = log_def.copy()

        self.shrinkage_info_: list[dict] = []

        for conf in np.unique(confs):
            idx = np.where(confs == conf)[0]
            p   = len(idx)

            if p < 3:
                continue

            X_att   = log_att[idx]
            X_def   = log_def[idx]
            sig_att = sigma_sq[idx]
            sig_def = sigma_sq[idx]

            t0_att  = np.full(p, X_att.mean())
            t0_def  = np.full(p, X_def.mean())

            js_att, sf_att = james_stein_scaled(X_att, t0_att, sig_att, self.positive_part)
            js_def, sf_def = james_stein_scaled(X_def, t0_def, sig_def, self.positive_part)

            # also compute per-team heteroscedastic version
            js_att_h = james_stein_hetero(X_att, t0_att, sig_att, self.positive_part)
            js_def_h = james_stein_hetero(X_def, t0_def, sig_def, self.positive_part)

            # store the heteroscedastic (per-team) version as primary
            js_log_att[idx] = js_att_h
            js_log_def[idx] = js_def_h

            # diagnostics
            d_att = X_att - t0_att
            norm_att = float(np.dot(d_att, d_att))
            med_sig  = float(np.median(sig_att))

            self.shrinkage_info_.append({
                "confederation":    conf,
                "n_teams":          p,
                "avg_n_matches":    float(n_match[idx].mean()),
                "median_sigma_sq":  med_sig,
                "norm_sq_att":      norm_att,
                "sf_att_scaled":    sf_att,
                "sf_def_scaled":    sf_def,
            })

        self.attack_  = pd.Series(np.exp(js_log_att), index=teams)
        self.defence_ = pd.Series(np.exp(js_log_def), index=teams)
        self.home_adv_ = self.dc.home_adv_

        self.attack_naive_  = self.dc.attack_.copy()
        self.defence_naive_ = self.dc.defence_.copy()

    # ── Prediction helpers ────────────────────────────────────────────────────

    def goal_lambdas(self, home: str, away: str, neutral: bool = True
                     ) -> tuple[float, float]:
        ha = 1.0 if neutral else self.home_adv_
        lh = self.attack_[home] * self.defence_[away] * ha
        la = self.attack_[away] * self.defence_[home]
        return lh, la

    def win_draw_loss(self, home: str, away: str, neutral: bool = True,
                      max_goals: int = 10) -> tuple[float, float, float]:
        from scipy.stats import poisson
        lh, la = self.goal_lambdas(home, away, neutral)
        gh = np.arange(max_goals + 1)
        score_mat = np.outer(poisson.pmf(gh, lh), poisson.pmf(gh, la))
        return (float(np.tril(score_mat, -1).sum()),
                float(np.trace(score_mat)),
                float(np.triu(score_mat, +1).sum()))

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def shrinkage_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.shrinkage_info_).set_index("confederation")

    def comparison_table(self) -> pd.DataFrame:
        teams = self.dc.teams_
        rows  = []
        for t in teams:
            conf = self._conf_for(t)
            n    = self.dc.n_matches_[t]
            rows.append({
                "team":            t,
                "confederation":   conf,
                "n_matches":       int(n),
                "att_naive":       float(self.attack_naive_[t]),
                "att_js":          float(self.attack_[t]),
                "def_naive":       float(self.defence_naive_[t]),
                "def_js":          float(self.defence_[t]),
                "att_pct_change":  100*(self.attack_[t] - self.attack_naive_[t]) / self.attack_naive_[t],
            })
        df = pd.DataFrame(rows).set_index("team")
        df["strength_naive"] = df["att_naive"] / df["def_naive"]
        df["strength_js"]    = df["att_js"]    / df["def_js"]
        return df.sort_values("strength_js", ascending=False)


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data import load_all

    team_df, match_df, _ = load_all()
    dc = DixonColesModel().fit(match_df)
    js = JSEstimator(dc, team_df)

    print("=" * 60)
    print("SHRINKAGE DIAGNOSTICS PER CONFEDERATION")
    print("=" * 60)
    st = js.shrinkage_table()
    print(st[["n_teams","avg_n_matches","median_sigma_sq","norm_sq_att","sf_att_scaled"]].to_string())

    print("\n" + "=" * 60)
    print("TOP 15 TEAMS: NAIVE vs JS STRENGTH")
    print("=" * 60)
    cmp = js.comparison_table()
    print(cmp[["n_matches","strength_naive","strength_js","att_pct_change"]].head(15).to_string())

    print("\n" + "=" * 60)
    print("SMALLEST-SAMPLE TEAMS — JS shifts most here:")
    print("=" * 60)
    print(cmp.sort_values("n_matches")[["n_matches","strength_naive","strength_js","att_pct_change"]].head(12).to_string())
