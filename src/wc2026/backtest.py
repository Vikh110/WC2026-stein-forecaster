"""
Backtest: compare Naive MLE vs JS estimator.

Three analyses — all fast (seconds, not minutes):

1. Analytical risk: From the paper's Appendix, for X ~ N(theta, sigma^2 * I_p):
     R(theta_hat_0,  theta) = p * sigma^2
     R(theta_hat_JS, theta) = p * sigma^2 - (p-2)^2 * sigma^4 * E[1/||X||^2]
   We evaluate this per-confederation using our fitted log-attack vector
   and observed match counts (sigma_i^2 = 1/n_i).

2. WC historical log-loss: Score naive vs JS on actual WC 2010-2022 matches.
   Two model fits total — fast.

3. Regime table: Show how risk reduction scales with sample size using
   a simple closed-form approximation (no re-fitting needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from wc2026.data import load_all
from wc2026.dixon_coles import DixonColesModel
from wc2026.js_shrinkage import JSEstimator


# ── 1. Analytical risk (paper Appendix formula) ───────────────────────────────

def analytical_risk(dc: DixonColesModel, team_df: pd.DataFrame,
                    n_mc: int = 50_000, seed: int = 0) -> pd.DataFrame:
    """
    Per-confederation risk comparison using the exact JS risk formula.

    For group of p teams with log-attack X ~ N(theta, sigma^2 * I_p):
      R_naive = p * sigma_bar^2
      R_JS    = R_naive - (p-2)^2 * sigma_bar^4 * E[1 / ||X - theta_0||^2]

    E[1/||X-theta_0||^2] is estimated via Monte Carlo sampling from
    N(X_obs, sigma^2 * I_p) — treating observed X as the true theta.
    """
    rng    = np.random.default_rng(seed)
    teams  = dc.teams_
    confs  = np.array([team_df["confederation"].get(t, "UNKNOWN") for t in teams])
    log_att = np.log(dc.attack_.values)
    n_match = dc.n_matches_.values.astype(float)
    sigma_sq = 1.0 / np.maximum(n_match, 1.0)

    rows = []
    for conf in np.unique(confs):
        idx   = np.where(confs == conf)[0]
        p     = len(idx)
        if p < 3:
            continue

        X     = log_att[idx]
        sig   = sigma_sq[idx]
        sig_bar = float(np.median(sig))
        theta_0 = X.mean() * np.ones(p)

        # Monte Carlo for E[1/||X - theta_0||^2]
        samples = X[None, :] + rng.standard_normal((n_mc, p)) * np.sqrt(sig_bar)
        d       = samples - theta_0[None, :]
        norms_sq = np.sum(d ** 2, axis=1)
        norms_sq = np.maximum(norms_sq, 1e-12)
        e_inv_norm = float(np.mean(1.0 / norms_sq))

        r_naive = p * sig_bar
        r_js    = r_naive - (p - 2) ** 2 * sig_bar ** 2 * e_inv_norm
        r_js    = max(r_js, 0.0)   # positive part

        rows.append({
            "confederation": conf,
            "n_teams":       p,
            "avg_n_matches": float(n_match[idx].mean()),
            "sigma_bar_sq":  round(sig_bar, 6),
            "risk_naive":    round(r_naive, 6),
            "risk_js":       round(r_js,    6),
            "reduction_%":   round(100 * (r_naive - r_js) / r_naive, 2),
        })

    return pd.DataFrame(rows).set_index("confederation")


# ── 2. WC historical log-loss ─────────────────────────────────────────────────

def match_log_loss(model, matches: pd.DataFrame) -> float:
    """Average negative log-likelihood per match under the Poisson model."""
    total, n = 0.0, 0
    for _, row in matches.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])
        if h not in model.attack_.index or a not in model.attack_.index:
            continue
        try:
            g_h, g_a = int(row["home_score"]), int(row["away_score"])
        except (ValueError, TypeError):
            continue
        lh, la = model.goal_lambdas(h, a, neutral=True)
        total += poisson.logpmf(g_h, lh) + poisson.logpmf(g_a, la)
        n     += 1
    return -total / max(n, 1)


def wc_log_loss_comparison(wc_history: pd.DataFrame,
                            team_df: pd.DataFrame,
                            match_df: pd.DataFrame) -> pd.DataFrame:
    dc = DixonColesModel().fit(match_df)
    js = JSEstimator(dc, team_df, positive_part=True)

    rows = []
    for year in [2010, 2014, 2018, 2022]:
        sub = wc_history[wc_history["year"] == year]
        if len(sub) < 5:
            continue
        ll_n = match_log_loss(dc, sub)
        ll_j = match_log_loss(js, sub)
        rows.append({
            "year":           year,
            "n_matches":      len(sub),
            "log_loss_naive": round(ll_n, 4),
            "log_loss_js":    round(ll_j, 4),
            "improvement_%":  round(100 * (ll_n - ll_j) / ll_n, 3),
        })
    return pd.DataFrame(rows)


# ── 3. Closed-form risk vs sample size ────────────────────────────────────────

def risk_vs_nmatches(dc: DixonColesModel, team_df: pd.DataFrame,
                     n_mc: int = 50_000, seed: int = 1) -> pd.DataFrame:
    """
    Show how JS risk reduction varies with sample size by analytically
    computing risk at hypothetical n_match values (no re-fitting needed).
    Uses the paper's formula with the observed X and varying sigma_bar.
    """
    rng = np.random.default_rng(seed)
    teams   = dc.teams_
    confs   = np.array([team_df["confederation"].get(t, "UNKNOWN") for t in teams])
    log_att = np.log(dc.attack_.values)

    # use all teams together (p = all qualified)
    idx = np.where(confs != "UNKNOWN")[0]
    X   = log_att[idx]
    p   = len(X)
    theta_0 = X.mean() * np.ones(p)

    rows = []
    for n_matches in [5, 10, 15, 20, 30, 50, 75, 100]:
        sig_bar = 1.0 / n_matches
        samples = X[None, :] + rng.standard_normal((n_mc, p)) * np.sqrt(sig_bar)
        d       = samples - theta_0[None, :]
        norms_sq = np.maximum(np.sum(d ** 2, axis=1), 1e-12)
        e_inv    = float(np.mean(1.0 / norms_sq))

        r_naive = p * sig_bar
        r_js    = max(0.0, r_naive - (p - 2) ** 2 * sig_bar ** 2 * e_inv)
        rows.append({
            "n_matches_per_team": n_matches,
            "risk_naive":         round(r_naive, 5),
            "risk_js":            round(r_js,    5),
            "reduction_%":        round(100 * (r_naive - r_js) / r_naive, 2),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")

    team_df, match_df, wc_history = load_all()
    dc = DixonColesModel().fit(match_df)
    js = JSEstimator(dc, team_df, positive_part=True)

    print("=" * 62)
    print("1. ANALYTICAL RISK (paper Appendix formula) PER CONFEDERATION")
    print("=" * 62)
    ar = analytical_risk(dc, team_df)
    print(ar.to_string())

    print("\n" + "=" * 62)
    print("2. WC HISTORICAL LOG-LOSS: NAIVE vs JS")
    print("=" * 62)
    ll = wc_log_loss_comparison(wc_history, team_df, match_df)
    print(ll.to_string(index=False))

    print("\n" + "=" * 62)
    print("3. RISK REDUCTION vs MATCHES PER TEAM (all 49 teams, p=49)")
    print("   (replicates Figure 1 shape from Samworth 2005)")
    print("=" * 62)
    rv = risk_vs_nmatches(dc, team_df)
    print(rv.to_string(index=False))
