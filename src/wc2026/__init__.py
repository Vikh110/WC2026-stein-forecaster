"""
wc2026 — FIFA World Cup 2026 Stein-Shrinkage Forecaster
========================================================

Apply James-Stein shrinkage estimation (Stein 1956, Efron-Morris 1977)
to simultaneously estimate strength parameters for all 48 WC2026 teams,
yielding provably lower total risk than the naive MLE.

Modules
-------
data         : team catalogue and synthetic match generator
dixon_coles  : naive MLE Poisson goals model
js_shrinkage : heteroscedastic James-Stein estimator
simulator    : Monte Carlo WC2026 tournament
backtest     : analytical risk evaluation
pipeline     : full orchestrator
"""

__version__ = "1.0.0"
__author__  = "Vishwas Khandelwal"

from wc2026.data import load_all, build_team_table, generate_matches
from wc2026.dixon_coles import DixonColesModel
from wc2026.js_shrinkage import JSEstimator
from wc2026.simulator import make_draw, monte_carlo
from wc2026.backtest import analytical_risk, wc_log_loss_comparison, risk_vs_nmatches

__all__ = [
    "load_all",
    "build_team_table",
    "generate_matches",
    "DixonColesModel",
    "JSEstimator",
    "make_draw",
    "monte_carlo",
    "analytical_risk",
    "wc_log_loss_comparison",
    "risk_vs_nmatches",
]
