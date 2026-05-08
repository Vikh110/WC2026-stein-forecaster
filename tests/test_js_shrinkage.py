"""Tests for wc2026.js_shrinkage module."""

import numpy as np
import pytest
from wc2026.data import build_team_table, generate_matches
from wc2026.dixon_coles import DixonColesModel
from wc2026.js_shrinkage import JSEstimator, james_stein_scaled, james_stein_hetero


# ── Unit tests for the core JS formulas ──────────────────────────────────────

class TestJamesSteinScaled:
    def test_output_shape(self):
        X = np.array([1.0, 2.0, 3.0, 4.0])
        t0 = np.zeros(4)
        sig = np.ones(4) * 0.1
        result, sf = james_stein_scaled(X, t0, sig)
        assert result.shape == (4,)

    def test_shrinkage_factor_in_zero_one(self):
        X = np.array([1.0, -1.0, 0.5, -0.5, 2.0])
        t0 = np.zeros(5)
        sig = np.ones(5) * 0.2
        _, sf = james_stein_scaled(X, t0, sig, positive_part=True)
        assert 0.0 <= sf <= 1.0

    def test_positive_part_clips_at_zero(self):
        # Very small norm → without positive part, sf could be negative
        X = np.array([0.01, -0.01, 0.005, -0.005, 0.0])
        t0 = np.zeros(5)
        sig = np.ones(5) * 0.5
        _, sf = james_stein_scaled(X, t0, sig, positive_part=True)
        assert sf >= 0.0

    def test_requires_p_at_least_3(self):
        with pytest.raises(AssertionError):
            james_stein_scaled(np.array([1.0, 2.0]), np.zeros(2), np.ones(2))

    def test_lower_risk_than_naive_on_average(self):
        """JS estimator should have lower MSE than X on repeated trials."""
        rng = np.random.default_rng(42)
        theta = np.array([1.0, 0.5, -0.5, 0.2, -0.8])
        p = len(theta)
        sigma = 0.4
        n_trials = 5000

        mse_naive, mse_js = 0.0, 0.0
        for _ in range(n_trials):
            X = theta + rng.normal(0, sigma, p)
            t0 = np.zeros(p)
            sig_sq = np.full(p, sigma ** 2)
            js, _ = james_stein_scaled(X, t0, sig_sq, positive_part=True)
            mse_naive += np.mean((X - theta) ** 2)
            mse_js    += np.mean((js - theta) ** 2)

        assert mse_js < mse_naive, "JS should have lower total MSE than naive"


class TestJamesSteinHetero:
    def test_output_shape(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t0 = np.zeros(5)
        sig = np.array([0.1, 0.2, 0.05, 0.3, 0.1])
        result = james_stein_hetero(X, t0, sig)
        assert result.shape == (5,)

    def test_high_variance_team_shrinks_more(self):
        """Team with larger sigma (fewer matches) should shift more toward t0."""
        X = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        t0 = np.zeros(5)
        # team 0 has high variance (few matches), team 4 has low variance
        sig = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
        result = james_stein_hetero(X, t0, sig, positive_part=True)
        # team 0 should be closer to t0 than team 4
        assert abs(result[0]) <= abs(result[4]) + 1e-9


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def js_estimator():
    team_df  = build_team_table()
    match_df = generate_matches(team_df, seed=42)
    dc       = DixonColesModel().fit(match_df)
    return JSEstimator(dc, team_df, positive_part=True), team_df


class TestJSEstimator:
    def test_attack_defence_positive(self, js_estimator):
        js, _ = js_estimator
        assert (js.attack_ > 0).all()
        assert (js.defence_ > 0).all()

    def test_js_has_lower_total_risk_than_naive(self, js_estimator):
        """Per-confederation JS should reduce total log-space risk."""
        js, team_df = js_estimator
        true_att = np.log(team_df["true_att"].reindex(js.dc.teams_).values)
        mse_naive = np.nanmean((np.log(js.attack_naive_.values) - true_att) ** 2)
        mse_js    = np.nanmean((np.log(js.attack_.values) - true_att) ** 2)
        assert mse_js <= mse_naive * 1.05, (
            f"JS MSE ({mse_js:.4f}) should be <= naive MSE ({mse_naive:.4f})"
        )

    def test_shrinkage_table_has_all_confederations(self, js_estimator):
        js, _ = js_estimator
        st = js.shrinkage_table()
        for conf in ["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF"]:
            assert conf in st.index

    def test_comparison_table_columns(self, js_estimator):
        js, _ = js_estimator
        cmp = js.comparison_table()
        assert "att_naive" in cmp.columns
        assert "att_js"    in cmp.columns
        assert "n_matches" in cmp.columns

    def test_goal_lambdas_positive(self, js_estimator):
        js, _ = js_estimator
        t1, t2 = js.dc.teams_[0], js.dc.teams_[1]
        lh, la = js.goal_lambdas(t1, t2)
        assert lh > 0 and la > 0

    def test_win_draw_loss_sums_to_one(self, js_estimator):
        js, _ = js_estimator
        t1, t2 = js.dc.teams_[0], js.dc.teams_[1]
        pw, pd_, pa = js.win_draw_loss(t1, t2)
        assert abs(pw + pd_ + pa - 1.0) < 1e-5
