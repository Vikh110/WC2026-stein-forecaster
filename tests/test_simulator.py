"""Tests for wc2026.simulator module."""

import numpy as np
import pytest
from wc2026.data import build_team_table, generate_matches
from wc2026.dixon_coles import DixonColesModel
from wc2026.js_shrinkage import JSEstimator
from wc2026.simulator import make_draw, simulate_tournament, monte_carlo, play_group


@pytest.fixture(scope="module")
def setup():
    team_df  = build_team_table()
    match_df = generate_matches(team_df, seed=42)
    dc       = DixonColesModel().fit(match_df)
    js       = JSEstimator(dc, team_df)
    strength = js.attack_ / js.defence_
    conf     = team_df["confederation"].reindex(dc.teams_).fillna("UNKNOWN")
    groups   = make_draw(dc.teams_, strength, conf,
                         rng=np.random.default_rng(2026))
    return js, groups, team_df


class TestMakeDraw:
    def test_produces_16_groups(self, setup):
        _, groups, _ = setup
        assert len(groups) == 16

    def test_each_group_has_3_teams(self, setup):
        _, groups, _ = setup
        for g in groups:
            assert len(g) == 3

    def test_all_teams_assigned_once(self, setup):
        js, groups, _ = setup
        all_in_groups = [t for g in groups for t in g]
        assert len(all_in_groups) == len(set(all_in_groups)), "Duplicate teams in draw"

    def test_no_two_conmebol_in_same_group(self, setup):
        """Draw algorithm tries to avoid same-conf groups; at most 1 violation allowed
        when CONMEBOL pot-3 teams (Bolivia) can't be placed elsewhere."""
        js, groups, team_df = setup
        violations = []
        for g in groups:
            confs = [team_df["confederation"].get(t, "?") for t in g]
            if confs.count("CONMEBOL") > 1:
                violations.append(g)
        assert len(violations) <= 1, f"Too many same-CONMEBOL groups: {violations}"


class TestPlayGroup:
    def test_returns_all_teams(self, setup):
        js, groups, _ = setup
        rng = np.random.default_rng(0)
        result = play_group(groups[0], js, rng)
        assert set(result) == set(groups[0])

    def test_returns_3_teams(self, setup):
        js, groups, _ = setup
        rng = np.random.default_rng(0)
        result = play_group(groups[0], js, rng)
        assert len(result) == 3


class TestSimulateTournament:
    def test_all_teams_get_result(self, setup):
        js, groups, _ = setup
        rng = np.random.default_rng(0)
        result = simulate_tournament(js, groups, rng)
        all_teams = {t for g in groups for t in g}
        assert set(result.keys()) == all_teams

    def test_exactly_one_winner(self, setup):
        js, groups, _ = setup
        rng = np.random.default_rng(0)
        result = simulate_tournament(js, groups, rng)
        winners = [t for t, r in result.items() if r == "Winner"]
        assert len(winners) == 1

    def test_exactly_one_runner_up(self, setup):
        js, groups, _ = setup
        rng = np.random.default_rng(0)
        result = simulate_tournament(js, groups, rng)
        runners = [t for t, r in result.items() if r == "Runner-up"]
        assert len(runners) == 1

    def test_valid_exit_labels(self, setup):
        js, groups, _ = setup
        rng = np.random.default_rng(0)
        result = simulate_tournament(js, groups, rng)
        valid = {"Group Stage", "R32", "R16", "QF", "SF", "3rd", "Runner-up", "Winner"}
        for t, r in result.items():
            assert r in valid, f"Team {t} has invalid label: {r}"


class TestMonteCarlo:
    def test_output_shape(self, setup):
        js, groups, _ = setup
        probs = monte_carlo(js, groups, n_simulations=500, seed=0)
        assert len(probs) == 48  # all teams

    def test_win_probs_sum_to_one(self, setup):
        js, groups, _ = setup
        probs = monte_carlo(js, groups, n_simulations=1000, seed=0)
        total = probs["p_winner"].sum()
        assert abs(total - 1.0) < 0.02, f"Win probs sum to {total:.3f}, expected ~1.0"

    def test_probabilities_in_0_1(self, setup):
        js, groups, _ = setup
        probs = monte_carlo(js, groups, n_simulations=500, seed=0)
        for col in probs.columns:
            assert (probs[col] >= 0).all()
            assert (probs[col] <= 1).all()

    def test_strong_teams_have_higher_win_prob(self, setup):
        js, groups, _ = setup
        probs = monte_carlo(js, groups, n_simulations=2000, seed=0)
        if "Brazil" in probs.index and "Jordan" in probs.index:
            assert probs.loc["Brazil", "p_winner"] > probs.loc["Jordan", "p_winner"]
