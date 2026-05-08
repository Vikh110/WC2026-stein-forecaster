"""Tests for wc2026.dixon_coles module."""

import numpy as np
import pandas as pd
import pytest
from wc2026.data import build_team_table, generate_matches
from wc2026.dixon_coles import DixonColesModel


@pytest.fixture(scope="module")
def fitted_model():
    team_df = build_team_table()
    match_df = generate_matches(team_df, seed=42)
    return DixonColesModel().fit(match_df), match_df


def test_fit_returns_self(fitted_model):
    model, _ = fitted_model
    assert isinstance(model, DixonColesModel)


def test_attack_defence_positive(fitted_model):
    model, _ = fitted_model
    assert (model.attack_ > 0).all()
    assert (model.defence_ > 0).all()


def test_home_advantage_reasonable(fitted_model):
    model, _ = fitted_model
    # Home advantage should be between 1.0 and 1.5 for international football
    assert 1.0 <= model.home_adv_ <= 1.5


def test_teams_list_non_empty(fitted_model):
    model, _ = fitted_model
    assert len(model.teams_) >= 10


def test_n_matches_per_team_positive(fitted_model):
    model, _ = fitted_model
    assert (model.n_matches_ > 0).all()


def test_goal_lambdas_positive(fitted_model):
    model, _ = fitted_model
    t1, t2 = model.teams_[0], model.teams_[1]
    lh, la = model.goal_lambdas(t1, t2, neutral=True)
    assert lh > 0
    assert la > 0


def test_win_draw_loss_sums_to_one(fitted_model):
    model, _ = fitted_model
    t1, t2 = model.teams_[0], model.teams_[1]
    pw, pd_, pa = model.win_draw_loss(t1, t2, neutral=True)
    assert abs(pw + pd_ + pa - 1.0) < 1e-6


def test_win_draw_loss_non_negative(fitted_model):
    model, _ = fitted_model
    t1, t2 = model.teams_[0], model.teams_[1]
    pw, pd_, pa = model.win_draw_loss(t1, t2)
    assert pw >= 0 and pd_ >= 0 and pa >= 0


def test_summary_sorted_by_strength(fitted_model):
    model, _ = fitted_model
    s = model.summary()
    strengths = s["strength"].values
    assert (strengths[:-1] >= strengths[1:]).all()


def test_strong_team_beats_weak_team(fitted_model):
    """Brazil/Argentina should have higher win probability vs Jordan/New Zealand."""
    model, _ = fitted_model
    strong = "Argentina" if "Argentina" in model.teams_ else model.teams_[0]
    weak   = "New Zealand" if "New Zealand" in model.teams_ else model.teams_[-1]
    pw, _, pa = model.win_draw_loss(strong, weak)
    assert pw > pa, f"Expected {strong} to be favoured over {weak}"
