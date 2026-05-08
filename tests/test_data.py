"""Tests for wc2026.data module."""

import pandas as pd
import pytest
from wc2026.data import build_team_table, generate_matches, TEAMS


def test_team_table_shape():
    df = build_team_table()
    assert len(df) == len(TEAMS)
    assert set(df.columns) >= {"confederation", "fifa_pts", "true_att", "true_def"}


def test_team_table_parameters_positive():
    df = build_team_table()
    assert (df["true_att"] > 0).all()
    assert (df["true_def"] > 0).all()


def test_team_table_attack_ordering():
    df = build_team_table()
    # Argentina has highest FIFA pts → highest attack
    assert df.loc["Argentina", "true_att"] > df.loc["New Zealand", "true_att"]
    # New Zealand has lowest pts → highest defence weakness (more goals conceded)
    assert df.loc["Argentina", "true_def"] < df.loc["New Zealand", "true_def"]


def test_generate_matches_columns():
    team_df = build_team_table()
    mdf = generate_matches(team_df, seed=0)
    assert set(mdf.columns) == {"home_team", "away_team", "home_score", "away_score"}


def test_generate_matches_non_empty():
    team_df = build_team_table()
    mdf = generate_matches(team_df, seed=0)
    assert len(mdf) > 50


def test_generate_matches_goals_non_negative():
    team_df = build_team_table()
    mdf = generate_matches(team_df, seed=0)
    assert (mdf["home_score"] >= 0).all()
    assert (mdf["away_score"] >= 0).all()


def test_generate_matches_reproducible():
    team_df = build_team_table()
    m1 = generate_matches(team_df, seed=99)
    m2 = generate_matches(team_df, seed=99)
    pd.testing.assert_frame_equal(m1, m2)


def test_confederations_present():
    df = build_team_table()
    expected = {"UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"}
    assert expected.issubset(set(df["confederation"]))
