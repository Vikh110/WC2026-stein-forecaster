"""
WC 2026 Monte Carlo tournament simulator.

Format (FIFA official):
  - 48 teams in 16 groups of 3
  - Each team plays 2 group-stage games (round-robin within group)
  - Top 2 from each group advance → 32 teams
  - Knockout: R32 (16 games) → R16 → QF → SF → Final + 3rd place
  - All matches treated as neutral-venue
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Match engine ──────────────────────────────────────────────────────────────

def simulate_match(model, home: str, away: str,
                   rng: np.random.Generator,
                   neutral: bool = True,
                   allow_draw: bool = True) -> tuple[int, int]:
    lh, la = model.goal_lambdas(home, away, neutral=neutral)
    for _ in range(200):
        g_h = int(rng.poisson(lh))
        g_a = int(rng.poisson(la))
        if allow_draw or g_h != g_a:
            return g_h, g_a
    # deterministic fallback
    pw, _, pa = model.win_draw_loss(home, away, neutral=neutral)
    return (1, 0) if rng.random() < pw / (pw + pa + 1e-9) else (0, 1)


# ── Group stage ───────────────────────────────────────────────────────────────

def play_group(teams: list[str], model,
               rng: np.random.Generator) -> list[str]:
    pts = defaultdict(int)
    gd  = defaultdict(int)
    gf  = defaultdict(int)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a    = teams[i], teams[j]
            g_h, g_a = simulate_match(model, h, a, rng, allow_draw=True)
            gf[h] += g_h; gf[a] += g_a
            gd[h] += g_h - g_a; gd[a] += g_a - g_h
            if g_h > g_a:    pts[h] += 3
            elif g_h < g_a:  pts[a] += 3
            else:             pts[h] += 1; pts[a] += 1
    return sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)


# ── Draw ──────────────────────────────────────────────────────────────────────

def make_draw(teams: list[str], strength: pd.Series,
              confederation: pd.Series,
              n_groups: int = 16,
              rng: np.random.Generator | None = None) -> list[list[str]]:
    """Pot-based draw: top 16 seeds in Pot 1, next 16 in Pot 2, rest in Pot 3."""
    if rng is None:
        rng = np.random.default_rng(0)

    ranked = [t for t in strength.sort_values(ascending=False).index
              if t in teams][:48]
    pots   = [ranked[:16], ranked[16:32], ranked[32:48]]
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    r      = random.Random(int(rng.integers(1_000_000)))

    for pot_idx, pot in enumerate(pots):
        shuffled = list(pot)
        r.shuffle(shuffled)
        for team in shuffled:
            conf = confederation.get(team, "?")
            candidates = [g for g in range(n_groups)
                          if len(groups[g]) == pot_idx
                          and not (conf == "CONMEBOL" and
                                   any(confederation.get(t) == "CONMEBOL"
                                       for t in groups[g]))]
            if not candidates:
                candidates = [g for g in range(n_groups)
                              if len(groups[g]) == pot_idx]
            g_idx = r.choice(candidates)
            groups[g_idx].append(team)

    return groups


# ── Full tournament ───────────────────────────────────────────────────────────

# Assign exit labels in order of rounds reached
ROUND_ORDER = ["Group Stage", "R32", "R16", "QF", "SF", "3rd", "Runner-up", "Winner"]
ROUND_RANK  = {r: i for i, r in enumerate(ROUND_ORDER)}


def simulate_tournament(model, groups: list[list[str]],
                        rng: np.random.Generator) -> dict[str, str]:
    """
    Returns {team: exit_label} where label ∈ ROUND_ORDER.
    "R32" means advanced from groups but lost in the round of 32.
    "R16" means won R32 but lost in R16, etc.
    """
    result: dict[str, str] = {}

    # ── Group stage ──
    qualifiers: list[list[str]] = []
    for g in groups:
        ranked_g = play_group(g, model, rng)
        qualifiers.append(ranked_g)
        result[ranked_g[0]] = "R32_advancing"
        result[ranked_g[1]] = "R32_advancing"
        for t in ranked_g[2:]:
            result[t] = "Group Stage"

    # ── Knockout bracket ──
    # Pair group winners vs runners-up from adjacent groups
    # (0-winner vs 1-runner, 1-winner vs 0-runner, 2-winner vs 3-runner, ...)
    r32: list[tuple[str, str]] = []
    for i in range(0, 16, 2):
        r32.append((qualifiers[i][0],   qualifiers[i+1][1]))
        r32.append((qualifiers[i+1][0], qualifiers[i][1]))

    def knockout_stage(pairs: list[tuple[str, str]],
                       loser_label: str) -> list[tuple[str, str]]:
        """Play pairs, label losers, return next-round pairs of winners."""
        winners: list[str] = []
        for h, a in pairs:
            g_h, g_a = simulate_match(model, h, a, rng,
                                      neutral=True, allow_draw=False)
            w, l = (h, a) if g_h > g_a else (a, h)
            result[l] = loser_label
            winners.append(w)
        return [(winners[j], winners[j+1])
                for j in range(0, len(winners), 2)]

    r16_pairs  = knockout_stage(r32,       loser_label="R32")
    qf_pairs   = knockout_stage(r16_pairs, loser_label="R16")
    sf_pairs   = knockout_stage(qf_pairs,  loser_label="QF")

    # Semi-finals: produce 2 finalists + 2 third-place contestants
    finalists, third_place_teams = [], []
    for h, a in sf_pairs:
        g_h, g_a = simulate_match(model, h, a, rng, neutral=True, allow_draw=False)
        w, l = (h, a) if g_h > g_a else (a, h)
        finalists.append(w)
        third_place_teams.append(l)
        result[l] = "SF"

    # 3rd-place play-off
    g_h, g_a = simulate_match(model, third_place_teams[0], third_place_teams[1],
                               rng, neutral=True, allow_draw=False)
    third = third_place_teams[0] if g_h > g_a else third_place_teams[1]
    fourth = third_place_teams[1] if g_h > g_a else third_place_teams[0]
    result[third]  = "3rd"
    result[fourth] = "SF"   # 4th place is still "SF exit"

    # Final
    g_h, g_a = simulate_match(model, finalists[0], finalists[1],
                               rng, neutral=True, allow_draw=False)
    winner = finalists[0] if g_h > g_a else finalists[1]
    runner = finalists[1] if g_h > g_a else finalists[0]
    result[winner] = "Winner"
    result[runner] = "Runner-up"

    # clean up advancement markers
    for t in list(result):
        if result[t] == "R32_advancing":
            result[t] = "R32"   # shouldn't happen but safety net

    return result


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo(model, groups: list[list[str]],
                n_simulations: int = 100_000,
                seed: int = 0) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      team, p_group_exit, p_r32, p_r16, p_qf, p_sf, p_3rd, p_final, p_winner
    These are CUMULATIVE (p_qf = P(reach QF or better)).
    """
    rng       = np.random.default_rng(seed)
    all_teams = [t for g in groups for t in g]
    counts: dict[str, dict[str, int]] = {t: defaultdict(int) for t in all_teams}

    for _ in range(n_simulations):
        res = simulate_tournament(model, groups, rng)
        for team, stage in res.items():
            counts[team][stage] += 1

    rows = []
    for team in all_teams:
        c = counts[team]
        n = n_simulations
        # cumulative: p_r16 = P(teams that reached R16 OR BETTER)
        p_gs  = c.get("Group Stage", 0) / n
        p_r32 = c.get("R32",         0) / n  # advanced but lost in R32
        p_r16 = c.get("R16",         0) / n  # won R32, lost R16
        p_qf  = c.get("QF",          0) / n
        p_f   = (c.get("Runner-up", 0) + c.get("Winner", 0)) / n
        p_w   = c.get("Winner",      0) / n

        # cumulative reach probabilities
        p_reach_r32 = 1 - p_gs           # advanced from groups
        p_reach_r16 = 1 - p_gs - p_r32   # won at least 1 KO game
        p_reach_qf  = 1 - p_gs - p_r32 - p_r16
        p_reach_sf  = 1 - p_gs - p_r32 - p_r16 - p_qf
        p_reach_f   = p_f
        rows.append({
            "team":         team,
            "p_reach_r32":  round(p_reach_r32, 4),
            "p_reach_r16":  round(p_reach_r16, 4),
            "p_reach_qf":   round(p_reach_qf,  4),
            "p_reach_sf":   round(p_reach_sf,  4),
            "p_reach_final":round(p_reach_f,   4),
            "p_winner":     round(p_w,          4),
        })

    return (pd.DataFrame(rows)
              .set_index("team")
              .sort_values("p_winner", ascending=False))


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data import load_all
    from dixon_coles import DixonColesModel
    from js_shrinkage import JSEstimator

    team_df, match_df, _ = load_all()
    dc = DixonColesModel().fit(match_df)
    js = JSEstimator(dc, team_df)

    strength    = js.attack_ / js.defence_
    conf_series = team_df["confederation"].reindex(js.dc.teams_).fillna("UNKNOWN")

    rng_draw = np.random.default_rng(2026)
    groups   = make_draw(js.dc.teams_, strength, conf_series, rng=rng_draw)

    print("Groups:")
    for i, g in enumerate(groups):
        print(f"  Group {chr(65+i)}: {g}")

    print("\nRunning 20,000 simulations...")
    probs = monte_carlo(js, groups, n_simulations=20_000, seed=42)
    print("\nTop 20 by P(win):")
    print(probs.head(20).to_string())
