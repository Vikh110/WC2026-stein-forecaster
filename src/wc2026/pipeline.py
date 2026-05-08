"""
Main pipeline: WC2026 Stein-Shrinkage Forecaster.

Runs end-to-end and saves all outputs to ../outputs/:
  - fig1_risk_vs_nmatches.png   — replicates Figure 1 from Samworth (2005)
  - fig2_win_probabilities.png  — WC2026 win probs naive vs JS
  - fig3_shrinkage_shifts.png   — which teams JS moved most
  - fig4_group_heatmap.png      — group-stage exit probabilities
  - wc2026_groups.csv           — draw
  - wc2026_probs_naive.csv      — naive simulation results
  - wc2026_probs_js.csv         — JS simulation results
  - backtest_summary.csv        — analytical risk table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from wc2026.data import load_all
from wc2026.dixon_coles import DixonColesModel
from wc2026.js_shrinkage import JSEstimator
from wc2026.simulator import make_draw, monte_carlo
from wc2026.backtest import analytical_risk, wc_log_loss_comparison, risk_vs_nmatches

OUT = Path(__file__).parent.parent.parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

N_SIMS   = 100_000
DRAW_SEED = 2026
SIM_SEED  = 42

CONF_COLORS = {
    "UEFA":     "#003399",
    "CONMEBOL": "#009900",
    "CONCACAF": "#CC0000",
    "CAF":      "#FF8800",
    "AFC":      "#CC00CC",
    "OFC":      "#888888",
    "UNKNOWN":  "#444444",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _conf_colors(teams: list[str], team_df: pd.DataFrame) -> list[str]:
    return [CONF_COLORS.get(team_df["confederation"].get(t, "UNKNOWN"), "#444")
            for t in teams]


# ── Figure 1: Risk vs sample size ─────────────────────────────────────────────

def plot_risk_curve(dc: DixonColesModel, team_df: pd.DataFrame) -> None:
    rv = risk_vs_nmatches(dc, team_df, n_mc=100_000, seed=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rv["n_matches_per_team"], rv["risk_naive"],
            "b--", lw=2, label=r"Naive MLE $\hat\theta^0$")
    ax.plot(rv["n_matches_per_team"], rv["risk_js"],
            "r-",  lw=2, label=r"JS Estimator $\hat\theta^{JS+}$")
    ax.fill_between(rv["n_matches_per_team"],
                    rv["risk_js"], rv["risk_naive"],
                    alpha=0.15, color="green",
                    label="Risk saved by JS")

    ax.set_xlabel("Matches per team (sample size)", fontsize=12)
    ax.set_ylabel(r"Risk  $R(\hat\theta, \theta)$", fontsize=12)
    ax.set_title("Stein's Paradox Applied to Football:\n"
                 "JS estimator always beats naive MLE (p = 49 teams)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # annotate reduction at qualifying-era sample size
    row = rv[rv["n_matches_per_team"] == 20].iloc[0]
    ax.annotate(f"~{row['reduction_%']:.0f}% reduction\n"
                f"at qualifying era\n(~20 games/team)",
                xy=(20, (row["risk_naive"] + row["risk_js"]) / 2),
                xytext=(40, row["risk_naive"] * 0.7),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10, color="green")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_risk_vs_nmatches.png", dpi=150)
    plt.close(fig)
    print("  Saved fig1_risk_vs_nmatches.png")


# ── Figure 2: Win probabilities naive vs JS ───────────────────────────────────

def plot_win_probs(probs_naive: pd.DataFrame, probs_js: pd.DataFrame,
                  team_df: pd.DataFrame, top_n: int = 20) -> None:
    top_teams = probs_js["p_winner"].nlargest(top_n).index.tolist()

    pn = probs_naive.loc[top_teams, "p_winner"].values * 100
    pj = probs_js.loc[top_teams,    "p_winner"].values * 100

    x  = np.arange(top_n)
    w  = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, pn, w, color="steelblue",  alpha=0.7, label="Naive MLE")
    ax.bar(x + w/2, pj, w, color="firebrick",  alpha=0.7, label="JS Estimator")

    # confederation colour strips along x-axis
    for i, t in enumerate(top_teams):
        c = CONF_COLORS.get(team_df["confederation"].get(t, "?"), "#888")
        ax.axvspan(i - 0.48, i + 0.48, ymin=0, ymax=0.02,
                   color=c, alpha=0.8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(top_teams, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("P(Win World Cup)  %", fontsize=12)
    ax.set_title("WC 2026 Win Probabilities: Naive MLE vs James-Stein Estimator\n"
                 f"(100,000 Monte Carlo simulations)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # confederation legend
    patches = [mpatches.Patch(color=c, label=k)
               for k, c in CONF_COLORS.items() if k != "UNKNOWN"]
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], *patches],
              labels=[*ax.get_legend_handles_labels()[1],
                      *[k for k in CONF_COLORS if k != "UNKNOWN"]],
              fontsize=9, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_win_probabilities.png", dpi=150)
    plt.close(fig)
    print("  Saved fig2_win_probabilities.png")


# ── Figure 3: Shrinkage shifts ────────────────────────────────────────────────

def plot_shrinkage(js: JSEstimator) -> None:
    cmp     = js.comparison_table().reset_index()
    cmp     = cmp.sort_values("att_pct_change", key=abs, ascending=False).head(25)
    teams   = cmp["team"].tolist()
    shifts  = cmp["att_pct_change"].values
    n_match = cmp["n_matches"].values
    bar_colors = [CONF_COLORS.get(r["confederation"], "#888")
                  for _, r in cmp.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(teams)), shifts, color=bar_colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(
        [f"{t}  (n={n})" for t, n in zip(teams, n_match)], fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Change in Attack Parameter after JS Shrinkage  (%)", fontsize=11)
    ax.set_title("James-Stein Effect: Teams Most Affected by Shrinkage\n"
                 "Weak teams (small n) shift most toward confederation mean",
                 fontsize=12)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=c, label=k)
               for k, c in CONF_COLORS.items() if k != "UNKNOWN"]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_shrinkage_shifts.png", dpi=150)
    plt.close(fig)
    print("  Saved fig3_shrinkage_shifts.png")


# ── Figure 4: Group-stage exit heatmap ───────────────────────────────────────

def plot_group_heatmap(probs_js: pd.DataFrame,
                       groups: list[list[str]]) -> None:
    rounds = ["p_reach_r32", "p_reach_r16", "p_reach_qf",
              "p_reach_sf", "p_reach_final", "p_winner"]
    labels = ["Advance\nfrom Groups", "Round\nof 16", "Quarter\nFinal",
              "Semi\nFinal", "Final", "Win"]

    fig, ax = plt.subplots(figsize=(12, 14))
    all_t   = [t for g in groups for t in g]
    data    = probs_js.loc[all_t, rounds].values * 100

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=60)
    plt.colorbar(im, ax=ax, label="Probability (%)", shrink=0.5)

    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(all_t)))
    ax.set_yticklabels(all_t, fontsize=8)

    # add group separators and labels
    group_labels = [chr(65 + i) for i in range(16)]
    for g_i, grp in enumerate(groups):
        start = g_i * 3
        if g_i > 0:
            ax.axhline(start - 0.5, color="white", lw=1.5)
        ax.text(-0.7, start + 1, f"G{group_labels[g_i]}",
                ha="right", va="center", fontsize=7,
                fontweight="bold", color="navy")

    ax.set_title("WC 2026 — JS Estimator Tournament Probabilities\n"
                 "(all 48 teams, grouped by draw)", fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_group_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4_group_heatmap.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("WC 2026 STEIN-SHRINKAGE FORECASTER")
    print("=" * 62)

    # ── Load data & fit models ──
    print("\n[1/6] Loading data and fitting models...")
    team_df, match_df, wc_history = load_all()
    dc_naive = DixonColesModel().fit(match_df)
    js       = JSEstimator(dc_naive, team_df, positive_part=True)
    print(f"      {len(team_df)} teams, {len(match_df)} matches")

    # ── Draw ──
    print("\n[2/6] Making tournament draw...")
    strength_js   = js.attack_ / js.defence_
    conf_series   = team_df["confederation"].reindex(js.dc.teams_).fillna("UNKNOWN")
    rng_draw      = np.random.default_rng(DRAW_SEED)
    groups        = make_draw(js.dc.teams_, strength_js, conf_series, rng=rng_draw)

    groups_df = pd.DataFrame([
        {"group": chr(65+i), "team": t,
         "confederation": team_df["confederation"].get(t, "?"),
         "fifa_pts": team_df["fifa_pts"].get(t, 0)}
        for i, g in enumerate(groups) for t in g
    ])
    groups_df.to_csv(OUT / "wc2026_groups.csv", index=False)
    print("      Groups saved to wc2026_groups.csv")

    # ── Simulate: Naive ──
    print(f"\n[3/6] Simulating {N_SIMS:,} tournaments (Naive MLE)...")
    probs_naive = monte_carlo(dc_naive, groups, n_simulations=N_SIMS, seed=SIM_SEED)
    probs_naive.to_csv(OUT / "wc2026_probs_naive.csv")
    print("      Saved wc2026_probs_naive.csv")

    # ── Simulate: JS ──
    print(f"\n[4/6] Simulating {N_SIMS:,} tournaments (JS Estimator)...")
    probs_js = monte_carlo(js, groups, n_simulations=N_SIMS, seed=SIM_SEED)
    probs_js.to_csv(OUT / "wc2026_probs_js.csv")
    print("      Saved wc2026_probs_js.csv")

    # ── Backtest ──
    print("\n[5/6] Running backtest analyses...")
    ar = analytical_risk(dc_naive, team_df)
    ar.to_csv(OUT / "backtest_summary.csv")

    ll = wc_log_loss_comparison(wc_history, team_df, match_df)
    print("\n  WC Historical Log-Loss (Naive vs JS):")
    print(ll.to_string(index=False))

    # ── Plots ──
    print("\n[6/6] Generating figures...")
    plot_risk_curve(dc_naive, team_df)
    plot_win_probs(probs_naive, probs_js, team_df)
    plot_shrinkage(js)
    plot_group_heatmap(probs_js, groups)

    # ── Final summary ──
    print("\n" + "=" * 62)
    print("TOP 10 WC 2026 PREDICTIONS (JS Estimator)")
    print("=" * 62)
    top10 = probs_js.head(10)[["p_reach_r16","p_reach_qf","p_reach_sf",
                                "p_reach_final","p_winner"]].copy()
    top10.columns = ["P(R16+)", "P(QF+)", "P(SF+)", "P(Final)", "P(Win)"]
    top10 = top10.map(lambda x: f"{x*100:.1f}%")
    print(top10.to_string())

    print("\n" + "=" * 62)
    print("ANALYTICAL RISK REDUCTION PER CONFEDERATION")
    print("=" * 62)
    print(ar[["n_teams","avg_n_matches","risk_naive","risk_js","reduction_%"]].to_string())

    print("\nAll outputs saved to:", OUT)


if __name__ == "__main__":
    main()
