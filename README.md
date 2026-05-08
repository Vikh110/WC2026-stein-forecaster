# WC2026 Stein-Shrinkage Forecaster

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/vishwaskhandelwal94/wc2026-stein-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/vishwaskhandelwal94/wc2026-stein-forecaster/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A rigorous application of **James-Stein shrinkage estimation** to international football, built on top of a Dixon-Coles Poisson goals model fit on **3,700+ real match results**.

---

## What this project actually shows

This is not primarily a "WC 2026 prediction tool." The central question is:

> *When does James-Stein shrinkage beat naive maximum-likelihood estimation for football team strength — and by how much?*

The honest answer, backed by the analytical risk formula from Efron & Morris (1977):

| Matches per team | JS risk reduction over naive MLE |
|:---:|:---:|
| 5 (early qualifying) | ~38% |
| 10 | ~28% |
| 20 | ~18% |
| 50 | ~9% |
| **100 (our WC2026 data)** | **~5%** |

**The benefit is real but decays rapidly.** By the time teams have played 100+ matches — which is where our WC2026 data sits — naive MLE is already stable and JS provides only a modest refinement. Our own backtest on WC 2010/2014/2018 showed ≤1% log-loss change in either direction.

The place where JS genuinely dominates is at the *start* of a qualifying campaign, when teams have 5–15 matches and MLE estimates are noisy. That is the Efron-Morris baseball paper's regime: 45 at-bats per player.

---

## What the model is built on

- **3,719 real international fixtures** (2018–present) from the [martj42/international_results](https://github.com/martj42/international_results) dataset
- **Exponential time decay** (λ = 0.003/day): recent form weighted higher than older results
- **Neutral venue handling**: home advantage not applied to tournament matches
- **Dixon-Coles Poisson goals model**: MLE via L-BFGS-B, one attack + one defence parameter per team
- **Efron-Morris heteroscedastic JS**: teams with fewer effective matches shrink more toward their confederation mean

---

## WC 2026 forecast (with honest caveat)

Predictions from 100,000 Monte Carlo tournament simulations using JS-shrunk parameters. These reflect actual recent match results — not FIFA ranking points or synthetic data.

| Team | P(Win) | P(Final) | P(SF+) | Why |
|------|:------:|:--------:|:------:|-----|
| Argentina | 15.0% | 23.0% | 33.4% | Reigning champions, Copa winners |
| Spain | 11.3% | 18.0% | 26.9% | Euro 2024 winners |
| France | 10.4% | 19.5% | 29.8% | Consistently top-ranked |
| England | 9.2% | 17.4% | 27.0% | Strong qualifying campaign |
| Morocco | 7.1% | 14.2% | 27.4% | 2022 semi-finalists |
| Brazil | 6.6% | 12.4% | 27.3% | Poor Copa America & qualifier form |
| Japan | 6.4% | 11.4% | 18.9% | Beat Germany + Spain in 2022 groups |

Morocco above Brazil and Japan in the top 7 are both backed by real results — not assumptions.

**Caveat:** JS shrinkage changes these numbers by at most 1–4 percentage points versus naive MLE. The forecast quality is driven by Dixon-Coles on real data; JS is a principled but small refinement at this sample size.

---

## Project structure

```
wc2026-stein-forecaster/
├── src/wc2026/
│   ├── data.py          # load_real_matches() — 3700+ real fixtures, time-decay weights
│   ├── dixon_coles.py   # Poisson goals MLE (weighted, neutral-venue-aware)
│   ├── js_shrinkage.py  # Efron-Morris heteroscedastic JS estimator
│   ├── simulator.py     # Tournament draw + Monte Carlo (100k runs)
│   ├── backtest.py      # Risk analysis + js_benefit_by_sample_size()
│   └── pipeline.py      # End-to-end runner
├── notebooks/
│   └── WC2026_Stein_Forecaster.ipynb
├── tests/               # 45 tests across all modules
└── .github/workflows/ci.yml
```

---

## Quickstart

```bash
git clone https://github.com/vishwaskhandelwal94/wc2026-stein-forecaster
cd wc2026-stein-forecaster
pip install -e ".[dev]"

# Run the full pipeline (~2 min, fetches real data)
wc2026-run

# Or open the notebook for the full walkthrough
jupyter lab notebooks/WC2026_Stein_Forecaster.ipynb
```

---

## Key concepts

**Dixon-Coles model** — each match is modelled as two independent Poisson draws:
```
Goals_home ~ Poisson(α_home × β_away × γ)   [γ = home advantage]
Goals_away ~ Poisson(α_away × β_home)
```
Parameters estimated via MLE with L-BFGS-B.

**James-Stein estimator** — for p ≥ 3 simultaneous parameters, naive MLE θ̂⁰ is inadmissible. The JS estimator:
```
θ̂ᴶˢ⁺ = θ₀ + max(0, 1 − (p−2)σ̄²/‖X−θ₀‖²) · (X − θ₀)
```
provably has lower total risk. We use the Efron-Morris (1977) heteroscedastic extension so teams with fewer matches shrink more.

**Why the benefit decays with n** — as n → ∞, σ² = 1/n → 0, the shrinkage factor → 1 (no shrinkage), and naive MLE becomes optimal. Stein's benefit lives in the sparse-data regime.

---

## References

- Stein, C. (1956). *Inadmissibility of the usual estimator for the mean of a multivariate normal distribution.*
- James, W. & Stein, C. (1961). *Estimation with quadratic loss.*
- Efron, B. & Morris, C. (1977). *Stein's paradox in statistics.* Scientific American.
- Dixon, M.J. & Coles, S.G. (1997). *Modelling association football scores and inefficiencies in the football betting market.*
- Karlis, D. & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate Poisson models.*

---

## Author

**Vishwas Khandelwal** · [vishwaskhandelwal94@gmail.com](mailto:vishwaskhandelwal94@gmail.com) · MIT License · 2026
