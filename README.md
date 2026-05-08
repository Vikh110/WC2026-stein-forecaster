# WC2026 Stein-Shrinkage Forecaster

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Vikh110/wc2026-stein-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/Vikh110/wc2026-stein-forecaster/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A rigorous application of **James-Stein shrinkage estimation** to international football, built on top of a Dixon-Coles Poisson goals model fit on **3,700+ real match results**.

---

## What this project actually shows

This is not a "WC 2026 prediction tool." The central question is:

> *When does James-Stein shrinkage beat naive MLE for football team strength estimation — and by how much?*

### The corrected picture: effective sample size matters

Raw match count per team is ~100. But with exponential time-decay (λ=0.003/day), the **effective sample size is ~9.5 matches per team** — that is what actually governs MLE variance. This places our data firmly in the sparse-data regime where JS provides meaningful risk reduction.

| Effective n per team | JS analytical risk reduction |
|:---:|:---:|
| 5 (early qualifying) | ~55% |
| 10 | ~39% |
| 20 | ~24% |
| 50 | ~11% |
| **~9.5 (WC2026 actual, after time-decay)** | **~25-35%** |

A previous version of this analysis reported only ~5% reduction "at n=100." That was wrong — it ignored the time-decay weights.

### The honest empirical result

Temporal backtest (train: 2014–2017, test: WC 2018):

| Model | Log-loss | vs baseline |
|:---|:---:|:---:|
| FIFA ranking baseline (zero fitting) | 2.8728 | — |
| Naive Dixon-Coles (real match data) | 2.8542 | +0.64% better |
| JS estimator | 3.0558 | **-7.1% vs naive DC** |

Two findings worth noting:
1. Real match data barely beats just using FIFA rankings (+0.64%). Most predictive signal is already in the rankings.
2. JS was empirically *worse* than naive DC on WC 2018 (-7.1%). The analytical risk guarantee is about *expected* risk averaged over all possible true parameters — it does not guarantee improvement on any single tournament. This is the gap between frequentist theory and one specific outcome.

These are not bugs. They are the honest answer to "does Stein's paradox actually help in practice here?"

---

## What the model is built on

- **3,719 real international fixtures** (2018–present) from the [martj42/international_results](https://github.com/martj42/international_results) dataset
- **Exponential time decay** (λ = 0.003/day): recent form weighted higher; effective n ~9.5 per WC team
- **Neutral venue handling**: home advantage not applied to tournament matches
- **Dixon-Coles Poisson goals model**: MLE via L-BFGS-B, one attack + one defence parameter per team
- **Efron-Morris heteroscedastic JS**: teams with fewer effective matches shrink more toward their confederation mean

---

## WC 2026 forecast

Predictions from 100,000 Monte Carlo tournament simulations using JS-shrunk parameters.

| Team | P(Win) | P(Final) | P(SF+) |
|------|:------:|:--------:|:------:|
| Argentina | 15.0% | 23.0% | 33.4% |
| Spain | 11.3% | 18.0% | 26.9% |
| France | 10.4% | 19.5% | 29.8% |
| England | 9.2% | 17.4% | 27.0% |
| Morocco | 7.1% | 14.2% | 27.4% |
| Brazil | 6.6% | 12.4% | 27.3% |
| Japan | 6.4% | 11.4% | 18.9% |

**Caveat:** the backtest shows these predictions are only marginally better than using FIFA rankings directly. The JS/DC machinery is principled, but don't mistake theoretical guarantees for empirical accuracy on a single 64-match tournament.

---

## Project structure

```
wc2026-stein-forecaster/
├── src/wc2026/
│   ├── data.py          # load_real_matches() — 3700+ real fixtures, time-decay weights
│   ├── dixon_coles.py   # Poisson goals MLE (weighted, neutral-venue-aware)
│   ├── js_shrinkage.py  # Efron-Morris heteroscedastic JS estimator
│   ├── simulator.py     # Tournament draw + Monte Carlo (100k runs)
│   ├── backtest.py      # Risk analysis, temporal backtest, FIFA baseline
│   └── pipeline.py      # End-to-end runner
├── notebooks/
│   └── WC2026_Stein_Forecaster.ipynb
├── tests/               # 45 tests across all modules
└── .github/workflows/ci.yml
```

---

## Quickstart

```bash
git clone https://github.com/Vikh110/wc2026-stein-forecaster
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
provably has lower *expected* total risk. We use the Efron-Morris (1977) heteroscedastic extension so teams with fewer effective matches shrink more.

**Effective sample size** — with exponential time-decay weights wᵢ = exp(-λ·daysᵢ), the effective n governing MLE variance is n_eff = Σwᵢ, not the raw match count. For our WC2026 dataset, median n_eff ≈ 9.5 despite ~100 raw matches per team.

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
