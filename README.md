# WC 2026 Stein-Shrinkage Forecaster

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/vishwaskhandelwal94/wc2026-stein-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/vishwaskhandelwal94/wc2026-stein-forecaster/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **"Perhaps the most surprising result in Statistics"** — Dr Richard J. Samworth, Statslab Cambridge

A FIFA World Cup 2026 forecaster that applies **Stein's Paradox (1956)** to football team strength estimation. The James–Stein estimator provably dominates the naive Maximum Likelihood Estimator when estimating 3 or more parameters simultaneously — and with 48 teams across 6 confederations, the World Cup is a perfect natural experiment.

---

## The Core Idea

When you estimate team strengths from historical match data, the "obvious" approach is to treat each team independently: use each team's own results to estimate their own strength. Stein (1956) proved this is **inadmissible** when estimating 3+ parameters simultaneously.

The James–Stein estimator *shrinks* each team's estimate toward the confederation average:

```
θ̂ᴶˢ = θ₀ + (1 - (p-2)·σ²/‖X-θ₀‖²)₊ · (X - θ₀)
```

- **Strong teams** (many matches, precise estimates) → barely moved  
- **Weak teams** (few matches, noisy estimates) → pulled hard toward confederation mean  
- **Total risk** across all 48 teams is provably lower — always, for any true values

This is exactly the baseball batting-average example from Samworth (2005), applied to football.

---

## Results

| Confederation | Teams | Avg Matches | Risk Reduction (JS vs Naive) |
|:---|:---:|:---:|:---:|
| AFC | 8 | 34 | **42.5%** |
| CAF | 10 | 35 | **41.5%** |
| UEFA | 16 | 53 | **36.0%** |
| CONMEBOL | 7 | 45 | **34.3%** |
| CONCACAF | 7 | 35 | **23.4%** |

### WC 2026 Top Predictions (JS Estimator, 100,000 simulations)

| Team | P(Win) | P(Final) | P(Semi-Final) |
|:---|:---:|:---:|:---:|
| 🇧🇷 Brazil | **19.7%** | 28.9% | 38.7% |
| 🇦🇷 Argentina | 11.0% | 20.4% | 31.7% |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 9.9% | 18.8% | 32.8% |
| 🇫🇷 France | 9.7% | 15.9% | 23.1% |
| 🇪🇸 Spain | 9.4% | 17.7% | 28.0% |

*Results will be verified when the tournament concludes in July 2026.*

---

## Visual Outputs

| Figure | Description |
|:---|:---|
| `fig1_risk_vs_nmatches.png` | JS always beats naive MLE — replicates Figure 1 from Samworth (2005) |
| `fig2_win_probabilities.png` | WC 2026 win probabilities, naive vs JS, coloured by confederation |
| `fig3_shrinkage_shifts.png` | Which teams JS moved most (low-data teams shift most) |
| `fig4_group_heatmap.png` | All 48 teams × 6 tournament stages probability heatmap |

---

## Project Structure

```
wc2026-stein-forecaster/
├── src/wc2026/
│   ├── __init__.py
│   ├── data.py           # 49 WC2026 teams + synthetic match generator
│   ├── dixon_coles.py    # Naive MLE Poisson goals model (θ̂⁰)
│   ├── js_shrinkage.py   # Efron-Morris heteroscedastic JS estimator (θ̂ᴶˢ⁺)
│   ├── simulator.py      # Monte Carlo WC2026 tournament (16 groups of 3)
│   ├── backtest.py       # Analytical risk + WC historical log-loss
│   └── pipeline.py       # Full orchestrator — runs everything
├── notebooks/
│   └── WC2026_Stein_Forecaster.ipynb   # End-to-end walkthrough
├── tests/
│   ├── test_data.py
│   ├── test_dixon_coles.py
│   ├── test_js_shrinkage.py
│   └── test_simulator.py
├── scripts/
│   └── run_pipeline.py   # CLI entry point
├── outputs/              # Generated plots + CSVs (git-ignored)
├── docs/
│   └── methodology.md    # Mathematical derivation
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/vishwaskhandelwal94/wc2026-stein-forecaster.git
cd wc2026-stein-forecaster
pip install -e ".[notebook]"
```

### 2. Run the full pipeline

```bash
python scripts/run_pipeline.py
# or
wc2026-run
```

Outputs are saved to `outputs/`. Takes ~2–3 minutes (200,000 tournament simulations).

### 3. Open the notebook

```bash
jupyter notebook notebooks/WC2026_Stein_Forecaster.ipynb
```

The notebook walks through every step with explanations, math, and inline plots.

### 4. Run tests

```bash
pytest --cov=wc2026 --cov-report=term-missing
```

---

## Methodology

The full mathematical derivation is in [`docs/methodology.md`](docs/methodology.md).

**Model:** Poisson goals model (Dixon-Coles variant)
```
Goals_home ~ Poisson(α_h · β_a · γ)
Goals_away ~ Poisson(α_a · β_h)
```
where `α_i` = attack strength, `β_i` = defence weakness, `γ` = home advantage.

**Shrinkage:** Efron-Morris (1977) heteroscedastic James-Stein estimator applied to log-parameters, with confederation-level shrinkage targets and per-team variance `σᵢ² ≈ 1/nᵢ` from MLE asymptotics.

**Simulation:** 100,000 Monte Carlo runs of the full WC2026 format (16 groups of 3 → R32 → R16 → QF → SF → Final).

---

## References

1. Stein, C. (1956). *Inadmissibility of the usual estimator for the mean of a multivariate normal distribution.* Proc. Third Berkeley Symposium, 1, 197–206.
2. James, W. & Stein, C. (1961). *Estimation with quadratic loss.* Proc. Fourth Berkeley Symposium, 1, 361–380.
3. Efron, B. & Morris, C. (1977). *Stein's paradox in statistics.* Scientific American, 236(5), 119–127.
4. Samworth, R.J. (2005). *Small confidence sets for the mean of a spherically symmetric distribution.* JRSS-B, 67, 343–361.
5. Dixon, M.J. & Coles, S.G. (1997). *Modelling association football scores and inefficiencies in the football betting market.* Applied Statistics, 46(2), 265–280.

---

## License

MIT © 2026 Vishwas Khandelwal
