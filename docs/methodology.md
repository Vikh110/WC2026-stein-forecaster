# Methodology

## 1. Stein's Paradox

Let $X_1, \ldots, X_p$ be independent with $X_i \sim \mathcal{N}(\theta_i, \sigma^2)$.
The naive estimator $\hat{\theta}^0 = X$ has risk $R(\hat{\theta}^0, \theta) = p\sigma^2$.

Stein (1956) proved that for $p \geq 3$, $\hat{\theta}^0$ is **inadmissible**: the James–Stein estimator

$$\hat{\theta}^{JS} = \left(1 - \frac{(p-2)\sigma^2}{\|X\|^2}\right) X$$

satisfies $R(\hat{\theta}^{JS}, \theta) < R(\hat{\theta}^0, \theta)$ for all $\theta \in \mathbb{R}^p$.

The positive-part version $\hat{\theta}^{JS+}$ replaces the shrinkage factor with $\max(0, \cdot)$ and further dominates $\hat{\theta}^{JS}$.

## 2. Heteroscedastic Extension (Efron–Morris 1977)

When $X_i \sim \mathcal{N}(\theta_i, \sigma_i^2)$ with unequal variances, the estimator becomes

$$\hat{\theta}^{JS+}_i = \theta_{0,i} + \left(1 - B \cdot \sigma_i^2\right)_+ (X_i - \theta_{0,i})$$

where $B = (p-2) / \sum_i (X_i - \theta_{0,i})^2 / \sigma_i^2$ and $\theta_0$ is the shrinkage target.

Teams with more data ($\sigma_i^2 = 1/n_i$ small) are shrunk less. Teams with few matches are pulled harder toward the confederation mean.

## 3. Football Strength Model

**Poisson goals model:**

$$\text{Goals}_{home} \sim \text{Poisson}(\alpha_h \cdot \beta_a \cdot \gamma), \quad \text{Goals}_{away} \sim \text{Poisson}(\alpha_a \cdot \beta_h)$$

Parameters estimated by MLE via L-BFGS-B. We apply JS shrinkage to $\log \alpha_i$ and $\log \beta_i$ separately, using $\sigma_i^2 \approx 1/n_i$ from asymptotic MLE theory.

## 4. Analytical Risk Formula

From the Appendix of Samworth (2005), for $X \sim \mathcal{N}(\theta, \sigma^2 I_p)$:

$$R(\hat{\theta}^{JS+}, \theta) = p\sigma^2 - (p-2)^2 \sigma^4 \cdot \mathbb{E}\left[\frac{1}{\|X - \theta_0\|^2}\right]$$

We estimate $\mathbb{E}[1/\|X\|^2]$ via Monte Carlo sampling from $\mathcal{N}(X_{obs}, \sigma^2 I_p)$.
