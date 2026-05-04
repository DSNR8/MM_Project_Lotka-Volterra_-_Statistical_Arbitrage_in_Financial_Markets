# Lotka–Volterra & Statistical Arbitrage in Financial Markets

## Overview
This project explores the application of dynamic systems theory and statistical arbitrage in financial markets. Specifically, it combines:
1. **Lotka–Volterra models** (predator–prey & competition dynamics)
2. **Cointegration-based pairs trading**
3. **Factor-neutral portfolio construction**
4. **Optimal execution** via Euler–Lagrange (Almgren–Chriss framework)

The goal is to bridge theoretical econophysics modeling with practical quantitative trading strategies.

## Methodology

### 1. Dynamic Systems Modeling
We interpret financial assets as interacting biological systems:
* **SPY ↔ VIX** → Predator–Prey dynamics
* **KO ↔ PEP** → Competitive dynamics

The continuous-time system:
$$ \frac{1}{X} \frac{dX}{dt} = f(X, Y) $$
is approximated via log transformations, discrete differences, and multiple linear regression.

### 2. Statistical Arbitrage (Core Strategy)
The trading strategy is based on mean-reverting spreads. We estimate the hedge ratio:
$$ Y_t = \alpha + \beta X_t + \epsilon_t $$
Construct the spread:
$$ S_t = Y_t - (\alpha + \beta X_t) $$
Normalize via rolling Z-score:
$$ Z_t = \frac{S_t - \mu}{\sigma} $$
**Trading rules:**
* Long the spread when $Z < -2.0$
* Short the spread when $Z > 2.0$

### 3. Factor-Neutral Extension
To reduce systemic market exposure, we isolate idiosyncratic risk by neutralizing the market factor (SPY):
$$ S_t = Y_t - (\alpha + \beta_1 X_t + \beta_2 M_t) $$
where $M_t$ is the market factor (SPY log prices).

### 4. Optimal Execution
To minimize market impact when executing the theoretical signals, we implement an Almgren–Chriss optimal execution model based on the Principle of Least Action, solving the Euler-Lagrange equation:
$$ q''(t) - \gamma^2 q(t) = 0 $$
which yields the optimal trading trajectory:
$$ q(t) = Q_0 \frac{\sinh(\gamma (T - t))}{\sinh(\gamma T)} $$
This dynamically balances market impact (slippage) and execution risk (variance).

## Results
The strategy is evaluated using Walk-Forward (Rolling Window) Out-of-Sample validation. We compare cumulative returns across:
1. Buy & Hold (SPY)
2. Simple Pairs Trading (KO vs PEP)
3. Factor-Neutral Strategy (Market Isolated)

Note: The Lotka–Volterra parameter estimation is performed in-sample for interpretability, not for predictive evaluation.

## Limitations
* Linear approximation of complex nonlinear market dynamics.
* Static thresholds ($Z = \pm 2$) for trading signals.
* Parameter instability across volatile market regimes.
* The current backtest iteration does not fully incorporate simulated transaction costs or bid-ask spread friction.

## Future Work
* **Kalman Filter** for continuous, dynamic updating of hedge ratios.
* **Ornstein–Uhlenbeck** modeling for advanced spread mean-reversion analytics.
* **Regime-Switching Models** (e.g., Hidden Markov Models) to dynamically adjust parameters.
* **Reinforcement Learning** for adaptive optimal execution.

---
**Disclaimer:** *This project is for academic research and educational purposes only. It is not a production trading system or financial advice.*
