"""HFMetal Python quickstart example.

Run from the build directory:
    PYTHONPATH=python python3 python/examples/quickstart.py
"""
import numpy as np
import _hfmetal as hfm

# --- Returns ---
prices = np.array([100.0, 101.5, 99.8, 102.3, 103.1, 100.9, 104.2])
log_r = hfm.log_returns(prices)
simple_r = hfm.simple_returns(prices)
print("Log returns:", log_r)
print("Simple returns:", simple_r)

# --- Realized measures ---
np.random.seed(42)
intraday = np.random.normal(0, 0.01, 500)
rv = hfm.realized_variance(intraday)
rvol = hfm.realized_volatility(intraday)
bv = hfm.bipower_variation(intraday)
print(f"\nRV = {rv:.6f}, RVol = {rvol:.6f}, BV = {bv:.6f}")

rm = hfm.compute_realized_measures(intraday)
print(f"Jump statistic = {rm.jump_statistic:.4f}")

# --- OLS ---
n = 200
x = np.linspace(0, 1, n)
X = np.column_stack([np.ones(n), x, x**2])
y = 1.0 + 2.0 * x - 0.5 * x**2 + np.random.normal(0, 0.05, n)

res = hfm.ols(y, X, covariance="newey_west", hac_lag=5)
print("\n" + res.summary())

# --- Rolling OLS ---
roll = hfm.rolling_ols(y, X, window=50, step=5)
print(f"Rolling OLS: {roll.n_windows} windows")
print(f"First window betas: {roll.betas[0]}")
print(f"Last window betas:  {roll.betas[-1]}")

# --- AR model ---
ar_data = np.zeros(500)
for t in range(1, 500):
    ar_data[t] = 0.3 + 0.6 * ar_data[t-1] + np.random.normal(0, 0.5)
ar_res = hfm.ar(ar_data, p=1)
print(f"\nAR(1) coef: intercept={ar_res.coefficients[0]:.4f}, phi={ar_res.coefficients[1]:.4f}")
print(f"AIC = {ar_res.aic:.2f}, BIC = {ar_res.bic:.2f}")

# --- VAR model ---
T = 300
Y_var = np.zeros((T, 2))
for t in range(1, T):
    Y_var[t, 0] = 0.5 * Y_var[t-1, 0] + 0.1 * Y_var[t-1, 1] + np.random.normal(0, 0.3)
    Y_var[t, 1] = 0.2 * Y_var[t-1, 0] + 0.4 * Y_var[t-1, 1] + np.random.normal(0, 0.3)
var_res = hfm.var(Y_var, p=1)
print(f"\nVAR(1) coefficient matrix:\n{var_res.coefficients}")
print(f"Residual covariance:\n{var_res.sigma_u}")

# --- Panel fixed effects ---
n_firms, n_periods = 20, 50
N = n_firms * n_periods
y_panel = np.random.randn(N)
X_panel = np.random.randn(N, 2)
entity_ids = np.repeat(np.arange(n_firms, dtype=np.int64), n_periods)
time_ids = np.tile(np.arange(n_periods, dtype=np.int64), n_firms)
panel_res = hfm.fixed_effects(y_panel, X_panel, entity_ids, time_ids,
                                cluster_entity=True, cluster_time=True)
print("\n" + panel_res.summary())

# --- Bootstrap ---
data = np.random.normal(5.0, 2.0, 100)
def mean_stat(x):
    return np.array([np.mean(x)])
boot_res = hfm.bootstrap(data, mean_stat, n_bootstrap=2000, seed=123)
print(f"\nBootstrap mean: {boot_res.mean[0]:.4f} ± {boot_res.std_error[0]:.4f}")
print(f"95% CI: [{boot_res.ci_lower[0]:.4f}, {boot_res.ci_upper[0]:.4f}]")

print("\nDone!")
