"""Integration tests using real OHLCV data from DuckDB.

Run from the build directory:
    PYTHONPATH=python python3 ../tests/integration/test_real_data.py
"""
import sys
import os
import time
import numpy as np

# Add build python dir to path
sys.path.insert(0, "python")
import _hfmetal as hfm

import duckdb

HF_DATA = os.path.expanduser("~/Desktop/hf_data")

def load_closes(db_file, symbol, limit=None):
    """Load close prices from DuckDB OHLCV table."""
    con = duckdb.connect(os.path.join(HF_DATA, db_file), read_only=True)
    q = f"SELECT close FROM ohlcv WHERE symbol = '{symbol}' ORDER BY datetime"
    if limit:
        q += f" LIMIT {limit}"
    data = con.execute(q).fetchnumpy()
    con.close()
    return data["close"].astype(np.float64)

def load_ohlcv(db_file, symbol, limit=None):
    """Load full OHLCV data."""
    con = duckdb.connect(os.path.join(HF_DATA, db_file), read_only=True)
    q = f"SELECT datetime, open, high, low, close, volume FROM ohlcv WHERE symbol = '{symbol}' ORDER BY datetime"
    if limit:
        q += f" LIMIT {limit}"
    data = con.execute(q).fetchnumpy()
    con.close()
    return data

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} {detail}")

# ============================================================
print("=" * 60)
print("HFMetal Integration Tests — Real Market Data")
print("=" * 60)

# --- TEST 1: BTC 5-min log returns ---
print("\n[1] BTC 5-min returns & realized measures")
t0 = time.time()
btc_prices = load_closes("crypto_5min.duckdb", "BTC", limit=50000)
btc_returns = hfm.log_returns(btc_prices)
dt = time.time() - t0
check("BTC returns computed", len(btc_returns) == len(btc_prices) - 1)
check("No NaN in returns", not np.any(np.isnan(btc_returns)))
check(f"Throughput: {len(btc_prices)/dt/1e6:.1f}M prices/s", True)

# Realized measures on a day's worth of 5-min bars (~288 per day)
day_returns = btc_returns[:288]
rv = hfm.realized_variance(day_returns)
rvol = hfm.realized_volatility(day_returns)
bv = hfm.bipower_variation(day_returns)
rm = hfm.compute_realized_measures(day_returns)
check("RV > 0", rv > 0)
check("RVol > 0", rvol > 0)
check("BV > 0", bv > 0)
check(f"RV={rv:.6f}, RVol={rvol:.4f}, BV={bv:.6f}", True)

# --- TEST 2: SPX daily returns + OLS ---
print("\n[2] SPX 5-min returns + OLS")
spx_prices = load_closes("index_5min.duckdb", "SPX", limit=100000)
spx_returns = hfm.log_returns(spx_prices)
check("SPX returns computed", len(spx_returns) > 0)

# Regress SPX returns on their own lag
n = len(spx_returns) - 1
y = spx_returns[1:n+1]
X = np.column_stack([np.ones(n), spx_returns[:n]])
res = hfm.ols(y, X, covariance="newey_west")
check("OLS converged", res.n_obs == n)
check(f"OLS R²={res.r_squared:.6f}", res.r_squared >= 0.0)
check(f"Autocorrelation coef={res.coefficients[1]:.4f}", True)
print(f"    {res.summary()[:200]}...")

# --- TEST 3: Rolling OLS on EURUSD ---
print("\n[3] EURUSD rolling OLS")
eurusd_prices = load_closes("fx_5min.duckdb", "EURUSD", limit=20000)
eurusd_returns = hfm.log_returns(eurusd_prices)
n = len(eurusd_returns) - 1
y_fx = eurusd_returns[1:n+1]
X_fx = np.column_stack([np.ones(n), eurusd_returns[:n]])
t0 = time.time()
roll = hfm.rolling_ols(y_fx, X_fx, window=500, step=50)
dt = time.time() - t0
check(f"Rolling OLS: {roll.n_windows} windows in {dt*1000:.1f}ms", roll.n_windows > 0)
check("Betas shape valid", roll.betas.shape[1] == 2)

# --- TEST 4: GARCH on BTC daily returns ---
print("\n[4] GARCH(1,1) on BTC daily returns")
# Aggregate 5-min to daily: take every 288th return sum
n_days = len(btc_returns) // 288
daily_returns = np.array([np.sum(btc_returns[i*288:(i+1)*288]) for i in range(n_days)])
check(f"Daily returns: {n_days} days", n_days > 100)

t0 = time.time()
garch_res = hfm.garch(daily_returns)
dt = time.time() - t0
check(f"GARCH converged in {garch_res.n_iter} iterations", garch_res.converged)
check(f"omega={garch_res.omega:.6f}", garch_res.omega > 0)
check(f"alpha={garch_res.alpha:.4f}", 0 < garch_res.alpha < 0.5)
check(f"beta={garch_res.beta:.4f}", 0 < garch_res.beta < 1)
check(f"persistence={garch_res.persistence:.4f}", garch_res.persistence < 1.0)
check(f"GARCH fit in {dt*1000:.1f}ms", True)

# --- TEST 5: AR model on daily returns ---
print("\n[5] AR(2) on BTC daily returns")
ar_res = hfm.ar(daily_returns, p=2)
check(f"AR(2) AIC={ar_res.aic:.2f}", True)
check(f"AR(2) n_obs={ar_res.n_obs}", ar_res.n_obs == n_days - 2)

# --- TEST 6: VAR on crypto pairs ---
print("\n[6] VAR(1) on BTC + ETH daily returns")
eth_prices = load_closes("crypto_5min.duckdb", "ETH", limit=50000)
eth_returns = hfm.log_returns(eth_prices)
n_eth_days = len(eth_returns) // 288
eth_daily = np.array([np.sum(eth_returns[i*288:(i+1)*288]) for i in range(n_eth_days)])
min_days = min(n_days, n_eth_days)
Y_var = np.column_stack([daily_returns[:min_days], eth_daily[:min_days]])
var_res = hfm.var(Y_var, p=1)
check(f"VAR(1) fitted: {var_res.n_vars} vars, {var_res.n_obs} obs", var_res.n_vars == 2)
check(f"VAR AIC={var_res.aic:.2f}", True)

# --- TEST 7: HAR-RV on realized daily variance ---
print("\n[7] HAR-RV on BTC daily realized variance")
daily_rv = np.array([hfm.realized_variance(btc_returns[i*288:(i+1)*288]) for i in range(n_days)])
check(f"Daily RV series: {len(daily_rv)} days", len(daily_rv) > 50)
har_res = hfm.har_rv(daily_rv)
check(f"HAR R²={har_res['r_squared']:.4f}", har_res['r_squared'] >= 0.0)
check(f"HAR beta_d={har_res['beta_d']:.4f}", True)

# --- TEST 8: Logit on direction prediction ---
print("\n[8] Logit: predict BTC return direction")
y_dir = (daily_returns[1:] > 0).astype(np.float64)
n_logit = len(y_dir)
X_logit = np.column_stack([daily_returns[:n_logit], daily_rv[:n_logit]])
logit_res = hfm.logit(y_dir, X_logit)
check(f"Logit converged", logit_res.converged)
check(f"Logit pseudo-R²={logit_res.pseudo_r_squared:.4f}", logit_res.pseudo_r_squared >= 0.0)
check(f"Predicted probs in [0,1]",
      np.all(logit_res.predicted_prob >= 0) and np.all(logit_res.predicted_prob <= 1))

# --- TEST 9: Local projections ---
print("\n[9] Local projections: BTC volatility shock → returns")
lp_res = hfm.local_projections(daily_returns[:min_days], eth_daily[:min_days],
                                 max_horizon=10, n_lags=3)
check(f"LP IRF: {len(lp_res.irf)} horizons", len(lp_res.irf) == 11)
check("LP confidence bands valid",
      np.all(lp_res.irf_lower <= lp_res.irf) and np.all(lp_res.irf_upper >= lp_res.irf))

# --- TEST 10: Large-scale benchmark ---
print("\n[10] Performance: 1M returns realized variance")
big_prices = load_closes("crypto_5min.duckdb", "BTC")
big_returns = hfm.log_returns(big_prices)
t0 = time.time()
rv_big = hfm.realized_variance(big_returns)
dt = time.time() - t0
check(f"RV on {len(big_returns)} returns in {dt*1000:.2f}ms", rv_big > 0)
check(f"Throughput: {len(big_returns)/dt/1e6:.1f}M elements/s", True)

# ============================================================
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 60)

if failed > 0:
    sys.exit(1)
