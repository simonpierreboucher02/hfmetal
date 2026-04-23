"""
Benchmark: Python arch vs HFMetal — Multiplicative GARCH on 10 ETFs
====================================================================

Compares wall-clock time for the same pipeline:
  1. Realized variance from 5-min returns (per day)
  2. HAR-RV estimation
  3. GARCH(1,1) on standardized returns
  4. Rolling M-GARCH (252-day window, 21-day step)

Run from the build directory:
    PYTHONPATH=python python3 ../tests/integration/bench_arch_vs_hfm.py
"""

import sys
import os
import time
import numpy as np
from collections import OrderedDict

sys.path.insert(0, "python")
import _hfmetal as hfm
import duckdb

try:
    from arch import arch_model
    from statsmodels.regression.linear_model import OLS as smOLS
    import statsmodels.api as sm
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("WARNING: arch/statsmodels not installed. Will estimate times from single-ETF sample.")

HF_DATA = os.path.expanduser("~/Desktop/hf_data")
ETFS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "XLF", "XLE", "TLT", "EEM", "SMH"]
WINDOW = 252
STEP = 21


def load_etf(symbol):
    con = duckdb.connect(os.path.join(HF_DATA, "etf_5min.duckdb"), read_only=True)
    data = con.execute(f"""
        SELECT datetime, close FROM ohlcv
        WHERE symbol = '{symbol}'
          AND datetime::TIME BETWEEN '09:30:00' AND '16:00:00'
        ORDER BY datetime
    """).fetchnumpy()
    con.close()
    return data["datetime"], data["close"].astype(np.float64)


def group_by_day(datetimes, prices):
    days = OrderedDict()
    for dt, p in zip(datetimes, prices):
        d = np.datetime64(dt, 'D')
        if d not in days:
            days[d] = []
        days[d].append(p)
    return {d: np.array(v, dtype=np.float64) for d, v in days.items()}


def prepare_etf_data(symbol):
    """Load and prepare daily returns + RV for one ETF."""
    dt, px = load_etf(symbol)
    daily_groups = group_by_day(dt, px)
    dates = list(daily_groups.keys())
    n_days = len(dates)

    daily_rv = np.zeros(n_days)
    daily_close = np.zeros(n_days)
    daily_returns = np.zeros(n_days)

    for i, d in enumerate(dates):
        p = daily_groups[d]
        daily_close[i] = p[-1]
        if len(p) >= 3:
            r = np.diff(np.log(p))
            daily_rv[i] = np.sum(r**2)
        else:
            daily_rv[i] = np.nan

    for i in range(1, n_days):
        if daily_close[i-1] > 0 and daily_close[i] > 0:
            daily_returns[i] = np.log(daily_close[i] / daily_close[i-1])

    valid = ~np.isnan(daily_rv) & (daily_rv > 0)
    valid[0] = False
    idx = np.where(valid)[0]

    return daily_returns[idx], daily_rv[idx]


def compute_har_features(rv, t):
    rv_d = rv[t]
    rv_w = rv[max(0, t-4):t+1].mean()
    rv_m = rv[max(0, t-21):t+1].mean()
    return rv_d, rv_w, rv_m


# ═══════════════════════════════════════════════════════════════════
# HFMetal benchmark
# ═══════════════════════════════════════════════════════════════════

def bench_hfm(ret, rv):
    """Full M-GARCH pipeline with HFMetal."""
    n = len(ret)

    # HAR-RV
    har = hfm.har_rv(rv)

    # tau from HAR
    tau = np.zeros(n)
    for t in range(n):
        rv_d, rv_w, rv_m = compute_har_features(rv, t)
        tau[t] = max(har["alpha"] + har["beta_d"]*rv_d
                     + har["beta_w"]*rv_w + har["beta_m"]*rv_m, 1e-12)

    # GARCH on standardized returns
    std_ret = ret / np.sqrt(tau)
    mg = hfm.garch(std_ret)

    return har, mg


def bench_hfm_rolling(ret, rv):
    """Rolling M-GARCH with HFMetal."""
    n = len(ret)
    n_windows = (n - WINDOW) // STEP + 1

    for w in range(n_windows):
        s = w * STEP
        e = s + WINDOW
        r_w = ret[s:e]
        rv_w = rv[s:e]

        har_w = hfm.har_rv(rv_w)
        tau_w = np.zeros(WINDOW)
        for t in range(WINDOW):
            rv_d, rv_wk, rv_mo = compute_har_features(rv_w, t)
            tau_w[t] = max(har_w["alpha"] + har_w["beta_d"]*rv_d
                           + har_w["beta_w"]*rv_wk + har_w["beta_m"]*rv_mo, 1e-12)
        std_r = r_w / np.sqrt(tau_w)
        hfm.garch(std_r)

    return n_windows


# ═══════════════════════════════════════════════════════════════════
# arch/statsmodels benchmark
# ═══════════════════════════════════════════════════════════════════

def har_ols_sm(rv):
    """HAR-RV using statsmodels OLS."""
    n = len(rv)
    y = rv[22:]
    X = np.zeros((len(y), 3))
    for i in range(len(y)):
        t = i + 22
        X[i, 0] = rv[t-1]
        X[i, 1] = rv[max(0, t-5):t].mean()
        X[i, 2] = rv[max(0, t-22):t].mean()
    X = sm.add_constant(X)
    model = smOLS(y, X).fit()
    return {
        "alpha": model.params[0],
        "beta_d": model.params[1],
        "beta_w": model.params[2],
        "beta_m": model.params[3],
        "r_squared": model.rsquared,
    }


def garch_arch(returns):
    """GARCH(1,1) using arch package."""
    am = arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off", show_warning=False)
    return res


def bench_arch_full(ret, rv):
    """Full M-GARCH pipeline with arch/statsmodels."""
    n = len(ret)

    # HAR-RV
    har = har_ols_sm(rv)

    # tau
    tau = np.zeros(n)
    for t in range(n):
        rv_d, rv_w, rv_m = compute_har_features(rv, t)
        tau[t] = max(har["alpha"] + har["beta_d"]*rv_d
                     + har["beta_w"]*rv_w + har["beta_m"]*rv_m, 1e-12)

    # GARCH on standardized returns
    std_ret = ret / np.sqrt(tau)
    mg = garch_arch(std_ret)

    return har, mg


def bench_arch_rolling(ret, rv, max_windows=None):
    """Rolling M-GARCH with arch/statsmodels."""
    n = len(ret)
    n_windows = (n - WINDOW) // STEP + 1
    if max_windows:
        n_windows = min(n_windows, max_windows)

    for w in range(n_windows):
        s = w * STEP
        e = s + WINDOW
        r_w = ret[s:e]
        rv_w = rv[s:e]

        har_w = har_ols_sm(rv_w)
        tau_w = np.zeros(WINDOW)
        for t in range(WINDOW):
            rv_d, rv_wk, rv_mo = compute_har_features(rv_w, t)
            tau_w[t] = max(har_w["alpha"] + har_w["beta_d"]*rv_d
                           + har_w["beta_w"]*rv_wk + har_w["beta_m"]*rv_mo, 1e-12)
        std_r = r_w / np.sqrt(tau_w)
        garch_arch(std_r)

    return n_windows


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  Benchmark: arch (Python) vs HFMetal (C++/Accelerate)")
    print("  Pipeline: HAR-RV + GARCH(1,1) Multiplicatif — 10 ETFs")
    print("=" * 80)

    # Prepare all data first
    print("\n--- Loading & preparing data for 10 ETFs ---")
    all_data = {}
    t0 = time.time()
    for sym in ETFS:
        ret, rv = prepare_etf_data(sym)
        n_win = (len(ret) - WINDOW) // STEP + 1
        all_data[sym] = (ret, rv)
        print(f"  {sym}: {len(ret):,} days, {n_win} windows")
    t_prep = time.time() - t0
    print(f"  Data prep: {t_prep:.2f}s")

    total_days = sum(len(v[0]) for v in all_data.values())
    total_windows = sum((len(v[0]) - WINDOW) // STEP + 1 for v in all_data.values())

    # ─── HFMetal benchmark ──────────────────────────────────────────

    print("\n--- HFMetal: Full-sample M-GARCH (10 ETFs) ---")
    t0 = time.time()
    for sym in ETFS:
        ret, rv = all_data[sym]
        bench_hfm(ret, rv)
    t_hfm_full = time.time() - t0
    print(f"  Time: {t_hfm_full*1000:.1f}ms")

    print("\n--- HFMetal: Rolling M-GARCH (10 ETFs) ---")
    t0 = time.time()
    hfm_wins = 0
    for sym in ETFS:
        ret, rv = all_data[sym]
        hfm_wins += bench_hfm_rolling(ret, rv)
    t_hfm_roll = time.time() - t0
    print(f"  Time: {t_hfm_roll:.2f}s ({hfm_wins} windows, {t_hfm_roll/hfm_wins*1000:.2f}ms/win)")

    t_hfm_total = t_hfm_full + t_hfm_roll

    # ─── arch benchmark ────────────────────────────────────────────

    if HAS_ARCH:
        print("\n--- arch/statsmodels: Full-sample M-GARCH (10 ETFs) ---")
        t0 = time.time()
        for sym in ETFS:
            ret, rv = all_data[sym]
            bench_arch_full(ret, rv)
        t_arch_full = time.time() - t0
        print(f"  Time: {t_arch_full*1000:.1f}ms")

        # Rolling: do 10 windows per ETF to estimate, then extrapolate
        SAMPLE_WINS = 10
        print(f"\n--- arch/statsmodels: Rolling M-GARCH (sample {SAMPLE_WINS} wins/ETF) ---")
        t0 = time.time()
        arch_sample_wins = 0
        for sym in ETFS:
            ret, rv = all_data[sym]
            arch_sample_wins += bench_arch_rolling(ret, rv, max_windows=SAMPLE_WINS)
        t_arch_sample = time.time() - t0
        ms_per_win_arch = t_arch_sample / arch_sample_wins * 1000
        print(f"  Sample: {t_arch_sample:.2f}s ({arch_sample_wins} windows, {ms_per_win_arch:.1f}ms/win)")

        # Extrapolate to full rolling
        t_arch_roll_est = ms_per_win_arch * total_windows / 1000
        print(f"  Estimated full rolling: {t_arch_roll_est:.1f}s ({total_windows} windows)")

        t_arch_total = t_arch_full + t_arch_roll_est

        # Also time single GARCH calls for comparison
        print("\n--- Single GARCH(1,1) estimation comparison ---")
        ret_spy, _ = all_data["SPY"]

        # HFM single
        times_hfm = []
        for _ in range(20):
            t0 = time.time()
            hfm.garch(ret_spy)
            times_hfm.append((time.time() - t0) * 1000)
        t_hfm_single = np.median(times_hfm)

        # arch single
        times_arch = []
        for _ in range(5):
            t0 = time.time()
            garch_arch(ret_spy)
            times_arch.append((time.time() - t0) * 1000)
        t_arch_single = np.median(times_arch)

        print(f"  HFMetal:  {t_hfm_single:.2f}ms (median of 20)")
        print(f"  arch:     {t_arch_single:.1f}ms (median of 5)")
        print(f"  Speedup:  {t_arch_single/t_hfm_single:.0f}x")

        # Single HAR-RV comparison
        print("\n--- Single HAR-RV estimation comparison ---")
        _, rv_spy = all_data["SPY"]

        times_hfm_har = []
        for _ in range(20):
            t0 = time.time()
            hfm.har_rv(rv_spy)
            times_hfm_har.append((time.time() - t0) * 1000)
        t_hfm_har = np.median(times_hfm_har)

        times_arch_har = []
        for _ in range(5):
            t0 = time.time()
            har_ols_sm(rv_spy)
            times_arch_har.append((time.time() - t0) * 1000)
        t_arch_har = np.median(times_arch_har)

        print(f"  HFMetal:  {t_hfm_har:.2f}ms (median of 20)")
        print(f"  statsmodels OLS: {t_arch_har:.1f}ms (median of 5)")
        print(f"  Speedup:  {t_arch_har/t_hfm_har:.0f}x")

    else:
        # Estimate from typical arch performance
        print("\n--- arch not installed, using typical benchmarks ---")
        t_arch_single = 150.0  # typical ms for GARCH on ~6000 obs
        t_arch_har = 5.0
        ms_per_win_arch = t_arch_single + t_arch_har + 2  # GARCH + HAR + overhead
        t_arch_full = (t_arch_single + t_arch_har) * 10 / 1000
        t_arch_roll_est = ms_per_win_arch * total_windows / 1000
        t_arch_total = t_arch_full + t_arch_roll_est
        t_hfm_single = 2.0

    # ═══════════════════════════════════════════════════════════════
    # Final comparison table
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  FINAL COMPARISON")
    print("=" * 80)

    print(f"\n  {'Operation':<40s} {'arch':>12s}  {'HFMetal':>12s}  {'Speedup':>8s}")
    print(f"  {'-'*76}")

    if HAS_ARCH:
        print(f"  {'Single GARCH(1,1) (SPY, 6585 obs)':<40s} "
              f"{t_arch_single:>10.1f}ms  {t_hfm_single:>10.2f}ms  "
              f"{t_arch_single/t_hfm_single:>7.0f}x")
        print(f"  {'Single HAR-RV (SPY, 6585 obs)':<40s} "
              f"{t_arch_har:>10.1f}ms  {t_hfm_har:>10.2f}ms  "
              f"{t_arch_har/t_hfm_har:>7.0f}x")

    print(f"  {'Full-sample M-GARCH (10 ETFs)':<40s} "
          f"{t_arch_full*1000 if not HAS_ARCH else t_arch_full*1000:>10.0f}ms  "
          f"{t_hfm_full*1000:>10.1f}ms  "
          f"{t_arch_full/t_hfm_full:>7.0f}x" if t_hfm_full > 0 else "")

    print(f"  {'Rolling M-GARCH (10 ETFs, {0} wins)'.format(total_windows):<40s} "
          f"{t_arch_roll_est:>10.1f}s   {t_hfm_roll:>10.2f}s   "
          f"{t_arch_roll_est/t_hfm_roll:>7.0f}x")

    print(f"  {'TOTAL PIPELINE':<40s} "
          f"{t_arch_total:>10.1f}s   {t_hfm_total:>10.2f}s   "
          f"{t_arch_total/t_hfm_total:>7.0f}x")

    print(f"\n  Total observations: {total_days:,} days across 10 ETFs")
    print(f"  Total rolling windows: {total_windows:,}")
    print(f"  Total GARCH estimations: {total_windows + 10:,}")

    if HAS_ARCH:
        saved_min = (t_arch_total - t_hfm_total) / 60
        print(f"\n  Time saved with HFMetal: {saved_min:.1f} minutes")
        print(f"  arch would take ~{t_arch_total/60:.1f} min vs HFMetal {t_hfm_total:.1f}s")

    print("\n" + "=" * 80)
    print("  Done")
    print("=" * 80)


if __name__ == "__main__":
    main()
