"""
Rolling-Window Multiplicative GARCH on REAL SPY ETF data
========================================================

Model:  sigma²_t = tau_t × g_t
  tau_t : HAR-RV on daily realized variance  (slow / intraday component)
  g_t   : GARCH(1,1) on r_t / sqrt(tau_t)   (fast / daily component)

Data sources (DuckDB):
  - etf_5min.duckdb  ->  SPY 5-min OHLCV  (~1M bars, 2000-2026)
  - etf_1min.duckdb  ->  SPY 1-min OHLCV  (~4.5M bars, 2000-2026)

Run from the build directory:
    PYTHONPATH=python python3 ../tests/integration/test_spy_mgarch.py
"""

import sys
import os
import time
import numpy as np
from collections import OrderedDict

sys.path.insert(0, "python")
import _hfmetal as hfm
import duckdb

HF_DATA = os.path.expanduser("~/Desktop/hf_data")


def load_spy_5min(limit=None):
    con = duckdb.connect(os.path.join(HF_DATA, "etf_5min.duckdb"), read_only=True)
    q = """
        SELECT datetime, close
        FROM ohlcv
        WHERE symbol = 'SPY'
          AND datetime::TIME BETWEEN '09:30:00' AND '16:00:00'
        ORDER BY datetime
    """
    if limit:
        q += f" LIMIT {limit}"
    data = con.execute(q).fetchnumpy()
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


def print_header(title):
    print(f"\n{'='*76}")
    print(f"  {title}")
    print(f"{'='*76}")


def print_section(title):
    print(f"\n--- {title} {'-'*(60 - len(title))}")


def main():
    t_total = time.time()

    print_header("SPY Multiplicative GARCH -- Real Data Analysis")
    print("  Model: sigma^2_t = tau_t * g_t")
    print("  tau_t : HAR-RV (intraday realized variance -> slow component)")
    print("  g_t   : GARCH(1,1) on standardized returns (fast component)")

    # == 1. Load 5-min SPY data ==

    print_section("Loading SPY 5-min data")
    t0 = time.time()
    dt_5m, px_5m = load_spy_5min()
    t_load = time.time() - t0
    print(f"  Loaded {len(px_5m):,} bars in {t_load:.2f}s")
    print(f"  Period: {dt_5m[0]} -> {dt_5m[-1]}")
    print(f"  Price range: ${px_5m.min():.2f} - ${px_5m.max():.2f}")

    # == 2. Group by day & compute intraday returns ==

    print_section("Grouping by day & computing intraday log returns")
    t0 = time.time()
    daily_groups = group_by_day(dt_5m, px_5m)
    dates = list(daily_groups.keys())
    n_days = len(dates)
    print(f"  Trading days: {n_days:,}")

    daily_rv = np.zeros(n_days)
    daily_rvol = np.zeros(n_days)
    daily_bv = np.zeros(n_days)
    daily_jump = np.zeros(n_days)
    daily_close = np.zeros(n_days)
    daily_returns = np.zeros(n_days)
    bars_per_day = np.zeros(n_days, dtype=int)

    for i, d in enumerate(dates):
        prices_day = daily_groups[d]
        bars_per_day[i] = len(prices_day)
        daily_close[i] = prices_day[-1]

        if len(prices_day) >= 3:
            intra_ret = hfm.log_returns(prices_day)
            rm = hfm.compute_realized_measures(intra_ret)
            daily_rv[i] = rm.realized_variance
            daily_rvol[i] = rm.realized_volatility
            daily_bv[i] = rm.bipower_variation
            daily_jump[i] = rm.jump_statistic
        else:
            daily_rv[i] = np.nan
            daily_rvol[i] = np.nan
            daily_bv[i] = np.nan
            daily_jump[i] = np.nan

    for i in range(1, n_days):
        if daily_close[i - 1] > 0 and daily_close[i] > 0:
            daily_returns[i] = np.log(daily_close[i] / daily_close[i - 1])

    t_proc = time.time() - t0
    print(f"  Processed in {t_proc:.2f}s")
    print(f"  Avg bars/day: {bars_per_day.mean():.1f}")
    print(f"  Min/Max bars: {bars_per_day.min()} / {bars_per_day.max()}")

    valid = ~np.isnan(daily_rv) & ~np.isnan(daily_returns) & (daily_rv > 0)
    valid[0] = False
    idx = np.where(valid)[0]

    ret = daily_returns[idx]
    rv = daily_rv[idx]
    rvol = daily_rvol[idx]
    bv = daily_bv[idx]
    jump = daily_jump[idx]
    valid_dates = [dates[i] for i in idx]
    n = len(ret)

    print(f"  Valid observations: {n:,}")

    # == 3. Descriptive statistics ==

    print_section("Descriptive Statistics")

    def stats_line(name, arr):
        return (f"  {name:<16s} "
                f"mean={arr.mean():>12.8f}  "
                f"std={arr.std():>12.8f}  "
                f"min={arr.min():>12.8f}  "
                f"med={np.median(arr):>12.8f}  "
                f"max={arr.max():>12.8f}")

    print(stats_line("Daily return", ret))
    print(stats_line("Realized var", rv))
    print(stats_line("Realized vol", rvol))
    print(stats_line("Bipower var", bv))
    print(stats_line("Jump stat", jump))

    ann_rvol = np.sqrt(rv.mean() * 252) * 100
    print(f"\n  Annualized RVol (from mean RV): {ann_rvol:.2f}%")
    print(f"  Daily return (ann):             {ret.mean()*252*100:.2f}%")
    print(f"  Sharpe (RV-based):              {ret.mean()/np.sqrt(rv.mean())*np.sqrt(252):.2f}")

    # == 4. Full-sample Standard GARCH(1,1) ==

    print_section("Full-Sample Standard GARCH(1,1)")
    t0 = time.time()
    std_garch = hfm.garch(ret)
    t_garch = time.time() - t0

    print(f"  omega      = {std_garch.omega:.8f}")
    print(f"  alpha      = {std_garch.alpha:.6f}")
    print(f"  beta       = {std_garch.beta:.6f}")
    print(f"  persistence= {std_garch.persistence:.6f}")
    print(f"  uncond var = {std_garch.unconditional_var:.8f}")
    print(f"  LogLik     = {std_garch.log_likelihood:.4f}")
    print(f"  AIC        = {std_garch.aic:.4f}")
    print(f"  BIC        = {std_garch.bic:.4f}")
    print(f"  converged  = {std_garch.converged}")
    print(f"  time       = {t_garch*1000:.1f}ms ({n:,} obs)")

    ann_vol_garch = np.sqrt(std_garch.unconditional_var * 252) * 100
    print(f"  Ann. uncond vol = {ann_vol_garch:.2f}%")

    # == 5. Full-sample HAR-RV ==

    print_section("Full-Sample HAR-RV (tau_t component)")
    t0 = time.time()
    har = hfm.har_rv(rv)
    t_har = time.time() - t0

    print(f"  alpha   = {har['alpha']:.8f}")
    print(f"  beta_d  = {har['beta_d']:.6f}  (daily RV)")
    print(f"  beta_w  = {har['beta_w']:.6f}  (weekly avg RV)")
    print(f"  beta_m  = {har['beta_m']:.6f}  (monthly avg RV)")
    print(f"  R^2     = {har['r_squared']:.6f}")
    print(f"  time    = {t_har*1000:.1f}ms")

    # == 6. Full-sample Multiplicative GARCH ==

    print_section("Full-Sample Multiplicative GARCH")
    t0 = time.time()

    tau = np.full(n, rv.mean())
    for t in range(n):
        rv_d = rv[t]
        rv_w = rv[max(0, t-4):t+1].mean()
        rv_m = rv[max(0, t-21):t+1].mean()
        fitted = har['alpha'] + har['beta_d']*rv_d + har['beta_w']*rv_w + har['beta_m']*rv_m
        tau[t] = max(fitted, 1e-12)

    std_ret = ret / np.sqrt(tau)

    mg_garch = hfm.garch(std_ret)
    g = np.array(mg_garch.conditional_var)

    total_var = tau * g
    total_vol = np.sqrt(total_var)

    t_mg = time.time() - t0

    print(f"\n  HAR-RV (slow component tau_t):")
    print(f"    R^2     = {har['r_squared']:.6f}")
    print(f"    beta_d  = {har['beta_d']:.6f}")
    print(f"    beta_w  = {har['beta_w']:.6f}")
    print(f"    beta_m  = {har['beta_m']:.6f}")
    print(f"\n  GARCH(1,1) on standardized returns (fast component g_t):")
    print(f"    omega   = {mg_garch.omega:.8f}")
    print(f"    alpha   = {mg_garch.alpha:.6f}")
    print(f"    beta    = {mg_garch.beta:.6f}")
    print(f"    persist = {mg_garch.persistence:.6f}")
    print(f"    LogLik  = {mg_garch.log_likelihood:.4f}")
    print(f"    AIC     = {mg_garch.aic:.4f}")
    print(f"    converged = {mg_garch.converged}")
    print(f"\n  Total time = {t_mg*1000:.1f}ms")

    # == 7. Comparison table ==

    print_section("Comparison: Standard GARCH vs Multiplicative GARCH")
    print(f"  {'':30s} {'Std GARCH':>14s}  {'M-GARCH (g_t)':>14s}")
    print(f"  {'-'*62}")
    print(f"  {'omega':30s} {std_garch.omega:>14.8f}  {mg_garch.omega:>14.8f}")
    print(f"  {'alpha':30s} {std_garch.alpha:>14.6f}  {mg_garch.alpha:>14.6f}")
    print(f"  {'beta':30s} {std_garch.beta:>14.6f}  {mg_garch.beta:>14.6f}")
    print(f"  {'persistence (alpha+beta)':30s} {std_garch.persistence:>14.6f}  {mg_garch.persistence:>14.6f}")
    print(f"  {'LogLik':30s} {std_garch.log_likelihood:>14.4f}  {mg_garch.log_likelihood:>14.4f}")
    print(f"  {'AIC':30s} {std_garch.aic:>14.4f}  {mg_garch.aic:>14.4f}")
    print()
    print(f"  HAR-RV R^2 = {har['r_squared']:.4f} -> intraday info explains "
          f"{har['r_squared']*100:.1f}% of daily variance")
    print(f"  Persistence drops from {std_garch.persistence:.4f} -> "
          f"{mg_garch.persistence:.4f}")
    print(f"  -> M-GARCH separates slow (intraday RV) and fast (GARCH) dynamics")

    # == 8. Rolling window M-GARCH ==

    WINDOW = 252
    STEP = 21

    print_section(f"Rolling Window M-GARCH (window={WINDOW}, step={STEP})")

    n_windows = (n - WINDOW) // STEP + 1
    print(f"  Windows: {n_windows}")

    roll_har_r2 = np.zeros(n_windows)
    roll_har_bd = np.zeros(n_windows)
    roll_har_bw = np.zeros(n_windows)
    roll_har_bm = np.zeros(n_windows)
    roll_g_omega = np.zeros(n_windows)
    roll_g_alpha = np.zeros(n_windows)
    roll_g_beta = np.zeros(n_windows)
    roll_g_persist = np.zeros(n_windows)
    roll_g_aic = np.zeros(n_windows)
    roll_converged = np.zeros(n_windows, dtype=bool)
    roll_vol_fcast = np.zeros(n_windows)
    roll_start_date = []
    roll_end_date = []

    t0 = time.time()

    for w in range(n_windows):
        s = w * STEP
        e = s + WINDOW
        r_w = ret[s:e]
        rv_w = rv[s:e]

        try:
            har_w = hfm.har_rv(rv_w)
            roll_har_r2[w] = har_w['r_squared']
            roll_har_bd[w] = har_w['beta_d']
            roll_har_bw[w] = har_w['beta_w']
            roll_har_bm[w] = har_w['beta_m']

            tau_w = np.zeros(WINDOW)
            for t in range(WINDOW):
                rv_d = rv_w[t]
                rv_wk = rv_w[max(0, t-4):t+1].mean()
                rv_mo = rv_w[max(0, t-21):t+1].mean()
                fitted = (har_w['alpha'] + har_w['beta_d']*rv_d
                          + har_w['beta_w']*rv_wk + har_w['beta_m']*rv_mo)
                tau_w[t] = max(fitted, 1e-12)
        except Exception:
            tau_w = np.maximum(rv_w, 1e-12)

        std_r = r_w / np.sqrt(tau_w)
        try:
            garch_w = hfm.garch(std_r)
            roll_g_omega[w] = garch_w.omega
            roll_g_alpha[w] = garch_w.alpha
            roll_g_beta[w] = garch_w.beta
            roll_g_persist[w] = garch_w.persistence
            roll_g_aic[w] = garch_w.aic
            roll_converged[w] = garch_w.converged

            last_g = garch_w.conditional_var[-1]
            last_tau = tau_w[-1]
            roll_vol_fcast[w] = np.sqrt(last_tau * last_g * 252) * 100
        except Exception:
            roll_converged[w] = False

        roll_start_date.append(str(valid_dates[s]))
        roll_end_date.append(str(valid_dates[e - 1]))

    t_roll = time.time() - t0
    print(f"  Estimated {n_windows} windows in {t_roll:.2f}s "
          f"({t_roll/n_windows*1000:.1f}ms/window)")
    print(f"  Converged: {roll_converged.sum()}/{n_windows}")

    # == 9. Rolling parameter table ==

    print_section("Rolling Parameter Evolution")
    print(f"\n  {'Win':>4s}  {'Start':>10s}  {'End':>10s}  "
          f"{'HAR_R2':>8s}  {'G_alpha':>8s}  {'G_beta':>8s}  "
          f"{'Persist':>8s}  {'VolF%':>7s}  {'Conv':>4s}")
    print(f"  {'-'*82}")

    for w in range(n_windows):
        print(f"  {w+1:4d}  {roll_start_date[w]:>10s}  {roll_end_date[w]:>10s}  "
              f"{roll_har_r2[w]:8.4f}  {roll_g_alpha[w]:8.5f}  {roll_g_beta[w]:8.5f}  "
              f"{roll_g_persist[w]:8.4f}  {roll_vol_fcast[w]:6.1f}%  "
              f"{'Y' if roll_converged[w] else 'N':>4s}")

    # == 10. Rolling summary statistics ==

    print_section("Rolling Parameter Summary")
    conv = roll_converged
    print(f"\n  {'':18s} {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}")
    print(f"  {'-'*62}")

    def pstat(name, arr):
        a = arr[conv] if conv.sum() > 0 else arr
        print(f"  {name:18s} {a.mean():10.6f}  {a.std():10.6f}  "
              f"{a.min():10.6f}  {a.max():10.6f}")

    pstat("GARCH alpha", roll_g_alpha)
    pstat("GARCH beta", roll_g_beta)
    pstat("Persistence", roll_g_persist)
    pstat("HAR-RV R^2", roll_har_r2)
    pstat("HAR beta_d", roll_har_bd)
    pstat("HAR beta_w", roll_har_bw)
    pstat("HAR beta_m", roll_har_bm)

    vf = roll_vol_fcast[conv] if conv.sum() > 0 else roll_vol_fcast
    print(f"  {'Vol forecast (%)':18s} {vf.mean():9.1f}%  {vf.std():9.1f}%  "
          f"{vf.min():9.1f}%  {vf.max():9.1f}%")

    # == 11. Regime analysis ==

    print_section("Regime Analysis (based on rolling volatility)")

    vf_all = roll_vol_fcast[conv]
    if len(vf_all) > 0:
        low_thresh = np.percentile(vf_all, 25)
        high_thresh = np.percentile(vf_all, 75)

        low_regime = vf_all[vf_all <= low_thresh]
        mid_regime = vf_all[(vf_all > low_thresh) & (vf_all <= high_thresh)]
        high_regime = vf_all[vf_all > high_thresh]

        print(f"  Low vol regime  (<= {low_thresh:.1f}%): "
              f"{len(low_regime)} windows, avg = {low_regime.mean():.1f}%")
        print(f"  Mid vol regime  ({low_thresh:.1f}-{high_thresh:.1f}%): "
              f"{len(mid_regime)} windows, avg = {mid_regime.mean():.1f}%")
        print(f"  High vol regime (> {high_thresh:.1f}%): "
              f"{len(high_regime)} windows, avg = {high_regime.mean():.1f}%")

        persist_conv = roll_g_persist[conv]

        low_mask = vf_all <= low_thresh
        high_mask = vf_all > high_thresh

        print(f"\n  Avg GARCH persistence:")
        print(f"    Low vol:  {persist_conv[low_mask].mean():.4f}")
        print(f"    High vol: {persist_conv[high_mask].mean():.4f}")

    # == 12. Volatility decomposition ==

    print_section("Volatility Decomposition (last 20 days, full sample)")

    n_disp = min(20, len(total_vol))
    print(f"\n  {'Day':>5s}  {'Date':>10s}  {'tau_t':>12s}  {'g_t':>12s}  "
          f"{'sigma^2_t':>12s}  {'AnnVol%':>8s}")
    print(f"  {'-'*68}")

    for i in range(n - n_disp, n):
        ann_v = np.sqrt(total_var[i] * 252) * 100
        print(f"  {i:5d}  {str(valid_dates[i]):>10s}  {tau[i]:12.8f}  {g[i]:12.6f}  "
              f"{total_var[i]:12.8f}  {ann_v:7.1f}%")

    # == 13. One-step-ahead forecast ==

    print_section("One-Step-Ahead Volatility Forecast")

    rv_d_last = rv[-1]
    rv_w_last = rv[-5:].mean()
    rv_m_last = rv[-22:].mean()
    tau_fcast = har['alpha'] + har['beta_d']*rv_d_last + har['beta_w']*rv_w_last + har['beta_m']*rv_m_last
    tau_fcast = max(tau_fcast, 1e-12)

    last_g = g[-1]
    last_std_r = ret[-1] / np.sqrt(tau[-1])
    g_fcast = mg_garch.omega + mg_garch.alpha*last_std_r**2 + mg_garch.beta*last_g

    sigma2_fcast = tau_fcast * g_fcast
    vol_fcast_ann = np.sqrt(sigma2_fcast * 252) * 100

    print(f"  tau_{{T+1}}     = {tau_fcast:.10f}  (HAR-RV)")
    print(f"  g_{{T+1}}       = {g_fcast:.10f}  (GARCH)")
    print(f"  sigma^2_{{T+1}} = {sigma2_fcast:.10f}  (total)")
    print(f"  Annualized   = {vol_fcast_ann:.2f}%")

    # == 14. Bonus: 1-min data ==

    print_section("Bonus: 1-min SPY Realized Measures (last 2 years)")
    t0 = time.time()

    con = duckdb.connect(os.path.join(HF_DATA, "etf_1min.duckdb"), read_only=True)
    data_1m = con.execute("""
        SELECT datetime, close
        FROM etf_1min
        WHERE ticker = 'SPY'
          AND datetime >= '2024-01-01'
          AND datetime::TIME BETWEEN '09:31:00' AND '16:00:00'
        ORDER BY datetime
    """).fetchnumpy()
    con.close()

    dt_1m = data_1m["datetime"]
    px_1m = data_1m["close"].astype(np.float64)
    daily_1m = group_by_day(dt_1m, px_1m)

    print(f"  Loaded {len(px_1m):,} 1-min bars, {len(daily_1m)} days in {time.time()-t0:.2f}s")

    rv_1m = []
    bv_1m = []
    dates_1m = []
    for d, prices in daily_1m.items():
        if len(prices) >= 10:
            intra_r = hfm.log_returns(prices)
            rm = hfm.compute_realized_measures(intra_r)
            rv_1m.append(rm.realized_variance)
            bv_1m.append(rm.bipower_variation)
            dates_1m.append(d)

    rv_1m = np.array(rv_1m)
    bv_1m = np.array(bv_1m)

    print(f"  Valid days: {len(rv_1m)}")
    print(f"  Mean RV:  {rv_1m.mean():.8f}  (ann vol: {np.sqrt(rv_1m.mean()*252)*100:.1f}%)")
    print(f"  Mean BV:  {bv_1m.mean():.8f}")

    if len(rv_1m) > 30:
        har_1m = hfm.har_rv(rv_1m)
        print(f"  HAR-RV R^2:   {har_1m['r_squared']:.4f}")
        print(f"  HAR beta_d:   {har_1m['beta_d']:.4f}")
        print(f"  HAR beta_w:   {har_1m['beta_w']:.4f}")
        print(f"  HAR beta_m:   {har_1m['beta_m']:.4f}")

        ret_1m_daily = np.zeros(len(rv_1m))
        dates_1m_list = list(daily_1m.keys())
        closes_1m = [daily_1m[d][-1] for d in dates_1m_list if len(daily_1m[d]) >= 10]
        for i in range(1, len(closes_1m)):
            ret_1m_daily[i] = np.log(closes_1m[i] / closes_1m[i-1])

        tau_1m = np.zeros(len(rv_1m))
        for t_i in range(len(rv_1m)):
            rv_d = rv_1m[t_i]
            rv_wk = rv_1m[max(0, t_i-4):t_i+1].mean()
            rv_mo = rv_1m[max(0, t_i-21):t_i+1].mean()
            tau_1m[t_i] = max(har_1m['alpha'] + har_1m['beta_d']*rv_d
                              + har_1m['beta_w']*rv_wk + har_1m['beta_m']*rv_mo, 1e-12)

        valid_1m = ret_1m_daily[1:]
        tau_1m_v = tau_1m[1:]
        std_1m = valid_1m / np.sqrt(tau_1m_v)

        garch_1m = hfm.garch(std_1m)
        print(f"\n  1-min M-GARCH fast component:")
        print(f"    omega      = {garch_1m.omega:.8f}")
        print(f"    alpha      = {garch_1m.alpha:.6f}")
        print(f"    beta       = {garch_1m.beta:.6f}")
        print(f"    persistence= {garch_1m.persistence:.6f}")
        print(f"    converged  = {garch_1m.converged}")

    # == 15. 5-min vs 1-min RV comparison ==

    print_section("5-min vs 1-min Realized Variance (overlapping period)")

    dates_5m_set = set(valid_dates)
    dates_1m_set = set(dates_1m)
    overlap = sorted(dates_5m_set & dates_1m_set)

    if len(overlap) > 10:
        rv_5m_overlap = []
        rv_1m_overlap = []
        for d in overlap:
            idx_5 = valid_dates.index(d)
            idx_1 = dates_1m.index(d)
            rv_5m_overlap.append(rv[idx_5])
            rv_1m_overlap.append(rv_1m[idx_1])

        rv_5m_o = np.array(rv_5m_overlap)
        rv_1m_o = np.array(rv_1m_overlap)
        corr = np.corrcoef(rv_5m_o, rv_1m_o)[0, 1]
        ratio = rv_1m_o / rv_5m_o

        print(f"  Overlapping days: {len(overlap)}")
        print(f"  Correlation(RV_5m, RV_1m): {corr:.4f}")
        print(f"  Mean ratio RV_1m/RV_5m:    {ratio.mean():.4f}")
        print(f"  Std  ratio RV_1m/RV_5m:    {ratio.std():.4f}")

    # == Summary ==

    t_elapsed = time.time() - t_total

    print_header("Summary")
    print(f"  SPY data: {n:,} trading days ({str(valid_dates[0])} -> {str(valid_dates[-1])})")
    print(f"  5-min bars processed: {len(px_5m):,}")
    print(f"  1-min bars processed: {len(px_1m):,}")
    print(f"  Rolling M-GARCH windows: {n_windows}")
    print(f"  Total elapsed: {t_elapsed:.2f}s")
    print(f"\n  Key findings:")
    print(f"    HAR-RV R^2 = {har['r_squared']:.4f} -> intraday RV is highly predictable")
    print(f"    Std GARCH persistence = {std_garch.persistence:.4f}")
    print(f"    M-GARCH persistence   = {mg_garch.persistence:.4f}")
    print(f"    -> Intraday info reduces GARCH persistence substantially")
    print(f"    -> Better volatility decomposition via multiplicative structure")
    print(f"\n  1-step forecast: {vol_fcast_ann:.1f}% annualized volatility")

    print_header("Done")


if __name__ == "__main__":
    main()
