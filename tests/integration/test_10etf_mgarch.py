"""
Rolling-Window Multiplicative GARCH on 10 ETFs — Real Data
===========================================================

ETFs:  SPY  QQQ  IWM  DIA  GLD  XLF  XLE  TLT  EEM  SMH
       equity  tech  small  dow  gold  fin  energy  bonds  EM  semi

Data: etf_5min.duckdb — all 5-min OHLCV bars (RTH only: 09:30–16:00)

Run from the build directory:
    PYTHONPATH=python python3 ../tests/integration/test_10etf_mgarch.py
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

ETFS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "XLF", "XLE", "TLT", "EEM", "SMH"]
LABELS = {
    "SPY": "S&P 500",    "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "DIA": "Dow Jones",  "GLD": "Gold",        "XLF": "Financials",
    "XLE": "Energy",     "TLT": "20Y+ Bonds",  "EEM": "Emerg. Mkts",
    "SMH": "Semicons",
}

WINDOW = 252
STEP = 21


def print_header(title):
    w = 80
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")


def print_section(title):
    print(f"\n--- {title} {'-'*(64 - len(title))}")


def load_etf_5min(symbol):
    con = duckdb.connect(os.path.join(HF_DATA, "etf_5min.duckdb"), read_only=True)
    data = con.execute(f"""
        SELECT datetime, close
        FROM ohlcv
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


def process_etf(symbol):
    """Load, compute daily RV, returns, and run full + rolling M-GARCH."""
    t0 = time.time()

    # Load
    dt, px = load_etf_5min(symbol)
    n_bars = len(px)
    t_load = time.time() - t0

    # Group by day
    daily_groups = group_by_day(dt, px)
    dates = list(daily_groups.keys())
    n_days = len(dates)

    daily_rv = np.zeros(n_days)
    daily_bv = np.zeros(n_days)
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
            daily_bv[i] = rm.bipower_variation
        else:
            daily_rv[i] = np.nan

    for i in range(1, n_days):
        if daily_close[i - 1] > 0 and daily_close[i] > 0:
            daily_returns[i] = np.log(daily_close[i] / daily_close[i - 1])

    valid = ~np.isnan(daily_rv) & (daily_rv > 0)
    valid[0] = False
    idx = np.where(valid)[0]

    ret = daily_returns[idx]
    rv = daily_rv[idx]
    bv = daily_bv[idx]
    vdates = [dates[i] for i in idx]
    n = len(ret)

    result = {
        "symbol": symbol,
        "n_bars": n_bars,
        "n_days": n,
        "first": str(vdates[0]),
        "last": str(vdates[-1]),
        "avg_bars": bars_per_day[idx].mean(),
    }

    # Descriptive stats
    result["ret_mean"] = ret.mean()
    result["ret_std"] = ret.std()
    result["rv_mean"] = rv.mean()
    result["ann_rvol"] = np.sqrt(rv.mean() * 252) * 100
    result["ann_ret"] = ret.mean() * 252 * 100
    result["sharpe"] = (ret.mean() / np.sqrt(rv.mean())) * np.sqrt(252) if rv.mean() > 0 else 0

    # Standard GARCH
    t1 = time.time()
    sg = hfm.garch(ret)
    result["sg_time_ms"] = (time.time() - t1) * 1000
    result["sg_omega"] = sg.omega
    result["sg_alpha"] = sg.alpha
    result["sg_beta"] = sg.beta
    result["sg_persist"] = sg.persistence
    result["sg_ll"] = sg.log_likelihood
    result["sg_aic"] = sg.aic
    result["sg_converged"] = sg.converged
    result["sg_uncond_vol"] = np.sqrt(sg.unconditional_var * 252) * 100

    # HAR-RV
    t1 = time.time()
    har = hfm.har_rv(rv)
    result["har_time_ms"] = (time.time() - t1) * 1000
    result["har_r2"] = har["r_squared"]
    result["har_bd"] = har["beta_d"]
    result["har_bw"] = har["beta_w"]
    result["har_bm"] = har["beta_m"]
    result["har_alpha"] = har["alpha"]

    # Multiplicative GARCH
    t1 = time.time()
    tau = np.zeros(n)
    for t in range(n):
        rv_d = rv[t]
        rv_w = rv[max(0, t-4):t+1].mean()
        rv_m = rv[max(0, t-21):t+1].mean()
        tau[t] = max(har["alpha"] + har["beta_d"]*rv_d + har["beta_w"]*rv_w + har["beta_m"]*rv_m, 1e-12)

    std_ret = ret / np.sqrt(tau)
    mg = hfm.garch(std_ret)
    g = np.array(mg.conditional_var)
    total_var = tau * g

    result["mg_time_ms"] = (time.time() - t1) * 1000
    result["mg_omega"] = mg.omega
    result["mg_alpha"] = mg.alpha
    result["mg_beta"] = mg.beta
    result["mg_persist"] = mg.persistence
    result["mg_aic"] = mg.aic
    result["mg_converged"] = mg.converged

    # 1-step forecast
    rv_d_last = rv[-1]
    rv_w_last = rv[-5:].mean()
    rv_m_last = rv[-22:].mean()
    tau_f = max(har["alpha"] + har["beta_d"]*rv_d_last + har["beta_w"]*rv_w_last + har["beta_m"]*rv_m_last, 1e-12)
    last_std_r = ret[-1] / np.sqrt(tau[-1])
    g_f = mg.omega + mg.alpha * last_std_r**2 + mg.beta * g[-1]
    result["vol_forecast"] = np.sqrt(tau_f * g_f * 252) * 100

    # Rolling M-GARCH
    t1 = time.time()
    n_windows = max(0, (n - WINDOW) // STEP + 1)
    roll_persist = np.zeros(n_windows)
    roll_har_r2 = np.zeros(n_windows)
    roll_vol = np.zeros(n_windows)
    roll_g_alpha = np.zeros(n_windows)
    roll_g_beta = np.zeros(n_windows)
    roll_conv = np.zeros(n_windows, dtype=bool)

    for w in range(n_windows):
        s = w * STEP
        e = s + WINDOW
        r_w = ret[s:e]
        rv_w = rv[s:e]

        try:
            har_w = hfm.har_rv(rv_w)
            tau_w = np.zeros(WINDOW)
            for t in range(WINDOW):
                rv_d = rv_w[t]
                rv_wk = rv_w[max(0, t-4):t+1].mean()
                rv_mo = rv_w[max(0, t-21):t+1].mean()
                tau_w[t] = max(har_w["alpha"] + har_w["beta_d"]*rv_d
                               + har_w["beta_w"]*rv_wk + har_w["beta_m"]*rv_mo, 1e-12)
        except Exception:
            tau_w = np.maximum(rv_w, 1e-12)

        std_r = r_w / np.sqrt(tau_w)
        try:
            gw = hfm.garch(std_r)
            roll_persist[w] = gw.persistence
            roll_g_alpha[w] = gw.alpha
            roll_g_beta[w] = gw.beta
            roll_conv[w] = gw.converged
            roll_vol[w] = np.sqrt(tau_w[-1] * gw.conditional_var[-1] * 252) * 100
            roll_har_r2[w] = har_w["r_squared"]
        except Exception:
            roll_conv[w] = False

    result["roll_time_ms"] = (time.time() - t1) * 1000
    result["n_windows"] = n_windows
    result["n_converged"] = int(roll_conv.sum())

    if roll_conv.sum() > 0:
        vc = roll_vol[roll_conv]
        pc = roll_persist[roll_conv]
        ac = roll_g_alpha[roll_conv]
        bc = roll_g_beta[roll_conv]
        hc = roll_har_r2[roll_conv]

        result["roll_vol_mean"] = vc.mean()
        result["roll_vol_min"] = vc.min()
        result["roll_vol_max"] = vc.max()
        result["roll_vol_std"] = vc.std()
        result["roll_persist_mean"] = pc.mean()
        result["roll_persist_std"] = pc.std()
        result["roll_alpha_mean"] = ac.mean()
        result["roll_beta_mean"] = bc.mean()
        result["roll_har_r2_mean"] = hc.mean()
        result["roll_har_r2_std"] = hc.std()

        # Regime counts
        p25 = np.percentile(vc, 25)
        p75 = np.percentile(vc, 75)
        result["regime_low_n"] = int((vc <= p25).sum())
        result["regime_low_vol"] = vc[vc <= p25].mean()
        result["regime_high_n"] = int((vc > p75).sum())
        result["regime_high_vol"] = vc[vc > p75].mean()
    else:
        for k in ["roll_vol_mean", "roll_vol_min", "roll_vol_max", "roll_vol_std",
                   "roll_persist_mean", "roll_persist_std", "roll_alpha_mean",
                   "roll_beta_mean", "roll_har_r2_mean", "roll_har_r2_std"]:
            result[k] = 0.0

    result["total_ms"] = (time.time() - t0) * 1000

    return result


def main():
    t_total = time.time()

    print_header("Multiplicative GARCH — 10 ETFs — Real 5-min Data")
    print(f"  Model: sigma^2_t = tau_t * g_t")
    print(f"  tau_t: HAR-RV on daily realized variance (intraday component)")
    print(f"  g_t  : GARCH(1,1) on r_t/sqrt(tau_t) (daily component)")
    print(f"  Rolling window: {WINDOW} days, step: {STEP} days")
    print(f"  ETFs: {', '.join(ETFS)}")

    # Process all ETFs
    print_section("Processing all ETFs")
    results = []
    for sym in ETFS:
        t0 = time.time()
        r = process_etf(sym)
        elapsed = time.time() - t0
        print(f"  {sym:>4s}  {r['n_bars']:>8,} bars  {r['n_days']:>6,} days  "
              f"{r['n_windows']:>4} windows  {elapsed:.2f}s  "
              f"{'OK' if r['sg_converged'] and r['mg_converged'] else 'WARN'}")
        results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # Summary Tables
    # ═══════════════════════════════════════════════════════════════

    print_section("Data Summary")
    print(f"  {'ETF':>4s}  {'Name':<14s}  {'Bars':>9s}  {'Days':>6s}  {'Period':>23s}  {'Bars/d':>6s}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['symbol']:>4s}  {LABELS[r['symbol']]:<14s}  {r['n_bars']:>9,}  {r['n_days']:>6,}  "
              f"{r['first']} -> {r['last'][-5:]}  {r['avg_bars']:>5.0f}")
    total_bars = sum(r["n_bars"] for r in results)
    total_days = sum(r["n_days"] for r in results)
    print(f"  {'':>4s}  {'TOTAL':<14s}  {total_bars:>9,}  {total_days:>6,}")

    # Descriptive stats
    print_section("Descriptive Statistics")
    print(f"  {'ETF':>4s}  {'Ann Ret%':>8s}  {'Ann RVol%':>9s}  {'Sharpe':>7s}  "
          f"{'Daily Std':>10s}  {'Mean RV':>12s}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['symbol']:>4s}  {r['ann_ret']:>7.1f}%  {r['ann_rvol']:>8.1f}%  "
              f"{r['sharpe']:>7.2f}  {r['ret_std']:>10.6f}  {r['rv_mean']:>12.8f}")

    # Standard GARCH comparison
    print_section("Standard GARCH(1,1) — Full Sample")
    print(f"  {'ETF':>4s}  {'omega':>12s}  {'alpha':>8s}  {'beta':>8s}  "
          f"{'Persist':>8s}  {'Uncond%':>8s}  {'AIC':>12s}  {'ms':>6s}")
    print(f"  {'-'*76}")
    for r in results:
        print(f"  {r['symbol']:>4s}  {r['sg_omega']:>12.8f}  {r['sg_alpha']:>8.4f}  "
              f"{r['sg_beta']:>8.4f}  {r['sg_persist']:>8.4f}  "
              f"{r['sg_uncond_vol']:>7.1f}%  {r['sg_aic']:>12.1f}  {r['sg_time_ms']:>5.1f}")

    # HAR-RV comparison
    print_section("HAR-RV (Slow Component tau_t) — Full Sample")
    print(f"  {'ETF':>4s}  {'R^2':>8s}  {'beta_d':>8s}  {'beta_w':>8s}  "
          f"{'beta_m':>8s}  {'alpha':>12s}  {'ms':>6s}")
    print(f"  {'-'*62}")
    for r in results:
        print(f"  {r['symbol']:>4s}  {r['har_r2']:>8.4f}  {r['har_bd']:>8.4f}  "
              f"{r['har_bw']:>8.4f}  {r['har_bm']:>8.4f}  "
              f"{r['har_alpha']:>12.8f}  {r['har_time_ms']:>5.1f}")

    # M-GARCH comparison
    print_section("Multiplicative GARCH (Fast Component g_t) — Full Sample")
    print(f"  {'ETF':>4s}  {'omega':>10s}  {'alpha':>8s}  {'beta':>8s}  "
          f"{'Persist':>8s}  {'AIC':>12s}  {'Fcast%':>7s}  {'ms':>6s}")
    print(f"  {'-'*74}")
    for r in results:
        print(f"  {r['symbol']:>4s}  {r['mg_omega']:>10.6f}  {r['mg_alpha']:>8.4f}  "
              f"{r['mg_beta']:>8.4f}  {r['mg_persist']:>8.4f}  "
              f"{r['mg_aic']:>12.1f}  {r['vol_forecast']:>6.1f}%  {r['mg_time_ms']:>5.1f}")

    # Persistence comparison: Std vs M-GARCH
    print_section("Persistence Comparison: Standard vs Multiplicative GARCH")
    print(f"  {'ETF':>4s}  {'Std Persist':>11s}  {'M-GARCH Persist':>15s}  "
          f"{'Delta':>8s}  {'HAR R^2':>8s}  {'Interpretation':<30s}")
    print(f"  {'-'*82}")
    for r in results:
        delta = r["mg_persist"] - r["sg_persist"]
        if r["mg_persist"] < r["sg_persist"] - 0.05:
            interp = "RV absorbs shock dynamics"
        elif abs(delta) < 0.05:
            interp = "Similar dynamics"
        else:
            interp = "Slow component dominates"
        print(f"  {r['symbol']:>4s}  {r['sg_persist']:>11.4f}  {r['mg_persist']:>15.4f}  "
              f"{delta:>+8.4f}  {r['har_r2']:>8.4f}  {interp}")

    # Rolling M-GARCH summary
    print_section("Rolling M-GARCH Summary")
    print(f"  {'ETF':>4s}  {'Wins':>5s}  {'Conv':>5s}  {'Vol Mean':>8s}  "
          f"{'Vol Min':>8s}  {'Vol Max':>8s}  {'Persist':>8s}  "
          f"{'HAR R^2':>8s}  {'ms':>7s}  {'ms/win':>7s}")
    print(f"  {'-'*82}")
    for r in results:
        ms_win = r["roll_time_ms"] / r["n_windows"] if r["n_windows"] > 0 else 0
        print(f"  {r['symbol']:>4s}  {r['n_windows']:>5}  {r['n_converged']:>5}  "
              f"{r.get('roll_vol_mean',0):>7.1f}%  {r.get('roll_vol_min',0):>7.1f}%  "
              f"{r.get('roll_vol_max',0):>7.1f}%  {r.get('roll_persist_mean',0):>8.4f}  "
              f"{r.get('roll_har_r2_mean',0):>8.4f}  {r['roll_time_ms']:>6.0f}  {ms_win:>6.1f}")

    # Regime analysis
    print_section("Regime Analysis (rolling vol quartiles)")
    print(f"  {'ETF':>4s}  {'Low Vol':>16s}  {'High Vol':>16s}  "
          f"{'Low Persist':>11s}  {'High Persist':>12s}")
    print(f"  {'-'*68}")
    for r in results:
        if "regime_low_vol" in r:
            print(f"  {r['symbol']:>4s}  {r['regime_low_n']:>3} win @ "
                  f"{r['regime_low_vol']:>5.1f}%  "
                  f"{r['regime_high_n']:>3} win @ {r['regime_high_vol']:>5.1f}%")

    # Cross-ETF ranking
    print_section("Cross-ETF Rankings")

    by_rvol = sorted(results, key=lambda r: r["ann_rvol"])
    print(f"\n  By Annualized RVol (low -> high):")
    for i, r in enumerate(by_rvol):
        print(f"    {i+1:2d}. {r['symbol']:>4s} ({LABELS[r['symbol']]:<14s}): {r['ann_rvol']:>5.1f}%")

    by_sharpe = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    print(f"\n  By Sharpe Ratio (high -> low):")
    for i, r in enumerate(by_sharpe):
        print(f"    {i+1:2d}. {r['symbol']:>4s} ({LABELS[r['symbol']]:<14s}): {r['sharpe']:>+.2f}")

    by_har = sorted(results, key=lambda r: r["har_r2"], reverse=True)
    print(f"\n  By HAR-RV R^2 (highest intraday predictability):")
    for i, r in enumerate(by_har):
        print(f"    {i+1:2d}. {r['symbol']:>4s} ({LABELS[r['symbol']]:<14s}): {r['har_r2']:.4f}")

    by_forecast = sorted(results, key=lambda r: r["vol_forecast"])
    print(f"\n  By Current Vol Forecast (low -> high):")
    for i, r in enumerate(by_forecast):
        print(f"    {i+1:2d}. {r['symbol']:>4s} ({LABELS[r['symbol']]:<14s}): {r['vol_forecast']:>5.1f}%")

    # Performance summary
    t_elapsed = time.time() - t_total

    print_section("Performance")
    total_bars = sum(r["n_bars"] for r in results)
    total_wins = sum(r["n_windows"] for r in results)
    total_garch_ms = sum(r["sg_time_ms"] + r["mg_time_ms"] for r in results)
    total_roll_ms = sum(r["roll_time_ms"] for r in results)

    print(f"  Total 5-min bars:        {total_bars:>10,}")
    print(f"  Total trading days:      {total_days:>10,}")
    print(f"  Total rolling windows:   {total_wins:>10,}")
    print(f"  GARCH estimations:       {total_wins * 2 + 20:>10,}  (HAR+GARCH per window + full)")
    print(f"  Full-sample GARCH time:  {total_garch_ms:>9.1f}ms")
    print(f"  Rolling estimation time: {total_roll_ms:>9.0f}ms")
    print(f"  Total elapsed:           {t_elapsed:>9.2f}s")
    print(f"  Throughput:              {total_bars/t_elapsed:>9,.0f} bars/s")
    print(f"  Rolling speed:           {total_roll_ms/total_wins:>9.1f} ms/window")

    print_header("Done")


if __name__ == "__main__":
    main()
