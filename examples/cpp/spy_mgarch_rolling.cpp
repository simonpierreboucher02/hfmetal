// Rolling-window Multiplicative GARCH on SPY ETF
// Combines intraday realized variance (tau_t) with daily GARCH(1,1) (g_t)
// Total variance: sigma²_t = tau_t * g_t
//
// Model:
//   r_t        = sqrt(sigma²_t) * z_t,          z_t ~ N(0,1)
//   sigma²_t   = tau_t * g_t                     (multiplicative decomposition)
//   tau_t       from HAR-RV on daily realized variance (slow/intraday component)
//   g_t         GARCH(1,1) on demeaned r_t / sqrt(tau_t) (fast/daily component)
//
// Rolling windows: re-estimate both components every `step` days

#include "hfm/hfm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

using namespace hfm;

// ─── Simulated SPY data generator ───────────────────────────────────────────

struct SPYData {
    std::vector<f64> daily_prices;
    std::vector<f64> daily_returns;
    std::vector<std::vector<f64>> intraday_returns; // [day][bar]
    std::vector<f64> daily_rv;                      // realized variance per day
    std::vector<f64> daily_rvol;                    // realized volatility per day
    std::vector<f64> daily_bv;                      // bipower variation per day
    std::vector<f64> daily_jump;                    // jump statistic per day
    std::size_t n_days;
    std::size_t bars_per_day;
};

SPYData simulate_spy(std::size_t n_days, std::size_t bars_per_day, uint64_t seed) {
    SPYData data;
    data.n_days = n_days;
    data.bars_per_day = bars_per_day;

    std::mt19937_64 rng(seed);
    std::normal_distribution<f64> norm(0.0, 1.0);

    // SPY-like parameters
    constexpr f64 annual_drift = 0.08;
    constexpr f64 daily_drift = annual_drift / 252.0;
    constexpr f64 base_vol = 0.18;          // 18% annualized
    const f64 daily_base_vol = base_vol / std::sqrt(252.0);

    // GARCH-like volatility dynamics for simulation
    constexpr f64 sim_omega = 0.000005;
    constexpr f64 sim_alpha = 0.08;
    constexpr f64 sim_beta = 0.88;

    f64 price = 450.0; // SPY starting price
    f64 h = daily_base_vol * daily_base_vol;

    data.daily_prices.push_back(price);
    data.intraday_returns.resize(n_days);
    data.daily_rv.resize(n_days);
    data.daily_rvol.resize(n_days);
    data.daily_bv.resize(n_days);
    data.daily_jump.resize(n_days);

    f64 bar_scale = 1.0 / std::sqrt(static_cast<f64>(bars_per_day));

    for (std::size_t d = 0; d < n_days; ++d) {
        f64 daily_vol = std::sqrt(h);
        f64 bar_vol = daily_vol * bar_scale;

        // Intraday U-shape volatility pattern
        data.intraday_returns[d].resize(bars_per_day);
        f64 daily_return = 0.0;

        for (std::size_t b = 0; b < bars_per_day; ++b) {
            f64 t_frac = static_cast<f64>(b) / static_cast<f64>(bars_per_day);
            f64 u_shape = 1.0 + 0.5 * (4.0 * (t_frac - 0.5) * (t_frac - 0.5));
            f64 intra_vol = bar_vol * u_shape;
            f64 r = daily_drift / static_cast<f64>(bars_per_day) + intra_vol * norm(rng);
            data.intraday_returns[d][b] = r;
            daily_return += r;
        }

        data.daily_returns.push_back(daily_return);
        price *= std::exp(daily_return);
        data.daily_prices.push_back(price);

        // Compute realized measures from intraday returns
        Series<f64> intra_series(data.intraday_returns[d]);
        auto rm = compute_realized_measures(intra_series);
        data.daily_rv[d] = rm.realized_variance;
        data.daily_rvol[d] = rm.realized_volatility;
        data.daily_bv[d] = rm.bipower_variation;
        data.daily_jump[d] = rm.jump_statistic;

        // Update GARCH variance for next day
        f64 e = daily_return - daily_drift;
        h = sim_omega + sim_alpha * e * e + sim_beta * h;
    }

    return data;
}

// ─── Multiplicative GARCH components ────────────────────────────────────────

struct MGARCHResult {
    // Slow component (tau): from HAR-RV on realized variance
    Vector<f64> tau;           // tau_t series (predicted RV from HAR)
    f64 har_alpha = 0.0;
    f64 har_beta_d = 0.0;
    f64 har_beta_w = 0.0;
    f64 har_beta_m = 0.0;
    f64 har_r2 = 0.0;

    // Fast component (g): GARCH(1,1) on r_t / sqrt(tau_t)
    f64 garch_omega = 0.0;
    f64 garch_alpha = 0.0;
    f64 garch_beta = 0.0;
    f64 garch_persistence = 0.0;
    Vector<f64> g;             // g_t conditional variance series
    f64 garch_ll = 0.0;
    f64 garch_aic = 0.0;

    // Combined
    Vector<f64> total_var;     // sigma²_t = tau_t * g_t
    Vector<f64> total_vol;     // sigma_t

    bool converged = false;
    std::size_t n_obs = 0;
};

MGARCHResult estimate_mgarch(const std::vector<f64>& daily_returns,
                              const std::vector<f64>& daily_rv,
                              std::size_t start, std::size_t end) {
    MGARCHResult result;
    std::size_t n = end - start;
    result.n_obs = n;

    if (n < 30) {
        return result;
    }

    // ── Step 1: HAR-RV on realized variance → tau_t (slow component) ──

    // Build daily RV series for HAR
    std::vector<f64> rv_window(daily_rv.begin() + static_cast<std::ptrdiff_t>(start),
                               daily_rv.begin() + static_cast<std::ptrdiff_t>(end));
    Series<f64> rv_series(rv_window);

    auto har_res = har_rv(rv_series, HAROptions{}.set_lags(1, 5, 22));
    if (!har_res) {
        // Fallback: use raw RV as tau
        result.tau = Vector<f64>(n);
        for (std::size_t i = 0; i < n; ++i) {
            result.tau[i] = std::max(rv_window[i], 1e-10);
        }
    } else {
        auto& hr = har_res.value();
        result.har_alpha = hr.alpha;
        result.har_beta_d = hr.beta_d;
        result.har_beta_w = hr.beta_w;
        result.har_beta_m = hr.beta_m;
        result.har_r2 = hr.r_squared;

        // Compute HAR fitted values as tau
        result.tau = Vector<f64>(n);
        for (std::size_t t = 0; t < n; ++t) {
            f64 rv_d = rv_window[t];
            f64 rv_w = 0.0, rv_m = 0.0;
            std::size_t w_count = 0, m_count = 0;

            for (std::size_t j = 0; j < 5 && t >= j; ++j) {
                rv_w += rv_window[t - j];
                ++w_count;
            }
            rv_w = (w_count > 0) ? rv_w / static_cast<f64>(w_count) : rv_d;

            for (std::size_t j = 0; j < 22 && t >= j; ++j) {
                rv_m += rv_window[t - j];
                ++m_count;
            }
            rv_m = (m_count > 0) ? rv_m / static_cast<f64>(m_count) : rv_d;

            f64 tau = hr.alpha + hr.beta_d * rv_d + hr.beta_w * rv_w + hr.beta_m * rv_m;
            result.tau[t] = std::max(tau, 1e-10);
        }
    }

    // ── Step 2: Standardize returns by sqrt(tau_t) ──

    Vector<f64> std_returns(n);
    for (std::size_t t = 0; t < n; ++t) {
        f64 r = daily_returns[start + t];
        std_returns[t] = r / std::sqrt(result.tau[t]);
    }

    // ── Step 3: GARCH(1,1) on standardized returns → g_t (fast component) ──

    auto garch_res = garch(std_returns, GARCHOptions{}.set_max_iter(1000).set_tol(1e-10));
    if (garch_res) {
        auto& gr = garch_res.value();
        result.garch_omega = gr.omega;
        result.garch_alpha = gr.alpha;
        result.garch_beta = gr.beta;
        result.garch_persistence = gr.persistence;
        result.g = std::move(gr.conditional_var);
        result.garch_ll = gr.log_likelihood;
        result.garch_aic = gr.aic;
        result.converged = gr.converged;
    } else {
        result.g = Vector<f64>(n, 1.0);
    }

    // ── Step 4: Total variance = tau_t * g_t ──

    result.total_var = Vector<f64>(n);
    result.total_vol = Vector<f64>(n);
    for (std::size_t t = 0; t < n; ++t) {
        result.total_var[t] = result.tau[t] * result.g[t];
        result.total_vol[t] = std::sqrt(result.total_var[t]);
    }

    return result;
}

// ─── Rolling window results container ───────────────────────────────────────

struct RollingMGARCHResult {
    std::vector<std::size_t> window_starts;
    std::vector<std::size_t> window_ends;

    // HAR-RV parameters per window
    std::vector<f64> har_alpha;
    std::vector<f64> har_beta_d;
    std::vector<f64> har_beta_w;
    std::vector<f64> har_beta_m;
    std::vector<f64> har_r2;

    // GARCH parameters per window
    std::vector<f64> garch_omega;
    std::vector<f64> garch_alpha;
    std::vector<f64> garch_beta;
    std::vector<f64> garch_persistence;
    std::vector<f64> garch_aic;
    std::vector<bool> converged;

    // End-of-window volatility forecast
    std::vector<f64> vol_forecast;

    std::size_t n_windows = 0;
};

RollingMGARCHResult rolling_mgarch(const std::vector<f64>& daily_returns,
                                    const std::vector<f64>& daily_rv,
                                    std::size_t window, std::size_t step) {
    RollingMGARCHResult result;
    std::size_t n = daily_returns.size();

    if (n < window) return result;

    std::size_t n_windows = (n - window) / step + 1;
    result.n_windows = n_windows;

    result.window_starts.resize(n_windows);
    result.window_ends.resize(n_windows);
    result.har_alpha.resize(n_windows);
    result.har_beta_d.resize(n_windows);
    result.har_beta_w.resize(n_windows);
    result.har_beta_m.resize(n_windows);
    result.har_r2.resize(n_windows);
    result.garch_omega.resize(n_windows);
    result.garch_alpha.resize(n_windows);
    result.garch_beta.resize(n_windows);
    result.garch_persistence.resize(n_windows);
    result.garch_aic.resize(n_windows);
    result.converged.resize(n_windows);
    result.vol_forecast.resize(n_windows);

    for (std::size_t w = 0; w < n_windows; ++w) {
        std::size_t s = w * step;
        std::size_t e = s + window;

        result.window_starts[w] = s;
        result.window_ends[w] = e;

        auto mg = estimate_mgarch(daily_returns, daily_rv, s, e);

        result.har_alpha[w] = mg.har_alpha;
        result.har_beta_d[w] = mg.har_beta_d;
        result.har_beta_w[w] = mg.har_beta_w;
        result.har_beta_m[w] = mg.har_beta_m;
        result.har_r2[w] = mg.har_r2;

        result.garch_omega[w] = mg.garch_omega;
        result.garch_alpha[w] = mg.garch_alpha;
        result.garch_beta[w] = mg.garch_beta;
        result.garch_persistence[w] = mg.garch_persistence;
        result.garch_aic[w] = mg.garch_aic;
        result.converged[w] = mg.converged;

        // Volatility forecast: last total_vol value (annualized)
        if (!mg.total_vol.empty()) {
            result.vol_forecast[w] = mg.total_vol[mg.total_vol.size() - 1] * std::sqrt(252.0);
        }
    }

    return result;
}

// ─── Display helpers ────────────────────────────────────────────────────────

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(72, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " " << std::string(60 - title.size(), '-') << "\n";
}

int main() {
    std::cout << std::fixed;

    print_separator("SPY Multiplicative GARCH — Rolling Window Estimation");
    std::cout << "  Model: sigma^2_t = tau_t * g_t\n";
    std::cout << "  tau_t : HAR-RV (slow component from intraday realized variance)\n";
    std::cout << "  g_t   : GARCH(1,1) on r_t/sqrt(tau_t) (fast daily component)\n";

    // ── 1. Simulate SPY data ────────────────────────────────────────────────

    constexpr std::size_t N_DAYS = 756;       // ~3 years of trading days
    constexpr std::size_t BARS_PER_DAY = 78;  // 5-min bars (6.5h trading day)
    constexpr std::size_t WINDOW = 252;       // 1-year rolling window
    constexpr std::size_t STEP = 21;          // Re-estimate monthly

    print_subsection("Data Generation");
    auto spy = simulate_spy(N_DAYS, BARS_PER_DAY, 20240423ULL);
    std::cout << std::setprecision(0);
    std::cout << "  Trading days    : " << spy.n_days << "\n";
    std::cout << "  Bars/day        : " << spy.bars_per_day << " (5-min)\n";
    std::cout << "  Total bars      : " << spy.n_days * spy.bars_per_day << "\n";
    std::cout << std::setprecision(2);
    std::cout << "  SPY start price : $" << spy.daily_prices.front() << "\n";
    std::cout << "  SPY end price   : $" << spy.daily_prices.back() << "\n";

    // ── 2. Descriptive statistics on realized measures ──────────────────────

    print_subsection("Realized Measures Summary (daily)");

    auto compute_stats = [](const std::vector<f64>& v) {
        f64 mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<f64>(v.size());
        f64 var = 0.0;
        for (auto x : v) var += (x - mean) * (x - mean);
        var /= static_cast<f64>(v.size());
        f64 mn = *std::min_element(v.begin(), v.end());
        f64 mx = *std::max_element(v.begin(), v.end());
        auto sorted = v;
        std::sort(sorted.begin(), sorted.end());
        f64 med = sorted[sorted.size() / 2];
        return std::make_tuple(mean, std::sqrt(var), mn, med, mx);
    };

    auto [rv_mean, rv_sd, rv_min, rv_med, rv_max] = compute_stats(spy.daily_rv);
    auto [bv_mean, bv_sd, bv_min, bv_med, bv_max] = compute_stats(spy.daily_bv);

    std::cout << std::setprecision(8);
    std::cout << "                    Mean         Std          Min          Median       Max\n";
    std::cout << "  RV        " << std::setw(13) << rv_mean
              << std::setw(13) << rv_sd
              << std::setw(13) << rv_min
              << std::setw(13) << rv_med
              << std::setw(13) << rv_max << "\n";
    std::cout << "  BV        " << std::setw(13) << bv_mean
              << std::setw(13) << bv_sd
              << std::setw(13) << bv_min
              << std::setw(13) << bv_med
              << std::setw(13) << bv_max << "\n";

    std::cout << std::setprecision(4);
    std::cout << "  Avg RVol (ann) : " << std::sqrt(rv_mean * 252.0) * 100.0 << "%\n";

    // ── 3. Full-sample Multiplicative GARCH ─────────────────────────────────

    print_subsection("Full-Sample Multiplicative GARCH Estimation");
    auto full_mg = estimate_mgarch(spy.daily_returns, spy.daily_rv, 0, spy.n_days);

    std::cout << std::setprecision(6);
    std::cout << "\n  HAR-RV (slow component tau_t):\n";
    std::cout << "    alpha   = " << std::setw(12) << full_mg.har_alpha << "\n";
    std::cout << "    beta_d  = " << std::setw(12) << full_mg.har_beta_d << "  (daily RV)\n";
    std::cout << "    beta_w  = " << std::setw(12) << full_mg.har_beta_w << "  (weekly avg RV)\n";
    std::cout << "    beta_m  = " << std::setw(12) << full_mg.har_beta_m << "  (monthly avg RV)\n";
    std::cout << "    R^2     = " << std::setw(12) << full_mg.har_r2 << "\n";

    std::cout << "\n  GARCH(1,1) (fast component g_t on standardized returns):\n";
    std::cout << "    omega   = " << std::setw(12) << full_mg.garch_omega << "\n";
    std::cout << "    alpha   = " << std::setw(12) << full_mg.garch_alpha << "\n";
    std::cout << "    beta    = " << std::setw(12) << full_mg.garch_beta << "\n";
    std::cout << "    alpha+beta = " << std::setw(9) << full_mg.garch_persistence << "\n";
    std::cout << "    LogLik  = " << std::setw(12) << full_mg.garch_ll << "\n";
    std::cout << "    AIC     = " << std::setw(12) << full_mg.garch_aic << "\n";
    std::cout << "    Converged: " << (full_mg.converged ? "YES" : "NO") << "\n";

    // Show last few total volatilities
    std::cout << "\n  Total annualized volatility (last 10 days):\n";
    for (std::size_t i = full_mg.total_vol.size() - 10; i < full_mg.total_vol.size(); ++i) {
        std::cout << "    Day " << std::setw(3) << i << " : "
                  << std::setprecision(2) << full_mg.total_vol[i] * std::sqrt(252.0) * 100.0
                  << "%" << std::setprecision(6) << "\n";
    }

    // ── 4. Compare: standard GARCH vs Multiplicative GARCH ──────────────────

    print_subsection("Comparison: Standard GARCH(1,1) vs Multiplicative GARCH");

    Vector<f64> all_returns(spy.daily_returns);
    auto std_garch = garch(all_returns, GARCHOptions{}.set_max_iter(1000));
    if (std_garch) {
        auto& sg = std_garch.value();
        std::cout << "  Standard GARCH(1,1):\n";
        std::cout << "    omega      = " << sg.omega << "\n";
        std::cout << "    alpha      = " << sg.alpha << "\n";
        std::cout << "    beta       = " << sg.beta << "\n";
        std::cout << "    persistence= " << sg.persistence << "\n";
        std::cout << "    LogLik     = " << sg.log_likelihood << "\n";
        std::cout << "    AIC        = " << sg.aic << "\n\n";

        std::cout << "  Multiplicative GARCH:\n";
        std::cout << "    HAR-RV R^2       = " << full_mg.har_r2 << "\n";
        std::cout << "    GARCH persistence= " << full_mg.garch_persistence << "\n";
        std::cout << "    GARCH AIC        = " << full_mg.garch_aic << "\n";
        std::cout << "\n  -> M-GARCH captures intraday info via RV, reducing\n";
        std::cout << "     persistence in the fast component (better separation).\n";
    }

    // ── 5. Rolling window estimation ────────────────────────────────────────

    print_subsection("Rolling Window Multiplicative GARCH");
    std::cout << "  Window size : " << WINDOW << " days (1 year)\n";
    std::cout << "  Step size   : " << STEP << " days (monthly)\n";

    auto rolling = rolling_mgarch(spy.daily_returns, spy.daily_rv, WINDOW, STEP);
    std::cout << "  N windows   : " << rolling.n_windows << "\n";

    // ── 6. Rolling parameter evolution ──────────────────────────────────────

    print_subsection("Rolling Parameter Evolution");

    std::cout << std::setprecision(6);
    std::cout << "\n  Window  Days        HAR_R2    GARCH_alpha  GARCH_beta   Persist.   AnnVol(%)\n";
    std::cout << "  " << std::string(86, '-') << "\n";

    for (std::size_t w = 0; w < rolling.n_windows; ++w) {
        std::cout << "  " << std::setw(3) << w + 1
                  << "    [" << std::setw(3) << rolling.window_starts[w]
                  << "-" << std::setw(3) << rolling.window_ends[w] << "]"
                  << std::setw(10) << rolling.har_r2[w]
                  << std::setw(13) << rolling.garch_alpha[w]
                  << std::setw(13) << rolling.garch_beta[w]
                  << std::setw(11) << rolling.garch_persistence[w]
                  << std::setw(10) << std::setprecision(2) << rolling.vol_forecast[w] * 100.0
                  << (rolling.converged[w] ? "" : " *")
                  << std::setprecision(6) << "\n";
    }
    std::cout << "  (* = GARCH did not converge)\n";

    // ── 7. Summary statistics of rolling parameters ─────────────────────────

    print_subsection("Rolling Parameter Summary Statistics");

    auto rolling_stats = [](const std::vector<f64>& v) {
        f64 mn = *std::min_element(v.begin(), v.end());
        f64 mx = *std::max_element(v.begin(), v.end());
        f64 mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<f64>(v.size());
        f64 var = 0.0;
        for (auto x : v) var += (x - mean) * (x - mean);
        f64 sd = std::sqrt(var / static_cast<f64>(v.size()));
        return std::make_tuple(mean, sd, mn, mx);
    };

    std::cout << std::setprecision(6);
    std::cout << "\n                    Mean         Std          Min          Max\n";

    auto [ga_mean, ga_sd, ga_min, ga_max] = rolling_stats(rolling.garch_alpha);
    std::cout << "  GARCH alpha "
              << std::setw(13) << ga_mean << std::setw(13) << ga_sd
              << std::setw(13) << ga_min << std::setw(13) << ga_max << "\n";

    auto [gb_mean, gb_sd, gb_min, gb_max] = rolling_stats(rolling.garch_beta);
    std::cout << "  GARCH beta  "
              << std::setw(13) << gb_mean << std::setw(13) << gb_sd
              << std::setw(13) << gb_min << std::setw(13) << gb_max << "\n";

    auto [gp_mean, gp_sd, gp_min, gp_max] = rolling_stats(rolling.garch_persistence);
    std::cout << "  Persistence "
              << std::setw(13) << gp_mean << std::setw(13) << gp_sd
              << std::setw(13) << gp_min << std::setw(13) << gp_max << "\n";

    auto [hr_mean, hr_sd, hr_min, hr_max] = rolling_stats(rolling.har_r2);
    std::cout << "  HAR-RV R^2  "
              << std::setw(13) << hr_mean << std::setw(13) << hr_sd
              << std::setw(13) << hr_min << std::setw(13) << hr_max << "\n";

    auto [vf_mean, vf_sd, vf_min, vf_max] = rolling_stats(rolling.vol_forecast);
    std::cout << std::setprecision(2);
    std::cout << "  Vol forecast"
              << std::setw(11) << vf_mean * 100.0 << "%"
              << std::setw(10) << vf_sd * 100.0 << "%"
              << std::setw(10) << vf_min * 100.0 << "%"
              << std::setw(10) << vf_max * 100.0 << "%\n";

    // ── 8. Volatility decomposition: tau vs g contribution ──────────────────

    print_subsection("Volatility Decomposition (full sample, last 20 days)");

    std::cout << std::setprecision(6);
    std::cout << "\n  Day   tau_t (RV)     g_t (GARCH)   sigma^2_t      AnnVol(%)\n";
    std::cout << "  " << std::string(66, '-') << "\n";

    std::size_t disp_start = full_mg.tau.size() - 20;
    for (std::size_t i = disp_start; i < full_mg.tau.size(); ++i) {
        std::cout << "  " << std::setw(3) << i
                  << std::setw(14) << full_mg.tau[i]
                  << std::setw(14) << full_mg.g[i]
                  << std::setw(14) << full_mg.total_var[i]
                  << std::setw(12) << std::setprecision(2)
                  << full_mg.total_vol[i] * std::sqrt(252.0) * 100.0 << "%"
                  << std::setprecision(6) << "\n";
    }

    // ── 9. One-step-ahead forecast ──────────────────────────────────────────

    print_subsection("One-Step-Ahead Volatility Forecast");

    // tau forecast from HAR
    std::size_t T = spy.n_days - 1;
    f64 rv_d = spy.daily_rv[T];
    f64 rv_w = 0.0, rv_m = 0.0;
    for (std::size_t j = 0; j < 5 && T >= j; ++j) rv_w += spy.daily_rv[T - j];
    rv_w /= 5.0;
    for (std::size_t j = 0; j < 22 && T >= j; ++j) rv_m += spy.daily_rv[T - j];
    rv_m /= 22.0;

    f64 tau_forecast = full_mg.har_alpha + full_mg.har_beta_d * rv_d
                     + full_mg.har_beta_w * rv_w + full_mg.har_beta_m * rv_m;
    tau_forecast = std::max(tau_forecast, 1e-10);

    // g forecast from GARCH
    f64 last_g = full_mg.g[full_mg.g.size() - 1];
    f64 last_std_ret = spy.daily_returns.back() / std::sqrt(full_mg.tau[full_mg.tau.size() - 1]);
    f64 g_forecast = full_mg.garch_omega + full_mg.garch_alpha * last_std_ret * last_std_ret
                   + full_mg.garch_beta * last_g;

    f64 sigma2_forecast = tau_forecast * g_forecast;
    f64 vol_forecast = std::sqrt(sigma2_forecast) * std::sqrt(252.0);

    std::cout << std::setprecision(8);
    std::cout << "  tau_{T+1}    = " << tau_forecast << "  (HAR-RV forecast)\n";
    std::cout << "  g_{T+1}      = " << g_forecast << "  (GARCH forecast)\n";
    std::cout << "  sigma^2_{T+1}= " << sigma2_forecast << "  (total)\n";
    std::cout << std::setprecision(2);
    std::cout << "  Annualized   = " << vol_forecast * 100.0 << "%\n";

    print_separator("Done");

    return 0;
}
