#include "hfm/diagnostics/statistical_tests.hpp"
#include "hfm/linalg/solver.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <algorithm>

namespace hfm {

namespace {

f64 regularized_gamma_lower(f64 a, f64 x) {
    if (x < 0.0) return 0.0;
    if (x == 0.0) return 0.0;

    if (x < a + 1.0) {
        f64 sum = 1.0 / a;
        f64 term = 1.0 / a;
        for (int n = 1; n < 200; ++n) {
            term *= x / (a + static_cast<f64>(n));
            sum += term;
            if (std::abs(term) < 1e-15 * std::abs(sum)) break;
        }
        return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
    }

    f64 f = 1.0;
    f64 c = 1.0;
    f64 d = 1.0 / (x - a + 1.0);
    f = d;
    for (int n = 1; n < 200; ++n) {
        f64 an = -static_cast<f64>(n) * (static_cast<f64>(n) - a);
        f64 bn = x - a + 1.0 + 2.0 * static_cast<f64>(n);
        d = bn + an * d;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = bn + an / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        f64 delta = c * d;
        f *= delta;
        if (std::abs(delta - 1.0) < 1e-15) break;
    }
    return 1.0 - f * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

f64 quantile_sorted(const std::vector<f64>& sorted, f64 p) {
    std::size_t n = sorted.size();
    f64 idx = p * static_cast<f64>(n - 1);
    std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    std::size_t hi = std::min(lo + 1, n - 1);
    f64 frac = idx - static_cast<f64>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

} // namespace

// ========== Distribution utilities ==========

f64 chi2_cdf(f64 x, f64 df) {
    if (x <= 0.0) return 0.0;
    return regularized_gamma_lower(df / 2.0, x / 2.0);
}

f64 chi2_sf(f64 x, f64 df) {
    return 1.0 - chi2_cdf(x, df);
}

f64 normal_cdf(f64 x) {
    return 0.5 * std::erfc(-x * std::numbers::sqrt2 / 2.0);
}

f64 normal_sf(f64 x) {
    return 1.0 - normal_cdf(x);
}

// ========== Descriptive statistics ==========

Result<DescriptiveStats> descriptive_stats(const Vector<f64>& x) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = x.size();
    if (n < 4) {
        return Status::error(ErrorCode::InvalidArgument,
                             "descriptive_stats: need at least 4 observations");
    }

    DescriptiveStats ds;
    ds.n_obs = n;
    f64 fn = static_cast<f64>(n);

    ds.mean = x.mean();

    f64 m2 = 0.0, m3 = 0.0, m4 = 0.0;
    ds.min = x[0];
    ds.max = x[0];
    for (std::size_t i = 0; i < n; ++i) {
        f64 d = x[i] - ds.mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
        if (x[i] < ds.min) ds.min = x[i];
        if (x[i] > ds.max) ds.max = x[i];
    }

    ds.variance = m2 / (fn - 1.0);
    ds.std_dev = std::sqrt(ds.variance);

    f64 s2 = m2 / fn;
    f64 s = std::sqrt(s2);
    if (s > 0.0) {
        ds.skewness = (m3 / fn) / (s * s * s);
        ds.excess_kurtosis = (m4 / fn) / (s2 * s2) - 3.0;
    }

    std::vector<f64> sorted(x.data(), x.data() + n);
    std::sort(sorted.begin(), sorted.end());
    ds.median = quantile_sorted(sorted, 0.5);
    ds.q25 = quantile_sorted(sorted, 0.25);
    ds.q75 = quantile_sorted(sorted, 0.75);

    auto end = std::chrono::high_resolution_clock::now();
    ds.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return ds;
}

// ========== Jarque-Bera ==========

Result<JarqueBeraResult> jarque_bera(const Vector<f64>& x) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = x.size();
    if (n < 8) {
        return Status::error(ErrorCode::InvalidArgument,
                             "jarque_bera: need at least 8 observations");
    }

    f64 fn = static_cast<f64>(n);
    f64 mu = x.mean();

    f64 m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 d = x[i] - mu;
        f64 d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }

    f64 s2 = m2 / fn;
    f64 s = std::sqrt(s2);

    JarqueBeraResult result;
    result.n_obs = n;
    if (s < 1e-15) {
        result.skewness = 0.0;
        result.excess_kurtosis = 0.0;
        result.statistic = 0.0;
        result.p_value = 1.0;
    } else {
        result.skewness = (m3 / fn) / (s * s * s);
        result.excess_kurtosis = (m4 / fn) / (s2 * s2) - 3.0;
        result.statistic = (fn / 6.0) * (result.skewness * result.skewness +
                           result.excess_kurtosis * result.excess_kurtosis / 4.0);
        result.p_value = chi2_sf(result.statistic, 2.0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== Durbin-Watson ==========

Result<DurbinWatsonResult> durbin_watson(const Vector<f64>& residuals) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = residuals.size();
    if (n < 3) {
        return Status::error(ErrorCode::InvalidArgument,
                             "durbin_watson: need at least 3 observations");
    }

    f64 num = 0.0;
    f64 den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        den += residuals[i] * residuals[i];
        if (i > 0) {
            f64 diff = residuals[i] - residuals[i - 1];
            num += diff * diff;
        }
    }

    DurbinWatsonResult result;
    result.n_obs = n;
    result.statistic = (den > 0.0) ? num / den : 2.0;

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== Autocorrelation ==========

Vector<f64> autocorrelation(const Vector<f64>& x, std::size_t max_lag) {
    std::size_t n = x.size();
    f64 mu = x.mean();

    f64 gamma0 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 d = x[i] - mu;
        gamma0 += d * d;
    }

    Vector<f64> acf(max_lag);
    if (gamma0 < 1e-30) {
        return acf;
    }
    for (std::size_t k = 1; k <= max_lag; ++k) {
        f64 gk = 0.0;
        for (std::size_t t = k; t < n; ++t) {
            gk += (x[t] - mu) * (x[t - k] - mu);
        }
        acf[k - 1] = gk / gamma0;
    }
    return acf;
}

// ========== Ljung-Box ==========

Result<LjungBoxResult> ljung_box(const Vector<f64>& x, std::size_t n_lags) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = x.size();
    if (n < n_lags + 2) {
        return Status::error(ErrorCode::InvalidArgument,
                             "ljung_box: insufficient observations for given lags");
    }

    auto acf = autocorrelation(x, n_lags);
    f64 fn = static_cast<f64>(n);

    f64 Q = 0.0;
    for (std::size_t k = 0; k < n_lags; ++k) {
        f64 rho_k = acf[k];
        Q += (rho_k * rho_k) / (fn - static_cast<f64>(k + 1));
    }
    Q *= fn * (fn + 2.0);

    LjungBoxResult result;
    result.n_obs = n;
    result.n_lags = n_lags;
    result.statistic = Q;
    result.p_value = chi2_sf(Q, static_cast<f64>(n_lags));

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== Breusch-Pagan ==========

Result<BreuschPaganResult> breusch_pagan(const Vector<f64>& residuals,
                                          const Matrix<f64>& X) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = residuals.size();
    std::size_t k = X.cols();

    if (n != X.rows() || n < k + 2) {
        return Status::error(ErrorCode::InvalidArgument,
                             "breusch_pagan: dimension mismatch or too few observations");
    }

    f64 sigma2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sigma2 += residuals[i] * residuals[i];
    }
    sigma2 /= static_cast<f64>(n);

    Vector<f64> e2(n);
    for (std::size_t i = 0; i < n; ++i) {
        e2[i] = residuals[i] * residuals[i] / sigma2 - 1.0;
    }

    auto aux = solve_least_squares(X, e2);
    if (!aux) {
        return Status::error(ErrorCode::SingularMatrix,
                             "breusch_pagan: auxiliary regression failed");
    }

    auto fitted = matvec(X, aux.value());
    f64 ess = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ess += fitted[i] * fitted[i];
    }

    BreuschPaganResult result;
    result.n_obs = n;
    result.df = k;
    result.statistic = ess / 2.0;
    result.p_value = chi2_sf(result.statistic, static_cast<f64>(k));

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== White test ==========

Result<WhiteTestResult> white_test(const Vector<f64>& residuals,
                                    const Matrix<f64>& X) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = residuals.size();
    std::size_t k = X.cols();

    if (n != X.rows() || n < k + 2) {
        return Status::error(ErrorCode::InvalidArgument,
                             "white_test: dimension mismatch or too few observations");
    }

    std::size_t n_sq_cross = k * (k + 1) / 2;
    std::size_t n_aux_cols = 1 + k + n_sq_cross;
    Matrix<f64> Z(n, n_aux_cols, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        Z(i, 0) = 1.0;
        for (std::size_t j = 0; j < k; ++j) {
            Z(i, 1 + j) = X(i, j);
        }
        std::size_t col = 1 + k;
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t l = j; l < k; ++l) {
                Z(i, col) = X(i, j) * X(i, l);
                ++col;
            }
        }
    }

    Vector<f64> e2(n);
    for (std::size_t i = 0; i < n; ++i) {
        e2[i] = residuals[i] * residuals[i];
    }

    auto aux = solve_least_squares(Z, e2);
    if (!aux) {
        return Status::error(ErrorCode::SingularMatrix,
                             "white_test: auxiliary regression failed");
    }

    auto fitted = matvec(Z, aux.value());
    f64 e2_mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) e2_mean += e2[i];
    e2_mean /= static_cast<f64>(n);

    f64 ess = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 d = fitted[i] - e2_mean;
        ess += d * d;
        f64 d2 = e2[i] - e2_mean;
        ss_tot += d2 * d2;
    }

    f64 r2 = (ss_tot > 0.0) ? ess / ss_tot : 0.0;
    std::size_t df = k + n_sq_cross;

    WhiteTestResult result;
    result.n_obs = n;
    result.df = df;
    result.statistic = static_cast<f64>(n) * r2;
    result.p_value = chi2_sf(result.statistic, static_cast<f64>(df));

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== ARCH-LM ==========

Result<ArchLMResult> arch_lm(const Vector<f64>& residuals, std::size_t n_lags) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = residuals.size();
    if (n < n_lags + 2) {
        return Status::error(ErrorCode::InvalidArgument,
                             "arch_lm: insufficient observations for given lags");
    }

    Vector<f64> e2(n);
    for (std::size_t i = 0; i < n; ++i) {
        e2[i] = residuals[i] * residuals[i];
    }

    std::size_t T = n - n_lags;
    Matrix<f64> Z(T, n_lags + 1);
    Vector<f64> y(T);

    for (std::size_t t = 0; t < T; ++t) {
        std::size_t idx = t + n_lags;
        y[t] = e2[idx];
        Z(t, 0) = 1.0;
        for (std::size_t j = 0; j < n_lags; ++j) {
            Z(t, j + 1) = e2[idx - j - 1];
        }
    }

    auto beta = solve_least_squares(Z, y);
    if (!beta) {
        return Status::error(ErrorCode::SingularMatrix,
                             "arch_lm: regression failed");
    }

    auto fitted = matvec(Z, beta.value());
    f64 y_mean = y.mean();
    f64 ss_reg = 0.0, ss_tot = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        f64 dr = fitted[t] - y_mean;
        ss_reg += dr * dr;
        f64 dt = y[t] - y_mean;
        ss_tot += dt * dt;
    }

    f64 r2 = (ss_tot > 0.0) ? ss_reg / ss_tot : 0.0;

    ArchLMResult result;
    result.n_obs = n;
    result.n_lags = n_lags;
    result.statistic = static_cast<f64>(T) * r2;
    result.p_value = chi2_sf(result.statistic, static_cast<f64>(n_lags));

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== Augmented Dickey-Fuller ==========

Result<ADFResult> adf_test(const Vector<f64>& y, std::size_t max_lag,
                            ADFType type) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = y.size();
    if (n < 20) {
        return Status::error(ErrorCode::InvalidArgument,
                             "adf_test: need at least 20 observations");
    }

    if (max_lag == 0) {
        max_lag = static_cast<std::size_t>(
            std::floor(std::pow(static_cast<f64>(n - 1), 1.0 / 3.0)));
    }

    Vector<f64> dy(n - 1);
    for (std::size_t i = 1; i < n; ++i) {
        dy[i - 1] = y[i] - y[i - 1];
    }

    std::size_t T = n - 1 - max_lag;
    if (T < 10) {
        return Status::error(ErrorCode::InvalidArgument,
                             "adf_test: too many lags for sample size");
    }

    std::size_t n_det = 0;
    if (type == ADFType::Intercept) n_det = 1;
    else if (type == ADFType::InterceptAndTrend) n_det = 2;

    std::size_t k = 1 + max_lag + n_det;
    Matrix<f64> X(T, k);
    Vector<f64> yy(T);

    for (std::size_t t = 0; t < T; ++t) {
        std::size_t idx = t + max_lag;
        yy[t] = dy[idx];

        std::size_t col = 0;
        if (type == ADFType::Intercept || type == ADFType::InterceptAndTrend) {
            X(t, col++) = 1.0;
        }
        if (type == ADFType::InterceptAndTrend) {
            X(t, col++) = static_cast<f64>(idx + 1);
        }

        X(t, col++) = y[idx];

        for (std::size_t j = 0; j < max_lag; ++j) {
            X(t, col++) = dy[idx - j - 1];
        }
    }

    auto beta = solve_least_squares(X, yy);
    if (!beta) {
        return Status::error(ErrorCode::SingularMatrix,
                             "adf_test: regression failed");
    }

    auto fitted = matvec(X, beta.value());
    f64 sse = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        f64 r = yy[t] - fitted[t];
        sse += r * r;
    }
    f64 s2 = sse / static_cast<f64>(T - k);

    Matrix<f64> XtX = matmul_AtB(X, X);
    auto XtX_inv = invert_spd(XtX);
    if (!XtX_inv) {
        return Status::error(ErrorCode::SingularMatrix,
                             "adf_test: covariance matrix inversion failed");
    }

    std::size_t rho_idx = n_det;
    f64 rho_hat = beta.value()[rho_idx];
    f64 se_rho = std::sqrt(s2 * XtX_inv.value()(rho_idx, rho_idx));
    f64 t_stat = rho_hat / se_rho;

    ADFResult result;
    result.n_obs = n;
    result.n_lags = max_lag;
    result.statistic = t_stat;
    result.has_intercept = (type != ADFType::None);
    result.has_trend = (type == ADFType::InterceptAndTrend);

    if (type == ADFType::Intercept) {
        result.critical_1pct = -3.43;
        result.critical_5pct = -2.86;
        result.critical_10pct = -2.57;
    } else if (type == ADFType::InterceptAndTrend) {
        result.critical_1pct = -3.96;
        result.critical_5pct = -3.41;
        result.critical_10pct = -3.12;
    } else {
        result.critical_1pct = -2.58;
        result.critical_5pct = -1.95;
        result.critical_10pct = -1.62;
    }

    if (type == ADFType::Intercept) {
        f64 tau = t_stat;
        f64 p;
        if (tau < -3.96) p = 0.001;
        else if (tau < -3.43) p = 0.01;
        else if (tau < -2.86) p = 0.05;
        else if (tau < -2.57) p = 0.10;
        else if (tau < -1.94) p = 0.30;
        else if (tau < -1.62) p = 0.50;
        else p = 0.90;
        result.p_value = p;
    } else if (type == ADFType::InterceptAndTrend) {
        f64 tau = t_stat;
        if (tau < -3.96) result.p_value = 0.01;
        else if (tau < -3.41) result.p_value = 0.05;
        else if (tau < -3.12) result.p_value = 0.10;
        else if (tau < -2.58) result.p_value = 0.25;
        else result.p_value = 0.75;
    } else {
        f64 tau = t_stat;
        if (tau < -2.58) result.p_value = 0.01;
        else if (tau < -1.95) result.p_value = 0.05;
        else if (tau < -1.62) result.p_value = 0.10;
        else result.p_value = 0.50;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ========== KPSS ==========

Result<KPSSResult> kpss_test(const Vector<f64>& y, bool trend, std::size_t n_lags) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = y.size();
    if (n < 20) {
        return Status::error(ErrorCode::InvalidArgument,
                             "kpss_test: need at least 20 observations");
    }

    if (n_lags == 0) {
        n_lags = static_cast<std::size_t>(
            std::ceil(std::sqrt(static_cast<f64>(n)) * 0.75));
    }

    Vector<f64> residuals(n);
    if (trend) {
        Matrix<f64> X(n, 2);
        for (std::size_t i = 0; i < n; ++i) {
            X(i, 0) = 1.0;
            X(i, 1) = static_cast<f64>(i + 1);
        }
        auto beta = solve_least_squares(X, y);
        if (!beta) {
            return Status::error(ErrorCode::SingularMatrix,
                                 "kpss_test: regression failed");
        }
        auto fitted = matvec(X, beta.value());
        for (std::size_t i = 0; i < n; ++i) {
            residuals[i] = y[i] - fitted[i];
        }
    } else {
        f64 mu = y.mean();
        for (std::size_t i = 0; i < n; ++i) {
            residuals[i] = y[i] - mu;
        }
    }

    Vector<f64> S(n);
    S[0] = residuals[0];
    for (std::size_t i = 1; i < n; ++i) {
        S[i] = S[i - 1] + residuals[i];
    }

    f64 fn = static_cast<f64>(n);

    f64 gamma0 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        gamma0 += residuals[i] * residuals[i];
    }
    gamma0 /= fn;

    f64 s2 = gamma0;
    for (std::size_t lag = 1; lag <= n_lags; ++lag) {
        f64 gk = 0.0;
        for (std::size_t t = lag; t < n; ++t) {
            gk += residuals[t] * residuals[t - lag];
        }
        gk /= fn;
        f64 w = 1.0 - static_cast<f64>(lag) / (static_cast<f64>(n_lags) + 1.0);
        s2 += 2.0 * w * gk;
    }

    f64 eta = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        eta += S[i] * S[i];
    }
    eta /= (fn * fn);

    KPSSResult result;
    result.n_obs = n;
    result.n_lags = n_lags;
    result.trend = trend;
    result.statistic = (s2 > 0.0) ? eta / s2 : 0.0;

    if (trend) {
        result.critical_1pct = 0.216;
        result.critical_5pct = 0.146;
        result.critical_10pct = 0.119;
        if (result.statistic > 0.216) result.p_value = 0.01;
        else if (result.statistic > 0.146) result.p_value = 0.05;
        else if (result.statistic > 0.119) result.p_value = 0.10;
        else result.p_value = 0.50;
    } else {
        result.critical_1pct = 0.739;
        result.critical_5pct = 0.463;
        result.critical_10pct = 0.347;
        if (result.statistic > 0.739) result.p_value = 0.01;
        else if (result.statistic > 0.463) result.p_value = 0.05;
        else if (result.statistic > 0.347) result.p_value = 0.10;
        else result.p_value = 0.50;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

} // namespace hfm
