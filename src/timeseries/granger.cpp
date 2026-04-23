#include "hfm/timeseries/granger.hpp"
#include "hfm/linalg/solver.hpp"
#include <chrono>
#include <cmath>

namespace hfm {

namespace {

f64 betacf(f64 a, f64 b, f64 x) {
    constexpr int max_iter = 200;
    constexpr f64 eps = 1e-14;
    constexpr f64 tiny = 1e-30;

    f64 qab = a + b;
    f64 qap = a + 1.0;
    f64 qam = a - 1.0;
    f64 c = 1.0;
    f64 d = 1.0 - qab * x / qap;
    if (std::abs(d) < tiny) d = tiny;
    d = 1.0 / d;
    f64 h = d;

    for (int m = 1; m <= max_iter; ++m) {
        f64 fm = static_cast<f64>(m);
        f64 m2 = 2.0 * fm;

        f64 aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        f64 del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < eps) break;
    }
    return h;
}

f64 ibeta(f64 a, f64 b, f64 x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    f64 bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                      + a * std::log(x) + b * std::log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(a, b, x) / a;
    }
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

f64 f_sf(f64 x, f64 df1, f64 df2) {
    if (x <= 0.0) return 1.0;
    f64 t = df1 * x / (df1 * x + df2);
    return 1.0 - ibeta(df1 / 2.0, df2 / 2.0, t);
}

} // namespace

Result<GrangerResult> granger_causality(const Vector<f64>& y, const Vector<f64>& x,
                                         std::size_t n_lags) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = y.size();
    if (n != x.size()) {
        return Status::error(ErrorCode::DimensionMismatch,
                             "granger_causality: y and x must have same length");
    }
    if (n < 2 * n_lags + 5) {
        return Status::error(ErrorCode::InvalidArgument,
                             "granger_causality: insufficient observations");
    }

    std::size_t T = n - n_lags;

    // Restricted: y_t on lags of y only
    std::size_t k_r = n_lags + 1;
    Matrix<f64> Xr(T, k_r);
    Vector<f64> yy(T);
    for (std::size_t t = 0; t < T; ++t) {
        std::size_t idx = t + n_lags;
        yy[t] = y[idx];
        Xr(t, 0) = 1.0;
        for (std::size_t j = 0; j < n_lags; ++j) {
            Xr(t, j + 1) = y[idx - j - 1];
        }
    }
    auto beta_r = solve_least_squares(Xr, yy);
    if (!beta_r) {
        return Status::error(ErrorCode::SingularMatrix,
                             "granger_causality: restricted regression failed");
    }
    auto fitted_r = matvec(Xr, beta_r.value());
    f64 ssr_r = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        f64 r = yy[t] - fitted_r[t];
        ssr_r += r * r;
    }

    // Unrestricted: y_t on lags of y AND lags of x
    std::size_t k_u = 2 * n_lags + 1;
    Matrix<f64> Xu(T, k_u);
    for (std::size_t t = 0; t < T; ++t) {
        std::size_t idx = t + n_lags;
        Xu(t, 0) = 1.0;
        for (std::size_t j = 0; j < n_lags; ++j) {
            Xu(t, j + 1) = y[idx - j - 1];
        }
        for (std::size_t j = 0; j < n_lags; ++j) {
            Xu(t, n_lags + j + 1) = x[idx - j - 1];
        }
    }
    auto beta_u = solve_least_squares(Xu, yy);
    if (!beta_u) {
        return Status::error(ErrorCode::SingularMatrix,
                             "granger_causality: unrestricted regression failed");
    }
    auto fitted_u = matvec(Xu, beta_u.value());
    f64 ssr_u = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        f64 r = yy[t] - fitted_u[t];
        ssr_u += r * r;
    }

    f64 df1 = static_cast<f64>(n_lags);
    f64 df2 = static_cast<f64>(T - k_u);
    f64 F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2);

    GrangerResult result;
    result.n_lags = n_lags;
    result.n_obs = n;
    result.ssr_restricted = ssr_r;
    result.ssr_unrestricted = ssr_u;
    result.f_statistic = F;
    result.p_value = f_sf(F, df1, df2);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

} // namespace hfm
