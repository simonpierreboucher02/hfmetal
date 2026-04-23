#include "hfm/models/garch_t.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <array>

namespace hfm {

namespace {

f64 compute_garch_t_ll(const Vector<f64>& returns, f64 mu,
                        f64 omega, f64 alpha, f64 beta, f64 nu) {
    std::size_t n = returns.size();
    f64 fn = static_cast<f64>(n);

    if (omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + beta >= 1.0 || nu <= 2.0)
        return -1e20;

    f64 sample_var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 e = returns[i] - mu;
        sample_var += e * e;
    }
    sample_var /= fn;

    f64 log_const = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
                    - 0.5 * std::log((nu - 2.0) * std::numbers::pi);

    f64 h = sample_var;
    f64 ll = 0.0;

    for (std::size_t t = 0; t < n; ++t) {
        if (h < 1e-20) h = 1e-20;
        f64 e = returns[t] - mu;
        f64 z2 = e * e / h;
        ll += log_const - 0.5 * std::log(h)
              - (nu + 1.0) / 2.0 * std::log(1.0 + z2 / (nu - 2.0));
        h = omega + alpha * e * e + beta * h;
    }
    return ll;
}

f64 logit(f64 x) { return std::log(x / (1.0 - x)); }
f64 inv_logit(f64 x) { return 1.0 / (1.0 + std::exp(-x)); }

} // namespace

Result<GARCHTResult> garch_t(const Vector<f64>& returns, const GARCHTOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 10) {
        return Status::error(ErrorCode::InvalidArgument,
                             "garch_t: need at least 10 observations");
    }

    f64 mu = returns.mean();

    constexpr std::size_t dim = 4;
    std::array<f64, dim> best_u = {
        std::log(opts.omega_init),
        logit(opts.alpha_init / 0.5),
        logit(opts.beta_init / 0.999),
        std::log(opts.nu_init - 2.0)
    };

    auto eval = [&](const std::array<f64, dim>& u) -> f64 {
        f64 om = std::exp(u[0]);
        f64 al = inv_logit(u[1]) * 0.5;
        f64 be = inv_logit(u[2]) * 0.999;
        f64 v = std::exp(u[3]) + 2.01;
        if (al + be >= 0.999) be = 0.999 - al - 0.001;
        if (be < 0.0) be = 0.01;
        return -compute_garch_t_ll(returns, mu, om, al, be, v);
    };

    std::array<std::array<f64, dim>, dim + 1> simplex;
    std::array<f64, dim + 1> fvals;
    simplex[0] = best_u;
    fvals[0] = eval(best_u);
    for (std::size_t i = 0; i < dim; ++i) {
        simplex[i + 1] = best_u;
        simplex[i + 1][i] += 0.3;
        fvals[i + 1] = eval(simplex[i + 1]);
    }

    bool converged = false;
    std::size_t iter = 0;

    for (; iter < opts.max_iter; ++iter) {
        std::size_t worst = 0, best_idx = 0;
        for (std::size_t i = 1; i <= dim; ++i) {
            if (fvals[i] > fvals[worst]) worst = i;
            if (fvals[i] < fvals[best_idx]) best_idx = i;
        }
        if (std::abs(fvals[worst] - fvals[best_idx]) < opts.tol) {
            converged = true;
            break;
        }

        std::array<f64, dim> centroid{};
        for (std::size_t i = 0; i <= dim; ++i) {
            if (i == worst) continue;
            for (std::size_t j = 0; j < dim; ++j) centroid[j] += simplex[i][j];
        }
        for (std::size_t j = 0; j < dim; ++j) centroid[j] /= static_cast<f64>(dim);

        std::array<f64, dim> reflected;
        for (std::size_t j = 0; j < dim; ++j)
            reflected[j] = 2.0 * centroid[j] - simplex[worst][j];
        f64 fr = eval(reflected);

        if (fr < fvals[best_idx]) {
            std::array<f64, dim> expanded;
            for (std::size_t j = 0; j < dim; ++j)
                expanded[j] = 3.0 * centroid[j] - 2.0 * simplex[worst][j];
            f64 fe = eval(expanded);
            simplex[worst] = (fe < fr) ? expanded : reflected;
            fvals[worst] = std::min(fe, fr);
        } else if (fr < fvals[worst]) {
            simplex[worst] = reflected;
            fvals[worst] = fr;
        } else {
            std::array<f64, dim> contracted;
            for (std::size_t j = 0; j < dim; ++j)
                contracted[j] = 0.5 * (centroid[j] + simplex[worst][j]);
            f64 fc = eval(contracted);
            if (fc < fvals[worst]) {
                simplex[worst] = contracted;
                fvals[worst] = fc;
            } else {
                for (std::size_t i = 0; i <= dim; ++i) {
                    if (i == best_idx) continue;
                    for (std::size_t j = 0; j < dim; ++j)
                        simplex[i][j] = 0.5 * (simplex[i][j] + simplex[best_idx][j]);
                    fvals[i] = eval(simplex[i]);
                }
            }
        }
    }

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i <= dim; ++i) {
        if (fvals[i] < fvals[best_idx]) best_idx = i;
    }
    auto& bu = simplex[best_idx];
    f64 omega = std::exp(bu[0]);
    f64 alpha = inv_logit(bu[1]) * 0.5;
    f64 beta = inv_logit(bu[2]) * 0.999;
    f64 nu = std::exp(bu[3]) + 2.01;
    if (alpha + beta >= 0.999) beta = 0.999 - alpha - 0.001;
    if (beta < 0.0) beta = 0.01;

    f64 best_ll = compute_garch_t_ll(returns, mu, omega, alpha, beta, nu);

    GARCHTResult result;
    result.omega = omega;
    result.alpha = alpha;
    result.beta = beta;
    result.nu = nu;
    result.persistence = alpha + beta;
    result.unconditional_var = (result.persistence < 1.0) ?
        omega / (1.0 - result.persistence) : omega;
    result.n_obs = n;
    result.n_iter = iter;
    result.converged = converged;
    result.log_likelihood = best_ll;

    f64 fn = static_cast<f64>(n);
    result.aic = -2.0 * best_ll + 2.0 * 4.0;
    result.bic = -2.0 * best_ll + 4.0 * std::log(fn);

    result.conditional_var.resize(n);
    result.std_residuals.resize(n);
    f64 sample_var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 e = returns[i] - mu;
        sample_var += e * e;
    }
    sample_var /= fn;

    f64 h = sample_var;
    for (std::size_t t = 0; t < n; ++t) {
        result.conditional_var[t] = h;
        f64 e = returns[t] - mu;
        result.std_residuals[t] = e / std::sqrt(h);
        h = omega + alpha * e * e + beta * h;
        if (h < 1e-20) h = 1e-20;
    }

    if (opts.compute_std_errors) {
        f64 eps = 1e-5;
        std::array<f64, 4> params = {omega, alpha, beta, nu};
        result.std_errors.resize(4);
        result.t_stats.resize(4);
        for (std::size_t j = 0; j < 4; ++j) {
            auto pp = params; auto pm = params;
            f64 h_step = std::max(eps, std::abs(params[j]) * eps);
            pp[j] += h_step; pm[j] -= h_step;
            f64 ll_p = compute_garch_t_ll(returns, mu, pp[0], pp[1], pp[2], pp[3]);
            f64 ll_m = compute_garch_t_ll(returns, mu, pm[0], pm[1], pm[2], pm[3]);
            f64 d2 = (ll_p - 2.0 * best_ll + ll_m) / (h_step * h_step);
            f64 info = -d2;
            result.std_errors[j] = (info > 0.0) ? std::sqrt(1.0 / info) : 0.0;
            result.t_stats[j] = (result.std_errors[j] > 0.0) ?
                params[j] / result.std_errors[j] : 0.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

} // namespace hfm
