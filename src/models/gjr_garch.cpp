#include "hfm/models/gjr_garch.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <array>

namespace hfm {

namespace {

struct GJRParams {
    f64 omega, alpha, gamma, beta;
};

f64 compute_gjr_ll(const Vector<f64>& returns, f64 mu, const GJRParams& p) {
    std::size_t n = returns.size();
    f64 fn = static_cast<f64>(n);

    if (p.omega <= 0.0 || p.alpha < 0.0 || p.gamma < 0.0 || p.beta < 0.0)
        return -1e20;
    if (p.alpha + p.gamma / 2.0 + p.beta >= 1.0)
        return -1e20;

    f64 sample_var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 e = returns[i] - mu;
        sample_var += e * e;
    }
    sample_var /= fn;

    f64 h = sample_var;
    f64 ll = 0.0;
    f64 half_log_2pi = 0.5 * std::log(2.0 * std::numbers::pi);

    for (std::size_t t = 0; t < n; ++t) {
        if (h < 1e-20) h = 1e-20;
        f64 e = returns[t] - mu;
        ll += -half_log_2pi - 0.5 * std::log(h) - 0.5 * e * e / h;

        f64 indicator = (e < 0.0) ? 1.0 : 0.0;
        h = p.omega + p.alpha * e * e + p.gamma * e * e * indicator + p.beta * h;
    }
    return ll;
}

f64 logit(f64 x) { return std::log(x / (1.0 - x)); }
f64 inv_logit(f64 x) { return 1.0 / (1.0 + std::exp(-x)); }

GJRParams from_unconstrained(const std::array<f64, 4>& u) {
    GJRParams p;
    p.omega = std::exp(u[0]);
    p.alpha = inv_logit(u[1]) * 0.3;
    p.gamma = inv_logit(u[2]) * 0.3;
    p.beta = inv_logit(u[3]) * 0.99;
    if (p.alpha + p.gamma / 2.0 + p.beta >= 0.999) {
        p.beta = 0.999 - p.alpha - p.gamma / 2.0;
        if (p.beta < 0.0) p.beta = 0.01;
    }
    return p;
}

} // namespace

f64 gjr_garch_loglikelihood(const Vector<f64>& returns, f64 mu,
                              f64 omega, f64 alpha, f64 gamma, f64 beta) {
    return compute_gjr_ll(returns, mu, {omega, alpha, gamma, beta});
}

Result<GJRGARCHResult> gjr_garch(const Vector<f64>& returns, const GJRGARCHOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 10) {
        return Status::error(ErrorCode::InvalidArgument,
                             "gjr_garch: need at least 10 observations");
    }

    f64 mu = returns.mean();

    constexpr std::size_t dim = 4;
    std::array<f64, dim> best_u = {
        std::log(opts.omega_init),
        logit(opts.alpha_init / 0.3),
        logit(opts.gamma_init / 0.3),
        logit(opts.beta_init / 0.99)
    };

    auto eval = [&](const std::array<f64, dim>& u) -> f64 {
        auto p = from_unconstrained(u);
        return -compute_gjr_ll(returns, mu, p);
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
    auto best = from_unconstrained(simplex[best_idx]);
    f64 best_ll = -fvals[best_idx];

    GJRGARCHResult result;
    result.omega = best.omega;
    result.alpha = best.alpha;
    result.gamma = best.gamma;
    result.beta = best.beta;
    result.persistence = best.alpha + best.gamma / 2.0 + best.beta;
    result.unconditional_var = (result.persistence < 1.0) ?
        best.omega / (1.0 - result.persistence) : best.omega;
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
        f64 indicator = (e < 0.0) ? 1.0 : 0.0;
        h = best.omega + best.alpha * e * e + best.gamma * e * e * indicator + best.beta * h;
        if (h < 1e-20) h = 1e-20;
    }

    if (opts.compute_std_errors) {
        f64 eps = 1e-5;
        std::array<f64, 4> params = {best.omega, best.alpha, best.gamma, best.beta};
        result.std_errors.resize(4);
        result.t_stats.resize(4);
        for (std::size_t j = 0; j < 4; ++j) {
            std::array<f64, 4> pp = params, pm = params;
            f64 h_step = std::max(eps, std::abs(params[j]) * eps);
            pp[j] += h_step;
            pm[j] -= h_step;
            f64 ll_p = compute_gjr_ll(returns, mu, {pp[0], pp[1], pp[2], pp[3]});
            f64 ll_m = compute_gjr_ll(returns, mu, {pm[0], pm[1], pm[2], pm[3]});
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
