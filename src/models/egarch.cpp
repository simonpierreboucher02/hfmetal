#include "hfm/models/egarch.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <algorithm>

namespace hfm {

namespace {

struct EGARCHParams {
    f64 omega, alpha, gamma, beta;
};

f64 compute_egarch_ll(const Vector<f64>& returns, f64 mu, const EGARCHParams& p) {
    std::size_t n = returns.size();
    f64 fn = static_cast<f64>(n);
    f64 sample_var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 e = returns[i] - mu;
        sample_var += e * e;
    }
    sample_var /= fn;

    f64 log_h = std::log(sample_var);
    f64 ll = 0.0;
    f64 half_log_2pi = 0.5 * std::log(2.0 * std::numbers::pi);

    for (std::size_t t = 0; t < n; ++t) {
        f64 h = std::exp(log_h);
        f64 e = returns[t] - mu;
        ll += -half_log_2pi - 0.5 * log_h - 0.5 * e * e / h;

        if (t < n - 1) {
            f64 z = e / std::sqrt(h);
            log_h = p.omega + p.alpha * std::abs(z) + p.gamma * z + p.beta * log_h;
        }
    }
    return ll;
}

} // namespace

f64 egarch_loglikelihood(const Vector<f64>& returns, f64 mu,
                          f64 omega, f64 alpha, f64 gamma, f64 beta) {
    return compute_egarch_ll(returns, mu, {omega, alpha, gamma, beta});
}

Result<EGARCHResult> egarch(const Vector<f64>& returns, const EGARCHOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 10) {
        return Status::error(ErrorCode::InvalidArgument,
                             "egarch: need at least 10 observations");
    }

    f64 mu = returns.mean();
    EGARCHParams best{opts.omega_init, opts.alpha_init, opts.gamma_init, opts.beta_init};
    f64 best_ll = compute_egarch_ll(returns, mu, best);

    f64 simplex_size = 0.3;
    constexpr std::size_t dim = 4;
    std::array<EGARCHParams, dim + 1> simplex;
    std::array<f64, dim + 1> fvals;

    simplex[0] = best;
    fvals[0] = -best_ll;

    auto perturb = [&](std::size_t idx, f64 delta) {
        EGARCHParams p = best;
        switch (idx) {
            case 0: p.omega += delta; break;
            case 1: p.alpha += delta * 0.5; break;
            case 2: p.gamma += delta * 0.5; break;
            case 3: p.beta += delta * 0.3; break;
        }
        return p;
    };

    for (std::size_t i = 0; i < dim; ++i) {
        simplex[i + 1] = perturb(i, simplex_size);
        fvals[i + 1] = -compute_egarch_ll(returns, mu, simplex[i + 1]);
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

        EGARCHParams centroid{0, 0, 0, 0};
        for (std::size_t i = 0; i <= dim; ++i) {
            if (i == worst) continue;
            centroid.omega += simplex[i].omega;
            centroid.alpha += simplex[i].alpha;
            centroid.gamma += simplex[i].gamma;
            centroid.beta += simplex[i].beta;
        }
        f64 inv = 1.0 / static_cast<f64>(dim);
        centroid.omega *= inv;
        centroid.alpha *= inv;
        centroid.gamma *= inv;
        centroid.beta *= inv;

        EGARCHParams reflected{
            2.0 * centroid.omega - simplex[worst].omega,
            2.0 * centroid.alpha - simplex[worst].alpha,
            2.0 * centroid.gamma - simplex[worst].gamma,
            2.0 * centroid.beta - simplex[worst].beta
        };
        f64 fr = -compute_egarch_ll(returns, mu, reflected);

        if (fr < fvals[best_idx]) {
            EGARCHParams expanded{
                3.0 * centroid.omega - 2.0 * simplex[worst].omega,
                3.0 * centroid.alpha - 2.0 * simplex[worst].alpha,
                3.0 * centroid.gamma - 2.0 * simplex[worst].gamma,
                3.0 * centroid.beta - 2.0 * simplex[worst].beta
            };
            f64 fe = -compute_egarch_ll(returns, mu, expanded);
            if (fe < fr) {
                simplex[worst] = expanded;
                fvals[worst] = fe;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = fr;
            }
        } else if (fr < fvals[worst]) {
            simplex[worst] = reflected;
            fvals[worst] = fr;
        } else {
            EGARCHParams contracted{
                0.5 * (centroid.omega + simplex[worst].omega),
                0.5 * (centroid.alpha + simplex[worst].alpha),
                0.5 * (centroid.gamma + simplex[worst].gamma),
                0.5 * (centroid.beta + simplex[worst].beta)
            };
            f64 fc = -compute_egarch_ll(returns, mu, contracted);
            if (fc < fvals[worst]) {
                simplex[worst] = contracted;
                fvals[worst] = fc;
            } else {
                for (std::size_t i = 0; i <= dim; ++i) {
                    if (i == best_idx) continue;
                    simplex[i].omega = 0.5 * (simplex[i].omega + simplex[best_idx].omega);
                    simplex[i].alpha = 0.5 * (simplex[i].alpha + simplex[best_idx].alpha);
                    simplex[i].gamma = 0.5 * (simplex[i].gamma + simplex[best_idx].gamma);
                    simplex[i].beta = 0.5 * (simplex[i].beta + simplex[best_idx].beta);
                    fvals[i] = -compute_egarch_ll(returns, mu, simplex[i]);
                }
            }
        }
    }

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i <= dim; ++i) {
        if (fvals[i] < fvals[best_idx]) best_idx = i;
    }
    best = simplex[best_idx];
    best_ll = -fvals[best_idx];

    EGARCHResult result;
    result.omega = best.omega;
    result.alpha = best.alpha;
    result.gamma = best.gamma;
    result.beta = best.beta;
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

    f64 log_h = std::log(sample_var);
    for (std::size_t t = 0; t < n; ++t) {
        f64 h = std::exp(log_h);
        result.conditional_var[t] = h;
        f64 e = returns[t] - mu;
        result.std_residuals[t] = e / std::sqrt(h);

        if (t < n - 1) {
            f64 z = result.std_residuals[t];
            log_h = best.omega + best.alpha * std::abs(z) + best.gamma * z + best.beta * log_h;
        }
    }

    if (opts.compute_std_errors) {
        f64 eps = 1e-5;
        std::array<f64, 4> params = {best.omega, best.alpha, best.gamma, best.beta};
        result.std_errors.resize(4);
        result.t_stats.resize(4);

        for (std::size_t j = 0; j < 4; ++j) {
            std::array<f64, 4> p_plus = params, p_minus = params;
            f64 h = std::max(eps, std::abs(params[j]) * eps);
            p_plus[j] += h;
            p_minus[j] -= h;

            f64 ll_plus = compute_egarch_ll(returns, mu,
                {p_plus[0], p_plus[1], p_plus[2], p_plus[3]});
            f64 ll_minus = compute_egarch_ll(returns, mu,
                {p_minus[0], p_minus[1], p_minus[2], p_minus[3]});

            f64 d2 = (ll_plus - 2.0 * best_ll + ll_minus) / (h * h);
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
