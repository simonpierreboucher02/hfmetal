#include "hfm/models/garch.hpp"
#include <chrono>
#include <cmath>
#include <numbers>
#include <algorithm>

namespace hfm {

f64 garch_loglikelihood(const Vector<f64>& returns, f64 mu,
                         f64 omega, f64 alpha, f64 beta) {
    std::size_t n = returns.size();
    if (n < 2) return -1e30;
    if (omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + beta >= 1.0)
        return -1e30;

    f64 ll = 0.0;
    f64 var_unconditional = omega / (1.0 - alpha - beta);
    f64 h = var_unconditional;
    constexpr f64 log2pi = 1.8378770664093453; // log(2*pi)

    for (std::size_t t = 0; t < n; ++t) {
        if (h < 1e-20) h = 1e-20;
        f64 e = returns[t] - mu;
        ll += -0.5 * (log2pi + std::log(h) + e * e / h);
        if (t < n - 1) {
            h = omega + alpha * e * e + beta * h;
        }
    }
    return ll;
}

namespace {

struct GARCHParams {
    f64 mu, omega, alpha, beta;
};

f64 eval_ll(const Vector<f64>& returns, const GARCHParams& p) {
    return garch_loglikelihood(returns, p.mu, p.omega, p.alpha, p.beta);
}

// Nelder-Mead simplex optimization (internal params in log/logit space)
struct TransformedParams {
    f64 log_omega;
    f64 logit_alpha;
    f64 logit_beta;
};

f64 logit(f64 x) { return std::log(x / (1.0 - x)); }
f64 inv_logit(f64 x) { return 1.0 / (1.0 + std::exp(-x)); }

GARCHParams from_transformed(f64 mu, const TransformedParams& tp) {
    f64 omega = std::exp(tp.log_omega);
    f64 alpha = inv_logit(tp.logit_alpha) * 0.5; // constrain to [0, 0.5]
    f64 beta_max = 0.999 - alpha;
    f64 beta = inv_logit(tp.logit_beta) * beta_max;
    return {mu, omega, alpha, beta};
}

} // namespace

Result<GARCHResult> garch(const Vector<f64>& returns, const GARCHOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();

    if (n < 10) {
        return Status::error(ErrorCode::InvalidArgument, "garch: need at least 10 observations");
    }

    // Compute sample mean
    f64 mu = returns.mean();
    f64 sample_var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 e = returns[i] - mu;
        sample_var += e * e;
    }
    sample_var /= static_cast<f64>(n);

    // Initialize parameters
    f64 omega = opts.omega_init;
    f64 alpha = opts.alpha_init;
    f64 beta = opts.beta_init;

    // Nelder-Mead on transformed parameters
    struct Vertex {
        TransformedParams p;
        f64 val;
    };

    auto make_vertex = [&](f64 lo, f64 la, f64 lb) -> Vertex {
        TransformedParams tp{lo, la, lb};
        auto gp = from_transformed(mu, tp);
        return {tp, -eval_ll(returns, gp)};
    };

    f64 lo0 = std::log(omega);
    f64 la0 = logit(alpha / 0.5);
    f64 lb0 = logit(beta / (0.999 - alpha));

    constexpr std::size_t dim = 3;
    std::array<Vertex, dim + 1> simplex;
    simplex[0] = make_vertex(lo0, la0, lb0);
    simplex[1] = make_vertex(lo0 + 1.0, la0, lb0);
    simplex[2] = make_vertex(lo0, la0 + 1.0, lb0);
    simplex[3] = make_vertex(lo0, la0, lb0 + 1.0);

    auto get_arr = [](const TransformedParams& p) -> std::array<f64, 3> {
        return {p.log_omega, p.logit_alpha, p.logit_beta};
    };
    bool converged = false;
    std::size_t iter = 0;

    for (; iter < opts.max_iter; ++iter) {
        std::sort(simplex.begin(), simplex.end(),
                  [](const Vertex& a, const Vertex& b) { return a.val < b.val; });

        f64 range = std::abs(simplex[dim].val - simplex[0].val);
        if (range < opts.tol) { converged = true; break; }

        // Centroid (exclude worst)
        std::array<f64, 3> centroid = {0, 0, 0};
        for (std::size_t i = 0; i < dim; ++i) {
            auto a = get_arr(simplex[i].p);
            for (std::size_t j = 0; j < dim; ++j) centroid[j] += a[j];
        }
        for (auto& c : centroid) c /= static_cast<f64>(dim);

        auto worst = get_arr(simplex[dim].p);

        // Reflect
        std::array<f64, 3> reflected;
        for (std::size_t j = 0; j < dim; ++j)
            reflected[j] = 2.0 * centroid[j] - worst[j];
        auto vr = make_vertex(reflected[0], reflected[1], reflected[2]);

        if (vr.val < simplex[0].val) {
            // Expand
            std::array<f64, 3> expanded;
            for (std::size_t j = 0; j < dim; ++j)
                expanded[j] = 3.0 * centroid[j] - 2.0 * worst[j];
            auto ve = make_vertex(expanded[0], expanded[1], expanded[2]);
            simplex[dim] = (ve.val < vr.val) ? ve : vr;
        } else if (vr.val < simplex[dim - 1].val) {
            simplex[dim] = vr;
        } else {
            // Contract
            std::array<f64, 3> contracted;
            if (vr.val < simplex[dim].val) {
                for (std::size_t j = 0; j < dim; ++j)
                    contracted[j] = 0.5 * (centroid[j] + reflected[j]);
            } else {
                for (std::size_t j = 0; j < dim; ++j)
                    contracted[j] = 0.5 * (centroid[j] + worst[j]);
            }
            auto vc = make_vertex(contracted[0], contracted[1], contracted[2]);
            if (vc.val < simplex[dim].val) {
                simplex[dim] = vc;
            } else {
                // Shrink
                auto best = get_arr(simplex[0].p);
                for (std::size_t i = 1; i <= dim; ++i) {
                    auto a = get_arr(simplex[i].p);
                    std::array<f64, 3> shrunk;
                    for (std::size_t j = 0; j < dim; ++j)
                        shrunk[j] = 0.5 * (best[j] + a[j]);
                    simplex[i] = make_vertex(shrunk[0], shrunk[1], shrunk[2]);
                }
            }
        }
    }

    std::sort(simplex.begin(), simplex.end(),
              [](const Vertex& a, const Vertex& b) { return a.val < b.val; });

    auto best_params = from_transformed(mu, simplex[0].p);

    GARCHResult result;
    result.omega = best_params.omega;
    result.alpha = best_params.alpha;
    result.beta = best_params.beta;
    result.persistence = result.alpha + result.beta;
    result.unconditional_var = (result.persistence < 1.0)
        ? result.omega / (1.0 - result.persistence) : sample_var;
    result.n_obs = n;
    result.n_iter = iter;
    result.converged = converged;
    result.log_likelihood = eval_ll(returns, best_params);
    result.aic = -2.0 * result.log_likelihood + 2.0 * 4.0; // 4 params: mu, omega, alpha, beta
    result.bic = -2.0 * result.log_likelihood + std::log(static_cast<f64>(n)) * 4.0;

    // Compute conditional variance and standardized residuals
    result.conditional_var = Vector<f64>(n);
    result.std_residuals = Vector<f64>(n);
    f64 h = result.unconditional_var;
    for (std::size_t t = 0; t < n; ++t) {
        if (h < 1e-20) h = 1e-20;
        result.conditional_var[t] = h;
        f64 e = returns[t] - mu;
        result.std_residuals[t] = e / std::sqrt(h);
        h = result.omega + result.alpha * e * e + result.beta * h;
    }

    // Numerical standard errors via finite differences of Hessian
    if (opts.compute_std_errors) {
        result.std_errors = Vector<f64>(3);
        result.t_stats = Vector<f64>(3);
        std::array<f64, 3> params = {result.omega, result.alpha, result.beta};

        for (std::size_t i = 0; i < 3; ++i) {
            f64 h_step = std::max(1e-6, std::abs(params[i]) * 1e-4);
            auto p_plus = best_params;
            auto p_minus = best_params;
            f64* pp = (i == 0) ? &p_plus.omega : (i == 1) ? &p_plus.alpha : &p_plus.beta;
            f64* pm = (i == 0) ? &p_minus.omega : (i == 1) ? &p_minus.alpha : &p_minus.beta;
            *pp += h_step;
            *pm -= h_step;
            f64 ll_plus = eval_ll(returns, p_plus);
            f64 ll_minus = eval_ll(returns, p_minus);
            f64 ll_center = result.log_likelihood;
            f64 d2 = (ll_plus - 2.0 * ll_center + ll_minus) / (h_step * h_step);
            if (d2 < 0.0) {
                result.std_errors[i] = std::sqrt(-1.0 / d2);
            } else {
                result.std_errors[i] = 0.0;
            }
            result.t_stats[i] = (result.std_errors[i] > 0.0)
                ? params[i] / result.std_errors[i] : 0.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
