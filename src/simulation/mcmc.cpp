#include "hfm/simulation/mcmc.hpp"
#include "hfm/models/garch.hpp"
#include "hfm/runtime/thread_pool.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numbers>

namespace hfm {

namespace {

void compute_summary(MCMCChain& chain) {
    std::size_t n = chain.n_kept;
    std::size_t d = chain.n_dim;
    if (n == 0 || d == 0) return;

    chain.mean = Vector<f64>(d, 0.0);
    chain.std_dev = Vector<f64>(d, 0.0);
    chain.median = Vector<f64>(d);
    chain.ci_lower = Vector<f64>(d);
    chain.ci_upper = Vector<f64>(d);
    chain.posterior_cov = Matrix<f64>(d, d, 0.0);

    // Means
    for (std::size_t j = 0; j < d; ++j) {
        f64 sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += chain.samples(i, j);
        chain.mean[j] = sum / static_cast<f64>(n);
    }

    // Covariance, std, quantiles
    for (std::size_t j = 0; j < d; ++j) {
        std::vector<f64> vals(n);
        for (std::size_t i = 0; i < n; ++i) vals[i] = chain.samples(i, j);

        f64 ss = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            f64 diff = vals[i] - chain.mean[j];
            ss += diff * diff;
        }
        chain.std_dev[j] = std::sqrt(ss / static_cast<f64>(n - 1));

        std::sort(vals.begin(), vals.end());
        chain.median[j] = vals[n / 2];
        chain.ci_lower[j] = vals[static_cast<std::size_t>(0.025 * static_cast<f64>(n))];
        chain.ci_upper[j] = vals[static_cast<std::size_t>(0.975 * static_cast<f64>(n))];
    }

    for (std::size_t j1 = 0; j1 < d; ++j1) {
        for (std::size_t j2 = j1; j2 < d; ++j2) {
            f64 cov = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                cov += (chain.samples(i, j1) - chain.mean[j1])
                     * (chain.samples(i, j2) - chain.mean[j2]);
            }
            cov /= static_cast<f64>(n - 1);
            chain.posterior_cov(j1, j2) = cov;
            chain.posterior_cov(j2, j1) = cov;
        }
    }
}

f64 log_student_t_pdf(f64 x, f64 nu, f64 sigma2) {
    f64 z2 = x * x / sigma2;
    return std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
         - 0.5 * std::log(nu * std::numbers::pi * sigma2)
         - (nu + 1.0) / 2.0 * std::log(1.0 + z2 / nu);
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Diagnostics
// ═══════════════════════════════════════════════════════════════════════════

f64 effective_sample_size(const f64* chain, std::size_t n) {
    if (n < 4) return static_cast<f64>(n);

    f64 mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += chain[i];
    mean /= static_cast<f64>(n);

    f64 var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        f64 d = chain[i] - mean;
        var += d * d;
    }
    var /= static_cast<f64>(n);
    if (var < 1e-30) return static_cast<f64>(n);

    f64 sum_rho = 0.0;
    std::size_t max_lag = std::min(n - 1, n / 2);
    for (std::size_t lag = 1; lag < max_lag; ++lag) {
        f64 autocov = 0.0;
        for (std::size_t i = 0; i < n - lag; ++i)
            autocov += (chain[i] - mean) * (chain[i + lag] - mean);
        autocov /= static_cast<f64>(n);
        f64 rho = autocov / var;
        if (rho < 0.05) break;
        sum_rho += rho;
    }
    return std::max(1.0, static_cast<f64>(n) / (1.0 + 2.0 * sum_rho));
}

f64 gelman_rubin_rhat(const Matrix<f64>& chains) {
    std::size_t m = chains.rows();
    std::size_t n = chains.cols();
    if (m < 2 || n < 2) return 1.0;

    std::vector<f64> cm(m, 0.0), cv(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) cm[i] += chains(i, j);
        cm[i] /= static_cast<f64>(n);
        for (std::size_t j = 0; j < n; ++j) {
            f64 d = chains(i, j) - cm[i]; cv[i] += d * d;
        }
        cv[i] /= static_cast<f64>(n - 1);
    }
    f64 gm = 0.0;
    for (auto v : cm) gm += v;
    gm /= static_cast<f64>(m);

    f64 B = 0.0;
    for (auto v : cm) { f64 d = v - gm; B += d * d; }
    B *= static_cast<f64>(n) / static_cast<f64>(m - 1);

    f64 W = 0.0;
    for (auto v : cv) W += v;
    W /= static_cast<f64>(m);

    if (W < 1e-30) return 1.0;
    f64 var_hat = (1.0 - 1.0 / static_cast<f64>(n)) * W + B / static_cast<f64>(n);
    return std::sqrt(var_hat / W);
}

Vector<f64> chain_autocorrelation(const f64* chain, std::size_t n, std::size_t max_lag) {
    Vector<f64> acf(max_lag, 0.0);
    if (n < 2) return acf;
    f64 mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += chain[i];
    mean /= static_cast<f64>(n);
    f64 var = 0.0;
    for (std::size_t i = 0; i < n; ++i) { f64 d = chain[i] - mean; var += d * d; }
    if (var < 1e-30) return acf;
    for (std::size_t lag = 0; lag < max_lag && lag < n; ++lag) {
        f64 cov = 0.0;
        for (std::size_t i = 0; i < n - lag; ++i)
            cov += (chain[i] - mean) * (chain[i + lag] - mean);
        acf[lag] = cov / var;
    }
    return acf;
}

f64 geweke_test(const f64* chain, std::size_t n, f64 frac_first, f64 frac_last) {
    auto n1 = static_cast<std::size_t>(frac_first * static_cast<f64>(n));
    auto n2 = static_cast<std::size_t>(frac_last * static_cast<f64>(n));
    if (n1 < 2 || n2 < 2 || n1 + n2 > n) return 0.0;

    f64 mean_a = 0.0, mean_b = 0.0;
    for (std::size_t i = 0; i < n1; ++i) mean_a += chain[i];
    mean_a /= static_cast<f64>(n1);
    for (std::size_t i = n - n2; i < n; ++i) mean_b += chain[i];
    mean_b /= static_cast<f64>(n2);

    f64 var_a = 0.0, var_b = 0.0;
    for (std::size_t i = 0; i < n1; ++i) { f64 d = chain[i] - mean_a; var_a += d * d; }
    var_a /= static_cast<f64>(n1 - 1);
    for (std::size_t i = n - n2; i < n; ++i) { f64 d = chain[i] - mean_b; var_b += d * d; }
    var_b /= static_cast<f64>(n2 - 1);

    f64 se = std::sqrt(var_a / static_cast<f64>(n1) + var_b / static_cast<f64>(n2));
    return (se > 1e-30) ? (mean_a - mean_b) / se : 0.0;
}

void compute_chain_diagnostics(MCMCChain& chain) {
    std::size_t n = chain.n_kept;
    std::size_t d = chain.n_dim;
    chain.ess = Vector<f64>(d);
    chain.autocorr_lag1 = Vector<f64>(d);
    chain.geweke_z = Vector<f64>(d);

    for (std::size_t j = 0; j < d; ++j) {
        std::vector<f64> col(n);
        for (std::size_t i = 0; i < n; ++i) col[i] = chain.samples(i, j);
        chain.ess[j] = effective_sample_size(col.data(), n);
        auto acf = chain_autocorrelation(col.data(), n, 2);
        chain.autocorr_lag1[j] = (acf.size() > 1) ? acf[1] : 0.0;
        chain.geweke_z[j] = geweke_test(col.data(), n);
    }
}

f64 log_marginal_likelihood_harmonic(const Vector<f64>& log_likelihoods) {
    std::size_t n = log_likelihoods.size();
    if (n == 0) return 0.0;
    f64 max_nll = -1e30;
    for (std::size_t i = 0; i < n; ++i)
        max_nll = std::max(max_nll, -log_likelihoods[i]);
    f64 sum_exp = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sum_exp += std::exp(-log_likelihoods[i] - max_nll);
    return -(std::log(sum_exp / static_cast<f64>(n)) + max_nll);
}

f64 compute_waic(const Matrix<f64>& pll) {
    std::size_t S = pll.rows();
    std::size_t n = pll.cols();
    if (S < 2 || n == 0) return 0.0;

    f64 lppd = 0.0;
    f64 p_waic = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        f64 max_ll = -1e30;
        for (std::size_t s = 0; s < S; ++s) max_ll = std::max(max_ll, pll(s, j));
        f64 sum_exp = 0.0;
        for (std::size_t s = 0; s < S; ++s) sum_exp += std::exp(pll(s, j) - max_ll);
        lppd += std::log(sum_exp / static_cast<f64>(S)) + max_ll;

        f64 mean_ll = 0.0;
        for (std::size_t s = 0; s < S; ++s) mean_ll += pll(s, j);
        mean_ll /= static_cast<f64>(S);
        f64 var_ll = 0.0;
        for (std::size_t s = 0; s < S; ++s) {
            f64 d = pll(s, j) - mean_ll; var_ll += d * d;
        }
        p_waic += var_ll / static_cast<f64>(S - 1);
    }
    return -2.0 * (lppd - p_waic);
}

// ═══════════════════════════════════════════════════════════════════════════
// Metropolis-Hastings
// ═══════════════════════════════════════════════════════════════════════════

Result<MCMCChain> metropolis_hastings(const Vector<f64>& initial,
                                       const LogDensityFn& log_density,
                                       const MHOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t d = initial.size();
    if (d == 0) return Status::error(ErrorCode::InvalidArgument, "mh: empty initial");

    std::size_t total = opts.burn_in + opts.n_samples * opts.thin;
    MCMCChain chain;
    chain.n_dim = d;
    chain.n_total = total;
    chain.samples = Matrix<f64>(opts.n_samples, d);
    chain.log_posterior = Vector<f64>(opts.n_samples);
    chain.sampler = "metropolis_hastings";

    Vector<f64> scale(d);
    if (opts.proposal_scale.size() == d) scale = opts.proposal_scale;
    else for (std::size_t j = 0; j < d; ++j) scale[j] = std::abs(initial[j]) * 0.1 + 1e-6;

    std::mt19937_64 rng(opts.seed);
    std::normal_distribution<f64> norm(0.0, 1.0);
    std::uniform_real_distribution<f64> unif(0.0, 1.0);

    Vector<f64> current = initial;
    f64 current_lp = log_density(current);
    std::size_t accepted = 0, kept = 0, adapt_accepted = 0;

    for (std::size_t iter = 0; iter < total; ++iter) {
        Vector<f64> proposal(d);
        for (std::size_t j = 0; j < d; ++j)
            proposal[j] = current[j] + scale[j] * norm(rng);

        f64 prop_lp = log_density(proposal);
        if (std::log(unif(rng)) < prop_lp - current_lp) {
            current = proposal;
            current_lp = prop_lp;
            ++accepted;
            ++adapt_accepted;
        }

        if (opts.adapt && iter < opts.burn_in && iter > 0
            && (iter % opts.adapt_interval) == 0) {
            f64 rate = static_cast<f64>(adapt_accepted) / static_cast<f64>(opts.adapt_interval);
            f64 factor = (rate > opts.target_accept_rate) ? 1.1 : 0.9;
            for (std::size_t j = 0; j < d; ++j) scale[j] *= factor;
            adapt_accepted = 0;
        }

        if (iter >= opts.burn_in && ((iter - opts.burn_in) % opts.thin) == 0
            && kept < opts.n_samples) {
            for (std::size_t j = 0; j < d; ++j) chain.samples(kept, j) = current[j];
            chain.log_posterior[kept] = current_lp;
            ++kept;
        }
    }

    chain.n_kept = kept;
    chain.acceptance_rate = static_cast<f64>(accepted) / static_cast<f64>(total);
    chain.proposal_scale = scale;
    compute_summary(chain);
    compute_chain_diagnostics(chain);

    auto end = std::chrono::high_resolution_clock::now();
    chain.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return chain;
}

// ═══════════════════════════════════════════════════════════════════════════
// Adaptive Metropolis (Haario et al. 2001)
// ═══════════════════════════════════════════════════════════════════════════

Result<MCMCChain> adaptive_metropolis(const Vector<f64>& initial,
                                       const LogDensityFn& log_density,
                                       const AMOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t d = initial.size();
    if (d == 0) return Status::error(ErrorCode::InvalidArgument, "am: empty initial");

    std::size_t total = opts.burn_in + opts.n_samples * opts.thin;
    f64 sd = opts.scale_factor * opts.scale_factor / static_cast<f64>(d);

    MCMCChain chain;
    chain.n_dim = d;
    chain.n_total = total;
    chain.samples = Matrix<f64>(opts.n_samples, d);
    chain.log_posterior = Vector<f64>(opts.n_samples);
    chain.sampler = "adaptive_metropolis";

    // Initial diagonal covariance
    Vector<f64> iscale(d);
    if (opts.initial_scale.size() == d) iscale = opts.initial_scale;
    else for (std::size_t j = 0; j < d; ++j) iscale[j] = std::abs(initial[j]) * 0.1 + 1e-6;

    // Running mean and covariance (Welford online)
    Vector<f64> run_mean(d, 0.0);
    Matrix<f64> run_cov(d, d, 0.0);
    for (std::size_t j = 0; j < d; ++j) {
        run_mean[j] = initial[j];
        run_cov(j, j) = iscale[j] * iscale[j];
    }

    std::mt19937_64 rng(opts.seed);
    std::normal_distribution<f64> norm(0.0, 1.0);
    std::uniform_real_distribution<f64> unif(0.0, 1.0);

    Vector<f64> current = initial;
    f64 current_lp = log_density(current);
    std::size_t accepted = 0, kept = 0, total_seen = 0;

    for (std::size_t iter = 0; iter < total; ++iter) {
        ++total_seen;
        Vector<f64> proposal(d);

        if (iter < opts.adapt_start) {
            for (std::size_t j = 0; j < d; ++j)
                proposal[j] = current[j] + iscale[j] * norm(rng);
        } else {
            // Cholesky-free: use eigendecomposition-free approach
            // Sample z ~ N(0, I), then proposal = current + sd * C * z + eps * I * z
            Vector<f64> z(d);
            for (std::size_t j = 0; j < d; ++j) z[j] = norm(rng);
            for (std::size_t j = 0; j < d; ++j) {
                proposal[j] = current[j] + opts.epsilon * z[j];
                for (std::size_t k = 0; k < d; ++k) {
                    proposal[j] += sd * run_cov(j, k) * z[k];
                }
            }
        }

        f64 prop_lp = log_density(proposal);
        if (std::log(unif(rng)) < prop_lp - current_lp) {
            current = proposal;
            current_lp = prop_lp;
            ++accepted;
        }

        // Update running mean and covariance
        f64 n_f = static_cast<f64>(total_seen);
        Vector<f64> old_mean = run_mean;
        for (std::size_t j = 0; j < d; ++j)
            run_mean[j] = old_mean[j] + (current[j] - old_mean[j]) / n_f;
        if (total_seen > 1) {
            for (std::size_t j = 0; j < d; ++j) {
                for (std::size_t k = j; k < d; ++k) {
                    f64 delta_j = current[j] - old_mean[j];
                    f64 delta_k = current[k] - run_mean[k];
                    run_cov(j, k) += (delta_j * delta_k - run_cov(j, k)) / n_f;
                    run_cov(k, j) = run_cov(j, k);
                }
            }
        }

        if (iter >= opts.burn_in && ((iter - opts.burn_in) % opts.thin) == 0
            && kept < opts.n_samples) {
            for (std::size_t j = 0; j < d; ++j) chain.samples(kept, j) = current[j];
            chain.log_posterior[kept] = current_lp;
            ++kept;
        }
    }

    chain.n_kept = kept;
    chain.acceptance_rate = static_cast<f64>(accepted) / static_cast<f64>(total);
    compute_summary(chain);
    compute_chain_diagnostics(chain);

    auto end = std::chrono::high_resolution_clock::now();
    chain.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return chain;
}

// ═══════════════════════════════════════════════════════════════════════════
// Hamiltonian Monte Carlo
// ═══════════════════════════════════════════════════════════════════════════

namespace {

Vector<f64> numerical_gradient(const LogDensityFn& f, const Vector<f64>& theta, f64 eps = 1e-5) {
    std::size_t d = theta.size();
    Vector<f64> grad(d);
    for (std::size_t j = 0; j < d; ++j) {
        Vector<f64> tp = theta, tm = theta;
        tp[j] += eps;
        tm[j] -= eps;
        grad[j] = (f(tp) - f(tm)) / (2.0 * eps);
    }
    return grad;
}

} // namespace

Result<MCMCChain> hmc(const Vector<f64>& initial,
                       const LogDensityFn& log_density,
                       const GradientFn& gradient,
                       const HMCOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t d = initial.size();
    if (d == 0) return Status::error(ErrorCode::InvalidArgument, "hmc: empty initial");

    std::size_t total = opts.burn_in + opts.n_samples * opts.thin;
    MCMCChain chain;
    chain.n_dim = d;
    chain.n_total = total;
    chain.samples = Matrix<f64>(opts.n_samples, d);
    chain.log_posterior = Vector<f64>(opts.n_samples);
    chain.sampler = "hmc";

    // Mass matrix (diagonal)
    Vector<f64> mass_inv(d, 1.0);
    if (opts.mass_diag.size() == d) mass_inv = opts.mass_diag;

    f64 epsilon = opts.step_size;
    std::size_t L = opts.n_leapfrog;

    std::mt19937_64 rng(opts.seed);
    std::normal_distribution<f64> norm(0.0, 1.0);
    std::uniform_real_distribution<f64> unif(0.0, 1.0);

    Vector<f64> current = initial;
    f64 current_lp = log_density(current);
    std::size_t accepted = 0, kept = 0;

    // Dual averaging for step size adaptation
    f64 mu = std::log(10.0 * epsilon);
    f64 log_eps_bar = std::log(epsilon);
    f64 H_bar = 0.0;
    f64 gamma_da = 0.05, t0 = 10.0, kappa = 0.75;

    for (std::size_t iter = 0; iter < total; ++iter) {
        // Sample momentum
        Vector<f64> p(d);
        for (std::size_t j = 0; j < d; ++j)
            p[j] = norm(rng) / std::sqrt(mass_inv[j]);

        Vector<f64> q = current;
        Vector<f64> p0 = p;
        f64 current_H = -current_lp;
        for (std::size_t j = 0; j < d; ++j)
            current_H += 0.5 * p0[j] * p0[j] * mass_inv[j];

        // Leapfrog integration
        auto grad = gradient(q);
        for (std::size_t j = 0; j < d; ++j)
            p[j] += 0.5 * epsilon * grad[j];

        for (std::size_t l = 0; l < L - 1; ++l) {
            for (std::size_t j = 0; j < d; ++j)
                q[j] += epsilon * mass_inv[j] * p[j];
            grad = gradient(q);
            for (std::size_t j = 0; j < d; ++j)
                p[j] += epsilon * grad[j];
        }
        for (std::size_t j = 0; j < d; ++j)
            q[j] += epsilon * mass_inv[j] * p[j];
        grad = gradient(q);
        for (std::size_t j = 0; j < d; ++j)
            p[j] += 0.5 * epsilon * grad[j];

        // Negate momentum (symmetric)
        for (std::size_t j = 0; j < d; ++j) p[j] = -p[j];

        f64 prop_lp = log_density(q);
        f64 proposed_H = -prop_lp;
        for (std::size_t j = 0; j < d; ++j)
            proposed_H += 0.5 * p[j] * p[j] * mass_inv[j];

        f64 log_alpha = -(proposed_H - current_H);
        f64 alpha = std::min(1.0, std::exp(log_alpha));
        if (std::isnan(alpha)) alpha = 0.0;

        if (std::log(unif(rng)) < log_alpha) {
            current = q;
            current_lp = prop_lp;
            ++accepted;
        }

        // Dual averaging step size adaptation
        if (opts.adapt_step_size && iter < opts.burn_in) {
            f64 m = static_cast<f64>(iter + 1);
            H_bar = (1.0 - 1.0 / (m + t0)) * H_bar
                  + (opts.target_accept_rate - alpha) / (m + t0);
            f64 log_eps = mu - std::sqrt(m) / gamma_da * H_bar;
            epsilon = std::exp(log_eps);
            log_eps_bar = std::pow(m, -kappa) * log_eps
                        + (1.0 - std::pow(m, -kappa)) * log_eps_bar;
        }
        if (opts.adapt_step_size && iter == opts.burn_in) {
            epsilon = std::exp(log_eps_bar);
        }

        if (iter >= opts.burn_in && ((iter - opts.burn_in) % opts.thin) == 0
            && kept < opts.n_samples) {
            for (std::size_t j = 0; j < d; ++j) chain.samples(kept, j) = current[j];
            chain.log_posterior[kept] = current_lp;
            ++kept;
        }
    }

    chain.n_kept = kept;
    chain.acceptance_rate = static_cast<f64>(accepted) / static_cast<f64>(total);
    chain.proposal_scale = Vector<f64>({epsilon});
    compute_summary(chain);
    compute_chain_diagnostics(chain);

    auto end = std::chrono::high_resolution_clock::now();
    chain.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return chain;
}

Result<MCMCChain> hmc(const Vector<f64>& initial,
                       const LogDensityFn& log_density,
                       const HMCOptions& opts) {
    auto grad_fn = [&log_density](const Vector<f64>& theta) -> Vector<f64> {
        return numerical_gradient(log_density, theta);
    };
    return hmc(initial, log_density, grad_fn, opts);
}

// ═══════════════════════════════════════════════════════════════════════════
// Slice Sampling (Neal 2003)
// ═══════════════════════════════════════════════════════════════════════════

Result<MCMCChain> slice_sampling(const Vector<f64>& initial,
                                   const LogDensityFn& log_density,
                                   const SliceOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t d = initial.size();
    if (d == 0) return Status::error(ErrorCode::InvalidArgument, "slice: empty initial");

    std::size_t total = opts.burn_in + opts.n_samples * opts.thin;
    MCMCChain chain;
    chain.n_dim = d;
    chain.n_total = total;
    chain.samples = Matrix<f64>(opts.n_samples, d);
    chain.log_posterior = Vector<f64>(opts.n_samples);
    chain.sampler = "slice_sampling";

    Vector<f64> widths(d);
    if (opts.widths.size() == d) widths = opts.widths;
    else for (std::size_t j = 0; j < d; ++j) widths[j] = std::abs(initial[j]) * 0.5 + 0.1;

    std::mt19937_64 rng(opts.seed);
    std::uniform_real_distribution<f64> unif(0.0, 1.0);

    Vector<f64> current = initial;
    f64 current_lp = log_density(current);
    std::size_t kept = 0;

    for (std::size_t iter = 0; iter < total; ++iter) {
        // Update each dimension via slice sampling
        for (std::size_t j = 0; j < d; ++j) {
            f64 log_y = current_lp + std::log(unif(rng));

            // Stepping out
            f64 L = current[j] - widths[j] * unif(rng);
            f64 R = L + widths[j];

            Vector<f64> probe = current;
            for (std::size_t step = 0; step < opts.max_steps_out; ++step) {
                probe[j] = L;
                if (log_density(probe) <= log_y) break;
                L -= widths[j];
            }
            for (std::size_t step = 0; step < opts.max_steps_out; ++step) {
                probe[j] = R;
                if (log_density(probe) <= log_y) break;
                R += widths[j];
            }

            // Shrinking
            for (std::size_t shrink = 0; shrink < 200; ++shrink) {
                f64 x1 = L + unif(rng) * (R - L);
                probe[j] = x1;
                f64 lp = log_density(probe);
                if (lp > log_y) {
                    current[j] = x1;
                    current_lp = lp;
                    break;
                }
                if (x1 < current[j]) L = x1;
                else R = x1;
            }
        }

        if (iter >= opts.burn_in && ((iter - opts.burn_in) % opts.thin) == 0
            && kept < opts.n_samples) {
            for (std::size_t j = 0; j < d; ++j) chain.samples(kept, j) = current[j];
            chain.log_posterior[kept] = current_lp;
            ++kept;
        }
    }

    chain.n_kept = kept;
    chain.acceptance_rate = 1.0;
    compute_summary(chain);
    compute_chain_diagnostics(chain);

    auto end = std::chrono::high_resolution_clock::now();
    chain.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return chain;
}

// ═══════════════════════════════════════════════════════════════════════════
// Gibbs Sampler
// ═══════════════════════════════════════════════════════════════════════════

Result<MCMCChain> gibbs(const Vector<f64>& initial,
                         const std::vector<GibbsBlock>& blocks,
                         const GibbsOptions& opts) {
    return gibbs(initial, blocks, LogDensityFn{}, opts);
}

Result<MCMCChain> gibbs(const Vector<f64>& initial,
                         const std::vector<GibbsBlock>& blocks,
                         const LogDensityFn& log_density,
                         const GibbsOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t d = initial.size();
    if (d == 0 || blocks.empty())
        return Status::error(ErrorCode::InvalidArgument, "gibbs: empty initial or no blocks");

    std::size_t total = opts.burn_in + opts.n_samples * opts.thin;
    MCMCChain chain;
    chain.n_dim = d;
    chain.n_total = total;
    chain.samples = Matrix<f64>(opts.n_samples, d);
    chain.log_posterior = Vector<f64>(opts.n_samples, 0.0);
    chain.acceptance_rate = 1.0;
    chain.sampler = "gibbs";

    std::mt19937_64 rng(opts.seed);
    Vector<f64> current = initial;
    std::size_t kept = 0;

    for (std::size_t iter = 0; iter < total; ++iter) {
        for (auto& block : blocks) {
            auto drawn = block.sampler(current, rng);
            for (std::size_t k = 0; k < block.indices.size() && k < drawn.size(); ++k)
                current[block.indices[k]] = drawn[k];
        }
        if (iter >= opts.burn_in && ((iter - opts.burn_in) % opts.thin) == 0
            && kept < opts.n_samples) {
            for (std::size_t j = 0; j < d; ++j) chain.samples(kept, j) = current[j];
            if (log_density) chain.log_posterior[kept] = log_density(current);
            ++kept;
        }
    }

    chain.n_kept = kept;
    compute_summary(chain);
    compute_chain_diagnostics(chain);

    auto end = std::chrono::high_resolution_clock::now();
    chain.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return chain;
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-Chain Runner
// ═══════════════════════════════════════════════════════════════════════════

Result<MultiChainResult> multi_chain_mh(const Vector<f64>& initial,
                                          const LogDensityFn& log_density,
                                          const MHOptions& opts,
                                          const MultiChainOptions& mc) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t d = initial.size();
    std::size_t K = mc.n_chains;

    MultiChainResult result;
    result.chains.resize(K);

    // Disperse initial points
    std::mt19937_64 init_rng(opts.seed + 999);
    std::normal_distribution<f64> norm(0.0, 1.0);

    auto run_chain = [&](std::size_t k) {
        Vector<f64> init_k(d);
        std::mt19937_64 local_rng(opts.seed + 999 + k * 17);
        std::normal_distribution<f64> ln(0.0, 1.0);
        for (std::size_t j = 0; j < d; ++j)
            init_k[j] = initial[j] * (1.0 + 0.1 * ln(local_rng));

        MHOptions chain_opts = opts;
        chain_opts.seed = opts.seed + k * 12345;
        auto res = metropolis_hastings(init_k, log_density, chain_opts);
        if (res) result.chains[k] = std::move(res).value();
    };

    if (mc.parallel && K > 1) {
        auto& pool = global_thread_pool();
        std::vector<std::future<void>> futures;
        for (std::size_t k = 0; k < K; ++k)
            futures.push_back(pool.submit([&, k]() { run_chain(k); }));
        for (auto& f : futures) f.get();
    } else {
        for (std::size_t k = 0; k < K; ++k) run_chain(k);
    }

    // Compute R-hat per parameter
    result.rhat = Vector<f64>(d);
    result.pooled_mean = Vector<f64>(d, 0.0);
    result.pooled_std = Vector<f64>(d, 0.0);
    result.pooled_ess = Vector<f64>(d, 0.0);
    result.converged = true;

    for (std::size_t j = 0; j < d; ++j) {
        std::size_t n_per = result.chains[0].n_kept;
        Matrix<f64> param_chains(K, n_per);
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t i = 0; i < n_per && i < result.chains[k].n_kept; ++i)
                param_chains(k, i) = result.chains[k].samples(i, j);
        }
        result.rhat[j] = gelman_rubin_rhat(param_chains);
        if (result.rhat[j] > 1.1) result.converged = false;

        for (std::size_t k = 0; k < K; ++k) {
            result.pooled_mean[j] += result.chains[k].mean[j];
            result.pooled_ess[j] += result.chains[k].ess[j];
        }
        result.pooled_mean[j] /= static_cast<f64>(K);
        for (std::size_t k = 0; k < K; ++k) {
            f64 d2 = result.chains[k].mean[j] - result.pooled_mean[j];
            result.pooled_std[j] += result.chains[k].std_dev[j] * result.chains[k].std_dev[j] + d2 * d2;
        }
        result.pooled_std[j] = std::sqrt(result.pooled_std[j] / static_cast<f64>(K));
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

Result<MultiChainResult> multi_chain_hmc(const Vector<f64>& initial,
                                           const LogDensityFn& log_density,
                                           const GradientFn& gradient,
                                           const HMCOptions& opts,
                                           const MultiChainOptions& mc) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t d = initial.size();
    std::size_t K = mc.n_chains;

    MultiChainResult result;
    result.chains.resize(K);

    auto run_chain = [&](std::size_t k) {
        Vector<f64> init_k(d);
        std::mt19937_64 local_rng(opts.seed + 999 + k * 17);
        std::normal_distribution<f64> ln(0.0, 1.0);
        for (std::size_t j = 0; j < d; ++j)
            init_k[j] = initial[j] * (1.0 + 0.1 * ln(local_rng));
        HMCOptions chain_opts = opts;
        chain_opts.seed = opts.seed + k * 12345;
        auto res = hmc(init_k, log_density, gradient, chain_opts);
        if (res) result.chains[k] = std::move(res).value();
    };

    if (mc.parallel && K > 1) {
        auto& pool = global_thread_pool();
        std::vector<std::future<void>> futures;
        for (std::size_t k = 0; k < K; ++k)
            futures.push_back(pool.submit([&, k]() { run_chain(k); }));
        for (auto& f : futures) f.get();
    } else {
        for (std::size_t k = 0; k < K; ++k) run_chain(k);
    }

    result.rhat = Vector<f64>(d);
    result.pooled_mean = Vector<f64>(d, 0.0);
    result.pooled_std = Vector<f64>(d, 0.0);
    result.pooled_ess = Vector<f64>(d, 0.0);
    result.converged = true;

    for (std::size_t j = 0; j < d; ++j) {
        std::size_t n_per = result.chains[0].n_kept;
        Matrix<f64> param_chains(K, n_per);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t i = 0; i < n_per && i < result.chains[k].n_kept; ++i)
                param_chains(k, i) = result.chains[k].samples(i, j);
        result.rhat[j] = gelman_rubin_rhat(param_chains);
        if (result.rhat[j] > 1.1) result.converged = false;
        for (std::size_t k = 0; k < K; ++k) {
            result.pooled_mean[j] += result.chains[k].mean[j];
            result.pooled_ess[j] += result.chains[k].ess[j];
        }
        result.pooled_mean[j] /= static_cast<f64>(K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian GARCH(1,1)
// ═══════════════════════════════════════════════════════════════════════════

Result<BayesGARCHResult> bayesian_garch(const Vector<f64>& returns,
                                         const BayesGARCHOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 10) return Status::error(ErrorCode::InvalidArgument, "bayesian_garch: need >= 10 obs");

    f64 mu = returns.mean();
    const auto& prior = opts.prior;

    auto log_prior = [&](f64 omega, f64 alpha, f64 beta) -> f64 {
        if (omega <= 0 || alpha < 0 || alpha > 0.5 || beta < 0 || beta >= 1 || alpha + beta >= 1)
            return -1e30;
        f64 lp = (prior.omega_shape - 1.0) * std::log(omega) - prior.omega_rate * omega;
        f64 a_s = alpha / 0.5;
        if (a_s <= 0 || a_s >= 1) return -1e30;
        lp += (prior.alpha_a - 1) * std::log(a_s) + (prior.alpha_b - 1) * std::log(1 - a_s);
        if (beta <= 0 || beta >= 1) return -1e30;
        lp += (prior.beta_a - 1) * std::log(beta) + (prior.beta_b - 1) * std::log(1 - beta);
        return lp;
    };

    auto log_posterior = [&](const Vector<f64>& theta) -> f64 {
        f64 lp = log_prior(theta[0], theta[1], theta[2]);
        if (lp < -1e29) return -1e30;
        return garch_loglikelihood(returns, mu, theta[0], theta[1], theta[2]) + lp;
    };

    auto mle = garch(returns);
    Vector<f64> initial(3);
    if (mle) { initial[0] = mle.value().omega; initial[1] = mle.value().alpha; initial[2] = mle.value().beta; }
    else { initial[0] = 0.00001; initial[1] = 0.05; initial[2] = 0.90; }

    Vector<f64> prop_scale(3);
    if (opts.proposal_scale.size() == 3) prop_scale = opts.proposal_scale;
    else { prop_scale[0] = initial[0] * 0.3; prop_scale[1] = 0.02; prop_scale[2] = 0.02; }

    // Use Adaptive Metropolis for better mixing
    AMOptions am_opts;
    am_opts.n_samples = opts.n_samples;
    am_opts.burn_in = opts.burn_in;
    am_opts.thin = opts.thin;
    am_opts.seed = opts.seed;
    am_opts.initial_scale = prop_scale;

    auto chain_result = adaptive_metropolis(initial, log_posterior, am_opts);
    if (!chain_result) return chain_result.status();

    BayesGARCHResult result;
    result.chain = std::move(chain_result).value();

    result.omega_mean = result.chain.mean[0];
    result.alpha_mean = result.chain.mean[1];
    result.beta_mean = result.chain.mean[2];
    result.persistence_mean = result.alpha_mean + result.beta_mean;

    result.omega_ci_lower = result.chain.ci_lower[0]; result.omega_ci_upper = result.chain.ci_upper[0];
    result.alpha_ci_lower = result.chain.ci_lower[1]; result.alpha_ci_upper = result.chain.ci_upper[1];
    result.beta_ci_lower = result.chain.ci_lower[2]; result.beta_ci_upper = result.chain.ci_upper[2];

    result.conditional_var = Vector<f64>(n);
    f64 uncond = (result.persistence_mean < 1.0)
        ? result.omega_mean / (1.0 - result.persistence_mean) : 0.0001;
    f64 h = uncond;
    for (std::size_t t = 0; t < n; ++t) {
        if (h < 1e-20) h = 1e-20;
        result.conditional_var[t] = h;
        f64 e = returns[t] - mu;
        h = result.omega_mean + result.alpha_mean * e * e + result.beta_mean * h;
    }

    // WAIC computation
    std::size_t S = result.chain.n_kept;
    Matrix<f64> pll(S, n);
    constexpr f64 log2pi = 1.8378770664093453;
    for (std::size_t s = 0; s < S; ++s) {
        f64 o = result.chain.samples(s, 0);
        f64 a = result.chain.samples(s, 1);
        f64 b = result.chain.samples(s, 2);
        f64 ht = (a + b < 1.0) ? o / (1.0 - a - b) : 0.0001;
        for (std::size_t t = 0; t < n; ++t) {
            if (ht < 1e-20) ht = 1e-20;
            f64 et = returns[t] - mu;
            pll(s, t) = -0.5 * (log2pi + std::log(ht) + et * et / ht);
            ht = o + a * et * et + b * ht;
        }
    }
    result.waic = compute_waic(pll);
    result.marginal_likelihood = log_marginal_likelihood_harmonic(result.chain.log_posterior);

    f64 ll_at_mean = garch_loglikelihood(returns, mu, result.omega_mean, result.alpha_mean, result.beta_mean);
    f64 mean_ll = 0.0;
    for (std::size_t i = 0; i < S; ++i) mean_ll += result.chain.log_posterior[i];
    mean_ll /= static_cast<f64>(S);
    result.dic = -4.0 * mean_ll + 2.0 * (ll_at_mean + log_prior(result.omega_mean, result.alpha_mean, result.beta_mean));

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian GJR-GARCH(1,1) with Student-t
// ═══════════════════════════════════════════════════════════════════════════

Result<BayesGJRResult> bayesian_gjr_garch(const Vector<f64>& returns,
                                            const BayesGJROptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = returns.size();
    if (n < 10) return Status::error(ErrorCode::InvalidArgument, "bayesian_gjr: need >= 10 obs");

    f64 mu = returns.mean();
    std::size_t n_params = opts.estimate_nu ? 5 : 4;

    auto log_posterior = [&](const Vector<f64>& theta) -> f64 {
        f64 omega = theta[0], alpha = theta[1], gamma = theta[2], beta = theta[3];
        f64 nu = opts.estimate_nu ? theta[4] : opts.nu_init;

        if (omega <= 0 || alpha < 0 || gamma < 0 || beta < 0) return -1e30;
        if (alpha + gamma / 2.0 + beta >= 1.0) return -1e30;
        if (nu <= 2.0 || nu > 200.0) return -1e30;

        // Priors
        f64 lp = -1e5 * omega; // exponential on omega
        lp += -2.0 * std::log(1.0 + alpha * 10.0);
        lp += -2.0 * std::log(1.0 + gamma * 10.0);
        lp += 7.0 * std::log(beta) + 1.0 * std::log(1.0 - beta); // Beta(8,2)
        if (opts.estimate_nu) {
            lp += -1.5 * std::log(nu); // Jeffrey's-like prior on nu
        }

        // GJR-GARCH likelihood with Student-t
        f64 sample_var = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            f64 e = returns[i] - mu; sample_var += e * e;
        }
        sample_var /= static_cast<f64>(n);
        f64 h = sample_var;

        for (std::size_t t = 0; t < n; ++t) {
            if (h < 1e-20) h = 1e-20;
            f64 e = returns[t] - mu;
            lp += log_student_t_pdf(e, nu, h);
            f64 indicator = (e < 0.0) ? 1.0 : 0.0;
            if (t < n - 1)
                h = omega + (alpha + gamma * indicator) * e * e + beta * h;
        }

        return lp;
    };

    // Initial values
    Vector<f64> initial(n_params);
    auto mle = garch(returns);
    if (mle) {
        initial[0] = mle.value().omega;
        initial[1] = mle.value().alpha * 0.5;
        initial[2] = mle.value().alpha * 0.5;
        initial[3] = mle.value().beta;
    } else {
        initial[0] = 0.00001; initial[1] = 0.03; initial[2] = 0.05; initial[3] = 0.88;
    }
    if (opts.estimate_nu) initial[4] = opts.nu_init;

    Vector<f64> iscale(n_params);
    iscale[0] = initial[0] * 0.3;
    iscale[1] = 0.01;
    iscale[2] = 0.01;
    iscale[3] = 0.01;
    if (opts.estimate_nu) iscale[4] = 1.0;

    AMOptions am_opts;
    am_opts.n_samples = opts.n_samples;
    am_opts.burn_in = opts.burn_in;
    am_opts.thin = opts.thin;
    am_opts.seed = opts.seed;
    am_opts.initial_scale = iscale;

    auto chain_result = adaptive_metropolis(initial, log_posterior, am_opts);
    if (!chain_result) return chain_result.status();

    BayesGJRResult result;
    result.chain = std::move(chain_result).value();

    result.omega_mean = result.chain.mean[0];
    result.alpha_mean = result.chain.mean[1];
    result.gamma_mean = result.chain.mean[2];
    result.beta_mean = result.chain.mean[3];
    result.nu_mean = opts.estimate_nu ? result.chain.mean[4] : opts.nu_init;
    result.persistence_mean = result.alpha_mean + result.gamma_mean / 2.0 + result.beta_mean;

    result.omega_ci_lower = result.chain.ci_lower[0]; result.omega_ci_upper = result.chain.ci_upper[0];
    result.alpha_ci_lower = result.chain.ci_lower[1]; result.alpha_ci_upper = result.chain.ci_upper[1];
    result.gamma_ci_lower = result.chain.ci_lower[2]; result.gamma_ci_upper = result.chain.ci_upper[2];
    result.beta_ci_lower = result.chain.ci_lower[3]; result.beta_ci_upper = result.chain.ci_upper[3];
    if (opts.estimate_nu) {
        result.nu_ci_lower = result.chain.ci_lower[4];
        result.nu_ci_upper = result.chain.ci_upper[4];
    }

    // Conditional variance at posterior mean
    result.conditional_var = Vector<f64>(n);
    f64 h = result.omega_mean / std::max(1e-6, 1.0 - result.persistence_mean);
    for (std::size_t t = 0; t < n; ++t) {
        if (h < 1e-20) h = 1e-20;
        result.conditional_var[t] = h;
        f64 e = returns[t] - mu;
        f64 ind = (e < 0.0) ? 1.0 : 0.0;
        h = result.omega_mean + (result.alpha_mean + result.gamma_mean * ind) * e * e
          + result.beta_mean * h;
    }

    // WAIC
    std::size_t S = result.chain.n_kept;
    Matrix<f64> pll(S, n);
    for (std::size_t s = 0; s < S; ++s) {
        f64 o = result.chain.samples(s, 0);
        f64 a = result.chain.samples(s, 1);
        f64 g = result.chain.samples(s, 2);
        f64 b = result.chain.samples(s, 3);
        f64 nu = opts.estimate_nu ? result.chain.samples(s, 4) : opts.nu_init;
        f64 sv = 0.0;
        for (std::size_t i = 0; i < n; ++i) { f64 e = returns[i] - mu; sv += e*e; }
        f64 ht = sv / static_cast<f64>(n);
        for (std::size_t t = 0; t < n; ++t) {
            if (ht < 1e-20) ht = 1e-20;
            f64 et = returns[t] - mu;
            pll(s, t) = log_student_t_pdf(et, nu, ht);
            f64 ind = (et < 0.0) ? 1.0 : 0.0;
            ht = o + (a + g * ind) * et * et + b * ht;
        }
    }
    result.waic = compute_waic(pll);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Posterior Predictive Simulation
// ═══════════════════════════════════════════════════════════════════════════

Result<PredictiveResult> garch_predictive(const Vector<f64>& returns,
                                            const BayesGARCHResult& posterior,
                                            std::size_t n_ahead,
                                            std::size_t n_posterior,
                                            u64 seed) {
    std::size_t n = returns.size();
    std::size_t S = posterior.chain.n_kept;
    if (S == 0 || n < 2) return Status::error(ErrorCode::InvalidArgument, "predictive: empty posterior");

    f64 mu = returns.mean();
    std::size_t step = std::max(std::size_t(1), S / n_posterior);
    std::size_t actual = (S + step - 1) / step;

    PredictiveResult result;
    result.n_ahead = n_ahead;
    result.n_posterior = actual;
    result.simulated_returns = Matrix<f64>(actual, n_ahead);
    result.simulated_vol = Matrix<f64>(actual, n_ahead);

    std::mt19937_64 rng(seed);
    std::normal_distribution<f64> norm(0.0, 1.0);

    for (std::size_t s = 0; s < actual; ++s) {
        std::size_t idx = s * step;
        f64 omega = posterior.chain.samples(idx, 0);
        f64 alpha = posterior.chain.samples(idx, 1);
        f64 beta = posterior.chain.samples(idx, 2);

        // Replay variance to end of sample
        f64 h = (alpha + beta < 1.0) ? omega / (1.0 - alpha - beta) : 0.0001;
        for (std::size_t t = 0; t < n; ++t) {
            if (h < 1e-20) h = 1e-20;
            f64 e = returns[t] - mu;
            h = omega + alpha * e * e + beta * h;
        }

        // Simulate forward
        for (std::size_t t = 0; t < n_ahead; ++t) {
            if (h < 1e-20) h = 1e-20;
            result.simulated_vol(s, t) = std::sqrt(h);
            f64 z = norm(rng);
            f64 r = mu + std::sqrt(h) * z;
            result.simulated_returns(s, t) = r;
            h = omega + alpha * (r - mu) * (r - mu) + beta * h;
        }
    }

    // Summarize
    result.mean_forecast = Vector<f64>(n_ahead, 0.0);
    result.vol_forecast = Vector<f64>(n_ahead, 0.0);
    result.vol_ci_lower = Vector<f64>(n_ahead);
    result.vol_ci_upper = Vector<f64>(n_ahead);

    for (std::size_t t = 0; t < n_ahead; ++t) {
        std::vector<f64> vols(actual);
        for (std::size_t s = 0; s < actual; ++s) {
            result.mean_forecast[t] += result.simulated_returns(s, t);
            vols[s] = result.simulated_vol(s, t);
            result.vol_forecast[t] += vols[s];
        }
        result.mean_forecast[t] /= static_cast<f64>(actual);
        result.vol_forecast[t] /= static_cast<f64>(actual);
        std::sort(vols.begin(), vols.end());
        result.vol_ci_lower[t] = vols[static_cast<std::size_t>(0.025 * static_cast<f64>(actual))];
        result.vol_ci_upper[t] = vols[static_cast<std::size_t>(0.975 * static_cast<f64>(actual))];
    }

    return result;
}

Result<PredictiveResult> gjr_predictive(const Vector<f64>& returns,
                                          const BayesGJRResult& posterior,
                                          std::size_t n_ahead,
                                          std::size_t n_posterior,
                                          u64 seed) {
    std::size_t n = returns.size();
    std::size_t S = posterior.chain.n_kept;
    if (S == 0 || n < 2) return Status::error(ErrorCode::InvalidArgument, "gjr_predictive: empty posterior");

    bool has_nu = posterior.chain.n_dim >= 5;
    f64 mu = returns.mean();
    std::size_t step = std::max(std::size_t(1), S / n_posterior);
    std::size_t actual = (S + step - 1) / step;

    PredictiveResult result;
    result.n_ahead = n_ahead;
    result.n_posterior = actual;
    result.simulated_returns = Matrix<f64>(actual, n_ahead);
    result.simulated_vol = Matrix<f64>(actual, n_ahead);

    std::mt19937_64 rng(seed);

    for (std::size_t s = 0; s < actual; ++s) {
        std::size_t idx = s * step;
        f64 omega = posterior.chain.samples(idx, 0);
        f64 alpha = posterior.chain.samples(idx, 1);
        f64 gamma = posterior.chain.samples(idx, 2);
        f64 beta = posterior.chain.samples(idx, 3);
        f64 nu = has_nu ? posterior.chain.samples(idx, 4) : 8.0;

        f64 persist = alpha + gamma / 2.0 + beta;
        f64 h = (persist < 1.0) ? omega / (1.0 - persist) : 0.0001;
        for (std::size_t t = 0; t < n; ++t) {
            if (h < 1e-20) h = 1e-20;
            f64 e = returns[t] - mu;
            f64 ind = (e < 0.0) ? 1.0 : 0.0;
            h = omega + (alpha + gamma * ind) * e * e + beta * h;
        }

        std::chi_squared_distribution<f64> chi2(nu);
        std::normal_distribution<f64> norm(0.0, 1.0);

        for (std::size_t t = 0; t < n_ahead; ++t) {
            if (h < 1e-20) h = 1e-20;
            result.simulated_vol(s, t) = std::sqrt(h);
            f64 z = norm(rng) / std::sqrt(chi2(rng) / nu);
            f64 r = mu + std::sqrt(h) * z;
            result.simulated_returns(s, t) = r;
            f64 e = r - mu;
            f64 ind = (e < 0.0) ? 1.0 : 0.0;
            h = omega + (alpha + gamma * ind) * e * e + beta * h;
        }
    }

    result.mean_forecast = Vector<f64>(n_ahead, 0.0);
    result.vol_forecast = Vector<f64>(n_ahead, 0.0);
    result.vol_ci_lower = Vector<f64>(n_ahead);
    result.vol_ci_upper = Vector<f64>(n_ahead);

    for (std::size_t t = 0; t < n_ahead; ++t) {
        std::vector<f64> vols(actual);
        for (std::size_t s = 0; s < actual; ++s) {
            result.mean_forecast[t] += result.simulated_returns(s, t);
            vols[s] = result.simulated_vol(s, t);
            result.vol_forecast[t] += vols[s];
        }
        result.mean_forecast[t] /= static_cast<f64>(actual);
        result.vol_forecast[t] /= static_cast<f64>(actual);
        std::sort(vols.begin(), vols.end());
        result.vol_ci_lower[t] = vols[static_cast<std::size_t>(0.025 * static_cast<f64>(actual))];
        result.vol_ci_upper[t] = vols[static_cast<std::size_t>(0.975 * static_cast<f64>(actual))];
    }

    return result;
}

} // namespace hfm
