#pragma once

#include <vector>
#include <functional>
#include <random>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"

namespace hfm {

// ─── Common types ───────────────────────────────────────────────────────────

using LogDensityFn = std::function<f64(const Vector<f64>& theta)>;
using GradientFn = std::function<Vector<f64>(const Vector<f64>& theta)>;

struct MCMCChain {
    Matrix<f64> samples;              // n_kept x dim
    Vector<f64> log_posterior;         // log-posterior at each kept sample
    std::size_t n_dim = 0;
    std::size_t n_kept = 0;
    std::size_t n_total = 0;
    f64 acceptance_rate = 0.0;
    Vector<f64> proposal_scale;

    // Posterior summary
    Vector<f64> mean;
    Vector<f64> std_dev;
    Vector<f64> ci_lower;             // 2.5%
    Vector<f64> ci_upper;             // 97.5%
    Vector<f64> median;
    Matrix<f64> posterior_cov;        // d x d posterior covariance

    // Diagnostics
    Vector<f64> ess;
    Vector<f64> autocorr_lag1;
    Vector<f64> geweke_z;             // Geweke convergence z-scores
    Vector<f64> rhat;                 // per-param R-hat (multi-chain only)

    f64 elapsed_ms = 0.0;
    std::string sampler;              // which algorithm produced this chain
};

// ─── Metropolis-Hastings (diagonal proposal) ────────────────────────────────

struct MHOptions {
    std::size_t n_samples = 10000;
    std::size_t burn_in = 2000;
    std::size_t thin = 1;
    Vector<f64> proposal_scale;
    f64 target_accept_rate = 0.234;
    bool adapt = true;
    std::size_t adapt_interval = 100;
    u64 seed = 42;

    MHOptions& set_n_samples(std::size_t n) { n_samples = n; return *this; }
    MHOptions& set_burn_in(std::size_t n) { burn_in = n; return *this; }
    MHOptions& set_thin(std::size_t n) { thin = n; return *this; }
    MHOptions& set_proposal_scale(Vector<f64> s) { proposal_scale = std::move(s); return *this; }
    MHOptions& set_seed(u64 s) { seed = s; return *this; }
    MHOptions& set_adapt(bool v) { adapt = v; return *this; }
};

Result<MCMCChain> metropolis_hastings(const Vector<f64>& initial,
                                       const LogDensityFn& log_density,
                                       const MHOptions& opts = MHOptions{});

// ─── Adaptive Metropolis (Haario et al. 2001) ───────────────────────────────
//     Learns full covariance from chain history

struct AMOptions {
    std::size_t n_samples = 10000;
    std::size_t burn_in = 5000;
    std::size_t thin = 1;
    std::size_t adapt_start = 500;    // start adapting covariance after this many
    f64 scale_factor = 2.38;          // 2.38/sqrt(d) optimal scaling
    f64 epsilon = 1e-6;               // regularization for covariance
    Vector<f64> initial_scale;
    u64 seed = 42;

    AMOptions& set_n_samples(std::size_t n) { n_samples = n; return *this; }
    AMOptions& set_burn_in(std::size_t n) { burn_in = n; return *this; }
    AMOptions& set_thin(std::size_t n) { thin = n; return *this; }
    AMOptions& set_seed(u64 s) { seed = s; return *this; }
};

Result<MCMCChain> adaptive_metropolis(const Vector<f64>& initial,
                                       const LogDensityFn& log_density,
                                       const AMOptions& opts = AMOptions{});

// ─── Hamiltonian Monte Carlo (HMC) ─────────────────────────────────────────

struct HMCOptions {
    std::size_t n_samples = 5000;
    std::size_t burn_in = 2000;
    std::size_t thin = 1;
    f64 step_size = 0.01;             // leapfrog step size (epsilon)
    std::size_t n_leapfrog = 20;      // leapfrog steps per proposal (L)
    bool adapt_step_size = true;      // dual averaging for step size
    f64 target_accept_rate = 0.65;    // optimal for HMC
    Vector<f64> mass_diag;            // diagonal mass matrix (inverse)
    u64 seed = 42;

    HMCOptions& set_n_samples(std::size_t n) { n_samples = n; return *this; }
    HMCOptions& set_burn_in(std::size_t n) { burn_in = n; return *this; }
    HMCOptions& set_thin(std::size_t n) { thin = n; return *this; }
    HMCOptions& set_step_size(f64 e) { step_size = e; return *this; }
    HMCOptions& set_n_leapfrog(std::size_t l) { n_leapfrog = l; return *this; }
    HMCOptions& set_seed(u64 s) { seed = s; return *this; }
};

Result<MCMCChain> hmc(const Vector<f64>& initial,
                       const LogDensityFn& log_density,
                       const GradientFn& gradient,
                       const HMCOptions& opts = HMCOptions{});

// Numerical gradient version (finite differences)
Result<MCMCChain> hmc(const Vector<f64>& initial,
                       const LogDensityFn& log_density,
                       const HMCOptions& opts = HMCOptions{});

// ─── Slice Sampling (Neal 2003) ─────────────────────────────────────────────

struct SliceOptions {
    std::size_t n_samples = 10000;
    std::size_t burn_in = 2000;
    std::size_t thin = 1;
    Vector<f64> widths;               // initial bracket widths per dim
    std::size_t max_steps_out = 32;   // max stepping-out iterations
    u64 seed = 42;

    SliceOptions& set_n_samples(std::size_t n) { n_samples = n; return *this; }
    SliceOptions& set_burn_in(std::size_t n) { burn_in = n; return *this; }
    SliceOptions& set_thin(std::size_t n) { thin = n; return *this; }
    SliceOptions& set_widths(Vector<f64> w) { widths = std::move(w); return *this; }
    SliceOptions& set_seed(u64 s) { seed = s; return *this; }
};

Result<MCMCChain> slice_sampling(const Vector<f64>& initial,
                                   const LogDensityFn& log_density,
                                   const SliceOptions& opts = SliceOptions{});

// ─── Gibbs Sampler ──────────────────────────────────────────────────────────

using ConditionalSamplerFn = std::function<Vector<f64>(const Vector<f64>& theta,
                                                        std::mt19937_64& rng)>;

struct GibbsBlock {
    std::vector<std::size_t> indices;
    ConditionalSamplerFn sampler;
};

struct GibbsOptions {
    std::size_t n_samples = 10000;
    std::size_t burn_in = 2000;
    std::size_t thin = 1;
    u64 seed = 42;

    GibbsOptions& set_n_samples(std::size_t n) { n_samples = n; return *this; }
    GibbsOptions& set_burn_in(std::size_t n) { burn_in = n; return *this; }
    GibbsOptions& set_thin(std::size_t n) { thin = n; return *this; }
    GibbsOptions& set_seed(u64 s) { seed = s; return *this; }
};

Result<MCMCChain> gibbs(const Vector<f64>& initial,
                         const std::vector<GibbsBlock>& blocks,
                         const GibbsOptions& opts = GibbsOptions{});

Result<MCMCChain> gibbs(const Vector<f64>& initial,
                         const std::vector<GibbsBlock>& blocks,
                         const LogDensityFn& log_density,
                         const GibbsOptions& opts = GibbsOptions{});

// ─── Multi-Chain Runner ─────────────────────────────────────────────────────

struct MultiChainOptions {
    std::size_t n_chains = 4;
    bool parallel = true;             // run chains in parallel via thread pool
};

struct MultiChainResult {
    std::vector<MCMCChain> chains;
    Vector<f64> rhat;                 // Gelman-Rubin per param
    Vector<f64> pooled_mean;
    Vector<f64> pooled_std;
    Vector<f64> pooled_ess;
    bool converged = false;           // all rhat < 1.1
    f64 elapsed_ms = 0.0;
};

Result<MultiChainResult> multi_chain_mh(const Vector<f64>& initial,
                                          const LogDensityFn& log_density,
                                          const MHOptions& opts = MHOptions{},
                                          const MultiChainOptions& mc = MultiChainOptions{});

Result<MultiChainResult> multi_chain_hmc(const Vector<f64>& initial,
                                           const LogDensityFn& log_density,
                                           const GradientFn& gradient,
                                           const HMCOptions& opts = HMCOptions{},
                                           const MultiChainOptions& mc = MultiChainOptions{});

// ─── Bayesian GARCH(1,1) ───────────────────────────────────────────────────

struct BayesGARCHPrior {
    f64 omega_shape = 2.0;
    f64 omega_rate = 1e5;
    f64 alpha_a = 2.0;
    f64 alpha_b = 10.0;
    f64 beta_a = 8.0;
    f64 beta_b = 2.0;
};

struct BayesGARCHOptions {
    std::size_t n_samples = 10000;
    std::size_t burn_in = 5000;
    std::size_t thin = 1;
    u64 seed = 42;
    BayesGARCHPrior prior;
    Vector<f64> proposal_scale;

    BayesGARCHOptions& set_n_samples(std::size_t n) { n_samples = n; return *this; }
    BayesGARCHOptions& set_burn_in(std::size_t n) { burn_in = n; return *this; }
    BayesGARCHOptions& set_thin(std::size_t n) { thin = n; return *this; }
    BayesGARCHOptions& set_seed(u64 s) { seed = s; return *this; }
};

struct BayesGARCHResult {
    MCMCChain chain;

    f64 omega_mean = 0.0;
    f64 alpha_mean = 0.0;
    f64 beta_mean = 0.0;
    f64 persistence_mean = 0.0;

    f64 omega_ci_lower = 0.0, omega_ci_upper = 0.0;
    f64 alpha_ci_lower = 0.0, alpha_ci_upper = 0.0;
    f64 beta_ci_lower = 0.0, beta_ci_upper = 0.0;

    Vector<f64> conditional_var;

    f64 dic = 0.0;
    f64 waic = 0.0;
    f64 marginal_likelihood = 0.0;    // harmonic mean estimator
    f64 elapsed_ms = 0.0;
};

Result<BayesGARCHResult> bayesian_garch(const Vector<f64>& returns,
                                         const BayesGARCHOptions& opts = BayesGARCHOptions{});

// ─── Bayesian GJR-GARCH(1,1) with Student-t errors ─────────────────────────
//     sigma²_t = omega + (alpha + gamma * I_{e<0}) * e²_{t-1} + beta * sigma²_{t-1}
//     e_t | F_{t-1} ~ t_nu(0, sigma²_t)

struct BayesGJROptions {
    std::size_t n_samples = 10000;
    std::size_t burn_in = 5000;
    std::size_t thin = 1;
    u64 seed = 42;
    bool estimate_nu = true;          // estimate degrees of freedom
    f64 nu_init = 8.0;               // initial d.o.f.

    BayesGJROptions& set_n_samples(std::size_t n) { n_samples = n; return *this; }
    BayesGJROptions& set_burn_in(std::size_t n) { burn_in = n; return *this; }
    BayesGJROptions& set_thin(std::size_t n) { thin = n; return *this; }
    BayesGJROptions& set_seed(u64 s) { seed = s; return *this; }
    BayesGJROptions& set_estimate_nu(bool v) { estimate_nu = v; return *this; }
};

struct BayesGJRResult {
    MCMCChain chain;                  // [omega, alpha, gamma, beta, nu]

    f64 omega_mean = 0.0;
    f64 alpha_mean = 0.0;
    f64 gamma_mean = 0.0;            // leverage effect
    f64 beta_mean = 0.0;
    f64 nu_mean = 0.0;               // degrees of freedom
    f64 persistence_mean = 0.0;       // alpha + gamma/2 + beta

    f64 omega_ci_lower = 0.0, omega_ci_upper = 0.0;
    f64 alpha_ci_lower = 0.0, alpha_ci_upper = 0.0;
    f64 gamma_ci_lower = 0.0, gamma_ci_upper = 0.0;
    f64 beta_ci_lower = 0.0, beta_ci_upper = 0.0;
    f64 nu_ci_lower = 0.0, nu_ci_upper = 0.0;

    Vector<f64> conditional_var;
    f64 dic = 0.0;
    f64 waic = 0.0;
    f64 elapsed_ms = 0.0;
};

Result<BayesGJRResult> bayesian_gjr_garch(const Vector<f64>& returns,
                                            const BayesGJROptions& opts = BayesGJROptions{});

// ─── Posterior Predictive Simulation ────────────────────────────────────────

struct PredictiveResult {
    Matrix<f64> simulated_returns;    // n_posterior x n_ahead
    Matrix<f64> simulated_vol;        // n_posterior x n_ahead
    Vector<f64> mean_forecast;
    Vector<f64> vol_forecast;
    Vector<f64> vol_ci_lower;
    Vector<f64> vol_ci_upper;
    std::size_t n_ahead = 0;
    std::size_t n_posterior = 0;
};

Result<PredictiveResult> garch_predictive(const Vector<f64>& returns,
                                            const BayesGARCHResult& posterior,
                                            std::size_t n_ahead = 10,
                                            std::size_t n_posterior = 1000,
                                            u64 seed = 42);

Result<PredictiveResult> gjr_predictive(const Vector<f64>& returns,
                                          const BayesGJRResult& posterior,
                                          std::size_t n_ahead = 10,
                                          std::size_t n_posterior = 1000,
                                          u64 seed = 42);

// ─── Diagnostics ────────────────────────────────────────────────────────────

f64 effective_sample_size(const f64* chain, std::size_t n);
f64 gelman_rubin_rhat(const Matrix<f64>& chains);
Vector<f64> chain_autocorrelation(const f64* chain, std::size_t n, std::size_t max_lag);
f64 geweke_test(const f64* chain, std::size_t n,
                 f64 frac_first = 0.1, f64 frac_last = 0.5);

void compute_chain_diagnostics(MCMCChain& chain);

f64 log_marginal_likelihood_harmonic(const Vector<f64>& log_likelihoods);
f64 compute_waic(const Matrix<f64>& pointwise_ll);  // n_samples x n_obs

} // namespace hfm
