#include <gtest/gtest.h>
#include "hfm/simulation/mcmc.hpp"
#include "hfm/models/garch.hpp"
#include <cmath>
#include <random>

using namespace hfm;

// ─── Metropolis-Hastings Tests ──────────────────────────────────────────────

TEST(MCMC, MetropolisHastingsNormal) {
    // Sample from N(3, 2^2) using MH
    auto log_density = [](const Vector<f64>& theta) -> f64 {
        f64 x = theta[0];
        return -0.5 * (x - 3.0) * (x - 3.0) / 4.0;
    };

    Vector<f64> initial({0.0});
    auto result = metropolis_hastings(initial, log_density,
        MHOptions{}.set_n_samples(20000).set_burn_in(5000)
                    .set_proposal_scale(Vector<f64>({1.0})).set_seed(123));

    ASSERT_TRUE(result.is_ok());
    auto& chain = result.value();

    EXPECT_GT(chain.acceptance_rate, 0.15);
    EXPECT_LT(chain.acceptance_rate, 0.70);
    EXPECT_NEAR(chain.mean[0], 3.0, 0.15);
    EXPECT_NEAR(chain.std_dev[0], 2.0, 0.3);
    EXPECT_EQ(chain.n_kept, 20000u);
    EXPECT_GT(chain.ess[0], 500.0);
}

TEST(MCMC, MetropolisHastings2D) {
    // Sample from bivariate normal: mu = (1, -1), sigma = (1, 0.5)
    auto log_density = [](const Vector<f64>& theta) -> f64 {
        f64 x = theta[0] - 1.0;
        f64 y = theta[1] + 1.0;
        return -0.5 * (x * x + y * y / 0.25);
    };

    Vector<f64> initial({0.0, 0.0});
    auto result = metropolis_hastings(initial, log_density,
        MHOptions{}.set_n_samples(20000).set_burn_in(5000)
                    .set_proposal_scale(Vector<f64>({0.5, 0.3})));

    ASSERT_TRUE(result.is_ok());
    auto& chain = result.value();

    EXPECT_EQ(chain.n_dim, 2u);
    EXPECT_NEAR(chain.mean[0], 1.0, 0.15);
    EXPECT_NEAR(chain.mean[1], -1.0, 0.15);
    EXPECT_NEAR(chain.std_dev[0], 1.0, 0.2);
    EXPECT_NEAR(chain.std_dev[1], 0.5, 0.15);
}

TEST(MCMC, MetropolisAdaptation) {
    auto log_density = [](const Vector<f64>& theta) -> f64 {
        return -0.5 * theta[0] * theta[0];
    };

    // Start with bad proposal scale
    Vector<f64> initial({0.0});
    auto result = metropolis_hastings(initial, log_density,
        MHOptions{}.set_n_samples(10000).set_burn_in(3000)
                    .set_proposal_scale(Vector<f64>({100.0}))
                    .set_adapt(true));

    ASSERT_TRUE(result.is_ok());
    auto& chain = result.value();

    // Should adapt to reasonable acceptance rate
    EXPECT_GT(chain.acceptance_rate, 0.10);
    EXPECT_NEAR(chain.mean[0], 0.0, 0.2);
}

// ─── Gibbs Sampler Tests ────────────────────────────────────────────────────

TEST(MCMC, GibbsBivariateNormal) {
    // Gibbs for bivariate normal with correlation rho = 0.5
    // p(x|y) = N(rho*y, 1-rho^2)
    // p(y|x) = N(rho*x, 1-rho^2)
    f64 rho = 0.5;
    f64 cond_var = 1.0 - rho * rho;
    f64 cond_sd = std::sqrt(cond_var);

    std::vector<GibbsBlock> blocks;

    // Block 1: sample x | y
    blocks.push_back({{0}, [rho, cond_sd](const Vector<f64>& theta, std::mt19937_64& rng) {
        std::normal_distribution<f64> norm(rho * theta[1], cond_sd);
        return Vector<f64>({norm(rng)});
    }});

    // Block 2: sample y | x
    blocks.push_back({{1}, [rho, cond_sd](const Vector<f64>& theta, std::mt19937_64& rng) {
        std::normal_distribution<f64> norm(rho * theta[0], cond_sd);
        return Vector<f64>({norm(rng)});
    }});

    Vector<f64> initial({0.0, 0.0});
    auto result = gibbs(initial, blocks,
        GibbsOptions{}.set_n_samples(20000).set_burn_in(2000).set_seed(77));

    ASSERT_TRUE(result.is_ok());
    auto& chain = result.value();

    EXPECT_NEAR(chain.mean[0], 0.0, 0.1);
    EXPECT_NEAR(chain.mean[1], 0.0, 0.1);
    EXPECT_NEAR(chain.std_dev[0], 1.0, 0.15);
    EXPECT_NEAR(chain.std_dev[1], 1.0, 0.15);

    // Check correlation
    f64 cov = 0.0;
    for (std::size_t i = 0; i < chain.n_kept; ++i) {
        cov += (chain.samples(i, 0) - chain.mean[0])
             * (chain.samples(i, 1) - chain.mean[1]);
    }
    cov /= static_cast<f64>(chain.n_kept);
    f64 empirical_rho = cov / (chain.std_dev[0] * chain.std_dev[1]);
    EXPECT_NEAR(empirical_rho, rho, 0.1);
}

// ─── Bayesian GARCH Tests ───────────────────────────────────────────────────

TEST(MCMC, BayesianGARCH) {
    // Generate GARCH(1,1) data
    std::mt19937_64 rng(42);
    std::normal_distribution<f64> norm(0.0, 1.0);

    f64 true_omega = 0.00001;
    f64 true_alpha = 0.08;
    f64 true_beta = 0.88;

    std::size_t n = 2000;
    Vector<f64> returns(n);
    f64 h = true_omega / (1.0 - true_alpha - true_beta);

    for (std::size_t t = 0; t < n; ++t) {
        f64 z = norm(rng);
        returns[t] = std::sqrt(h) * z;
        f64 e = returns[t];
        h = true_omega + true_alpha * e * e + true_beta * h;
    }

    auto result = bayesian_garch(returns,
        BayesGARCHOptions{}.set_n_samples(5000).set_burn_in(3000).set_seed(123));

    ASSERT_TRUE(result.is_ok());
    auto& bg = result.value();

    // Posterior means should be in ballpark of true values
    EXPECT_GT(bg.omega_mean, 0.0);
    EXPECT_LT(bg.omega_mean, 0.001);
    EXPECT_GT(bg.alpha_mean, 0.01);
    EXPECT_LT(bg.alpha_mean, 0.3);
    EXPECT_GT(bg.beta_mean, 0.5);
    EXPECT_LT(bg.beta_mean, 0.99);
    EXPECT_GT(bg.persistence_mean, 0.7);
    EXPECT_LT(bg.persistence_mean, 1.0);

    // CI should contain true values (broad test)
    EXPECT_LE(bg.alpha_ci_lower, true_alpha + 0.05);
    EXPECT_GE(bg.alpha_ci_upper, true_alpha - 0.05);
    EXPECT_LE(bg.beta_ci_lower, true_beta + 0.05);
    EXPECT_GE(bg.beta_ci_upper, true_beta - 0.05);

    // Chain diagnostics
    EXPECT_GT(bg.chain.acceptance_rate, 0.05);
    EXPECT_GT(bg.chain.ess[0], 50.0);

    EXPECT_EQ(bg.conditional_var.size(), n);
}

// ─── Diagnostics Tests ──────────────────────────────────────────────────────

TEST(MCMC, EffectiveSampleSize) {
    // IID samples: ESS should be close to n
    std::mt19937_64 rng(42);
    std::normal_distribution<f64> norm(0.0, 1.0);
    std::size_t n = 5000;
    std::vector<f64> iid(n);
    for (std::size_t i = 0; i < n; ++i) iid[i] = norm(rng);

    f64 ess = effective_sample_size(iid.data(), n);
    EXPECT_GT(ess, static_cast<f64>(n) * 0.5);

    // Highly correlated chain: ESS should be much less than n
    std::vector<f64> corr(n);
    corr[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        corr[i] = 0.99 * corr[i - 1] + 0.01 * norm(rng);
    }
    f64 ess_corr = effective_sample_size(corr.data(), n);
    EXPECT_LT(ess_corr, static_cast<f64>(n) * 0.1);
}

TEST(MCMC, GelmanRubin) {
    // Two chains from same distribution: R-hat near 1
    std::size_t n = 1000;
    Matrix<f64> chains(2, n);
    std::mt19937_64 rng1(42), rng2(99);
    std::normal_distribution<f64> norm(0.0, 1.0);
    for (std::size_t j = 0; j < n; ++j) {
        chains(0, j) = norm(rng1);
        chains(1, j) = norm(rng2);
    }
    f64 rhat = gelman_rubin_rhat(chains);
    EXPECT_NEAR(rhat, 1.0, 0.1);

    // Two chains from different distributions: R-hat > 1
    for (std::size_t j = 0; j < n; ++j) {
        chains(0, j) = norm(rng1);
        chains(1, j) = norm(rng2) + 5.0;
    }
    f64 rhat_bad = gelman_rubin_rhat(chains);
    EXPECT_GT(rhat_bad, 1.5);
}

TEST(MCMC, ChainAutocorrelation) {
    std::mt19937_64 rng(42);
    std::normal_distribution<f64> norm(0.0, 1.0);
    std::size_t n = 5000;

    // IID: lag-1 acf near 0
    std::vector<f64> iid(n);
    for (std::size_t i = 0; i < n; ++i) iid[i] = norm(rng);
    auto acf = chain_autocorrelation(iid.data(), n, 5);
    EXPECT_NEAR(acf[0], 1.0, 0.01);
    EXPECT_NEAR(acf[1], 0.0, 0.05);

    // AR(1) with phi=0.9: high lag-1 acf
    std::vector<f64> ar(n);
    ar[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        ar[i] = 0.9 * ar[i - 1] + norm(rng);
    }
    auto acf_ar = chain_autocorrelation(ar.data(), n, 5);
    EXPECT_GT(acf_ar[1], 0.8);
}
