#include "hfm/hfm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>

using namespace hfm;

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== HFMetal Comprehensive Example ===\n\n";

    // --- Log returns ---
    Series<f64> prices({100.0, 101.5, 99.8, 102.3, 103.1, 100.9, 104.2});
    auto returns = log_returns(prices);
    std::cout << "Log returns: ";
    for (std::size_t i = 0; i < returns.size(); ++i)
        std::cout << returns[i] << " ";
    std::cout << "\n\n";

    // --- Realized measures ---
    std::size_t n_intraday = 500;
    std::vector<f64> intraday_data(n_intraday);
    uint64_t seed = 42;
    for (std::size_t i = 0; i < n_intraday; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        intraday_data[i] = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.01;
    }
    Series<f64> intraday(intraday_data);

    auto rm = compute_realized_measures(intraday);
    std::cout << "Realized Measures:\n";
    std::cout << "  RV  = " << rm.realized_variance << "\n";
    std::cout << "  RVol = " << rm.realized_volatility << "\n";
    std::cout << "  BV  = " << rm.bipower_variation << "\n";
    std::cout << "  Jump = " << rm.jump_statistic << "\n\n";

    // --- OLS with HAC ---
    std::size_t n = 200;
    Vector<f64> y(n);
    Matrix<f64> X(n, 3);
    seed = 123;
    for (std::size_t i = 0; i < n; ++i) {
        f64 x = static_cast<f64>(i) / static_cast<f64>(n);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.05;
        X(i, 0) = 1.0;
        X(i, 1) = x;
        X(i, 2) = x * x;
        y[i] = 1.0 + 2.0 * x - 0.5 * x * x + noise;
    }

    auto ols_res = ols(y, X, OLSOptions{}.set_covariance(CovarianceType::NeweyWest));
    if (ols_res) {
        std::cout << ols_res.value().summary() << "\n";
    }

    // --- Rolling OLS ---
    auto roll_res = rolling_ols(y, X, RollingOptions{}.set_window(50).set_step(10));
    if (roll_res) {
        std::cout << "Rolling OLS: " << roll_res.value().n_windows() << " windows\n\n";
    }

    // --- AR(1) ---
    Vector<f64> ar_data(500);
    ar_data[0] = 0.0;
    seed = 77;
    for (std::size_t t = 1; t < 500; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 noise = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.5;
        ar_data[t] = 0.3 + 0.6 * ar_data[t-1] + noise;
    }

    auto ar_res = ar(ar_data, AROptions{}.set_p(1));
    if (ar_res) {
        auto& r = ar_res.value();
        std::cout << "AR(1): intercept=" << r.coefficients[0]
                  << " phi=" << r.coefficients[1]
                  << " AIC=" << r.aic << " BIC=" << r.bic << "\n\n";
    }

    // --- VAR(1) ---
    Matrix<f64> Y_var(300, 2, 0.0);
    seed = 99;
    for (std::size_t t = 1; t < 300; ++t) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e1 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 e2 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.3;
        Y_var(t, 0) = 0.5 * Y_var(t-1, 0) + 0.1 * Y_var(t-1, 1) + e1;
        Y_var(t, 1) = 0.2 * Y_var(t-1, 0) + 0.4 * Y_var(t-1, 1) + e2;
    }

    auto var_res = var(Y_var, VAROptions{}.set_p(1));
    if (var_res) {
        auto& r = var_res.value();
        std::cout << "VAR(1): " << r.n_vars << " variables, " << r.n_obs << " obs\n";
        std::cout << "  AIC=" << r.aic << " BIC=" << r.bic << "\n\n";
    }

    // --- Execution planner ---
    auto& planner = ExecutionPlanner::instance();
    WorkloadDescriptor wd;
    wd.type = WorkloadType::BatchedRegression;
    wd.batch_count = 1000;
    wd.n_elements = 100000;
    auto chosen = planner.choose_backend(wd);
    std::cout << "Execution planner chose backend: " << static_cast<int>(chosen) << "\n";

    // --- Metal context ---
#ifdef HFM_METAL_ENABLED
    try {
        auto& ctx = global_metal_context();
        if (ctx.is_available()) {
            std::cout << "Metal device: " << ctx.device_name() << "\n";
            std::cout << "Max buffer: " << ctx.max_buffer_length() / (1024*1024) << " MB\n";
        }
    } catch (...) {}
#endif

    std::cout << "\nDone.\n";
    return 0;
}
