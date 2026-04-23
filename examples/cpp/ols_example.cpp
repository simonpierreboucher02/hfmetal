#include "hfm/hfmetal.hpp"
#include <iostream>
#include <cmath>

int main() {
    // Generate data: y = 2 + 1.5*x1 - 0.5*x2 + noise
    std::size_t n = 500;
    std::size_t k = 3;
    hfm::Matrix<hfm::f64> X(n, k, 0.0);
    hfm::Vector<hfm::f64> y(n);

    for (std::size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / 100.0;
        X(i, 0) = 1.0;                        // intercept
        X(i, 1) = t;                           // x1
        X(i, 2) = std::sin(t * 2.0);          // x2
        y[i] = 2.0 + 1.5 * X(i, 1) - 0.5 * X(i, 2)
               + std::sin(static_cast<double>(i) * 0.3) * 0.2;  // noise
    }

    // Classical OLS
    std::cout << "=== Classical OLS ===\n";
    auto result = hfm::ols(y, X);
    if (result.is_ok()) {
        std::cout << result.value().summary() << "\n";
    }

    // White robust SE
    std::cout << "=== White Robust ===\n";
    hfm::OLSOptions white_opts;
    white_opts.covariance = hfm::CovarianceType::White;
    auto result_white = hfm::ols(y, X, white_opts);
    if (result_white.is_ok()) {
        std::cout << result_white.value().summary() << "\n";
    }

    // Newey-West HAC
    std::cout << "=== Newey-West HAC ===\n";
    hfm::OLSOptions nw_opts;
    nw_opts.covariance = hfm::CovarianceType::NeweyWest;
    nw_opts.hac_lag = 5;
    auto result_nw = hfm::ols(y, X, nw_opts);
    if (result_nw.is_ok()) {
        std::cout << result_nw.value().summary() << "\n";
    }

    return 0;
}
