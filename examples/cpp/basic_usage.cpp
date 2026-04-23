#include "hfm/hfmetal.hpp"
#include <iostream>
#include <cmath>

int main() {
    // Generate synthetic prices
    std::size_t n = 1000;
    std::vector<hfm::f64> prices(n);
    prices[0] = 100.0;
    for (std::size_t i = 1; i < n; ++i) {
        prices[i] = prices[i - 1] * (1.0 + std::sin(static_cast<double>(i) * 0.05) * 0.005);
    }
    hfm::Series<hfm::f64> price_series(prices);

    // Compute log returns
    auto returns = hfm::log_returns(price_series);
    std::cout << "Computed " << returns.size() << " log returns\n";

    // Realized measures
    auto measures = hfm::compute_realized_measures(returns);
    std::cout << "Realized variance:   " << measures.realized_variance << "\n";
    std::cout << "Realized volatility: " << measures.realized_volatility << "\n";
    std::cout << "Bipower variation:   " << measures.bipower_variation << "\n";
    std::cout << "Jump statistic:      " << measures.jump_statistic << "\n";

    // Semivariance decomposition
    auto rsv_neg = hfm::realized_semivariance(returns, false);
    auto rsv_pos = hfm::realized_semivariance(returns, true);
    std::cout << "Neg semivariance:    " << rsv_neg << "\n";
    std::cout << "Pos semivariance:    " << rsv_pos << "\n";
    std::cout << "Sum (= RV):          " << rsv_neg + rsv_pos << "\n";

    // Metal context info
#ifdef HFM_METAL_ENABLED
    auto& metal = hfm::global_metal_context();
    if (metal.is_available()) {
        std::cout << "\nMetal device: " << metal.device_name() << "\n";
        std::cout << "Max buffer:   " << metal.max_buffer_length() / (1024*1024) << " MB\n";
    }
#endif

    return 0;
}
