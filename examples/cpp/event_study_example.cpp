#include "hfm/hfmetal.hpp"
#include <iostream>
#include <cmath>

int main() {
    // Simulate 2000 returns with events at specific points
    std::size_t n = 2000;
    std::vector<hfm::f64> returns(n);
    for (std::size_t i = 0; i < n; ++i) {
        returns[i] = std::sin(static_cast<double>(i) * 0.1) * 0.005;
    }

    // Add event effects: spikes at event locations
    std::vector<std::size_t> events{200, 500, 800, 1100, 1400, 1700};
    for (auto idx : events) {
        returns[idx] += 0.02;      // positive event impact
        returns[idx + 1] += 0.01;  // lingering effect
    }

    hfm::Series<hfm::f64> ret_series(returns);

    // Run event study
    hfm::EventStudyOptions opts;
    opts.window = {-20, 20};
    opts.compute_volatility_response = true;

    auto result = hfm::event_study(ret_series, events, opts);
    if (!result.is_ok()) {
        std::cerr << "Event study failed: " << result.status().message() << "\n";
        return 1;
    }

    auto& res = result.value();
    std::cout << "Event study results:\n";
    std::cout << "  Events: " << res.n_events << "\n";
    std::cout << "  Window: [" << res.window.left << ", " << res.window.right << "]\n";
    std::cout << "  Time: " << res.elapsed_ms << " ms\n\n";

    std::cout << "Mean CAR around events:\n";
    for (std::size_t w = 0; w < res.mean_car.size(); ++w) {
        auto t = static_cast<hfm::i64>(w) + res.window.left;
        if (t >= -5 && t <= 5) {
            std::cout << "  t=" << t << ": CAR=" << res.mean_car[w]
                      << "  vol=" << res.volatility_response[w] << "\n";
        }
    }

    return 0;
}
