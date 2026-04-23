#include "hfm/timeseries/rolling.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace hfm {

Series<f64> rolling_mean(const Series<f64>& data, std::size_t window) {
    if (data.size() < window) return Series<f64>();
    std::size_t n_out = data.size() - window + 1;
    std::vector<f64> out(n_out);

    // Use running sum for O(n) complexity
    f64 sum = 0.0;
    for (std::size_t i = 0; i < window; ++i) sum += data[i];
    out[0] = sum / static_cast<f64>(window);

    for (std::size_t i = 1; i < n_out; ++i) {
        sum += data[i + window - 1] - data[i - 1];
        out[i] = sum / static_cast<f64>(window);
    }
    return Series<f64>(std::move(out));
}

Series<f64> rolling_variance(const Series<f64>& data, std::size_t window) {
    if (data.size() < window) return Series<f64>();
    std::size_t n_out = data.size() - window + 1;
    std::vector<f64> out(n_out);
    f64 w = static_cast<f64>(window);

    for (std::size_t i = 0; i < n_out; ++i) {
        f64 sum = 0.0, sum_sq = 0.0;
        for (std::size_t j = 0; j < window; ++j) {
            f64 v = data[i + j];
            sum += v;
            sum_sq += v * v;
        }
        f64 mean = sum / w;
        out[i] = (sum_sq / w - mean * mean) * w / (w - 1.0);
    }
    return Series<f64>(std::move(out));
}

Series<f64> rolling_sum(const Series<f64>& data, std::size_t window) {
    if (data.size() < window) return Series<f64>();
    std::size_t n_out = data.size() - window + 1;
    std::vector<f64> out(n_out);

    f64 sum = 0.0;
    for (std::size_t i = 0; i < window; ++i) sum += data[i];
    out[0] = sum;

    for (std::size_t i = 1; i < n_out; ++i) {
        sum += data[i + window - 1] - data[i - 1];
        out[i] = sum;
    }
    return Series<f64>(std::move(out));
}

Series<f64> rolling_min(const Series<f64>& data, std::size_t window) {
    if (data.size() < window) return Series<f64>();
    std::size_t n_out = data.size() - window + 1;
    std::vector<f64> out(n_out);

    for (std::size_t i = 0; i < n_out; ++i) {
        f64 m = std::numeric_limits<f64>::max();
        for (std::size_t j = 0; j < window; ++j) {
            m = std::min(m, data[i + j]);
        }
        out[i] = m;
    }
    return Series<f64>(std::move(out));
}

Series<f64> rolling_max(const Series<f64>& data, std::size_t window) {
    if (data.size() < window) return Series<f64>();
    std::size_t n_out = data.size() - window + 1;
    std::vector<f64> out(n_out);

    for (std::size_t i = 0; i < n_out; ++i) {
        f64 m = std::numeric_limits<f64>::lowest();
        for (std::size_t j = 0; j < window; ++j) {
            m = std::max(m, data[i + j]);
        }
        out[i] = m;
    }
    return Series<f64>(std::move(out));
}

} // namespace hfm
