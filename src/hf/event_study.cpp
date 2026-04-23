#include "hfm/hf/event_study.hpp"
#include <chrono>
#include <cmath>

namespace hfm {

Result<Matrix<f64>> extract_event_windows(
    const Series<f64>& returns,
    const std::vector<std::size_t>& event_indices,
    const EventWindow& window) {

    i64 wsize = window.size();
    if (wsize <= 0) {
        return Status::error(ErrorCode::InvalidArgument, "invalid event window");
    }

    std::size_t n_events = event_indices.size();
    Matrix<f64> windows(n_events, static_cast<std::size_t>(wsize), 0.0);

    for (std::size_t e = 0; e < n_events; ++e) {
        auto eidx = static_cast<i64>(event_indices[e]);
        for (i64 w = window.left; w <= window.right; ++w) {
            i64 src = eidx + w;
            auto col = static_cast<std::size_t>(w - window.left);
            if (src >= 0 && static_cast<std::size_t>(src) < returns.size()) {
                windows(e, col) = returns[static_cast<std::size_t>(src)];
            }
        }
    }

    return windows;
}

Result<EventStudyResult> event_study(
    const Series<f64>& returns,
    const std::vector<std::size_t>& event_indices,
    const EventStudyOptions& opts) {

    auto start = std::chrono::high_resolution_clock::now();

    auto windows_result = extract_event_windows(returns, event_indices, opts.window);
    if (!windows_result) return windows_result.status();

    EventStudyResult result;
    result.abnormal_returns = std::move(windows_result).value();
    result.n_events = event_indices.size();
    result.window = opts.window;
    result.backend_used = Backend::CPU;

    std::size_t n = result.n_events;
    auto wsize = static_cast<std::size_t>(opts.window.size());

    // Cumulative abnormal returns
    if (opts.compute_car) {
        result.cumulative_ar = Matrix<f64>(n, wsize, 0.0);
        for (std::size_t e = 0; e < n; ++e) {
            f64 cumsum = 0.0;
            for (std::size_t w = 0; w < wsize; ++w) {
                cumsum += result.abnormal_returns(e, w);
                result.cumulative_ar(e, w) = cumsum;
            }
        }
    }

    // Mean CAR across events
    result.mean_car = Vector<f64>(wsize, 0.0);
    if (n > 0) {
        for (std::size_t w = 0; w < wsize; ++w) {
            f64 sum = 0.0;
            for (std::size_t e = 0; e < n; ++e) {
                sum += result.cumulative_ar(e, w);
            }
            result.mean_car[w] = sum / static_cast<f64>(n);
        }
    }

    // Volatility response
    if (opts.compute_volatility_response) {
        result.volatility_response = Vector<f64>(wsize, 0.0);
        for (std::size_t w = 0; w < wsize; ++w) {
            f64 sum_sq = 0.0;
            f64 sum = 0.0;
            for (std::size_t e = 0; e < n; ++e) {
                f64 r = result.abnormal_returns(e, w);
                sum += r;
                sum_sq += r * r;
            }
            if (n > 1) {
                f64 mean = sum / static_cast<f64>(n);
                result.volatility_response[w] = std::sqrt(
                    (sum_sq / static_cast<f64>(n) - mean * mean) *
                    static_cast<f64>(n) / static_cast<f64>(n - 1));
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
