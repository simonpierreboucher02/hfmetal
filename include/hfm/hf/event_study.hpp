#pragma once

#include <vector>
#include <string>
#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/data/series.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct EventWindow {
    i64 left = -60;
    i64 right = 60;
    i64 size() const { return right - left + 1; }
};

struct EventStudyOptions {
    EventWindow window{-60, 60};
    bool compute_car = true;
    bool compute_volatility_response = false;
    bool group_by_event_type = false;
    Backend backend = Backend::Auto;

    EventStudyOptions& set_left_window(i64 l) { window.left = l; return *this; }
    EventStudyOptions& set_right_window(i64 r) { window.right = r; return *this; }
};

struct EventStudyResult {
    Matrix<f64> abnormal_returns;   // n_events x window_size
    Matrix<f64> cumulative_ar;     // n_events x window_size
    Vector<f64> mean_car;          // window_size
    Vector<f64> volatility_response; // window_size (if computed)
    std::size_t n_events = 0;
    EventWindow window;
    Backend backend_used = Backend::CPU;
    f64 elapsed_ms = 0.0;
};

Result<Matrix<f64>> extract_event_windows(
    const Series<f64>& returns,
    const std::vector<std::size_t>& event_indices,
    const EventWindow& window);

Result<EventStudyResult> event_study(
    const Series<f64>& returns,
    const std::vector<std::size_t>& event_indices,
    const EventStudyOptions& opts = EventStudyOptions{});

} // namespace hfm
