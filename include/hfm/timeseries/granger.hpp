#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/vector.hpp"
#include "hfm/linalg/matrix.hpp"

namespace hfm {

struct GrangerResult {
    f64 f_statistic = 0.0;
    f64 p_value = 1.0;
    std::size_t n_lags = 0;
    std::size_t n_obs = 0;
    f64 ssr_restricted = 0.0;
    f64 ssr_unrestricted = 0.0;
    f64 elapsed_ms = 0.0;
};

// Granger causality: does x Granger-cause y?
// H0: lagged x does not help predict y beyond lagged y alone
Result<GrangerResult> granger_causality(const Vector<f64>& y,
                                         const Vector<f64>& x,
                                         std::size_t n_lags = 4);

} // namespace hfm
