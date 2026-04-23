#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include "hfm/linalg/matrix.hpp"
#include "hfm/linalg/vector.hpp"

namespace hfm {

struct FamaMacBethOptions {
    bool newey_west_correction = true;
    i64 hac_lag = -1;

    FamaMacBethOptions& set_nw(bool v) { newey_west_correction = v; return *this; }
    FamaMacBethOptions& set_lag(i64 lag) { hac_lag = lag; return *this; }
};

struct FamaMacBethResult {
    Vector<f64> gamma;          // average cross-sectional coefficients
    Vector<f64> std_errors;     // Fama-MacBeth SE (optionally NW corrected)
    Vector<f64> t_stats;
    Vector<f64> p_values;
    Matrix<f64> gamma_series;   // T x k time series of cross-sectional betas
    std::size_t n_periods = 0;
    std::size_t n_factors = 0;
    f64 elapsed_ms = 0.0;
};

// y_it = gamma_0t + gamma_1t * X1_it + ... + gamma_kt * Xk_it
// Returns: average gamma over time, with NW-corrected SE
// Panel data format: each row of Y and X corresponds to (entity, time) pair
// time_ids identifies which time period each row belongs to
Result<FamaMacBethResult> fama_macbeth(const Vector<f64>& y,
                                        const Matrix<f64>& X,
                                        const Vector<i64>& time_ids,
                                        const FamaMacBethOptions& opts = FamaMacBethOptions{});

} // namespace hfm
