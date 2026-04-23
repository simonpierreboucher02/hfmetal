#include "hfm/timeseries/irf.hpp"
#include "hfm/linalg/solver.hpp"
#include <chrono>
#include <cmath>

namespace hfm {

Result<IRFResult> var_irf(const VARResult& var_result, std::size_t n_horizons) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t k = var_result.n_vars;
    std::size_t p = var_result.p;

    if (k == 0 || p == 0) {
        return Status::error(ErrorCode::InvalidArgument, "var_irf: empty VAR result");
    }

    auto chol = cholesky(var_result.sigma_u);
    if (!chol) {
        return Status::error(ErrorCode::SingularMatrix,
                             "var_irf: Cholesky decomposition of Sigma_u failed");
    }
    Matrix<f64> P = chol.value();

    std::size_t offset = var_result.has_intercept ? 1 : 0;
    std::vector<Matrix<f64>> A(p, Matrix<f64>(k, k, 0.0));
    for (std::size_t l = 0; l < p; ++l) {
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                A[l](i, j) = var_result.coefficients(i, offset + l * k + j);
            }
        }
    }

    IRFResult result;
    result.n_horizons = n_horizons;
    result.n_vars = k;
    result.irf.resize(n_horizons + 1);

    std::vector<Matrix<f64>> Phi(n_horizons + 1, Matrix<f64>(k, k, 0.0));
    Phi[0] = Matrix<f64>::identity(k);

    for (std::size_t h = 1; h <= n_horizons; ++h) {
        Phi[h] = Matrix<f64>(k, k, 0.0);
        for (std::size_t j = 0; j < std::min(h, p); ++j) {
            auto term = matmul(Phi[h - j - 1], A[j]);
            for (std::size_t r = 0; r < k; ++r)
                for (std::size_t c = 0; c < k; ++c)
                    Phi[h](r, c) += term(r, c);
        }
    }

    for (std::size_t h = 0; h <= n_horizons; ++h) {
        result.irf[h] = matmul(Phi[h], P);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

Result<FEVDResult> var_fevd(const VARResult& var_result, std::size_t n_horizons) {
    auto irf_res = var_irf(var_result, n_horizons);
    if (!irf_res) {
        return Status::error(irf_res.status().code(), "var_fevd: " + irf_res.status().message());
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto& irf = irf_res.value();
    std::size_t k = irf.n_vars;

    FEVDResult result;
    result.n_horizons = n_horizons;
    result.n_vars = k;
    result.fevd.resize(n_horizons + 1, Matrix<f64>(k, k, 0.0));

    Matrix<f64> cumulative_mse(k, k, 0.0);

    for (std::size_t h = 0; h <= n_horizons; ++h) {
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                cumulative_mse(i, j) += irf.irf[h](i, j) * irf.irf[h](i, j);
            }
        }

        for (std::size_t i = 0; i < k; ++i) {
            f64 total = 0.0;
            for (std::size_t j = 0; j < k; ++j) total += cumulative_mse(i, j);
            for (std::size_t j = 0; j < k; ++j) {
                result.fevd[h](i, j) = (total > 0.0) ? cumulative_mse(i, j) / total : 0.0;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

Result<ForecastEvalResult> forecast_eval(const Vector<f64>& actual,
                                          const Vector<f64>& forecast) {
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = actual.size();
    if (n != forecast.size() || n == 0) {
        return Status::error(ErrorCode::DimensionMismatch,
                             "forecast_eval: actual and forecast must have same length");
    }

    ForecastEvalResult result;
    result.n_obs = n;
    f64 fn = static_cast<f64>(n);

    f64 sum_ae = 0.0, sum_se = 0.0, sum_ape = 0.0;
    f64 sum_rw_se = 0.0;
    f64 actual_mean = actual.mean();
    f64 ss_tot = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        f64 e = actual[i] - forecast[i];
        sum_ae += std::abs(e);
        sum_se += e * e;
        if (std::abs(actual[i]) > 1e-15) {
            sum_ape += std::abs(e / actual[i]);
        }
        if (i > 0) {
            f64 rw_e = actual[i] - actual[i - 1];
            sum_rw_se += rw_e * rw_e;
        }
        f64 d = actual[i] - actual_mean;
        ss_tot += d * d;
    }

    result.mae = sum_ae / fn;
    result.mse = sum_se / fn;
    result.rmse = std::sqrt(result.mse);
    result.mape = sum_ape / fn * 100.0;
    result.r_squared = (ss_tot > 0.0) ? 1.0 - sum_se / ss_tot : 0.0;

    if (n > 1 && sum_rw_se > 0.0) {
        result.theil_u = std::sqrt(sum_se / static_cast<f64>(n - 1)) /
                         std::sqrt(sum_rw_se / static_cast<f64>(n - 1));
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();
    return result;
}

} // namespace hfm
