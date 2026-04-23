#include "hfm/estimators/batched_ols.hpp"
#include "hfm/estimators/ols.hpp"
#include "hfm/runtime/thread_pool.hpp"
#include <chrono>
#include <thread>
#include <future>

namespace hfm {

Result<BatchedOLSResult> batched_ols(const Matrix<f64>& Y_columns,
                                      const Matrix<f64>& X,
                                      const BatchedOLSOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = X.rows();
    std::size_t k = X.cols();
    std::size_t n_reg = Y_columns.cols();

    if (Y_columns.rows() != n) {
        return Status::error(ErrorCode::DimensionMismatch, "batched_ols: Y.rows != X.rows");
    }

    BatchedOLSResult result;
    result.n_regressions = n_reg;
    result.betas = Matrix<f64>(n_reg, k, 0.0);
    result.std_errors = Matrix<f64>(n_reg, k, 0.0);
    result.t_stats = Matrix<f64>(n_reg, k, 0.0);
    result.r_squared = Vector<f64>(n_reg, 0.0);
    result.converged.resize(n_reg, false);
    result.backend_used = Backend::CPU;

    OLSOptions ols_opts;
    ols_opts.covariance = opts.covariance;
    ols_opts.hac_lag = opts.hac_lag;
    ols_opts.add_intercept = opts.add_intercept;

    auto& pool = global_thread_pool();
    std::vector<std::future<void>> futures;
    futures.reserve(n_reg);

    for (std::size_t r = 0; r < n_reg; ++r) {
        futures.push_back(pool.submit([&, r]() {
            Vector<f64> y = Y_columns.col(r);
            auto res = ols(y, X, ols_opts);
            if (res.is_ok()) {
                auto& val = res.value();
                for (std::size_t j = 0; j < k; ++j) {
                    result.betas(r, j) = val.coefficients()[j];
                    result.std_errors(r, j) = val.std_errors()[j];
                    result.t_stats(r, j) = val.t_stats()[j];
                }
                result.r_squared[r] = val.r_squared();
                result.converged[r] = true;
            }
        }));
    }

    for (auto& f : futures) f.get();
    result.n_succeeded = 0;
    for (bool c : result.converged) { if (c) ++result.n_succeeded; }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

Result<BatchedOLSResult> batched_ols(
    const std::vector<std::pair<Vector<f64>, Matrix<f64>>>& problems,
    const BatchedOLSOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n_reg = problems.size();
    if (n_reg == 0) {
        return Status::error(ErrorCode::InvalidArgument, "batched_ols: empty problems");
    }

    std::size_t k = problems[0].second.cols();

    BatchedOLSResult result;
    result.n_regressions = n_reg;
    result.betas = Matrix<f64>(n_reg, k, 0.0);
    result.std_errors = Matrix<f64>(n_reg, k, 0.0);
    result.t_stats = Matrix<f64>(n_reg, k, 0.0);
    result.r_squared = Vector<f64>(n_reg, 0.0);
    result.converged.resize(n_reg, false);
    result.backend_used = Backend::CPU;

    OLSOptions ols_opts;
    ols_opts.covariance = opts.covariance;
    ols_opts.hac_lag = opts.hac_lag;
    ols_opts.add_intercept = opts.add_intercept;

    auto& pool = global_thread_pool();
    std::vector<std::future<void>> futures;
    futures.reserve(n_reg);

    for (std::size_t r = 0; r < n_reg; ++r) {
        futures.push_back(pool.submit([&, r]() {
            auto res = ols(problems[r].first, problems[r].second, ols_opts);
            if (res.is_ok()) {
                auto& val = res.value();
                std::size_t kj = val.n_regressors();
                for (std::size_t j = 0; j < kj && j < k; ++j) {
                    result.betas(r, j) = val.coefficients()[j];
                    result.std_errors(r, j) = val.std_errors()[j];
                    result.t_stats(r, j) = val.t_stats()[j];
                }
                result.r_squared[r] = val.r_squared();
                result.converged[r] = true;
            }
        }));
    }

    for (auto& f : futures) f.get();
    result.n_succeeded = 0;
    for (bool c : result.converged) { if (c) ++result.n_succeeded; }

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
