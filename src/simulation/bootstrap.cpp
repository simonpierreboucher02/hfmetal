#include "hfm/simulation/bootstrap.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace hfm {

std::vector<std::size_t> generate_block_indices(std::size_t n, std::size_t block_size,
                                                 std::mt19937_64& rng) {
    std::vector<std::size_t> indices(n);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    if (block_size <= 1) {
        for (std::size_t i = 0; i < n; ++i) {
            indices[i] = dist(rng);
        }
    } else {
        std::size_t pos = 0;
        while (pos < n) {
            std::size_t start = dist(rng);
            for (std::size_t j = 0; j < block_size && pos < n; ++j, ++pos) {
                indices[pos] = (start + j) % n;
            }
        }
    }
    return indices;
}

namespace {

void compute_ci_and_pvalue(BootstrapResult& result) {
    std::size_t B = result.n_bootstrap;
    std::size_t dim = result.estimate.size();

    result.mean = Vector<f64>(dim, 0.0);
    result.std_error = Vector<f64>(dim, 0.0);
    result.ci_lower = Vector<f64>(dim);
    result.ci_upper = Vector<f64>(dim);
    result.p_value = Vector<f64>(dim);

    for (std::size_t d = 0; d < dim; ++d) {
        // Collect column
        std::vector<f64> vals(B);
        f64 sum = 0.0;
        for (std::size_t b = 0; b < B; ++b) {
            vals[b] = result.bootstrap_samples(b, d);
            sum += vals[b];
        }
        result.mean[d] = sum / static_cast<f64>(B);

        f64 ss = 0.0;
        for (std::size_t b = 0; b < B; ++b) {
            f64 diff = vals[b] - result.mean[d];
            ss += diff * diff;
        }
        result.std_error[d] = std::sqrt(ss / static_cast<f64>(B - 1));

        // Percentile CI
        std::sort(vals.begin(), vals.end());
        f64 alpha = 1.0 - result.confidence_level;
        auto lo_idx = static_cast<std::size_t>(std::floor(alpha / 2.0 * static_cast<f64>(B)));
        auto hi_idx = static_cast<std::size_t>(std::ceil((1.0 - alpha / 2.0) * static_cast<f64>(B))) - 1;
        lo_idx = std::min(lo_idx, B - 1);
        hi_idx = std::min(hi_idx, B - 1);
        result.ci_lower[d] = vals[lo_idx];
        result.ci_upper[d] = vals[hi_idx];

        // Two-sided p-value: fraction of bootstrap stats on the other side of zero from estimate
        std::size_t count = 0;
        for (std::size_t b = 0; b < B; ++b) {
            if (result.estimate[d] >= 0 && result.bootstrap_samples(b, d) <= 0) ++count;
            else if (result.estimate[d] < 0 && result.bootstrap_samples(b, d) >= 0) ++count;
        }
        result.p_value[d] = 2.0 * static_cast<f64>(count) / static_cast<f64>(B);
        result.p_value[d] = std::min(result.p_value[d], 1.0);
    }
}

} // namespace

Result<BootstrapResult> bootstrap(const Vector<f64>& data,
                                   const StatisticFn& statistic,
                                   const BootstrapOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    if (data.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "bootstrap: empty data");
    }

    auto original = statistic(data);
    std::size_t dim = original.size();
    std::size_t n = data.size();
    std::size_t B = opts.n_bootstrap;
    std::size_t block_size = (opts.type == BootstrapType::IID) ? 1 : opts.block_size;

    BootstrapResult result;
    result.estimate = original;
    result.bootstrap_samples = Matrix<f64>(B, dim, 0.0);
    result.confidence_level = opts.confidence_level;
    result.n_bootstrap = B;

    std::mt19937_64 rng(opts.seed);

    for (std::size_t b = 0; b < B; ++b) {
        auto indices = generate_block_indices(n, block_size, rng);
        Vector<f64> resampled(n);
        for (std::size_t i = 0; i < n; ++i) {
            resampled[i] = data[indices[i]];
        }
        auto stat = statistic(resampled);
        for (std::size_t d = 0; d < dim; ++d) {
            result.bootstrap_samples(b, d) = stat[d];
        }
    }

    compute_ci_and_pvalue(result);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

Result<BootstrapResult> bootstrap_pairs(const Vector<f64>& y,
                                         const Matrix<f64>& X,
                                         const PairedStatisticFn& statistic,
                                         const BootstrapOptions& opts) {
    auto start = std::chrono::high_resolution_clock::now();

    std::size_t n = y.size();
    if (n != X.rows()) {
        return Status::error(ErrorCode::DimensionMismatch, "bootstrap_pairs: y.size != X.rows");
    }

    auto original = statistic(y, X);
    std::size_t dim = original.size();
    std::size_t k = X.cols();
    std::size_t B = opts.n_bootstrap;
    std::size_t block_size = (opts.type == BootstrapType::IID) ? 1 : opts.block_size;

    BootstrapResult result;
    result.estimate = original;
    result.bootstrap_samples = Matrix<f64>(B, dim, 0.0);
    result.confidence_level = opts.confidence_level;
    result.n_bootstrap = B;

    std::mt19937_64 rng(opts.seed);

    for (std::size_t b = 0; b < B; ++b) {
        auto indices = generate_block_indices(n, block_size, rng);
        Vector<f64> y_boot(n);
        Matrix<f64> X_boot(n, k);
        for (std::size_t i = 0; i < n; ++i) {
            y_boot[i] = y[indices[i]];
            for (std::size_t j = 0; j < k; ++j) {
                X_boot(i, j) = X(indices[i], j);
            }
        }
        auto stat = statistic(y_boot, X_boot);
        for (std::size_t d = 0; d < dim; ++d) {
            result.bootstrap_samples(b, d) = stat[d];
        }
    }

    compute_ci_and_pvalue(result);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<f64, std::milli>(end - start).count();

    return result;
}

} // namespace hfm
