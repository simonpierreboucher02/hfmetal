#include "hfm/covariance/covariance.hpp"
#include "hfm/linalg/solver.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace hfm {

i64 newey_west_auto_lag(std::size_t n) {
    return static_cast<i64>(std::floor(std::pow(static_cast<f64>(n), 1.0 / 3.0)));
}

Matrix<f64> classical_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                                  std::size_t n, std::size_t k) {
    f64 s2 = 0.0;
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        s2 += residuals[i] * residuals[i];
    }
    s2 /= static_cast<f64>(n - k);

    Matrix<f64> XtX = matmul_AtB(X, X);
    auto XtX_inv = invert_spd(XtX);
    if (!XtX_inv) {
        return Matrix<f64>(k, k, 0.0);
    }

    Matrix<f64> cov(k, k);
    for (std::size_t i = 0; i < k * k; ++i) {
        cov.data()[i] = XtX_inv.value().data()[i] * s2;
    }
    return cov;
}

Matrix<f64> white_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                              const Matrix<f64>& XtX_inv) {
    std::size_t n = X.rows();
    std::size_t k = X.cols();

    // meat = sum_i e_i^2 * x_i * x_i'
    Matrix<f64> meat(k, k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        f64 ei2 = residuals[i] * residuals[i];
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t l = 0; l < k; ++l) {
                meat(j, l) += ei2 * X(i, j) * X(i, l);
            }
        }
    }

    // sandwich = (X'X)^{-1} * meat * (X'X)^{-1}
    // HC0 form, no small-sample correction
    auto temp = matmul(XtX_inv, meat);
    return matmul(temp, XtX_inv);
}

Matrix<f64> newey_west_covariance(const Matrix<f64>& X, const Vector<f64>& residuals,
                                   const Matrix<f64>& XtX_inv, i64 max_lag) {
    std::size_t n = X.rows();
    std::size_t k = X.cols();

    if (max_lag < 0) {
        max_lag = newey_west_auto_lag(n);
    }

    // Gamma_0: sum of e_i^2 * x_i * x_i'
    Matrix<f64> S(k, k, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        f64 ei = residuals[i];
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t l = 0; l < k; ++l) {
                S(j, l) += ei * ei * X(i, j) * X(i, l);
            }
        }
    }

    // Add cross-lag terms with Bartlett weights
    for (i64 lag = 1; lag <= max_lag; ++lag) {
        f64 w = 1.0 - static_cast<f64>(lag) / (static_cast<f64>(max_lag) + 1.0);
        for (std::size_t i = static_cast<std::size_t>(lag); i < n; ++i) {
            f64 ei = residuals[i];
            f64 ej = residuals[i - static_cast<std::size_t>(lag)];
            for (std::size_t j = 0; j < k; ++j) {
                for (std::size_t l = 0; l < k; ++l) {
                    f64 cross = ei * ej * (X(i, j) * X(i - static_cast<std::size_t>(lag), l)
                                         + X(i - static_cast<std::size_t>(lag), j) * X(i, l));
                    S(j, l) += w * cross;
                }
            }
        }
    }

    auto temp = matmul(XtX_inv, S);
    return matmul(temp, XtX_inv);
}

Matrix<f64> clustered_covariance(const Matrix<f64>& X,
                                  const Vector<f64>& residuals,
                                  const Matrix<f64>& XtX_inv,
                                  const Vector<i64>& cluster_ids) {
    std::size_t n = X.rows();
    std::size_t k = X.cols();

    std::unordered_map<i64, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i) {
        groups[cluster_ids[i]].push_back(i);
    }
    std::size_t G = groups.size();

    Matrix<f64> meat(k, k, 0.0);
    for (auto& [_, indices] : groups) {
        Vector<f64> score(k, 0.0);
        for (std::size_t i : indices) {
            for (std::size_t j = 0; j < k; ++j) {
                score[j] += residuals[i] * X(i, j);
            }
        }
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t l = 0; l < k; ++l) {
                meat(j, l) += score[j] * score[l];
            }
        }
    }

    f64 correction = (static_cast<f64>(G) / static_cast<f64>(G - 1)) *
                     (static_cast<f64>(n - 1) / static_cast<f64>(n - k));
    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t l = 0; l < k; ++l) {
            meat(j, l) *= correction;
        }
    }

    auto temp = matmul(XtX_inv, meat);
    return matmul(temp, XtX_inv);
}

Matrix<f64> twoway_clustered_covariance(const Matrix<f64>& X,
                                         const Vector<f64>& residuals,
                                         const Matrix<f64>& XtX_inv,
                                         const Vector<i64>& cluster1_ids,
                                         const Vector<i64>& cluster2_ids) {
    auto V1 = clustered_covariance(X, residuals, XtX_inv, cluster1_ids);
    auto V2 = clustered_covariance(X, residuals, XtX_inv, cluster2_ids);

    std::size_t n = X.rows();
    Vector<i64> intersection_ids(n);
    for (std::size_t i = 0; i < n; ++i) {
        i64 a = cluster1_ids[i];
        i64 b = cluster2_ids[i];
        intersection_ids[i] = (a + b) * (a + b + 1) / 2 + b;
    }
    auto V12 = clustered_covariance(X, residuals, XtX_inv, intersection_ids);

    std::size_t k = X.cols();
    Matrix<f64> result(k, k);
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            result(i, j) = V1(i, j) + V2(i, j) - V12(i, j);
        }
    }
    return result;
}

} // namespace hfm
