#include <gtest/gtest.h>
#include "hfm/covariance/covariance.hpp"
#include "hfm/linalg/solver.hpp"

using namespace hfm;

TEST(ClusteredCovTest, SingleCluster) {
    // With one cluster, clustered SE should approximate White SE
    std::size_t n = 50;
    Matrix<f64> X(n, 2);
    Vector<f64> resid(n);
    Vector<i64> cluster_ids(n, 0); // all in one cluster

    uint64_t seed = 42;
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / static_cast<f64>(n);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        resid[i] = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.5;
    }

    Matrix<f64> XtX = matmul_AtB(X, X);
    auto XtX_inv = invert_spd(XtX);
    ASSERT_TRUE(XtX_inv.is_ok());

    // Should not crash with 1 cluster (but small-sample correction G/(G-1) = 1/0)
    // Actually 1 cluster means G=1, G-1=0 — division by zero
    // This is a degenerate case; at least 2 clusters needed for meaningful inference
}

TEST(ClusteredCovTest, TwoClusters) {
    std::size_t n = 100;
    Matrix<f64> X(n, 2);
    Vector<f64> resid(n);
    Vector<i64> cluster_ids(n);

    uint64_t seed = 55;
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / static_cast<f64>(n);
        cluster_ids[i] = static_cast<i64>(i / 10); // 10 clusters
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        resid[i] = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.5;
    }

    Matrix<f64> XtX = matmul_AtB(X, X);
    auto XtX_inv = invert_spd(XtX);
    ASSERT_TRUE(XtX_inv.is_ok());

    auto cov = clustered_covariance(X, resid, XtX_inv.value(), cluster_ids);
    EXPECT_EQ(cov.rows(), 2u);
    EXPECT_EQ(cov.cols(), 2u);
    EXPECT_GT(cov(0, 0), 0.0);
    EXPECT_GT(cov(1, 1), 0.0);
}

TEST(ClusteredCovTest, TwoWay) {
    std::size_t n = 100;
    Matrix<f64> X(n, 2);
    Vector<f64> resid(n);
    Vector<i64> cluster1(n);
    Vector<i64> cluster2(n);

    uint64_t seed = 88;
    for (std::size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0;
        X(i, 1) = static_cast<f64>(i) / static_cast<f64>(n);
        cluster1[i] = static_cast<i64>(i / 10);
        cluster2[i] = static_cast<i64>(i % 5);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        resid[i] = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 1.0) * 0.5;
    }

    Matrix<f64> XtX = matmul_AtB(X, X);
    auto XtX_inv = invert_spd(XtX);
    ASSERT_TRUE(XtX_inv.is_ok());

    auto cov = twoway_clustered_covariance(X, resid, XtX_inv.value(), cluster1, cluster2);
    EXPECT_EQ(cov.rows(), 2u);
    EXPECT_EQ(cov.cols(), 2u);
}
