#include <gtest/gtest.h>
#include "hfm/linalg/decompositions.hpp"
#include <cmath>

using namespace hfm;

TEST(DecompositionsTest, SVDIdentity) {
    auto I = Matrix<f64>::identity(3);
    auto result = svd(I);
    ASSERT_TRUE(result.is_ok());
    auto& r = result.value();
    EXPECT_EQ(r.S.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(r.S[i], 1.0, 1e-10);
    }
}

TEST(DecompositionsTest, SVDRectangular) {
    Matrix<f64> A(4, 2, std::vector<f64>{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0
    });
    auto result = svd(A);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(result.value().S.size(), 2u);
    EXPECT_EQ(result.value().U.rows(), 4u);
    EXPECT_EQ(result.value().U.cols(), 2u);
    EXPECT_EQ(result.value().Vt.rows(), 2u);
    EXPECT_EQ(result.value().Vt.cols(), 2u);
    EXPECT_GT(result.value().S[0], result.value().S[1]);
}

TEST(DecompositionsTest, ConditionNumber) {
    auto I = Matrix<f64>::identity(3);
    f64 cond = condition_number(I);
    EXPECT_NEAR(cond, 1.0, 1e-10);

    Matrix<f64> A(2, 2, std::vector<f64>{1.0, 0.0, 0.0, 0.001});
    f64 cond_A = condition_number(A);
    EXPECT_NEAR(cond_A, 1000.0, 1.0);
}

TEST(DecompositionsTest, EigenSymmetric) {
    Matrix<f64> A(3, 3, std::vector<f64>{
        4.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0
    });
    auto result = eigen_symmetric(A);
    ASSERT_TRUE(result.is_ok());
    auto& r = result.value();
    EXPECT_EQ(r.eigenvalues.size(), 3u);
    // Eigenvalues should be ascending
    EXPECT_LE(r.eigenvalues[0], r.eigenvalues[1]);
    EXPECT_LE(r.eigenvalues[1], r.eigenvalues[2]);
    // All positive (SPD matrix)
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_GT(r.eigenvalues[i], 0.0);
    }
}

TEST(DecompositionsTest, EigenIdentity) {
    auto I = Matrix<f64>::identity(3);
    auto result = eigen_symmetric(I);
    ASSERT_TRUE(result.is_ok());
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(result.value().eigenvalues[i], 1.0, 1e-10);
    }
}

TEST(DecompositionsTest, PCABasic) {
    // 100 observations, 3 variables with known structure
    Matrix<f64> X(100, 3);
    uint64_t seed = 42;
    for (std::size_t i = 0; i < 100; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z1 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 10.0;
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f64 z2 = (static_cast<f64>(seed >> 33) / static_cast<f64>(1ULL << 31) - 0.5) * 2.0;
        X(i, 0) = z1;
        X(i, 1) = z1 * 0.8 + z2;
        X(i, 2) = z2 * 0.5;
    }
    auto result = pca(X, 2);
    ASSERT_TRUE(result.is_ok());
    auto& r = result.value();
    EXPECT_EQ(r.n_components, 2u);
    EXPECT_EQ(r.components.rows(), 100u);
    EXPECT_EQ(r.components.cols(), 2u);
    EXPECT_GT(r.explained_ratio[0], r.explained_ratio[1]);

    f64 total_ratio = 0.0;
    for (std::size_t i = 0; i < r.n_components; ++i) total_ratio += r.explained_ratio[i];
    EXPECT_GT(total_ratio, 0.8);
}

TEST(DecompositionsTest, EigenNonSquare) {
    Matrix<f64> A(2, 3, 1.0);
    EXPECT_FALSE(eigen_symmetric(A).is_ok());
}
