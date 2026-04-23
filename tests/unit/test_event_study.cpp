#include <gtest/gtest.h>
#include "hfm/hf/event_study.hpp"
#include <cmath>

using namespace hfm;

TEST(EventStudyTest, ExtractWindows) {
    // 20 returns, event at index 10, window [-3, +3]
    std::vector<f64> vals(20);
    for (std::size_t i = 0; i < 20; ++i) vals[i] = static_cast<f64>(i) * 0.01;
    Series<f64> returns(vals);

    std::vector<std::size_t> events{10};
    EventWindow window{-3, 3};
    auto result = extract_event_windows(returns, events, window);
    ASSERT_TRUE(result.is_ok());
    auto& mat = result.value();
    EXPECT_EQ(mat.rows(), 1u);
    EXPECT_EQ(mat.cols(), 7u);
    EXPECT_NEAR(mat(0, 0), 0.07, 1e-12); // index 7
    EXPECT_NEAR(mat(0, 3), 0.10, 1e-12); // index 10
    EXPECT_NEAR(mat(0, 6), 0.13, 1e-12); // index 13
}

TEST(EventStudyTest, MultipleEvents) {
    std::vector<f64> vals(100);
    for (std::size_t i = 0; i < 100; ++i) vals[i] = 0.01;
    Series<f64> returns(vals);

    std::vector<std::size_t> events{20, 50, 80};
    auto result = event_study(returns, events, EventStudyOptions{}.set_left_window(-5).set_right_window(5));
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_EQ(res.n_events, 3u);
    EXPECT_EQ(res.mean_car.size(), 11u);
}

TEST(EventStudyTest, BoundaryEvents) {
    // Event near start, window extends before data
    std::vector<f64> vals(20, 0.01);
    Series<f64> returns(vals);

    std::vector<std::size_t> events{2};
    EventWindow window{-5, 5};
    auto result = extract_event_windows(returns, events, window);
    ASSERT_TRUE(result.is_ok());
    auto& mat = result.value();
    // First 3 elements should be 0 (out of bounds), rest should be 0.01
    EXPECT_DOUBLE_EQ(mat(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(mat(0, 2), 0.0);
    EXPECT_NEAR(mat(0, 3), 0.01, 1e-12);
}

TEST(EventStudyTest, VolatilityResponse) {
    std::vector<f64> vals(200);
    for (std::size_t i = 0; i < 200; ++i) {
        vals[i] = std::sin(static_cast<f64>(i) * 0.5) * 0.01;
    }
    Series<f64> returns(vals);

    std::vector<std::size_t> events{50, 100, 150};
    EventStudyOptions opts;
    opts.window = {-10, 10};
    opts.compute_volatility_response = true;
    auto result = event_study(returns, events, opts);
    ASSERT_TRUE(result.is_ok());
    auto& res = result.value();
    EXPECT_EQ(res.volatility_response.size(), 21u);
    for (std::size_t i = 0; i < res.volatility_response.size(); ++i) {
        EXPECT_GE(res.volatility_response[i], 0.0);
    }
}
