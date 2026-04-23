#include <gtest/gtest.h>
#include "hfm/metal/metal_context.hpp"

using namespace hfm;

TEST(MetalContextTest, Create) {
    auto result = MetalContext::create();
    if (!result.is_ok()) {
        GTEST_SKIP() << "Metal not available";
    }
    auto& ctx = result.value();
    EXPECT_TRUE(ctx.is_available());
    EXPECT_FALSE(ctx.device_name().empty());
    EXPECT_GT(ctx.max_buffer_length(), 0u);
}

TEST(MetalContextTest, DeviceName) {
    auto result = MetalContext::create();
    if (!result.is_ok()) {
        GTEST_SKIP() << "Metal not available";
    }
    auto name = result.value().device_name();
    EXPECT_NE(name, "none");
}
