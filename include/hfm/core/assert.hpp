#pragma once

#include <stdexcept>
#include <string>
#include <source_location>

namespace hfm {

inline void hfm_assert(bool condition, const char* msg,
                        std::source_location loc = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(
            std::string(loc.file_name()) + ":" + std::to_string(loc.line()) +
            " in " + loc.function_name() + ": assertion failed: " + msg);
    }
}

#define HFM_ASSERT(cond, msg) ::hfm::hfm_assert((cond), (msg))
#define HFM_CHECK_DIM(a, b, msg) ::hfm::hfm_assert((a) == (b), (msg))

} // namespace hfm
