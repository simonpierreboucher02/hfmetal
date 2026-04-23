#include "hfm/data/timestamp.hpp"
#include <ctime>
#include <sstream>
#include <iomanip>

namespace hfm {

std::string Timestamp::to_string() const {
    auto sys_tp = std::chrono::time_point_cast<std::chrono::seconds>(
        std::chrono::time_point_cast<std::chrono::system_clock::duration>(tp_));
    auto time_t_val = std::chrono::system_clock::to_time_t(sys_tp);
    auto us = microseconds() % 1'000'000;

    std::tm tm_val{};
    gmtime_r(&time_t_val, &tm_val);

    std::ostringstream ss;
    ss << std::put_time(&tm_val, "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(6) << (us >= 0 ? us : -us);
    return ss.str();
}

} // namespace hfm
