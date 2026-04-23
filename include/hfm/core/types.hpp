#pragma once

#include <cstdint>
#include <cstddef>

namespace hfm {

using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

enum class Backend : u32 {
    Auto,
    CPU,
    Accelerate,
    Metal
};

enum class DType : u32 {
    F32,
    F64,
    I32,
    I64,
    Bool
};

enum class StorageOrder : u32 {
    RowMajor,
    ColMajor
};

struct ExecutionOptions {
    Backend backend = Backend::Auto;
    bool deterministic = false;
    bool prefer_gpu_if_available = true;
    std::size_t gpu_min_elements = 4096;

    ExecutionOptions& set_backend(Backend b) { backend = b; return *this; }
    ExecutionOptions& set_deterministic(bool v) { deterministic = v; return *this; }
    ExecutionOptions& set_gpu_min_elements(std::size_t n) { gpu_min_elements = n; return *this; }
};

} // namespace hfm
