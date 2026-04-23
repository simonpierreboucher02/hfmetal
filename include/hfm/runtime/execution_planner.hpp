#pragma once

#include "hfm/core/types.hpp"
#include <cstddef>

namespace hfm {

enum class WorkloadType : u32 {
    Elementwise,
    Reduction,
    MatMul,
    Solve,
    Rolling,
    Bootstrap,
    BatchedRegression,
    EventExtraction
};

struct WorkloadDescriptor {
    WorkloadType type = WorkloadType::Elementwise;
    std::size_t n_elements = 0;
    std::size_t n_rows = 0;
    std::size_t n_cols = 0;
    std::size_t batch_count = 1;
    std::size_t repetitions = 1;
    DType dtype = DType::F64;
    bool data_on_gpu = false;
};

class ExecutionPlanner {
public:
    Backend choose_backend(const WorkloadDescriptor& desc,
                           const ExecutionOptions& opts = ExecutionOptions{}) const;

    static ExecutionPlanner& instance();

private:
    ExecutionPlanner() = default;
    bool metal_available() const;
};

} // namespace hfm
