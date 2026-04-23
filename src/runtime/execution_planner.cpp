#include "hfm/runtime/execution_planner.hpp"
#include "hfm/metal/metal_context.hpp"

namespace hfm {

ExecutionPlanner& ExecutionPlanner::instance() {
    static ExecutionPlanner planner;
    return planner;
}

bool ExecutionPlanner::metal_available() const {
#ifdef HFM_METAL_ENABLED
    try {
        auto& ctx = global_metal_context();
        return ctx.is_available();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

Backend ExecutionPlanner::choose_backend(const WorkloadDescriptor& desc,
                                          const ExecutionOptions& opts) const {
    if (opts.backend != Backend::Auto) {
        return opts.backend;
    }

    if (opts.deterministic) {
        return Backend::CPU;
    }

    // F64 Metal support is limited on some hardware
    bool can_use_metal = opts.prefer_gpu_if_available && metal_available();

    // Small problems: CPU always
    if (desc.n_elements < opts.gpu_min_elements && !desc.data_on_gpu) {
        if (desc.type == WorkloadType::MatMul || desc.type == WorkloadType::Solve) {
            return Backend::Accelerate;
        }
        return Backend::CPU;
    }

    // Data already on GPU: stay there
    if (desc.data_on_gpu && can_use_metal) {
        return Backend::Metal;
    }

    switch (desc.type) {
        case WorkloadType::Elementwise:
            return (desc.n_elements >= opts.gpu_min_elements && can_use_metal)
                ? Backend::Metal : Backend::CPU;

        case WorkloadType::Reduction:
            return (desc.n_elements >= opts.gpu_min_elements * 5 && can_use_metal)
                ? Backend::Metal : Backend::CPU;

        case WorkloadType::MatMul:
        case WorkloadType::Solve:
            return Backend::Accelerate;

        case WorkloadType::Rolling:
            return (desc.n_elements >= opts.gpu_min_elements && desc.repetitions >= 10 && can_use_metal)
                ? Backend::Metal : Backend::CPU;

        case WorkloadType::Bootstrap:
            return (desc.batch_count >= 100 && desc.n_elements >= 1000 && can_use_metal)
                ? Backend::Metal : Backend::CPU;

        case WorkloadType::BatchedRegression:
            return (desc.batch_count >= 50 && can_use_metal)
                ? Backend::Metal : Backend::Accelerate;

        case WorkloadType::EventExtraction:
            return (desc.batch_count >= 100 && can_use_metal)
                ? Backend::Metal : Backend::CPU;
    }

    return Backend::CPU;
}

} // namespace hfm
