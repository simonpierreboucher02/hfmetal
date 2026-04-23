#include <gtest/gtest.h>
#include "hfm/runtime/execution_planner.hpp"

using namespace hfm;

TEST(ExecutionPlannerTest, SmallElementwiseCPU) {
    auto& planner = ExecutionPlanner::instance();
    WorkloadDescriptor desc;
    desc.type = WorkloadType::Elementwise;
    desc.n_elements = 100;
    auto backend = planner.choose_backend(desc);
    EXPECT_EQ(backend, Backend::CPU);
}

TEST(ExecutionPlannerTest, MatMulAccelerate) {
    auto& planner = ExecutionPlanner::instance();
    WorkloadDescriptor desc;
    desc.type = WorkloadType::MatMul;
    desc.n_elements = 100;
    auto backend = planner.choose_backend(desc);
    EXPECT_EQ(backend, Backend::Accelerate);
}

TEST(ExecutionPlannerTest, SolveAccelerate) {
    auto& planner = ExecutionPlanner::instance();
    WorkloadDescriptor desc;
    desc.type = WorkloadType::Solve;
    desc.n_elements = 500;
    auto backend = planner.choose_backend(desc);
    EXPECT_EQ(backend, Backend::Accelerate);
}

TEST(ExecutionPlannerTest, ForcedBackend) {
    auto& planner = ExecutionPlanner::instance();
    WorkloadDescriptor desc;
    desc.type = WorkloadType::Elementwise;
    desc.n_elements = 1000000;

    ExecutionOptions opts;
    opts.backend = Backend::CPU;
    auto backend = planner.choose_backend(desc, opts);
    EXPECT_EQ(backend, Backend::CPU);
}

TEST(ExecutionPlannerTest, DeterministicForcesCPU) {
    auto& planner = ExecutionPlanner::instance();
    WorkloadDescriptor desc;
    desc.type = WorkloadType::Elementwise;
    desc.n_elements = 1000000;

    ExecutionOptions opts;
    opts.deterministic = true;
    auto backend = planner.choose_backend(desc, opts);
    EXPECT_EQ(backend, Backend::CPU);
}
