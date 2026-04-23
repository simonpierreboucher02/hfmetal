#pragma once

#include "hfm/core/types.hpp"
#include "hfm/core/status.hpp"
#include <string>
#include <memory>

namespace hfm {

class MetalContext {
public:
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext(MetalContext&&) noexcept;
    MetalContext& operator=(MetalContext&&) noexcept;

    static Result<MetalContext> create(const std::string& metallib_path = "");

    bool is_available() const;
    std::string device_name() const;
    std::size_t max_buffer_length() const;
    std::size_t max_threads_per_threadgroup() const;

    void* device() const;
    void* command_queue() const;
    void* library() const;

    Status load_library(const std::string& path);
    Status compile_from_source(const std::string& shader_dir);

    void* new_buffer(std::size_t size) const;
    void* new_buffer_with_data(const void* data, std::size_t size) const;

    Status execute_kernel(const std::string& kernel_name,
                          void* const* buffers, std::size_t n_buffers,
                          std::size_t grid_size, std::size_t threadgroup_size = 0);

    Status execute_kernel_1d(const std::string& kernel_name,
                             void* const* buffers, std::size_t n_buffers,
                             std::size_t count);

private:
    MetalContext();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

MetalContext& global_metal_context();

} // namespace hfm
