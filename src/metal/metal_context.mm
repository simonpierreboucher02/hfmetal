#include "hfm/metal/metal_context.hpp"

#ifdef HFM_METAL_ENABLED

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mutex>

namespace hfm {

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipeline_cache = nil;

    id<MTLComputePipelineState> get_pipeline(const std::string& name) {
        NSString* key = [NSString stringWithUTF8String:name.c_str()];
        id<MTLComputePipelineState> cached = pipeline_cache[key];
        if (cached) return cached;

        id<MTLFunction> func = [library newFunctionWithName:key];
        if (!func) return nil;

        NSError* error = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
        if (pso) {
            pipeline_cache[key] = pso;
        }
        return pso;
    }
};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}
MetalContext::~MetalContext() = default;
MetalContext::MetalContext(MetalContext&&) noexcept = default;
MetalContext& MetalContext::operator=(MetalContext&&) noexcept = default;

Result<MetalContext> MetalContext::create(const std::string& metallib_path) {
    MetalContext ctx;
    ctx.impl_->device = MTLCreateSystemDefaultDevice();
    if (!ctx.impl_->device) {
        return Status::error(ErrorCode::MetalUnavailable, "No Metal device found");
    }

    ctx.impl_->queue = [ctx.impl_->device newCommandQueue];
    ctx.impl_->pipeline_cache = [NSMutableDictionary dictionary];

    std::string lib_path = metallib_path;
    if (lib_path.empty()) {
#ifdef HFM_METALLIB_PATH
        lib_path = HFM_METALLIB_PATH;
#endif
    }

    if (!lib_path.empty()) {
        auto status = ctx.load_library(lib_path);
        if (!status) {
            // Metallib not found, try runtime compilation
#ifdef HFM_METAL_SHADER_DIR
            auto rt_status = ctx.compile_from_source(HFM_METAL_SHADER_DIR);
            if (!rt_status) return rt_status;
#else
            return status;
#endif
        }
    }
#ifdef HFM_METAL_RUNTIME_COMPILE
    else {
#ifdef HFM_METAL_SHADER_DIR
        auto rt_status = ctx.compile_from_source(HFM_METAL_SHADER_DIR);
        if (!rt_status) return rt_status;
#endif
    }
#endif

    return ctx;
}

bool MetalContext::is_available() const {
    return impl_ && impl_->device != nil;
}

std::string MetalContext::device_name() const {
    if (!impl_->device) return "none";
    return std::string([[impl_->device name] UTF8String]);
}

std::size_t MetalContext::max_buffer_length() const {
    return [impl_->device maxBufferLength];
}

std::size_t MetalContext::max_threads_per_threadgroup() const {
    return 1024; // Apple Silicon standard
}

void* MetalContext::device() const { return (__bridge void*)impl_->device; }
void* MetalContext::command_queue() const { return (__bridge void*)impl_->queue; }
void* MetalContext::library() const { return (__bridge void*)impl_->library; }

Status MetalContext::load_library(const std::string& path) {
    NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:ns_path];
    NSError* error = nil;
    impl_->library = [impl_->device newLibraryWithURL:url error:&error];
    if (!impl_->library) {
        return Status::error(ErrorCode::MetalKernelError,
            "Failed to load metallib: " + std::string([[error localizedDescription] UTF8String]));
    }
    return Status::ok();
}

Status MetalContext::compile_from_source(const std::string& shader_dir) {
    NSString* dir = [NSString stringWithUTF8String:shader_dir.c_str()];
    NSFileManager* fm = [NSFileManager defaultManager];
    NSArray* files = [fm contentsOfDirectoryAtPath:dir error:nil];
    NSMutableString* combined = [NSMutableString string];

    for (NSString* file in files) {
        if ([file hasSuffix:@".metal"]) {
            NSString* path = [dir stringByAppendingPathComponent:file];
            NSString* src = [NSString stringWithContentsOfFile:path
                                                      encoding:NSUTF8StringEncoding
                                                         error:nil];
            if (src) {
                [combined appendString:src];
                [combined appendString:@"\n"];
            }
        }
    }

    if ([combined length] == 0) {
        return Status::error(ErrorCode::MetalKernelError, "No .metal sources found in " + shader_dir);
    }

    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;

    impl_->library = [impl_->device newLibraryWithSource:combined options:opts error:&error];
    if (!impl_->library) {
        return Status::error(ErrorCode::MetalKernelError,
            "Runtime shader compilation failed: " + std::string([[error localizedDescription] UTF8String]));
    }
    return Status::ok();
}

void* MetalContext::new_buffer(std::size_t size) const {
    id<MTLBuffer> buf = [impl_->device newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

void* MetalContext::new_buffer_with_data(const void* data, std::size_t size) const {
    id<MTLBuffer> buf = [impl_->device newBufferWithBytes:data
                                                   length:size
                                                  options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

Status MetalContext::execute_kernel(const std::string& kernel_name,
                                     void* const* buffers, std::size_t n_buffers,
                                     std::size_t grid_size, std::size_t threadgroup_size) {
    id<MTLComputePipelineState> pso = impl_->get_pipeline(kernel_name);
    if (!pso) {
        return Status::error(ErrorCode::MetalKernelError,
            "Kernel not found: " + kernel_name);
    }

    if (threadgroup_size == 0) {
        threadgroup_size = std::min(static_cast<std::size_t>([pso maxTotalThreadsPerThreadgroup]),
                                    grid_size);
    }

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];

    for (std::size_t i = 0; i < n_buffers; ++i) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffers[i];
        [enc setBuffer:buf offset:0 atIndex:i];
    }

    MTLSize grid = MTLSizeMake(grid_size, 1, 1);
    MTLSize tg = MTLSizeMake(threadgroup_size, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] == MTLCommandBufferStatusError) {
        return Status::error(ErrorCode::MetalKernelError,
            "Kernel execution failed: " + kernel_name);
    }

    return Status::ok();
}

Status MetalContext::execute_kernel_1d(const std::string& kernel_name,
                                        void* const* buffers, std::size_t n_buffers,
                                        std::size_t count) {
    return execute_kernel(kernel_name, buffers, n_buffers, count, 0);
}

static std::unique_ptr<MetalContext> g_metal_ctx;
static std::once_flag g_metal_init;

MetalContext& global_metal_context() {
    std::call_once(g_metal_init, []() {
        auto result = MetalContext::create();
        if (result) {
            g_metal_ctx = std::make_unique<MetalContext>(std::move(result).value());
        }
    });
    return *g_metal_ctx;
}

} // namespace hfm

#else // !HFM_METAL_ENABLED

namespace hfm {

struct MetalContext::Impl {};
MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}
MetalContext::~MetalContext() = default;
MetalContext::MetalContext(MetalContext&&) noexcept = default;
MetalContext& MetalContext::operator=(MetalContext&&) noexcept = default;

Result<MetalContext> MetalContext::create(const std::string&) {
    return Status::error(ErrorCode::MetalUnavailable, "Metal not enabled");
}

bool MetalContext::is_available() const { return false; }
std::string MetalContext::device_name() const { return "none"; }
std::size_t MetalContext::max_buffer_length() const { return 0; }
std::size_t MetalContext::max_threads_per_threadgroup() const { return 0; }
void* MetalContext::device() const { return nullptr; }
void* MetalContext::command_queue() const { return nullptr; }
void* MetalContext::library() const { return nullptr; }
Status MetalContext::load_library(const std::string&) {
    return Status::error(ErrorCode::MetalUnavailable, "Metal not enabled");
}
Status MetalContext::compile_from_source(const std::string&) {
    return Status::error(ErrorCode::MetalUnavailable, "Metal not enabled");
}
void* MetalContext::new_buffer(std::size_t) const { return nullptr; }
void* MetalContext::new_buffer_with_data(const void*, std::size_t) const { return nullptr; }
Status MetalContext::execute_kernel(const std::string&, void* const*, std::size_t, std::size_t, std::size_t) {
    return Status::error(ErrorCode::MetalUnavailable, "Metal not enabled");
}
Status MetalContext::execute_kernel_1d(const std::string&, void* const*, std::size_t, std::size_t) {
    return Status::error(ErrorCode::MetalUnavailable, "Metal not enabled");
}

MetalContext& global_metal_context() {
    static MetalContext dummy;
    return dummy;
}

} // namespace hfm

#endif
