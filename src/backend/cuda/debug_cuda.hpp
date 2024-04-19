// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once
#include <common/Logger.hpp>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <string>

namespace flare {
namespace cuda {
namespace kernel_logger {

inline auto getLogger() {
    static auto logger = common::loggerFactory("kernel");
    return logger;
}
}  // namespace kernel_logger
}  // namespace cuda
}  // namespace flare

template<>
struct fmt::formatter<dim3> : fmt::formatter<std::string> {
    // parse is inherited from formatter<string_view>.
    template<typename FormatContext>
    auto format(dim3 c, FormatContext& ctx) {
        std::string name = fmt::format("{} {} {}", c.x, c.y, c.z);
        return formatter<std::string>::format(name, ctx);
    }
};

#define CUDA_LAUNCH_SMEM(fn, blks, thrds, smem_size, ...)                   \
    do {                                                                    \
        {                                                                   \
            using namespace flare::cuda::kernel_logger;                 \
            FLY_TRACE(                                                       \
                "Launching {}: Blocks: [{}] Threads: [{}] "                 \
                "Shared Memory: {}",                                        \
                #fn, blks, thrds, smem_size);                               \
        }                                                                   \
        fn<<<blks, thrds, smem_size, flare::cuda::getActiveStream()>>>( \
            __VA_ARGS__);                                                   \
    } while (false)

#define CUDA_LAUNCH(fn, blks, thrds, ...) \
    CUDA_LAUNCH_SMEM(fn, blks, thrds, 0, __VA_ARGS__)

// FIXME: Add a special flag for debug
#ifndef NDEBUG

#define POST_LAUNCH_CHECK()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaStreamSynchronize(flare::cuda::getActiveStream())); \
    } while (0)

#else

#define POST_LAUNCH_CHECK()                                                 \
    do {                                                                    \
        if (flare::cuda::synchronize_calls()) {                         \
            CUDA_CHECK(                                                     \
                cudaStreamSynchronize(flare::cuda::getActiveStream())); \
        } else {                                                            \
            CUDA_CHECK(cudaPeekAtLastError());                              \
        }                                                                   \
    } while (0)

#endif
