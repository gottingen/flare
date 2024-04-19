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

#include <common/KernelInterface.hpp>
#include <common/Logger.hpp>

#include <backend.hpp>
#include <cl2hpp.hpp>
#include <string>

namespace flare {
namespace opencl {
namespace kernel_logger {
inline auto getLogger() -> clog::logger* {
    static auto logger = common::loggerFactory("kernel");
    return logger.get();
}
}  // namespace kernel_logger

struct Enqueuer {
    template<typename... Args>
    void operator()(std::string name, cl::Kernel ker,
                    const cl::EnqueueArgs& qArgs, Args&&... args) {
        auto launchOp = cl::KernelFunctor<Args...>(ker);
        using namespace kernel_logger;
        FLY_TRACE("Launching {}", name);
        launchOp(qArgs, std::forward<Args>(args)...);
    }
};

class Kernel
    : public common::KernelInterface<const cl::Program*, cl::Kernel, Enqueuer,
                                     cl::Buffer*> {
   public:
    using BaseClass =
        common::KernelInterface<ModuleType, KernelType, Enqueuer, DevPtrType>;

    Kernel() : BaseClass("", nullptr, cl::Kernel{nullptr, false}) {}
    Kernel(std::string name, ModuleType mod, KernelType ker)
        : BaseClass(name, mod, ker) {}

    // clang-format off
    [[deprecated("OpenCL backend doesn't need Kernel::getDevPtr method")]]
    DevPtrType getDevPtr(const char* name) final;
    // clang-format on

    void copyToReadOnly(DevPtrType dst, DevPtrType src, size_t bytes) final;

    void setFlag(DevPtrType dst, int* scalarValPtr,
                 const bool syncCopy = false) final;

    int getFlag(DevPtrType src) final;
};

}  // namespace opencl
}  // namespace flare
