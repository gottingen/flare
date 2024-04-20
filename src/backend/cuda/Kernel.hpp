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

#include <EnqueueArgs.hpp>
#include <backend.hpp>
#include <err_cuda.hpp>
#include <cstdlib>
#include <string>

namespace flare {
namespace cuda {

struct Enqueuer {
    static auto getLogger() {
        static auto logger = common::loggerFactory("kernel");
        return logger.get();
    };

    template<typename... Args>
    void operator()(std::string name, void* ker, const EnqueueArgs& qArgs,
                    Args... args) {
        void* params[] = {static_cast<void*>(&args)...};
        for (auto& event : qArgs.mEvents) {
            CU_CHECK(cuStreamWaitEvent(qArgs.mStream, event, 0));
        }
        FLY_TRACE(
            "Launching {}: Blocks: [{}, {}, {}] Threads: [{}, {}, {}] Shared "
            "Memory: {}",
            name, qArgs.mBlocks.x, qArgs.mBlocks.y, qArgs.mBlocks.z,
            qArgs.mThreads.x, qArgs.mThreads.y, qArgs.mThreads.z,
            qArgs.mSharedMemSize);
        CU_CHECK(cuLaunchKernel(static_cast<CUfunction>(ker), qArgs.mBlocks.x,
                                qArgs.mBlocks.y, qArgs.mBlocks.z,
                                qArgs.mThreads.x, qArgs.mThreads.y,
                                qArgs.mThreads.z, qArgs.mSharedMemSize,
                                qArgs.mStream, params, NULL));
    }
};

class Kernel
    : public common::KernelInterface<CUmodule, CUfunction, Enqueuer,
                                     CUdeviceptr> {
   public:
    using ModuleType = CUmodule;
    using KernelType = CUfunction;
    using DevPtrType = CUdeviceptr;
    using BaseClass =
        common::KernelInterface<ModuleType, KernelType, Enqueuer, DevPtrType>;

    Kernel() : BaseClass("", nullptr, nullptr) {}
    Kernel(std::string name, ModuleType mod, KernelType ker)
        : BaseClass(name, mod, ker) {}

    DevPtrType getDevPtr(const char* name) final;

    void copyToReadOnly(DevPtrType dst, DevPtrType src, size_t bytes) final;

    void setFlag(DevPtrType dst, int* scalarValPtr,
                 const bool syncCopy = false) final;

    int getFlag(DevPtrType src) final;
};

}  // namespace cuda
}  // namespace flare
