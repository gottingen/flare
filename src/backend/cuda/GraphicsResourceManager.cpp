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

#include <GraphicsResourceManager.hpp>

#include <common/graphics_common.hpp>
// cuda_gl_interop.h does not include OpenGL headers for ARM
// __gl_h_ should be defined by glad.h inclusion
#include <cuda_gl_interop.h>
#include <err_cuda.hpp>
#include <platform.hpp>

namespace flare {
namespace cuda {
GraphicsResourceManager::ShrdResVector
GraphicsResourceManager::registerResources(
    const std::vector<uint32_t>& resources) {
    ShrdResVector output;

    auto deleter = [](cudaGraphicsResource_t* handle) {
        // FIXME Having a CUDA_CHECK around unregister
        // call is causing invalid GL context.
        // Moving TheiaManager class singleton as data
        // member of DeviceManager with proper ordering
        // of member destruction doesn't help either.
        // Calling makeContextCurrent also doesn't help.
        cudaGraphicsUnregisterResource(*handle);
        delete handle;
    };

    for (auto id : resources) {
        cudaGraphicsResource_t r;
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &r, id, cudaGraphicsMapFlagsWriteDiscard));
        output.emplace_back(new cudaGraphicsResource_t(r), deleter);
    }

    return output;
}
}  // namespace cuda
}  // namespace flare
