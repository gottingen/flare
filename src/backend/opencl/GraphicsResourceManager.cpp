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
#include <platform.hpp>

namespace flare {
namespace opencl {
GraphicsResourceManager::ShrdResVector
GraphicsResourceManager::registerResources(
    const std::vector<uint32_t>& resources) {
    ShrdResVector output;

    for (auto id : resources) {
        output.emplace_back(new cl::BufferGL(
            getContext(), CL_MEM_WRITE_ONLY,  // NOLINT(hicpp-signed-bitwise)
            id, NULL));
    }

    return output;
}
}  // namespace opencl
}  // namespace flare
