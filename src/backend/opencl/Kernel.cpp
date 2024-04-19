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

#include <Kernel.hpp>

#include <backend.hpp>
#include <cl2hpp.hpp>
#include <common/defines.hpp>
#include <platform.hpp>

namespace flare {
namespace opencl {

Kernel::DevPtrType Kernel::getDevPtr(const char* name) {
    UNUSED(name);
    return nullptr;
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    getQueue().enqueueCopyBuffer(*src, *dst, 0, 0, bytes);
}

void Kernel::setFlag(Kernel::DevPtrType dst, int* scalarValPtr,
                     const bool syncCopy) {
    UNUSED(syncCopy);
    getQueue().enqueueFillBuffer(*dst, *scalarValPtr, 0, sizeof(int));
}

int Kernel::getFlag(Kernel::DevPtrType src) {
    int retVal = 0;
    getQueue().enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(int), &retVal);
    return retVal;
}

}  // namespace opencl
}  // namespace flare
