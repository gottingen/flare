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

#include <platform.hpp>

namespace flare {
namespace cuda {

Kernel::DevPtrType Kernel::getDevPtr(const char* name) {
    Kernel::DevPtrType out = 0;
    size_t size            = 0;
    CU_CHECK(cuModuleGetGlobal(&out, &size, this->getModuleHandle(), name));
    return out;
}

void Kernel::copyToReadOnly(Kernel::DevPtrType dst, Kernel::DevPtrType src,
                            size_t bytes) {
    CU_CHECK(cuMemcpyDtoDAsync(dst, src, bytes, getActiveStream()));
}

void Kernel::setFlag(Kernel::DevPtrType dst, int* scalarValPtr,
                     const bool syncCopy) {
    CU_CHECK(
        cuMemcpyHtoDAsync(dst, scalarValPtr, sizeof(int), getActiveStream()));
    if (syncCopy) { CU_CHECK(cuStreamSynchronize(getActiveStream())); }
}

int Kernel::getFlag(Kernel::DevPtrType src) {
    int retVal = 0;
    CU_CHECK(cuMemcpyDtoHAsync(&retVal, src, sizeof(int), getActiveStream()));
    CU_CHECK(cuStreamSynchronize(getActiveStream()));
    return retVal;
}

}  // namespace cuda
}  // namespace flare
