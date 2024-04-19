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

#include <fly/array.h>
#include <fly/opencl.h>
#include <cstring>

namespace fly {
template<>
FLY_API cl_mem *array::device() const {
    auto *mem_ptr = new cl_mem;
    void *dptr    = nullptr;
    fly_err err    = fly_get_device_ptr(&dptr, get());
    memcpy(mem_ptr, &dptr, sizeof(void *));
    if (err != FLY_SUCCESS) {
        throw fly::exception("Failed to get cl_mem from array object");
    }
    return mem_ptr;
}
}  // namespace fly
