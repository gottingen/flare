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

#include <cl2hpp.hpp>
#include <kernel/KParam.hpp>

namespace flare {
namespace opencl {

struct Param {
    cl::Buffer* data;
    KParam info;
    Param& operator=(const Param& other) = default;
    Param(const Param& other)            = default;
    Param(Param&& other)                 = default;

    dim_t* dims_ptr() { return info.dims; }
    dim_t* strides_ptr() { return info.strides; }

    // FLY_DEPRECATED("Use Array<T>")
    Param();
    // FLY_DEPRECATED("Use Array<T>")
    Param(cl::Buffer* data_, KParam info_);
    ~Param() = default;
};

// FLY_DEPRECATED("Use Array<T>")
Param makeParam(cl::Buffer& mem, int off, const int dims[4],
                const int strides[4]);
}  // namespace opencl
}  // namespace flare
