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

#include <Param.hpp>
#include <kernel/KParam.hpp>
#include <platform.hpp>
#include <fly/defines.h>

namespace flare {
namespace oneapi {

template<typename T>
Param<T> makeParam(sycl::buffer<T> &mem, int off, const int dims[4],
                   const int strides[4]) {
    Param<T> out;
    out.data        = &mem;
    out.info.offset = off;
    for (int i = 0; i < 4; i++) {
        out.info.dims[i]    = dims[i];
        out.info.strides[i] = strides[i];
    }
    return out;
}

}  // namespace oneapi
}  // namespace flare
