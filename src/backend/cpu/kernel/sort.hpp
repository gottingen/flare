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
#include <Param.hpp>
#include <err_cpu.hpp>
#include <math.hpp>
#include <algorithm>
#include <functional>
#include <numeric>

namespace flare {
namespace cpu {
namespace kernel {

// Based off of http://stackoverflow.com/a/12399290
template<typename T>
void sort0Iterative(Param<T> val, bool isAscending) {
    // initialize original index locations
    T *val_ptr = val.get();

    std::function<bool(T, T)> op = std::greater<T>();
    if (isAscending) { op = std::less<T>(); }

    T *comp_ptr = nullptr;
    for (dim_t w = 0; w < val.dims(3); w++) {
        dim_t valW = w * val.strides(3);
        for (dim_t z = 0; z < val.dims(2); z++) {
            dim_t valWZ = valW + z * val.strides(2);
            for (dim_t y = 0; y < val.dims(1); y++) {
                dim_t valOffset = valWZ + y * val.strides(1);

                comp_ptr = val_ptr + valOffset;
                std::sort(comp_ptr, comp_ptr + val.dims(0), op);
            }
        }
    }
    return;
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
