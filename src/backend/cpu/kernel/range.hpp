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
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cpu {
namespace kernel {

template<typename T, int dim>
void range(Param<T> output) {
    T* out = output.get();

    const dim4 dims    = output.dims();
    const dim4 strides = output.strides();

    for (dim_t w = 0; w < dims[3]; w++) {
        dim_t offW = w * strides[3];
        for (dim_t z = 0; z < dims[2]; z++) {
            dim_t offWZ = offW + z * strides[2];
            for (dim_t y = 0; y < dims[1]; y++) {
                dim_t offWZY = offWZ + y * strides[1];
                for (dim_t x = 0; x < dims[0]; x++) {
                    dim_t id = offWZY + x;
                    if (dim == 0) {
                        out[id] = x;
                    } else if (dim == 1) {
                        out[id] = y;
                    } else if (dim == 2) {
                        out[id] = z;
                    } else if (dim == 3) {
                        out[id] = w;
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
