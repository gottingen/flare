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

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void iota(Param<T> output, const fly::dim4& sdims) {
    const fly::dim4 dims    = output.dims();
    data_t<T>* out         = output.get();
    const fly::dim4 strides = output.strides();

    for (dim_t w = 0; w < dims[3]; w++) {
        dim_t offW = w * strides[3];
        dim_t valW = (w % sdims[3]) * sdims[0] * sdims[1] * sdims[2];
        for (dim_t z = 0; z < dims[2]; z++) {
            dim_t offWZ = offW + z * strides[2];
            dim_t valZ  = valW + (z % sdims[2]) * sdims[0] * sdims[1];
            for (dim_t y = 0; y < dims[1]; y++) {
                dim_t offWZY = offWZ + y * strides[1];
                dim_t valY   = valZ + (y % sdims[1]) * sdims[0];
                for (dim_t x = 0; x < dims[0]; x++) {
                    dim_t id = offWZY + x;
                    out[id]  = valY + (x % sdims[0]);
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
