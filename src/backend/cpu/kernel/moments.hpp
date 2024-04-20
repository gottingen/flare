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
#include <math.hpp>
#include <utility.hpp>
#include <fly/defines.h>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
void moments(Param<float> output, CParam<T> input, fly_moment_type moment) {
    T const *const in       = input.get();
    fly::dim4 const idims    = input.dims();
    fly::dim4 const istrides = input.strides();
    fly::dim4 const ostrides = output.strides();

    float *out = output.get();

    for (dim_t w = 0; w < idims[3]; w++) {
        for (dim_t z = 0; z < idims[2]; z++) {
            dim_t out_off = w * ostrides[3] + z * ostrides[2];
            for (dim_t y = 0; y < idims[1]; y++) {
                dim_t in_off =
                    y * istrides[1] + z * istrides[2] + w * istrides[3];
                for (dim_t x = 0; x < idims[0]; x++) {
                    dim_t m_off = 0;
                    float val   = in[in_off + x];
                    if ((moment & FLY_MOMENT_M00) > 0) {
                        out[out_off + m_off] += val;
                        m_off++;
                    }
                    if ((moment & FLY_MOMENT_M01) > 0) {
                        out[out_off + m_off] += x * val;
                        m_off++;
                    }
                    if ((moment & FLY_MOMENT_M10) > 0) {
                        out[out_off + m_off] += y * val;
                        m_off++;
                    }
                    if ((moment & FLY_MOMENT_M11) > 0) {
                        out[out_off + m_off] += x * y * val;
                        m_off++;
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
