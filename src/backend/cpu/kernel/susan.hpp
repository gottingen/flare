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
void susan_responses(Param<T> output, CParam<T> input, const dim_t idim0,
                     const dim_t idim1, const int radius, const float t,
                     const float g, const unsigned border_len) {
    T* resp_out = output.get();
    const T* in = input.get();

    const unsigned r = border_len;
    const int rSqrd  = radius * radius;

    for (dim_t y = r; y < idim1 - r; ++y) {
        for (dim_t x = r; x < idim0 - r; ++x) {
            const dim_t idx = y * idim0 + x;
            T m_0           = in[idx];
            float nM        = 0.0f;

            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    if (i * i + j * j < rSqrd) {
                        int p         = x + i;
                        int q         = y + j;
                        T m           = in[p + idim0 * q];
                        float exp_pow = std::pow((m - m_0) / t, 6.0);
                        float cM      = std::exp(-exp_pow);
                        nM += cM;
                    }
                }
            }

            resp_out[idx] = nM < g ? g - nM : T(0);
        }
    }
}

template<typename T>
void non_maximal(Param<float> xcoords, Param<float> ycoords,
                 Param<float> response, shared_ptr<unsigned> counter,
                 const dim_t idim0, const dim_t idim1, CParam<T> input,
                 const unsigned border_len, const unsigned max_corners) {
    float* x_out     = xcoords.get();
    float* y_out     = ycoords.get();
    float* resp_out  = response.get();
    unsigned* count  = counter.get();
    const T* resp_in = input.get();

    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = border_len + 1;

    for (dim_t y = r; y < idim1 - r; y++) {
        for (dim_t x = r; x < idim0 - r; x++) {
            const T v = resp_in[y * idim0 + x];

            // Find maximum neighborhood response
            T max_v;
            max_v = std::max(resp_in[(y - 1) * idim0 + x - 1],
                             resp_in[y * idim0 + x - 1]);
            max_v = std::max(max_v, resp_in[(y + 1) * idim0 + x - 1]);
            max_v = std::max(max_v, resp_in[(y - 1) * idim0 + x]);
            max_v = std::max(max_v, resp_in[(y + 1) * idim0 + x]);
            max_v = std::max(max_v, resp_in[(y - 1) * idim0 + x + 1]);
            max_v = std::max(max_v, resp_in[(y)*idim0 + x + 1]);
            max_v = std::max(max_v, resp_in[(y + 1) * idim0 + x + 1]);

            // Stores corner to {x,y,resp}_out if it's response is maximum
            // compared to its 8-neighborhood and greater or equal minimum
            // response
            if (v > max_v) {
                const dim_t idx = *count;
                *count += 1;
                if (idx < max_corners) {
                    x_out[idx]    = (float)x;
                    y_out[idx]    = (float)y;
                    resp_out[idx] = (float)v;
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
