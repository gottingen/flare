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
#include <fly/constants.h>
#include <algorithm>
#include <cmath>
#include "backend.hpp"

namespace flare {
namespace cpu {
static inline dim_t trimIndex(int const& idx, dim_t const& len) {
    int ret_val = idx;
    if (ret_val < 0) {
        int offset = (abs(ret_val) - 1) % len;
        ret_val    = offset;
    } else if (ret_val >= (int)len) {
        int offset = abs(ret_val) % len;
        ret_val    = len - offset - 1;
    }
    return ret_val;
}

static inline unsigned getIdx(fly::dim4 const& strides, int i, int j = 0,
                              int k = 0, int l = 0) {
    return (l * strides[3] + k * strides[2] + j * strides[1] + i * strides[0]);
}

template<typename T>
void gaussian1D(T* out, int const dim, double sigma = 0.0) {
    if (!(sigma > 0)) sigma = 0.25 * dim;

    T sum = (T)0;
    for (int i = 0; i < dim; i++) {
        int x = i - (dim - 1) / 2;
        T el  = 1. / std::sqrt(2 * fly::Pi * sigma * sigma) *
               std::exp(-((x * x) / (2 * (sigma * sigma))));
        out[i] = el;
        sum += el;
    }

    for (int k = 0; k < dim; k++) out[k] /= sum;
}
}  // namespace cpu
}  // namespace flare
