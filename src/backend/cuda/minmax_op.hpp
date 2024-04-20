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

#include <common/Binary.hpp>

namespace flare {
namespace cuda {

template<typename T>
static double cabs(const T &in) {
    return (double)in;
}

template<>
double cabs<char>(const char &in) {
    return (double)(in > 0);
}

template<>
double cabs<cfloat>(const cfloat &in) {
    return (double)abs(in);
}

template<>
double cabs<cdouble>(const cdouble &in) {
    return (double)abs(in);
}

template<fly_op_t op, typename T>
struct MinMaxOp {
    T m_val;
    uint m_idx;
    MinMaxOp(T val, uint idx) : m_val(val), m_idx(idx) {
        using flare::cuda::is_nan;
        if (is_nan(val)) { m_val = common::Binary<compute_t<T>, op>::init(); }
    }

    void operator()(T val, uint idx) {
        if ((cabs(val) < cabs(m_val) ||
             (cabs(val) == cabs(m_val) && idx > m_idx))) {
            m_val = val;
            m_idx = idx;
        }
    }
};

template<typename T>
struct MinMaxOp<fly_max_t, T> {
    T m_val;
    uint m_idx;
    MinMaxOp(T val, uint idx) : m_val(val), m_idx(idx) {
        using flare::cuda::is_nan;
        if (is_nan(val)) { m_val = common::Binary<T, fly_max_t>::init(); }
    }

    void operator()(T val, uint idx) {
        if ((cabs(val) > cabs(m_val) ||
             (cabs(val) == cabs(m_val) && idx <= m_idx))) {
            m_val = val;
            m_idx = idx;
        }
    }
};

}  // namespace cuda
}  // namespace flare
