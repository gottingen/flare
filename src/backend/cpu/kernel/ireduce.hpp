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
#include <common/Binary.hpp>
#include <common/half.hpp>
#include <algorithm>
#include <cmath>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T>
double cabs(const T in) {
    return (double)in;
}
static double cabs(const char in) { return (double)(in > 0); }
static double cabs(const cfloat &in) { return (double)abs(in); }
static double cabs(const cdouble &in) { return (double)abs(in); }

template<fly_op_t op, typename T>
struct MinMaxOp {
    T m_val;
    uint m_idx;
    MinMaxOp(T val, uint idx) : m_val(val), m_idx(idx) {
        using flare::cpu::is_nan;
        if (is_nan(val)) { m_val = common::Binary<T, op>::init(); }
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
        using flare::cpu::is_nan;
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

template<fly_op_t op, typename T, int D>
struct ireduce_dim {
    void operator()(Param<T> output, Param<uint> locParam,
                    const dim_t outOffset, CParam<T> input,
                    const dim_t inOffset, const int dim, CParam<uint> rlen) {
        const fly::dim4 odims    = output.dims();
        const fly::dim4 ostrides = output.strides();
        const fly::dim4 istrides = input.strides();
        const int D1            = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            ireduce_dim<op, T, D1>()(output, locParam,
                                     outOffset + i * ostrides[D1], input,
                                     inOffset + i * istrides[D1], dim, rlen);
        }
    }
};

template<fly_op_t op, typename T>
struct ireduce_dim<op, T, 0> {
    void operator()(Param<T> output, Param<uint> locParam,
                    const dim_t outOffset, CParam<T> input,
                    const dim_t inOffset, const int dim, CParam<uint> rlen) {
        const fly::dim4 idims    = input.dims();
        const fly::dim4 istrides = input.strides();

        T const *const in   = input.get();
        T *out              = output.get();
        uint *loc           = locParam.get();
        const uint *rlenptr = (rlen.get()) ? rlen.get() + outOffset : nullptr;

        dim_t stride = istrides[dim];
        MinMaxOp<op, T> Op(in[inOffset], 0);
        int lim =
            (rlenptr) ? std::min(idims[dim], (dim_t)*rlenptr) : idims[dim];
        for (dim_t i = 0; i < lim; i++) { Op(in[inOffset + i * stride], i); }

        out[outOffset] = Op.m_val;
        loc[outOffset] = Op.m_idx;
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
