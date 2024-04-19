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
#include <ireduce.hpp>
#include <kernel/ireduce.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>

#include <complex>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cpu {

template<fly_op_t op, typename T>
using ireduce_dim_func =
    std::function<void(Param<T>, Param<uint>, const dim_t, CParam<T>,
                       const dim_t, const int, CParam<uint>)>;

template<fly_op_t op, typename T>
void ireduce(Array<T> &out, Array<uint> &loc, const Array<T> &in,
             const int dim) {
    dim4 odims       = in.dims();
    odims[dim]       = 1;
    Array<uint> rlen = createEmptyArray<uint>(fly::dim4(0));
    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0, dim,
                       rlen);
}

template<fly_op_t op, typename T>
void rreduce(Array<T> &out, Array<uint> &loc, const Array<T> &in, const int dim,
             const Array<uint> &rlen) {
    dim4 odims = in.dims();
    odims[dim] = 1;

    static const ireduce_dim_func<op, T> ireduce_funcs[] = {
        kernel::ireduce_dim<op, T, 1>(), kernel::ireduce_dim<op, T, 2>(),
        kernel::ireduce_dim<op, T, 3>(), kernel::ireduce_dim<op, T, 4>()};

    getQueue().enqueue(ireduce_funcs[in.ndims() - 1], out, loc, 0, in, 0, dim,
                       rlen);
}

template<fly_op_t op, typename T>
T ireduce_all(unsigned *loc, const Array<T> &in) {
    getQueue().sync();

    fly::dim4 dims    = in.dims();
    fly::dim4 strides = in.strides();
    const T *inPtr   = in.get();

    kernel::MinMaxOp<op, T> Op(inPtr[0], 0);

    for (dim_t l = 0; l < dims[3]; l++) {
        dim_t off3 = l * strides[3];

        for (dim_t k = 0; k < dims[2]; k++) {
            dim_t off2 = k * strides[2];

            for (dim_t j = 0; j < dims[1]; j++) {
                dim_t off1 = j * strides[1];

                for (dim_t i = 0; i < dims[0]; i++) {
                    dim_t idx = i + off1 + off2 + off3;
                    Op(inPtr[idx], idx);
                }
            }
        }
    }

    *loc = Op.m_idx;
    return Op.m_val;
}

#define INSTANTIATE(ROp, T)                                           \
    template void ireduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim); \
    template void rreduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim,  \
                                  const Array<uint> &rlen);           \
    template T ireduce_all<ROp, T>(unsigned *loc, const Array<T> &in);

// min
INSTANTIATE(fly_min_t, float)
INSTANTIATE(fly_min_t, double)
INSTANTIATE(fly_min_t, cfloat)
INSTANTIATE(fly_min_t, cdouble)
INSTANTIATE(fly_min_t, int)
INSTANTIATE(fly_min_t, uint)
INSTANTIATE(fly_min_t, intl)
INSTANTIATE(fly_min_t, uintl)
INSTANTIATE(fly_min_t, char)
INSTANTIATE(fly_min_t, uchar)
INSTANTIATE(fly_min_t, short)
INSTANTIATE(fly_min_t, ushort)
INSTANTIATE(fly_min_t, half)

// max
INSTANTIATE(fly_max_t, float)
INSTANTIATE(fly_max_t, double)
INSTANTIATE(fly_max_t, cfloat)
INSTANTIATE(fly_max_t, cdouble)
INSTANTIATE(fly_max_t, int)
INSTANTIATE(fly_max_t, uint)
INSTANTIATE(fly_max_t, intl)
INSTANTIATE(fly_max_t, uintl)
INSTANTIATE(fly_max_t, char)
INSTANTIATE(fly_max_t, uchar)
INSTANTIATE(fly_max_t, short)
INSTANTIATE(fly_max_t, ushort)
INSTANTIATE(fly_max_t, half)

}  // namespace cpu
}  // namespace flare
