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

#include <Array.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/half.hpp>
#include <kernel/reduce.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <reduce.hpp>
#include <fly/dim4.hpp>

#include <complex>
#include <functional>

using fly::dim4;
using flare::common::Binary;
using flare::common::half;
using flare::common::Transform;
using flare::cpu::cdouble;

namespace flare {
namespace common {

template<>
struct Binary<cdouble, fly_add_t> {
    static cdouble init() { return cdouble(0, 0); }

    cdouble operator()(cdouble lhs, cdouble rhs) {
        return cdouble(real(lhs) + real(rhs), imag(lhs) + imag(rhs));
    }
};

}  // namespace common
namespace cpu {

template<fly_op_t op, typename Ti, typename To>
using reduce_dim_func = std::function<void(
    Param<To>, const dim_t, CParam<Ti>, const dim_t, const int, bool, double)>;

template<fly_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    dim4 odims = in.dims();
    odims[dim] = 1;

    Array<To> out = createEmptyArray<To>(odims);
    static const reduce_dim_func<op, Ti, To> reduce_funcs[4] = {
        kernel::reduce_dim<op, Ti, To, 1>(),
        kernel::reduce_dim<op, Ti, To, 2>(),
        kernel::reduce_dim<op, Ti, To, 3>(),
        kernel::reduce_dim<op, Ti, To, 4>()};

    getQueue().enqueue(reduce_funcs[in.ndims() - 1], out, 0, in, 0, dim,
                       change_nan, nanval);

    return out;
}

template<fly_op_t op, typename Ti, typename Tk, typename To>
using reduce_dim_func_by_key =
    std::function<void(Param<To> ovals, const dim_t ovOffset, CParam<Tk> keys,
                       CParam<Ti> vals, const dim_t vOffset, int *n_reduced,
                       const int dim, bool change_nan, double nanval)>;

template<fly_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    dim4 okdims = keys.dims();
    dim4 ovdims = vals.dims();

    int n_reduced;
    Array<Tk> fullsz_okeys = createEmptyArray<Tk>(okdims);
    getQueue().enqueue(kernel::n_reduced_keys<Tk>, fullsz_okeys, &n_reduced,
                       keys);
    getQueue().sync();

    okdims[0]   = n_reduced;
    ovdims[dim] = n_reduced;

    std::vector<fly_seq> index;
    for (int i = 0; i < keys.ndims(); ++i) {
        fly_seq s = {0.0, static_cast<double>(okdims[i]) - 1, 1.0};
        index.push_back(s);
    }
    Array<Tk> okeys = createSubArray<Tk>(fullsz_okeys, index, true);
    Array<To> ovals = createEmptyArray<To>(ovdims);

    static const reduce_dim_func_by_key<op, Ti, Tk, To> reduce_funcs[4] = {
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 1>(),
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 2>(),
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 3>(),
        kernel::reduce_dim_by_key<op, Ti, Tk, To, 4>()};

    getQueue().enqueue(reduce_funcs[vals.ndims() - 1], ovals, 0, keys, vals, 0,
                       &n_reduced, dim, change_nan, nanval);

    keys_out = okeys;
    vals_out = ovals;
}

template<fly_op_t op, typename Ti, typename To>
using reduce_all_func =
    std::function<void(Param<To>, CParam<Ti>, bool, double)>;

template<fly_op_t op, typename Ti, typename To>
Array<To> reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    in.eval();

    Array<To> out = createEmptyArray<To>(1);
    static const reduce_all_func<op, Ti, To> reduce_all_kernel =
        kernel::reduce_all<op, Ti, To>();
    getQueue().enqueue(reduce_all_kernel, out, in, change_nan, nanval);
    getQueue().sync();
    return out;
}

#define INSTANTIATE(ROp, Ti, To)                                               \
    template Array<To> reduce<ROp, Ti, To>(const Array<Ti> &in, const int dim, \
                                           bool change_nan, double nanval);    \
    template Array<To> reduce_all<ROp, Ti, To>(                                \
        const Array<Ti> &in, bool change_nan, double nanval);                  \
    template void reduce_by_key<ROp, Ti, int, To>(                             \
        Array<int> & keys_out, Array<To> & vals_out, const Array<int> &keys,   \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template void reduce_by_key<ROp, Ti, uint, To>(                            \
        Array<uint> & keys_out, Array<To> & vals_out, const Array<uint> &keys, \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval);

// min
INSTANTIATE(fly_min_t, float, float)
INSTANTIATE(fly_min_t, double, double)
INSTANTIATE(fly_min_t, cfloat, cfloat)
INSTANTIATE(fly_min_t, cdouble, cdouble)
INSTANTIATE(fly_min_t, int, int)
INSTANTIATE(fly_min_t, uint, uint)
INSTANTIATE(fly_min_t, intl, intl)
INSTANTIATE(fly_min_t, uintl, uintl)
INSTANTIATE(fly_min_t, char, char)
INSTANTIATE(fly_min_t, uchar, uchar)
INSTANTIATE(fly_min_t, short, short)
INSTANTIATE(fly_min_t, ushort, ushort)
INSTANTIATE(fly_min_t, half, half)

// max
INSTANTIATE(fly_max_t, float, float)
INSTANTIATE(fly_max_t, double, double)
INSTANTIATE(fly_max_t, cfloat, cfloat)
INSTANTIATE(fly_max_t, cdouble, cdouble)
INSTANTIATE(fly_max_t, int, int)
INSTANTIATE(fly_max_t, uint, uint)
INSTANTIATE(fly_max_t, intl, intl)
INSTANTIATE(fly_max_t, uintl, uintl)
INSTANTIATE(fly_max_t, char, char)
INSTANTIATE(fly_max_t, uchar, uchar)
INSTANTIATE(fly_max_t, short, short)
INSTANTIATE(fly_max_t, ushort, ushort)
INSTANTIATE(fly_max_t, half, half)

// sum
INSTANTIATE(fly_add_t, float, float)
INSTANTIATE(fly_add_t, double, double)
INSTANTIATE(fly_add_t, cfloat, cfloat)
INSTANTIATE(fly_add_t, cdouble, cdouble)
INSTANTIATE(fly_add_t, int, int)
INSTANTIATE(fly_add_t, int, float)
INSTANTIATE(fly_add_t, uint, uint)
INSTANTIATE(fly_add_t, uint, float)
INSTANTIATE(fly_add_t, intl, intl)
INSTANTIATE(fly_add_t, intl, double)
INSTANTIATE(fly_add_t, uintl, uintl)
INSTANTIATE(fly_add_t, uintl, double)
INSTANTIATE(fly_add_t, char, int)
INSTANTIATE(fly_add_t, char, float)
INSTANTIATE(fly_add_t, uchar, uint)
INSTANTIATE(fly_add_t, uchar, float)
INSTANTIATE(fly_add_t, short, int)
INSTANTIATE(fly_add_t, short, float)
INSTANTIATE(fly_add_t, ushort, uint)
INSTANTIATE(fly_add_t, ushort, float)
INSTANTIATE(fly_add_t, half, float)
INSTANTIATE(fly_add_t, half, half)

// mul
INSTANTIATE(fly_mul_t, float, float)
INSTANTIATE(fly_mul_t, double, double)
INSTANTIATE(fly_mul_t, cfloat, cfloat)
INSTANTIATE(fly_mul_t, cdouble, cdouble)
INSTANTIATE(fly_mul_t, int, int)
INSTANTIATE(fly_mul_t, uint, uint)
INSTANTIATE(fly_mul_t, intl, intl)
INSTANTIATE(fly_mul_t, uintl, uintl)
INSTANTIATE(fly_mul_t, char, int)
INSTANTIATE(fly_mul_t, uchar, uint)
INSTANTIATE(fly_mul_t, short, int)
INSTANTIATE(fly_mul_t, ushort, uint)
INSTANTIATE(fly_mul_t, half, float)

// count
INSTANTIATE(fly_notzero_t, float, uint)
INSTANTIATE(fly_notzero_t, double, uint)
INSTANTIATE(fly_notzero_t, cfloat, uint)
INSTANTIATE(fly_notzero_t, cdouble, uint)
INSTANTIATE(fly_notzero_t, int, uint)
INSTANTIATE(fly_notzero_t, uint, uint)
INSTANTIATE(fly_notzero_t, intl, uint)
INSTANTIATE(fly_notzero_t, uintl, uint)
INSTANTIATE(fly_notzero_t, char, uint)
INSTANTIATE(fly_notzero_t, uchar, uint)
INSTANTIATE(fly_notzero_t, short, uint)
INSTANTIATE(fly_notzero_t, ushort, uint)
INSTANTIATE(fly_notzero_t, half, uint)

// anytrue
INSTANTIATE(fly_or_t, float, char)
INSTANTIATE(fly_or_t, double, char)
INSTANTIATE(fly_or_t, cfloat, char)
INSTANTIATE(fly_or_t, cdouble, char)
INSTANTIATE(fly_or_t, int, char)
INSTANTIATE(fly_or_t, uint, char)
INSTANTIATE(fly_or_t, intl, char)
INSTANTIATE(fly_or_t, uintl, char)
INSTANTIATE(fly_or_t, char, char)
INSTANTIATE(fly_or_t, uchar, char)
INSTANTIATE(fly_or_t, short, char)
INSTANTIATE(fly_or_t, ushort, char)
INSTANTIATE(fly_or_t, half, char)

// alltrue
INSTANTIATE(fly_and_t, float, char)
INSTANTIATE(fly_and_t, double, char)
INSTANTIATE(fly_and_t, cfloat, char)
INSTANTIATE(fly_and_t, cdouble, char)
INSTANTIATE(fly_and_t, int, char)
INSTANTIATE(fly_and_t, uint, char)
INSTANTIATE(fly_and_t, intl, char)
INSTANTIATE(fly_and_t, uintl, char)
INSTANTIATE(fly_and_t, char, char)
INSTANTIATE(fly_and_t, uchar, char)
INSTANTIATE(fly_and_t, short, char)
INSTANTIATE(fly_and_t, ushort, char)
INSTANTIATE(fly_and_t, half, char)

}  // namespace cpu
}  // namespace flare
