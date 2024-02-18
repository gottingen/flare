/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/reduce.hpp>
#include <kernel/reduce_by_key.hpp>
#include <reduce.hpp>
#include <fly/dim4.hpp>
#include <complex>

using fly::dim4;
using std::swap;
namespace flare {
namespace opencl {
template<fly_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    dim4 odims    = in.dims();
    odims[dim]    = 1;
    Array<To> out = createEmptyArray<To>(odims);
    kernel::reduce<Ti, To, op>(out, in, dim, change_nan, nanval);
    return out;
}

template<fly_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    kernel::reduceByKey<op, Ti, Tk, To>(keys_out, vals_out, keys, vals, dim,
                                        change_nan, nanval);
}

template<fly_op_t op, typename Ti, typename To>
Array<To> reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    Array<To> out = createEmptyArray<To>(1);
    kernel::reduceAll<Ti, To, op>(out, in, change_nan, nanval);
    return out;
}

}  // namespace opencl
}  // namespace flare

#define INSTANTIATE(Op, Ti, To)                                                \
    template Array<To> reduce<Op, Ti, To>(const Array<Ti> &in, const int dim,  \
                                          bool change_nan, double nanval);     \
    template void reduce_by_key<Op, Ti, int, To>(                              \
        Array<int> & keys_out, Array<To> & vals_out, const Array<int> &keys,   \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template void reduce_by_key<Op, Ti, uint, To>(                             \
        Array<uint> & keys_out, Array<To> & vals_out, const Array<uint> &keys, \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template Array<To> reduce_all<Op, Ti, To>(const Array<Ti> &in,             \
                                              bool change_nan, double nanval);
