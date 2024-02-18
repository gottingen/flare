/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <ireduce.hpp>
#include <kernel/ireduce.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
#include <optypes.hpp>
#include <fly/dim4.hpp>
#include <complex>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace opencl {

template<fly_op_t op, typename T>
void ireduce(Array<T> &out, Array<uint> &loc, const Array<T> &in,
             const int dim) {
    Array<uint> rlen = createEmptyArray<uint>(fly::dim4(0));
    kernel::ireduce<T, op>(out, loc.get(), in, dim, rlen);
}

template<fly_op_t op, typename T>
void rreduce(Array<T> &out, Array<uint> &loc, const Array<T> &in, const int dim,
             const Array<uint> &rlen) {
    kernel::ireduce<T, op>(out, loc.get(), in, dim, rlen);
}

template<fly_op_t op, typename T>
T ireduce_all(unsigned *loc, const Array<T> &in) {
    return kernel::ireduceAll<T, op>(loc, in);
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
}  // namespace opencl
}  // namespace flare
