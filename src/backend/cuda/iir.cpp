/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <convolve.hpp>
#include <err_cuda.hpp>
#include <iir.hpp>
#include <kernel/iir.hpp>
#include <math.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {
template<typename T>
Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x) {
    FLY_BATCH_KIND type = x.ndims() == 1 ? FLY_BATCH_NONE : FLY_BATCH_SAME;
    if (x.ndims() != b.ndims()) {
        type = (x.ndims() < b.ndims()) ? FLY_BATCH_RHS : FLY_BATCH_LHS;
    }

    // Extract the first N elements
    Array<T> c = convolve<T, T>(x, b, type, 1, true);
    dim4 cdims = c.dims();
    cdims[0]   = x.dims()[0];
    c.resetDims(cdims);

    int num_a = a.dims()[0];

    if (num_a == 1) { return c; }

    dim4 ydims = c.dims();
    Array<T> y = createEmptyArray<T>(ydims);

    if (a.ndims() > 1) {
        kernel::iir<T, true>(y, c, a);
    } else {
        kernel::iir<T, false>(y, c, a);
    }
    return y;
}

#define INSTANTIATE(T)                                          \
    template Array<T> iir(const Array<T> &b, const Array<T> &a, \
                          const Array<T> &x);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
}  // namespace cuda
}  // namespace flare
