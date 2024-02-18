/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <rotate.hpp>

#include <kernel/rotate.hpp>

namespace flare {
namespace opencl {
template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const fly::dim4 &odims,
                const fly_interp_type method) {
    Array<T> out = createEmptyArray<T>(odims);

    switch (method) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER:
            kernel::rotate<T>(out, in, theta, method, 1);
            break;
        case FLY_INTERP_BILINEAR:
        case FLY_INTERP_BILINEAR_COSINE:
            kernel::rotate<T>(out, in, theta, method, 2);
            break;
        case FLY_INTERP_BICUBIC:
        case FLY_INTERP_BICUBIC_SPLINE:
            kernel::rotate<T>(out, in, theta, method, 3);
            break;
        default: FLY_ERROR("Unsupported interpolation type", FLY_ERR_ARG);
    }
    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> rotate(const Array<T> &in, const float theta, \
                             const fly::dim4 &odims,                 \
                             const fly_interp_type method);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
}  // namespace opencl
}  // namespace flare
