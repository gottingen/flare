/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <transform.hpp>

#include <err_oneapi.hpp>
#include <kernel/transform.hpp>

namespace flare {
namespace oneapi {

template<typename T>
void transform(Array<T> &out, const Array<T> &in, const Array<float> &tf,
               const fly_interp_type method, const bool inverse,
               const bool perspective) {
    switch (method) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER:
            kernel::transform<T>(out, in, tf, inverse, perspective, method, 1);
            break;
        case FLY_INTERP_BILINEAR:
        case FLY_INTERP_BILINEAR_COSINE:
            kernel::transform<T>(out, in, tf, inverse, perspective, method, 2);
            break;
        case FLY_INTERP_BICUBIC:
        case FLY_INTERP_BICUBIC_SPLINE:
            kernel::transform<T>(out, in, tf, inverse, perspective, method, 3);
            break;
        default: FLY_ERROR("Unsupported interpolation type", FLY_ERR_ARG);
    }
}

#define INSTANTIATE(T)                                                       \
    template void transform(Array<T> &out, const Array<T> &in,               \
                            const Array<float> &tf,                          \
                            const fly_interp_type method, const bool inverse, \
                            const bool perspective);

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

}  // namespace oneapi
}  // namespace flare
