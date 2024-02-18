/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <gradient.hpp>
#include <handle.hpp>
#include <fly/defines.h>
#include <fly/image.h>

using fly::dim4;
using flare::getArray;
using detail::cdouble;
using detail::cfloat;

template<typename T>
static inline void gradient(fly_array *grad0, fly_array *grad1,
                            const fly_array in) {
    gradient<T>(getArray<T>(*grad0), getArray<T>(*grad1), getArray<T>(in));
}

fly_err fly_gradient(fly_array *grows, fly_array *gcols, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 idims        = info.dims();

        DIM_ASSERT(2, info.elements() > 0);

        fly_array grad0;
        fly_array grad1;
        FLY_CHECK(fly_create_handle(&grad0, idims.ndims(), idims.get(), type));
        FLY_CHECK(fly_create_handle(&grad1, idims.ndims(), idims.get(), type));

        switch (type) {
            case f32: gradient<float>(&grad0, &grad1, in); break;
            case c32: gradient<cfloat>(&grad0, &grad1, in); break;
            case f64: gradient<double>(&grad0, &grad1, in); break;
            case c64: gradient<cdouble>(&grad0, &grad1, in); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*grows, grad0);
        std::swap(*gcols, grad1);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
