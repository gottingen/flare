/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_oneapi.hpp>
#include <identity.hpp>
#include <solve.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <platform.hpp>

namespace flare {
namespace oneapi {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    Array<T> I = identity<T>(in.dims());
    return solve<T>(in, I);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace oneapi
}  // namespace flare

#else  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace oneapi {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    ONEAPI_NOT_SUPPORTED("");
    FLY_ERROR("Linear Algebra is disabled on OneAPI backend",
             FLY_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace oneapi
}  // namespace flare

#endif
