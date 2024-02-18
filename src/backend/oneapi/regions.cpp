/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_oneapi.hpp>
// #include <kernel/regions.hpp>
#include <regions.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace oneapi {

template<typename T>
Array<T> regions(const Array<char> &in, fly_connectivity connectivity) {
    ONEAPI_NOT_SUPPORTED("regions Not supported");

    const fly::dim4 &dims = in.dims();
    Array<T> out         = createEmptyArray<T>(dims);
    // kernel::regions<T>(out, in, connectivity == FLY_CONNECTIVITY_8_4, 2);
    return out;
}

#define INSTANTIATE(T)                                  \
    template Array<T> regions<T>(const Array<char> &in, \
                                 fly_connectivity connectivity);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace oneapi
}  // namespace flare
