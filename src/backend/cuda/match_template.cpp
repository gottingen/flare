/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cuda.hpp>
#include <kernel/match_template.hpp>
#include <match_template.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {

template<typename inType, typename outType>
Array<outType> match_template(const Array<inType> &sImg,
                              const Array<inType> &tImg,
                              const fly::matchType mType) {
    Array<outType> out = createEmptyArray<outType>(sImg.dims());
    bool needMean = mType == FLY_ZSAD || mType == FLY_LSAD || mType == FLY_ZSSD ||
                    mType == FLY_LSSD || mType == FLY_ZNCC;
    kernel::matchTemplate<inType, outType>(out, sImg, tImg, mType, needMean);
    return out;
}

#define INSTANTIATE(in_t, out_t)                       \
    template Array<out_t> match_template<in_t, out_t>( \
        const Array<in_t> &, const Array<in_t> &, const fly::matchType);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(char, float)
INSTANTIATE(int, float)
INSTANTIATE(uint, float)
INSTANTIATE(uchar, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)

}  // namespace cuda
}  // namespace flare
