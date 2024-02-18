/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cpu.hpp>
#include <kernel/meanshift.hpp>
#include <math.hpp>
#include <meanshift.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>
#include <algorithm>
#include <cmath>

using fly::dim4;
using std::vector;

namespace flare {
namespace cpu {
template<typename T>
Array<T> meanshift(const Array<T> &in, const float &spatialSigma,
                   const float &chromaticSigma, const unsigned &numIterations,
                   const bool &isColor) {
    Array<T> out = createEmptyArray<T>(in.dims());

    if (isColor) {
        getQueue().enqueue(kernel::meanShift<T, true>, out, in, spatialSigma,
                           chromaticSigma, numIterations);
    } else {
        getQueue().enqueue(kernel::meanShift<T, false>, out, in, spatialSigma,
                           chromaticSigma, numIterations);
    }

    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> meanshift<T>(const Array<T> &, const float &, \
                                   const float &, const unsigned &, \
                                   const bool &);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
}  // namespace cpu
}  // namespace flare
