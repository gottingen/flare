/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/half.hpp>
#include "reduce_impl.hpp"

using flare::common::half;

namespace flare {
namespace cuda {
// max
INSTANTIATE(fly_max_t, float, float)
INSTANTIATE(fly_max_t, double, double)
INSTANTIATE(fly_max_t, cfloat, cfloat)
INSTANTIATE(fly_max_t, cdouble, cdouble)
INSTANTIATE(fly_max_t, int, int)
INSTANTIATE(fly_max_t, uint, uint)
INSTANTIATE(fly_max_t, intl, intl)
INSTANTIATE(fly_max_t, uintl, uintl)
INSTANTIATE(fly_max_t, char, char)
INSTANTIATE(fly_max_t, uchar, uchar)
INSTANTIATE(fly_max_t, short, short)
INSTANTIATE(fly_max_t, ushort, ushort)
INSTANTIATE(fly_max_t, half, half)
}  // namespace cuda
}  // namespace flare
