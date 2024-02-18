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
namespace opencl {
// min
INSTANTIATE(fly_min_t, float, float)
INSTANTIATE(fly_min_t, double, double)
INSTANTIATE(fly_min_t, cfloat, cfloat)
INSTANTIATE(fly_min_t, cdouble, cdouble)
INSTANTIATE(fly_min_t, int, int)
INSTANTIATE(fly_min_t, uint, uint)
INSTANTIATE(fly_min_t, intl, intl)
INSTANTIATE(fly_min_t, uintl, uintl)
INSTANTIATE(fly_min_t, char, char)
INSTANTIATE(fly_min_t, uchar, uchar)
INSTANTIATE(fly_min_t, short, short)
INSTANTIATE(fly_min_t, ushort, ushort)
INSTANTIATE(fly_min_t, half, half)
}  // namespace opencl
}  // namespace flare
