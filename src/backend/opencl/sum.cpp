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
// sum
INSTANTIATE(fly_add_t, float, float)
INSTANTIATE(fly_add_t, double, double)
INSTANTIATE(fly_add_t, cfloat, cfloat)
INSTANTIATE(fly_add_t, cdouble, cdouble)
INSTANTIATE(fly_add_t, int, int)
INSTANTIATE(fly_add_t, int, float)
INSTANTIATE(fly_add_t, uint, uint)
INSTANTIATE(fly_add_t, uint, float)
INSTANTIATE(fly_add_t, intl, intl)
INSTANTIATE(fly_add_t, intl, double)
INSTANTIATE(fly_add_t, uintl, uintl)
INSTANTIATE(fly_add_t, uintl, double)
INSTANTIATE(fly_add_t, char, int)
INSTANTIATE(fly_add_t, char, float)
INSTANTIATE(fly_add_t, uchar, uint)
INSTANTIATE(fly_add_t, uchar, float)
INSTANTIATE(fly_add_t, short, int)
INSTANTIATE(fly_add_t, short, float)
INSTANTIATE(fly_add_t, ushort, uint)
INSTANTIATE(fly_add_t, ushort, float)
INSTANTIATE(fly_add_t, half, half)
INSTANTIATE(fly_add_t, half, float)
}  // namespace opencl
}  // namespace flare
