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
// alltrue
INSTANTIATE(fly_and_t, float, char)
INSTANTIATE(fly_and_t, double, char)
INSTANTIATE(fly_and_t, cfloat, char)
INSTANTIATE(fly_and_t, cdouble, char)
INSTANTIATE(fly_and_t, int, char)
INSTANTIATE(fly_and_t, uint, char)
INSTANTIATE(fly_and_t, intl, char)
INSTANTIATE(fly_and_t, uintl, char)
INSTANTIATE(fly_and_t, char, char)
INSTANTIATE(fly_and_t, uchar, char)
INSTANTIATE(fly_and_t, short, char)
INSTANTIATE(fly_and_t, ushort, char)
INSTANTIATE(fly_and_t, half, char)
}  // namespace opencl
}  // namespace flare
