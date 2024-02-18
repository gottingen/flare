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
// mul
INSTANTIATE(fly_mul_t, float, float)
INSTANTIATE(fly_mul_t, double, double)
INSTANTIATE(fly_mul_t, cfloat, cfloat)
INSTANTIATE(fly_mul_t, cdouble, cdouble)
INSTANTIATE(fly_mul_t, int, int)
INSTANTIATE(fly_mul_t, uint, uint)
INSTANTIATE(fly_mul_t, intl, intl)
INSTANTIATE(fly_mul_t, uintl, uintl)
INSTANTIATE(fly_mul_t, char, int)
INSTANTIATE(fly_mul_t, uchar, uint)
INSTANTIATE(fly_mul_t, short, int)
INSTANTIATE(fly_mul_t, ushort, uint)
INSTANTIATE(fly_mul_t, half, float)
}  // namespace cuda
}  // namespace flare
