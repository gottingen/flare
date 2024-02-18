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
// count
INSTANTIATE(fly_notzero_t, float, uint)
INSTANTIATE(fly_notzero_t, double, uint)
INSTANTIATE(fly_notzero_t, cfloat, uint)
INSTANTIATE(fly_notzero_t, cdouble, uint)
INSTANTIATE(fly_notzero_t, int, uint)
INSTANTIATE(fly_notzero_t, uint, uint)
INSTANTIATE(fly_notzero_t, intl, uint)
INSTANTIATE(fly_notzero_t, uintl, uint)
INSTANTIATE(fly_notzero_t, char, uint)
INSTANTIATE(fly_notzero_t, uchar, uint)
INSTANTIATE(fly_notzero_t, short, uint)
INSTANTIATE(fly_notzero_t, ushort, uint)
INSTANTIATE(fly_notzero_t, half, uint)
}  // namespace opencl
}  // namespace flare
