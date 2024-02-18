/*******************************************************
 * Copyright (c) 2022, Flare
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
namespace oneapi {
// anytrue
INSTANTIATE(fly_or_t, float, char)
INSTANTIATE(fly_or_t, double, char)
INSTANTIATE(fly_or_t, cfloat, char)
INSTANTIATE(fly_or_t, cdouble, char)
INSTANTIATE(fly_or_t, int, char)
INSTANTIATE(fly_or_t, uint, char)
INSTANTIATE(fly_or_t, intl, char)
INSTANTIATE(fly_or_t, uintl, char)
INSTANTIATE(fly_or_t, char, char)
INSTANTIATE(fly_or_t, uchar, char)
INSTANTIATE(fly_or_t, short, char)
INSTANTIATE(fly_or_t, ushort, char)
INSTANTIATE(fly_or_t, half, char)
}  // namespace oneapi
}  // namespace flare
