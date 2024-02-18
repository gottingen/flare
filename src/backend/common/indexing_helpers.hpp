/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>

#include <array>

namespace flare {
namespace common {

// will generate indexes to flip input array
// of size original dims according to axes specified in flip
template<typename T>
static detail::Array<T> flip(const detail::Array<T>& in,
                             const std::array<bool, FLY_MAX_DIMS> flip) {
    std::vector<fly_seq> index(4, fly_span);
    const fly::dim4& dims = in.dims();

    for (int i = 0; i < FLY_MAX_DIMS; ++i) {
        if (flip[i]) {
            index[i] = {static_cast<double>(dims[i] - 1), 0.0, -1.0};
        }
    }
    return createSubArray(in, index);
}

}  // namespace common
}  // namespace flare
