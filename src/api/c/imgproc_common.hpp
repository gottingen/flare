// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <copy.hpp>
#include <logic.hpp>
#include <reduce.hpp>
#include <scan.hpp>

#include <cmath>

namespace flare {
namespace common {

template<typename To, typename Ti = To>
detail::Array<To> integralImage(const detail::Array<Ti>& in) {
    auto input                       = common::cast<To, Ti>(in);
    detail::Array<To> horizontalScan = detail::scan<fly_add_t, To, To>(input, 0);
    return detail::scan<fly_add_t, To, To>(horizontalScan, 1);
}

template<typename T>
detail::Array<T> threshold(const detail::Array<T>& in, T min, T max) {
    const fly::dim4 inDims = in.dims();

    auto MN    = detail::createValueArray(inDims, min);
    auto MX    = detail::createValueArray(inDims, max);
    auto below = detail::logicOp<T, fly_le_t>(in, MX, inDims);
    auto above = detail::logicOp<T, fly_ge_t>(in, MN, inDims);
    auto valid = detail::logicOp<char, fly_and_t>(below, above, inDims);

    return detail::arithOp<T, fly_mul_t>(in, common::cast<T, char>(valid),
                                        inDims);
}

template<typename To, typename Ti>
detail::Array<To> convRange(const detail::Array<Ti>& in,
                            const To newLow = To(0), const To newHigh = To(1)) {
    auto dims  = in.dims();
    auto input = common::cast<To, Ti>(in);
    To high =
        detail::getScalar<To>(detail::reduce_all<fly_max_t, To, To>(input));
    To low = detail::getScalar<To>(detail::reduce_all<fly_min_t, To, To>(input));
    To range = high - low;

    if (std::abs(range) < 1.0e-6) {
        if (low == To(0) && newLow == To(0)) {
            return input;
        } else {
            // Input is constant, use high as constant in converted range
            return detail::createValueArray(dims, newHigh);
        }
    }

    auto minArray = detail::createValueArray(dims, low);
    auto invDen   = detail::createValueArray(dims, To(1.0 / range));
    auto numer    = detail::arithOp<To, fly_sub_t>(input, minArray, dims);
    auto result   = detail::arithOp<To, fly_mul_t>(numer, invDen, dims);

    if (newLow != To(0) || newHigh != To(1)) {
        To newRange    = newHigh - newLow;
        auto newRngArr = detail::createValueArray(dims, newRange);
        auto newMinArr = detail::createValueArray(dims, newLow);
        auto scaledArr = detail::arithOp<To, fly_mul_t>(result, newRngArr, dims);

        result = detail::arithOp<To, fly_add_t>(newMinArr, scaledArr, dims);
    }
    return result;
}

}  // namespace common
}  // namespace flare
