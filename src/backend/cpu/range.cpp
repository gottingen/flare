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
#include <kernel/range.hpp>
#include <range.hpp>

#include <Array.hpp>
#include <err_cpu.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

using flare::common::half;

namespace flare {
namespace cpu {

template<typename T>
Array<T> range(const dim4& dims, const int seq_dim) {
    // Set dimension along which the sequence should be
    // Other dimensions are simply tiled
    int _seq_dim = seq_dim;
    if (seq_dim < 0) {
        _seq_dim = 0;  // column wise sequence
    }

    Array<T> out = createEmptyArray<T>(dims);
    switch (_seq_dim) {
        case 0: getQueue().enqueue(kernel::range<T, 0>, out); break;
        case 1: getQueue().enqueue(kernel::range<T, 1>, out); break;
        case 2: getQueue().enqueue(kernel::range<T, 2>, out); break;
        case 3: getQueue().enqueue(kernel::range<T, 3>, out); break;
        default: FLY_ERROR("Invalid rep selection", FLY_ERR_ARG);
    }

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> range<T>(const fly::dim4& dims, const int seq_dims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace flare
