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

#include <range.hpp>

#include <Array.hpp>
#include <err_cuda.hpp>
#include <kernel/range.hpp>
#include <math.hpp>

#include <stdexcept>

using flare::common::half;

namespace flare {
namespace cuda {
template<typename T>
Array<T> range(const dim4& dim, const int seq_dim) {
    // Set dimension along which the sequence should be
    // Other dimensions are simply tiled
    int _seq_dim = seq_dim;
    if (seq_dim < 0) {
        _seq_dim = 0;  // column wise sequence
    }

    if (_seq_dim < 0 || _seq_dim > 3) {
        FLY_ERROR("Invalid rep selection", FLY_ERR_ARG);
    }

    Array<T> out = createEmptyArray<T>(dim);
    kernel::range<T>(out, _seq_dim);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> range<T>(const fly::dim4& dims, const int seq_dim);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)
}  // namespace cuda
}  // namespace flare
