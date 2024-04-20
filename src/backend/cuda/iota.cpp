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

#include <Array.hpp>
#include <common/half.hpp>
#include <err_cuda.hpp>
#include <iota.hpp>
#include <kernel/iota.hpp>
#include <math.hpp>
#include <stdexcept>

using flare::common::half;

namespace flare {
namespace cuda {
template<typename T>
Array<T> iota(const dim4 &dims, const dim4 &tile_dims) {
    dim4 outdims = dims * tile_dims;

    Array<T> out = createEmptyArray<T>(outdims);
    kernel::iota<T>(out, dims);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> iota<T>(const fly::dim4 &dims, const fly::dim4 &tile_dims);

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
