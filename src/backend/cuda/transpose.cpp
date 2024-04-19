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
#include <kernel/transpose.hpp>
#include <transpose.hpp>
#include <fly/dim4.hpp>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cuda {

template<typename T>
Array<T> transpose(const Array<T> &in, const bool conjugate) {
    const dim4 &inDims = in.dims();

    dim4 outDims = dim4(inDims[1], inDims[0], inDims[2], inDims[3]);

    Array<T> out = createEmptyArray<T>(outDims);

    const bool is32multiple =
        inDims[0] % kernel::TILE_DIM == 0 && inDims[1] % kernel::TILE_DIM == 0;

    kernel::transpose<T>(out, in, conjugate, is32multiple);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> transpose(const Array<T> &in, const bool conjugate);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cuda
}  // namespace flare
