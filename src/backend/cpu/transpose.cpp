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
#include <kernel/transpose.hpp>
#include <transpose.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <platform.hpp>
#include <fly/dim4.hpp>

#include <cassert>
#include <utility>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cpu {

template<typename T>
Array<T> transpose(const Array<T> &in, const bool conjugate) {
    const dim4 &inDims = in.dims();
    const dim4 outDims = dim4(inDims[1], inDims[0], inDims[2], inDims[3]);
    // create an array with first two dimensions swapped
    Array<T> out = createEmptyArray<T>(outDims);

    getQueue().enqueue(kernel::transpose<T>, out, in, conjugate);

    return out;
}

template<typename T>
void transpose_inplace(Array<T> &in, const bool conjugate) {
    getQueue().enqueue(kernel::transpose_inplace<T>, in, conjugate);
}

#define INSTANTIATE(T)                                                     \
    template Array<T> transpose(const Array<T> &in, const bool conjugate); \
    template void transpose_inplace(Array<T> &in, const bool conjugate);

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

}  // namespace cpu
}  // namespace flare
