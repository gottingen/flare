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

#include <kernel/tile.hpp>
#include <tile.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <platform.hpp>

using flare::common::half;

namespace flare {
namespace cpu {

template<typename T>
Array<T> tile(const Array<T> &in, const fly::dim4 &tileDims) {
    const fly::dim4 &iDims = in.dims();
    fly::dim4 oDims        = iDims;
    oDims *= tileDims;

    if (iDims.elements() == 0 || oDims.elements() == 0) {
        throw std::runtime_error("Elements are 0");
    }

    Array<T> out = createEmptyArray<T>(oDims);

    getQueue().enqueue(kernel::tile<T>, out, in);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> tile<T>(const Array<T> &in, const fly::dim4 &tileDims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace flare
