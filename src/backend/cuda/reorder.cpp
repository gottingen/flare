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

#include <reorder.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_cuda.hpp>
#include <kernel/reorder.hpp>

#include <stdexcept>

using flare::common::half;

namespace flare {
namespace cuda {

template<typename T>
Array<T> reorder(const Array<T> &in, const fly::dim4 &rdims) {
    const fly::dim4 &iDims = in.dims();
    fly::dim4 oDims(0);
    for (int i = 0; i < 4; i++) { oDims[i] = iDims[rdims[i]]; }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::reorder<T>(out, in, rdims.get());

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> reorder<T>(const Array<T> &in, const fly::dim4 &rdims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cuda
}  // namespace flare
