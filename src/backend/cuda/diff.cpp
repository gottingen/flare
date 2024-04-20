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
#include <diff.hpp>
#include <err_cuda.hpp>
#include <kernel/diff.hpp>
#include <stdexcept>

namespace flare {
namespace cuda {

template<typename T>
Array<T> diff(const Array<T> &in, const int dim, const bool isDiff2) {
    const fly::dim4 &iDims = in.dims();
    fly::dim4 oDims        = iDims;
    oDims[dim] -= (isDiff2 + 1);

    if (iDims.elements() == 0 || oDims.elements() == 0) {
        FLY_ERROR("Elements are 0", FLY_ERR_SIZE);
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::diff<T>(out, in, in.ndims(), dim, isDiff2);

    return out;
}

template<typename T>
Array<T> diff1(const Array<T> &in, const int dim) {
    return diff<T>(in, dim, false);
}

template<typename T>
Array<T> diff2(const Array<T> &in, const int dim) {
    return diff<T>(in, dim, true);
}

#define INSTANTIATE(T)                                             \
    template Array<T> diff1<T>(const Array<T> &in, const int dim); \
    template Array<T> diff2<T>(const Array<T> &in, const int dim);

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

}  // namespace cuda
}  // namespace flare
