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

#include <diff.hpp>

#include <Array.hpp>
#include <kernel/diff.hpp>
#include <platform.hpp>

#include <fly/dim4.hpp>

namespace flare {
namespace cpu {

template<typename T>
Array<T> diff1(const Array<T> &in, const int dim) {
    // Decrement dimension of select dimension
    fly::dim4 dims = in.dims();
    dims[dim]--;

    Array<T> outArray = createEmptyArray<T>(dims);

    getQueue().enqueue(kernel::diff1<T>, outArray, in, dim);

    return outArray;
}

template<typename T>
Array<T> diff2(const Array<T> &in, const int dim) {
    // Decrement dimension of select dimension
    fly::dim4 dims = in.dims();
    dims[dim] -= 2;

    Array<T> outArray = createEmptyArray<T>(dims);

    getQueue().enqueue(kernel::diff2<T>, outArray, in, dim);

    return outArray;
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
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
}  // namespace flare
