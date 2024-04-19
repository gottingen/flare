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
#include <diagonal.hpp>
#include <err_cuda.hpp>
#include <kernel/diagonal.hpp>
#include <math.hpp>
#include <fly/dim4.hpp>

using flare::common::half;

namespace flare {
namespace cuda {
template<typename T>
Array<T> diagCreate(const Array<T> &in, const int num) {
    int size     = in.dims()[0] + std::abs(num);
    int batch    = in.dims()[1];
    Array<T> out = createEmptyArray<T>(dim4(size, size, batch));

    kernel::diagCreate<T>(out, in, num);

    return out;
}

template<typename T>
Array<T> diagExtract(const Array<T> &in, const int num) {
    const dim_t *idims = in.dims().get();
    dim_t size         = std::min(idims[0], idims[1]) - std::abs(num);
    Array<T> out       = createEmptyArray<T>(dim4(size, 1, idims[2], idims[3]));

    kernel::diagExtract<T>(out, in, num);

    return out;
}

#define INSTANTIATE_DIAGONAL(T)                                          \
    template Array<T> diagExtract<T>(const Array<T> &in, const int num); \
    template Array<T> diagCreate<T>(const Array<T> &in, const int num);

INSTANTIATE_DIAGONAL(float)
INSTANTIATE_DIAGONAL(double)
INSTANTIATE_DIAGONAL(cfloat)
INSTANTIATE_DIAGONAL(cdouble)
INSTANTIATE_DIAGONAL(int)
INSTANTIATE_DIAGONAL(uint)
INSTANTIATE_DIAGONAL(intl)
INSTANTIATE_DIAGONAL(uintl)
INSTANTIATE_DIAGONAL(char)
INSTANTIATE_DIAGONAL(uchar)
INSTANTIATE_DIAGONAL(short)
INSTANTIATE_DIAGONAL(ushort)
INSTANTIATE_DIAGONAL(half)

}  // namespace cuda
}  // namespace flare
