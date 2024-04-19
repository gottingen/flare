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

#include <unwrap.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <err_cuda.hpp>
#include <kernel/unwrap.hpp>

#include <stdexcept>

using flare::common::half;

namespace flare {
namespace cuda {

template<typename T>
Array<T> unwrap(const Array<T> &in, const dim_t wx, const dim_t wy,
                const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
                const dim_t dx, const dim_t dy, const bool is_column) {
    fly::dim4 idims = in.dims();

    dim_t nx = 1 + (idims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;
    dim_t ny = 1 + (idims[1] + 2 * py - (((wy - 1) * dy) + 1)) / sy;

    fly::dim4 odims(wx * wy, nx * ny, idims[2], idims[3]);

    if (!is_column) { std::swap(odims[0], odims[1]); }

    Array<T> outArray = createEmptyArray<T>(odims);
    kernel::unwrap<T>(outArray, in, wx, wy, sx, sy, px, py, dx, dy, nx,
                      is_column);

    return outArray;
}

#define INSTANTIATE(T)                                                      \
    template Array<T> unwrap<T>(                                            \
        const Array<T> &in, const dim_t wx, const dim_t wy, const dim_t sx, \
        const dim_t sy, const dim_t px, const dim_t py, const dim_t dx,     \
        const dim_t dy, const bool is_column);

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
#undef INSTANTIATE

}  // namespace cuda
}  // namespace flare
