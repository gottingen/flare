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
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <kernel/wrap.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <wrap.hpp>

using flare::common::half;

namespace flare {
namespace cpu {

template<typename T>
void wrap(Array<T> &out, const Array<T> &in, const dim_t wx, const dim_t wy,
          const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
          const bool is_column) {
    evalMultiple<T>(std::vector<Array<T> *>{const_cast<Array<T> *>(&in), &out});

    if (is_column) {
        getQueue().enqueue(kernel::wrap_dim<T, 1>, out, in, wx, wy, sx, sy, px,
                           py);
    } else {
        getQueue().enqueue(kernel::wrap_dim<T, 0>, out, in, wx, wy, sx, sy, px,
                           py);
    }
}

#define INSTANTIATE(T)                                                        \
    template void wrap<T>(Array<T> & out, const Array<T> &in, const dim_t wx, \
                          const dim_t wy, const dim_t sx, const dim_t sy,     \
                          const dim_t px, const dim_t py,                     \
                          const bool is_column);

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
#undef INSTANTIATE

template<typename T>
Array<T> wrap_dilated(const Array<T> &in, const dim_t ox, const dim_t oy,
                      const dim_t wx, const dim_t wy, const dim_t sx,
                      const dim_t sy, const dim_t px, const dim_t py,
                      const dim_t dx, const dim_t dy, const bool is_column) {
    fly::dim4 idims = in.dims();
    fly::dim4 odims(ox, oy, idims[2], idims[3]);

    Array<T> out = createValueArray<T>(odims, scalar<T>(0));
    out.eval();
    in.eval();

    getQueue().enqueue(kernel::wrap_dim_dilated<T>, out, in, wx, wy, sx, sy, px,
                       py, dx, dy, is_column);

    return out;
}

#define INSTANTIATE(T)                                                      \
    template Array<T> wrap_dilated<T>(                                      \
        const Array<T> &in, const dim_t ox, const dim_t oy, const dim_t wx, \
        const dim_t wy, const dim_t sx, const dim_t sy, const dim_t px,     \
        const dim_t py, const dim_t dx, const dim_t dy, const bool is_column);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(half)
#undef INSTANTIATE

}  // namespace cpu
}  // namespace flare
