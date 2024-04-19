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
#include <kernel/resize.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <resize.hpp>

namespace flare {
namespace cpu {

template<typename T>
Array<T> resize(const Array<T> &in, const dim_t odim0, const dim_t odim1,
                const fly_interp_type method) {
    fly::dim4 idims = in.dims();
    fly::dim4 odims(odim0, odim1, idims[2], idims[3]);
    // Create output placeholder
    Array<T> out = createValueArray(odims, static_cast<T>(0));

    switch (method) {
        case FLY_INTERP_NEAREST:
            getQueue().enqueue(kernel::resize<T, FLY_INTERP_NEAREST>, out, in);
            break;
        case FLY_INTERP_BILINEAR:
            getQueue().enqueue(kernel::resize<T, FLY_INTERP_BILINEAR>, out, in);
            break;
        case FLY_INTERP_LOWER:
            getQueue().enqueue(kernel::resize<T, FLY_INTERP_LOWER>, out, in);
            break;
        default: break;
    }
    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> resize<T>(const Array<T> &in, const dim_t odim0, \
                                const dim_t odim1,                     \
                                const fly_interp_type method);

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

}  // namespace cpu
}  // namespace flare
