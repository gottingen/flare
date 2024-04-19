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

#include <rotate.hpp>

#include <kernel/rotate.hpp>

namespace flare {
namespace opencl {
template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const fly::dim4 &odims,
                const fly_interp_type method) {
    Array<T> out = createEmptyArray<T>(odims);

    switch (method) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER:
            kernel::rotate<T>(out, in, theta, method, 1);
            break;
        case FLY_INTERP_BILINEAR:
        case FLY_INTERP_BILINEAR_COSINE:
            kernel::rotate<T>(out, in, theta, method, 2);
            break;
        case FLY_INTERP_BICUBIC:
        case FLY_INTERP_BICUBIC_SPLINE:
            kernel::rotate<T>(out, in, theta, method, 3);
            break;
        default: FLY_ERROR("Unsupported interpolation type", FLY_ERR_ARG);
    }
    return out;
}

#define INSTANTIATE(T)                                              \
    template Array<T> rotate(const Array<T> &in, const float theta, \
                             const fly::dim4 &odims,                 \
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
}  // namespace opencl
}  // namespace flare
