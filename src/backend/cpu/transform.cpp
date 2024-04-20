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
#include <kernel/transform.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <transform.hpp>

namespace flare {
namespace cpu {

template<typename T>
void transform(Array<T> &out, const Array<T> &in, const Array<float> &tf,
               const fly_interp_type method, const bool inverse,
               const bool perspective) {
    out.eval();
    in.eval();
    tf.eval();

    switch (method) {
        case FLY_INTERP_NEAREST:
        case FLY_INTERP_LOWER:
            getQueue().enqueue(kernel::transform<T, 1>, out, in, tf, inverse,
                               perspective, method);
            break;
        case FLY_INTERP_BILINEAR:
        case FLY_INTERP_BILINEAR_COSINE:
            getQueue().enqueue(kernel::transform<T, 2>, out, in, tf, inverse,
                               perspective, method);
            break;
        case FLY_INTERP_BICUBIC:
        case FLY_INTERP_BICUBIC_SPLINE:
            getQueue().enqueue(kernel::transform<T, 3>, out, in, tf, inverse,
                               perspective, method);
            break;
        default: FLY_ERROR("Unsupported interpolation type", FLY_ERR_ARG); break;
    }
}

#define INSTANTIATE(T)                                                       \
    template void transform(Array<T> &out, const Array<T> &in,               \
                            const Array<float> &tf,                          \
                            const fly_interp_type method, const bool inverse, \
                            const bool perspective);

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
