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
#include <utility.hpp>

namespace flare {
namespace cuda {

template<typename T>
Array<T> rotate(const Array<T> &in, const float theta, const fly::dim4 &odims,
                const fly_interp_type method) {
    Array<T> out = createEmptyArray<T>(odims);
    kernel::rotate<T>(out, in, theta, method, interpOrder(method));
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
}  // namespace cuda
}  // namespace flare