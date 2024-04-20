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

#include <triangle.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/triangle.hpp>
#include <fly/dim4.hpp>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cuda {

template<typename T>
void triangle(Array<T> &out, const Array<T> &in, const bool is_upper,
              const bool is_unit_diag) {
    kernel::triangle<T>(out, in, is_upper, is_unit_diag);
}

template<typename T>
Array<T> triangle(const Array<T> &in, const bool is_upper,
                  const bool is_unit_diag) {
    Array<T> out = createEmptyArray<T>(in.dims());
    triangle<T>(out, in, is_upper, is_unit_diag);
    return out;
}

#define INSTANTIATE(T)                                                  \
    template void triangle<T>(Array<T> &, const Array<T> &, const bool, \
                              const bool);                              \
    template Array<T> triangle<T>(const Array<T> &, const bool, const bool);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cuda
}  // namespace flare
