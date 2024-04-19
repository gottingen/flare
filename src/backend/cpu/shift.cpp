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
#include <kernel/shift.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <shift.hpp>

namespace flare {
namespace cpu {

template<typename T>
Array<T> shift(const Array<T> &in, const int sdims[4]) {
    Array<T> out = createEmptyArray<T>(in.dims());
    const fly::dim4 temp(sdims[0], sdims[1], sdims[2], sdims[3]);

    getQueue().enqueue(kernel::shift<T>, out, in, temp);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> shift<T>(const Array<T> &in, const int sdims[4]);

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
