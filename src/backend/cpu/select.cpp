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
#include <kernel/select.hpp>
#include <select.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <platform.hpp>
#include <queue.hpp>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cpu {

template<typename T>
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b) {
    getQueue().enqueue(kernel::select<T>, out, cond, a, b);
}

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const T &b) {
    getQueue().enqueue(kernel::select_scalar<T, flip>, out, cond, a, b);
}

#define INSTANTIATE(T)                                                   \
    template void select<T>(Array<T> & out, const Array<char> &cond,     \
                            const Array<T> &a, const Array<T> &b);       \
    template void select_scalar<T, true>(Array<T> & out,                 \
                                         const Array<char> &cond,        \
                                         const Array<T> &a, const T &b); \
    template void select_scalar<T, false>(Array<T> & out,                \
                                          const Array<char> &cond,       \
                                          const Array<T> &a, const T &b);

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

}  // namespace cpu
}  // namespace flare
