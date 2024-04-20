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
#include <identity.hpp>
#include <kernel/identity.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>

using flare::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy

namespace flare {
namespace cpu {

template<typename T>
Array<T> identity(const dim4& dims) {
    Array<T> out = createEmptyArray<T>(dims);

    getQueue().enqueue(kernel::identity<T>, out);

    return out;
}

#define INSTANTIATE_IDENTITY(T) \
    template Array<T> identity<T>(const fly::dim4& dims);

INSTANTIATE_IDENTITY(float)
INSTANTIATE_IDENTITY(double)
INSTANTIATE_IDENTITY(cfloat)
INSTANTIATE_IDENTITY(cdouble)
INSTANTIATE_IDENTITY(int)
INSTANTIATE_IDENTITY(uint)
INSTANTIATE_IDENTITY(intl)
INSTANTIATE_IDENTITY(uintl)
INSTANTIATE_IDENTITY(char)
INSTANTIATE_IDENTITY(uchar)
INSTANTIATE_IDENTITY(short)
INSTANTIATE_IDENTITY(ushort)
INSTANTIATE_IDENTITY(half)

}  // namespace cpu
}  // namespace flare
