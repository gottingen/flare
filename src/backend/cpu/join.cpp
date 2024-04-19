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
#include <common/half.hpp>
#include <join.hpp>
#include <kernel/join.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <algorithm>

using flare::common::half;

namespace flare {
namespace cpu {

template<typename T>
Array<T> join(const int dim, const Array<T> &first, const Array<T> &second) {
    // All dimensions except join dimension must be equal
    // Compute output dims
    fly::dim4 odims;
    fly::dim4 fdims = first.dims();
    fly::dim4 sdims = second.dims();

    for (int i = 0; i < 4; i++) {
        if (i == dim) {
            odims[i] = fdims[i] + sdims[i];
        } else {
            odims[i] = fdims[i];
        }
    }

    Array<T> out = createEmptyArray<T>(odims);
    std::vector<CParam<T>> v{first, second};
    getQueue().enqueue(kernel::join<T>, dim, out, v, 2);

    return out;
}

template<typename T>
void join(Array<T> &out, const int dim, const std::vector<Array<T>> &inputs) {
    const dim_t n_arrays = inputs.size();

    std::vector<Array<T> *> input_ptrs(inputs.size());
    std::transform(
        begin(inputs), end(inputs), begin(input_ptrs),
        [](const Array<T> &input) { return const_cast<Array<T> *>(&input); });
    evalMultiple(input_ptrs);
    std::vector<CParam<T>> inputParams(inputs.begin(), inputs.end());

    getQueue().enqueue(kernel::join<T>, dim, out, inputParams, n_arrays);
}

#define INSTANTIATE(T)                                              \
    template Array<T> join<T>(const int dim, const Array<T> &first, \
                              const Array<T> &second);

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
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

#undef INSTANTIATE

#define INSTANTIATE(T)                                   \
    template void join<T>(Array<T> & out, const int dim, \
                          const std::vector<Array<T>> &inputs);

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
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

#undef INSTANTIATE
}  // namespace cpu
}  // namespace flare
