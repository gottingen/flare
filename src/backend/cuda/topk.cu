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
#include <kernel/topk.hpp>
#include <topk.hpp>
#include <fly/dim4.hpp>

using flare::common::half;

namespace flare {
namespace cuda {
template<typename T>
void topk(Array<T>& ovals, Array<uint>& oidxs, const Array<T>& ivals,
          const int k, const int dim, const fly::topkFunction order) {
    dim4 outDims = ivals.dims();
    outDims[dim] = k;

    ovals = createEmptyArray<T>(outDims);
    oidxs = createEmptyArray<uint>(outDims);

    kernel::topk<T>(ovals, oidxs, ivals, k, dim, order);
}

#define INSTANTIATE(T)                                                         \
    template void topk<T>(Array<T>&, Array<uint>&, const Array<T>&, const int, \
                          const int, const fly::topkFunction);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(half)
}  // namespace cuda
}  // namespace flare
