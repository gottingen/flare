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
#include <err_oneapi.hpp>
#include <index.hpp>
#include <sort.hpp>
#include <sort_index.hpp>
#include <types.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using flare::common::half;

using std::iota;
using std::min;
using std::partial_sort_copy;
using std::transform;
using std::vector;

namespace flare {
namespace oneapi {
vector<fly_index_t> indexForTopK(const int k) {
    fly_index_t idx;
    idx.idx.seq = fly_seq{0.0, static_cast<double>(k) - 1.0, 1.0};
    idx.isSeq   = true;
    idx.isBatch = false;

    fly_index_t sp;
    sp.idx.seq = fly_span;
    sp.isSeq   = true;
    sp.isBatch = false;

    return vector<fly_index_t>({idx, sp, sp, sp});
}

template<typename T>
void topk(Array<T>& vals, Array<unsigned>& idxs, const Array<T>& in,
          const int k, const int dim, const fly::topkFunction order) {
    auto values  = createEmptyArray<T>(in.dims());
    auto indices = createEmptyArray<unsigned>(in.dims());
    sort_index(values, indices, in, dim, order & FLY_TOPK_MIN);
    auto indVec = indexForTopK(k);
    vals        = index<T>(values, indVec.data());
    idxs        = index<unsigned>(indices, indVec.data());
}

#define INSTANTIATE(T)                                                  \
    template void topk<T>(Array<T>&, Array<unsigned>&, const Array<T>&, \
                          const int, const int, const fly::topkFunction);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(half)

}  // namespace oneapi
}  // namespace flare
