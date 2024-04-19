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

#include <index.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <kernel/index.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>

#include <utility>
#include <vector>

using fly::dim4;
using flare::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy
using std::vector;

namespace flare {
namespace cpu {

template<typename T>
Array<T> index(const Array<T>& in, const fly_index_t idxrs[]) {
    vector<bool> isSeq(4);
    vector<fly_seq> seqs(4, fly_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (unsigned x = 0; x < isSeq.size(); ++x) {
        if (idxrs[x].isSeq) { seqs[x] = idxrs[x].idx.seq; }
        isSeq[x] = idxrs[x].isSeq;
    }

    // retrieve
    dim4 oDims = toDims(seqs, in.dims());

    vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read fly_array indexs
    for (unsigned x = 0; x < isSeq.size(); ++x) {
        if (!isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            // set output array ith dimension value
            oDims[x] = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    vector<CParam<uint>> idxParams(idxArrs.begin(), idxArrs.end());

    getQueue().enqueue(kernel::index<T>, out, in, in.getDataDims(),
                       std::move(isSeq), std::move(seqs), std::move(idxParams));

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> index<T>(const Array<T>& in, const fly_index_t idxrs[]);

INSTANTIATE(cdouble)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(float)
INSTANTIATE(uintl)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(int)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace flare
