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

#include <assign.hpp>
#include <kernel/assign.hpp>

#include <Array.hpp>
#include <Param.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <platform.hpp>
#include <types.hpp>

#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/index.h>
#include <fly/seq.h>

#include <utility>
#include <vector>

using fly::dim4;
using std::vector;

namespace flare {
namespace cpu {
template<typename T>
void assign(Array<T>& out, const fly_index_t idxrs[], const Array<T>& rhs) {
    vector<bool> isSeq(4);
    vector<fly_seq> seqs(4, fly_span);
    // create seq vector to retrieve output dimensions, offsets & offsets
    for (dim_t x = 0; x < 4; ++x) {
        if (idxrs[x].isSeq) { seqs[x] = idxrs[x].idx.seq; }
        isSeq[x] = idxrs[x].isSeq;
    }

    vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read fly_array indexs
    for (dim_t x = 0; x < 4; ++x) {
        if (!isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            idxArrs[x].eval();
        }
    }

    vector<CParam<uint>> idxParams(idxArrs.begin(), idxArrs.end());
    getQueue().enqueue(kernel::assign<T>, out, out.getDataDims(), rhs,
                       move(isSeq), move(seqs), move(idxParams));
}

#define INSTANTIATE(T)                                                \
    template void assign<T>(Array<T> & out, const fly_index_t idxrs[], \
                            const Array<T>& rhs);

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
INSTANTIATE(flare::common::half)

}  // namespace cpu
}  // namespace flare
