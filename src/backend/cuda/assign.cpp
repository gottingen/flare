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
#include <common/half.hpp>
#include <err_cuda.hpp>
#include <handle.hpp>
#include <fly/dim4.hpp>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cuda {

template<typename T>
void assign(Array<T>& out, const fly_index_t idxrs[], const Array<T>& rhs) {
    AssignKernelParam p;
    std::vector<fly_seq> seqs(4, fly_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (dim_t x = 0; x < 4; ++x) {
        if (idxrs[x].isSeq) { seqs[x] = idxrs[x].idx.seq; }
    }

    // retrieve dimensions, strides and offsets
    dim4 dDims = out.dims();
    // retrieve dimensions & strides for array
    // to which rhs is being copied to
    dim4 dstOffs  = toOffset(seqs, dDims);
    dim4 dstStrds = toStride(seqs, dDims);

    for (dim_t i = 0; i < 4; ++i) {
        p.isSeq[i] = idxrs[i].isSeq;
        p.offs[i]  = dstOffs[i];
        p.strds[i] = dstStrds[i];
    }

    std::vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4()));
    // look through indexs to read fly_array indexs
    for (dim_t x = 0; x < 4; ++x) {
        // set idxPtrs to null
        p.ptr[x] = 0;
        // set index pointers were applicable
        if (!p.isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            p.ptr[x]   = idxArrs[x].get();
        }
    }

    kernel::assign<T>(out, rhs, p);
}

#define INSTANTIATE(T)                                                \
    template void assign<T>(Array<T> & out, const fly_index_t idxrs[], \
                            const Array<T>& rhs);

INSTANTIATE(cdouble)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(float)
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
