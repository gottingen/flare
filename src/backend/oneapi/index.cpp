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
#include <err_oneapi.hpp>
#include <handle.hpp>
#include <kernel/assign_kernel_param.hpp>
#include <kernel/index.hpp>
#include <memory.hpp>
#include <fly/dim4.hpp>

using flare::common::half;
using flare::oneapi::IndexKernelParam;

namespace flare {
namespace oneapi {

template<typename T>
Array<T> index(const Array<T>& in, const fly_index_t idxrs[]) {
    IndexKernelParam p;
    std::vector<fly_seq> seqs(4, fly_span);
    // create seq vector to retrieve output
    // dimensions, offsets & offsets
    for (dim_t x = 0; x < 4; ++x) {
        if (idxrs[x].isSeq) { seqs[x] = idxrs[x].idx.seq; }
    }

    // retrieve dimensions, strides and offsets
    const dim4& iDims = in.dims();
    dim4 dDims        = in.getDataDims();
    dim4 oDims        = toDims(seqs, iDims);
    dim4 iOffs        = toOffset(seqs, dDims);
    dim4 iStrds       = in.strides();

    for (dim_t i = 0; i < 4; ++i) {
        p.isSeq[i] = idxrs[i].isSeq;
        p.offs[i]  = iOffs[i];
        p.strds[i] = iStrds[i];
    }

    std::vector<Array<uint>> idxArrs(4, createEmptyArray<uint>(dim4(1)));
    // look through indexs to read fly_array indexs
    for (dim_t x = 0; x < 4; ++x) {
        if (!p.isSeq[x]) {
            idxArrs[x] = castArray<uint>(idxrs[x].idx.arr);
            oDims[x]   = idxArrs[x].elements();
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    if (oDims.elements() == 0) { return out; }
    kernel::index<T>(out, in, p, idxArrs);

    return out;
}

#define INSTANTIATE(T) \
    template Array<T> index<T>(const Array<T>& in, const fly_index_t idxrs[]);

INSTANTIATE(cdouble)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace oneapi
}  // namespace flare
