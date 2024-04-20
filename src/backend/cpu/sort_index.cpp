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
#include <common/err_common.hpp>
#include <copy.hpp>
#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <range.hpp>
#include <reorder.hpp>
#include <sort_index.hpp>

#include <algorithm>
#include <numeric>

namespace flare {
namespace cpu {

template<typename T>
void sort_index(Array<T> &okey, Array<uint> &oval, const Array<T> &in,
                const uint dim, bool isAscending) {
    // okey is values, oval is indices
    okey = copyArray<T>(in);
    oval = range<uint>(in.dims(), dim);

    switch (dim) {
        case 0:
            getQueue().enqueue(kernel::sort0ByKey<T, uint>, okey, oval,
                               isAscending);
            break;
        case 1:
        case 2:
        case 3:
            getQueue().enqueue(kernel::sortByKeyBatched<T, uint>, okey, oval,
                               dim, isAscending);
            break;
        default: FLY_ERROR("Not Supported", FLY_ERR_NOT_SUPPORTED);
    }

    if (dim != 0) {
        fly::dim4 preorderDims = okey.dims();
        fly::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0]  = okey.dims()[dim];
        for (int i = 1; i <= static_cast<int>(dim); i++) {
            reorderDims[i - 1] = i;
            preorderDims[i]    = okey.dims()[i - 1];
        }

        okey.setDataDims(preorderDims);
        oval.setDataDims(preorderDims);

        okey = reorder<T>(okey, reorderDims);
        oval = reorder<uint>(oval, reorderDims);
    }
}

#define INSTANTIATE(T)                                              \
    template void sort_index<T>(Array<T> & val, Array<uint> & idx,  \
                                const Array<T> &in, const uint dim, \
                                bool isAscending);

INSTANTIATE(float)
INSTANTIATE(double)
// INSTANTIATE(cfloat)
// INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)

}  // namespace cpu
}  // namespace flare
