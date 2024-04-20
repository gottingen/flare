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
#include <copy.hpp>
#include <iota.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <range.hpp>
#include <reorder.hpp>
#include <sort.hpp>
#include <sort_by_key.hpp>
#include <algorithm>
#include <functional>

namespace flare {
namespace cpu {

template<typename T, int dim>
void sortBatched(Array<T>& val, bool isAscending) {
    fly::dim4 inDims = val.dims();

    // Sort dimension
    fly::dim4 tileDims(1);
    fly::dim4 seqDims = inDims;
    tileDims[dim]    = inDims[dim];
    seqDims[dim]     = 1;

    Array<uint> key = iota<uint>(seqDims, tileDims);

    Array<uint> resKey = createEmptyArray<uint>(dim4());
    Array<T> resVal    = createEmptyArray<T>(dim4());

    val.setDataDims(inDims.elements());
    key.setDataDims(inDims.elements());

    sort_by_key<T, uint>(resVal, resKey, val, key, 0, isAscending);

    // Needs to be ascending (true) in order to maintain the indices properly
    sort_by_key<uint, T>(key, val, resKey, resVal, 0, true);
    val.setDataDims(inDims);  // This is correct only for dim0
}

template<typename T>
void sort0(Array<T>& val, bool isAscending) {
    int higherDims = val.elements() / val.dims()[0];
    // TODO Make a better heurisitic
    if (higherDims > 10) {
        sortBatched<T, 0>(val, isAscending);
    } else {
        getQueue().enqueue(kernel::sort0Iterative<T>, val, isAscending);
    }
}

template<typename T>
Array<T> sort(const Array<T>& in, const unsigned dim, bool isAscending) {
    Array<T> out = copyArray<T>(in);
    switch (dim) {
        case 0: sort0<T>(out, isAscending); break;
        case 1: sortBatched<T, 1>(out, isAscending); break;
        case 2: sortBatched<T, 2>(out, isAscending); break;
        case 3: sortBatched<T, 3>(out, isAscending); break;
        default: FLY_ERROR("Not Supported", FLY_ERR_NOT_SUPPORTED);
    }

    if (dim != 0) {
        fly::dim4 preorderDims = out.dims();
        fly::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0]  = out.dims()[dim];
        for (int i = 1; i <= static_cast<int>(dim); i++) {
            reorderDims[i - 1] = i;
            preorderDims[i]    = out.dims()[i - 1];
        }

        out.setDataDims(preorderDims);
        out = reorder<T>(out, reorderDims);
    }
    return out;
}

#define INSTANTIATE(T)                                                \
    template Array<T> sort<T>(const Array<T>& in, const unsigned dim, \
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
