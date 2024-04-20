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
#include <err_cuda.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <reorder.hpp>
#include <sort.hpp>
#include <stdexcept>

namespace flare {
namespace cuda {
template<typename T>
Array<T> sort(const Array<T> &in, const unsigned dim, bool isAscending) {
    Array<T> out = copyArray<T>(in);
    switch (dim) {
        case 0: kernel::sort0<T>(out, isAscending); break;
        case 1: kernel::sortBatched<T>(out, 1, isAscending); break;
        case 2: kernel::sortBatched<T>(out, 2, isAscending); break;
        case 3: kernel::sortBatched<T>(out, 3, isAscending); break;
        default: FLY_ERROR("Not Supported", FLY_ERR_NOT_SUPPORTED);
    }

    if (dim != 0) {
        fly::dim4 preorderDims = out.dims();
        fly::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0]  = out.dims()[dim];
        for (int i = 1; i <= (int)dim; i++) {
            reorderDims[i - 1] = i;
            preorderDims[i]    = out.dims()[i - 1];
        }

        out.setDataDims(preorderDims);
        out = reorder<T>(out, reorderDims);
    }
    return out;
}

#define INSTANTIATE(T)                                                \
    template Array<T> sort<T>(const Array<T> &in, const unsigned dim, \
                              bool isAscending);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
}  // namespace cuda
}  // namespace flare
