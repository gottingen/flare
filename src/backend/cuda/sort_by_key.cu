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
#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <reorder.hpp>
#include <sort_by_key.hpp>
#include <stdexcept>

namespace flare {
namespace cuda {
template<typename Tk, typename Tv>
void sort_by_key(Array<Tk> &okey, Array<Tv> &oval, const Array<Tk> &ikey,
                 const Array<Tv> &ival, const uint dim, bool isAscending) {
    okey = copyArray<Tk>(ikey);
    oval = copyArray<Tv>(ival);

    switch (dim) {
        case 0: kernel::sort0ByKey<Tk, Tv>(okey, oval, isAscending); break;
        case 1:
        case 2:
        case 3:
            kernel::sortByKeyBatched<Tk, Tv>(okey, oval, dim, isAscending);
            break;
        default: FLY_ERROR("Not Supported", FLY_ERR_NOT_SUPPORTED);
    }

    if (dim != 0) {
        fly::dim4 preorderDims = okey.dims();
        fly::dim4 reorderDims(0, 1, 2, 3);
        reorderDims[dim] = 0;
        preorderDims[0]  = okey.dims()[dim];
        for (int i = 1; i <= (int)dim; i++) {
            reorderDims[i - 1] = i;
            preorderDims[i]    = okey.dims()[i - 1];
        }

        okey.setDataDims(preorderDims);
        oval.setDataDims(preorderDims);

        okey = reorder<Tk>(okey, reorderDims);
        oval = reorder<Tv>(oval, reorderDims);
    }
}

#define INSTANTIATE(Tk, Tv)                                        \
    template void sort_by_key<Tk, Tv>(                             \
        Array<Tk> & okey, Array<Tv> & oval, const Array<Tk> &ikey, \
        const Array<Tv> &ival, const uint dim, bool);

#define INSTANTIATE1(Tk)     \
    INSTANTIATE(Tk, float)   \
    INSTANTIATE(Tk, double)  \
    INSTANTIATE(Tk, cfloat)  \
    INSTANTIATE(Tk, cdouble) \
    INSTANTIATE(Tk, int)     \
    INSTANTIATE(Tk, uint)    \
    INSTANTIATE(Tk, short)   \
    INSTANTIATE(Tk, ushort)  \
    INSTANTIATE(Tk, char)    \
    INSTANTIATE(Tk, uchar)   \
    INSTANTIATE(Tk, intl)    \
    INSTANTIATE(Tk, uintl)

INSTANTIATE1(float)
INSTANTIATE1(double)
INSTANTIATE1(int)
INSTANTIATE1(uint)
INSTANTIATE1(short)
INSTANTIATE1(ushort)
INSTANTIATE1(char)
INSTANTIATE1(uchar)
INSTANTIATE1(intl)
INSTANTIATE1(uintl)

}  // namespace cuda
}  // namespace flare
