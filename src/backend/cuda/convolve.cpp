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
#include <cast.hpp>
#include <common/half.hpp>
#include <convolve.hpp>
#include <err_cuda.hpp>
#include <kernel/convolve.hpp>
#include <platform.hpp>
#include <fly/dim4.hpp>
#include <type_traits>

using fly::dim4;
using flare::common::half;
using std::conditional;
using std::is_same;

namespace flare {
namespace cuda {

template<typename T, typename accT>
Array<T> convolve(Array<T> const &signal, Array<accT> const &filter,
                  FLY_BATCH_KIND kind, const int rank, const bool expand) {
    const dim4 &sDims = signal.dims();
    const dim4 &fDims = filter.dims();

    dim4 oDims(1);
    if (expand) {
        for (int d = 0; d < FLY_MAX_DIMS; ++d) {
            if (kind == FLY_BATCH_NONE || kind == FLY_BATCH_RHS) {
                oDims[d] = sDims[d] + fDims[d] - 1;
            } else {
                oDims[d] = (d < rank ? sDims[d] + fDims[d] - 1 : sDims[d]);
            }
        }
    } else {
        oDims = sDims;
        if (kind == FLY_BATCH_RHS) {
            for (int i = rank; i < FLY_MAX_DIMS; ++i) { oDims[i] = fDims[i]; }
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::convolve_nd<T, accT>(out, signal, filter, kind, rank, expand);

    return out;
}

template<typename T, typename accT>
Array<T> convolve2(Array<T> const &signal, Array<accT> const &c_filter,
                   Array<accT> const &r_filter, const bool expand) {
    const dim4 &cfDims = c_filter.dims();
    const dim4 &rfDims = r_filter.dims();

    const dim_t cfLen = cfDims.elements();
    const dim_t rfLen = rfDims.elements();

    const dim4 &sDims = signal.dims();
    dim4 tDims        = sDims;
    dim4 oDims        = sDims;

    if (expand) {
        tDims[0] += cfLen - 1;
        oDims[0] += cfLen - 1;
        oDims[1] += rfLen - 1;
    }

    Array<T> temp = createEmptyArray<T>(tDims);
    Array<T> out  = createEmptyArray<T>(oDims);

    kernel::convolve2<T, accT>(temp, signal, c_filter, 0, expand);
    kernel::convolve2<T, accT>(out, temp, r_filter, 1, expand);

    return out;
}

#define INSTANTIATE(T, accT)                                                   \
    template Array<T> convolve<T, accT>(Array<T> const &, Array<accT> const &, \
                                        FLY_BATCH_KIND, const int, const bool); \
    template Array<T> convolve2<T, accT>(Array<T> const &,                     \
                                         Array<accT> const &,                  \
                                         Array<accT> const &, const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)
#undef INSTANTIATE

}  // namespace cuda
}  // namespace flare
