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

#include <fftconvolve.hpp>

#include <Array.hpp>
#include <fft.hpp>
#include <kernel/fftconvolve.hpp>
#include <fly/dim4.hpp>

#include <type_traits>

using fly::dim4;
using std::conditional;
using std::is_integral;
using std::is_same;

namespace flare {
namespace cuda {

template<typename T>
dim4 calcPackedSize(Array<T> const& i1, Array<T> const& i2, const int rank) {
    const dim4& i1d = i1.dims();
    const dim4& i2d = i2.dims();

    dim_t pd[FLY_MAX_DIMS] = {1, 1, 1, 1};

    dim_t max_d0 = (i1d[0] > i2d[0]) ? i1d[0] : i2d[0];
    dim_t min_d0 = (i1d[0] < i2d[0]) ? i1d[0] : i2d[0];
    pd[0]        = nextpow2(static_cast<unsigned>(
        static_cast<int>(ceil(max_d0 / 2.f)) + min_d0 - 1));

    for (int k = 1; k < FLY_MAX_DIMS; k++) {
        if (k < rank) {
            pd[k] = nextpow2(static_cast<unsigned>(i1d[k] + i2d[k] - 1));
        } else {
            pd[k] = i1d[k];
        }
    }

    return dim4(pd[0], pd[1], pd[2], pd[3]);
}

template<typename T>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, FLY_BATCH_KIND kind, const int rank) {
    using convT = typename conditional<is_integral<T>::value ||
                                           is_same<T, float>::value ||
                                           is_same<T, cfloat>::value,
                                       float, double>::type;
    using cT    = typename conditional<is_same<convT, float>::value, cfloat,
                                    cdouble>::type;

    const dim4& sDims = signal.dims();
    const dim4& fDims = filter.dims();

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

    const dim4 spDims       = calcPackedSize<T>(signal, filter, rank);
    const dim4 fpDims       = calcPackedSize<T>(filter, signal, rank);
    Array<cT> signal_packed = createEmptyArray<cT>(spDims);
    Array<cT> filter_packed = createEmptyArray<cT>(fpDims);

    kernel::packDataHelper<cT, T>(signal_packed, filter_packed, signal, filter);

    fft_inplace<cT>(signal_packed, rank, true);
    fft_inplace<cT>(filter_packed, rank, true);

    Array<T> out = createEmptyArray<T>(oDims);

    kernel::complexMultiplyHelper<T, cT>(signal_packed, filter_packed, kind);

    if (kind == FLY_BATCH_RHS) {
        fft_inplace<cT>(filter_packed, rank, false);
        kernel::reorderOutputHelper<T, cT>(out, filter_packed, signal, filter,
                                           expand, rank);
    } else {
        fft_inplace<cT>(signal_packed, rank, false);
        kernel::reorderOutputHelper<T, cT>(out, signal_packed, signal, filter,
                                           expand, rank);
    }

    return out;
}

#define INSTANTIATE(T)                                                 \
    template Array<T> fftconvolve<T>(Array<T> const&, Array<T> const&, \
                                     const bool, FLY_BATCH_KIND, const int);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(int)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(uintl)
INSTANTIATE(intl)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cuda
}  // namespace flare
