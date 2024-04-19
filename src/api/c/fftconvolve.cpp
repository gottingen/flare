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

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/dispatch.hpp>
#include <common/err_common.hpp>
#include <complex.hpp>
#include <fft_common.hpp>
#include <handle.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/signal.h>

#include <algorithm>
#include <type_traits>
#include <vector>

using fly::dim4;
using flare::common::cast;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createSubArray;
using detail::fftconvolve;
using detail::intl;
using detail::real;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::conditional;
using std::is_integral;
using std::is_same;
using std::max;
using std::swap;
using std::vector;

template<typename T>
fly_array fftconvolve_fallback(const fly_array signal, const fly_array filter,
                              const bool expand, const int baseDim) {
    using convT = typename conditional<is_integral<T>::value ||
                                           is_same<T, float>::value ||
                                           is_same<T, cfloat>::value,
                                       float, double>::type;
    using cT    = typename conditional<is_same<convT, float>::value, cfloat,
                                    cdouble>::type;

    const Array<cT> S = castArray<cT>(signal);
    const Array<cT> F = castArray<cT>(filter);
    const dim4 &sdims = S.dims();
    const dim4 &fdims = F.dims();
    dim4 odims(1, 1, 1, 1);
    dim4 psdims(1, 1, 1, 1);
    dim4 pfdims(1, 1, 1, 1);

    vector<fly_seq> index(FLY_MAX_DIMS);

    int count = 1;
    for (int i = 0; i < baseDim; i++) {
        dim_t tdim_i = sdims[i] + fdims[i] - 1;

        // Pad temporary buffers to power of 2 for performance
        odims[i]  = nextpow2(tdim_i);
        psdims[i] = nextpow2(tdim_i);
        pfdims[i] = nextpow2(tdim_i);

        // The normalization factor
        count *= odims[i];

        // Get the indexing params for output
        if (expand) {
            index[i].begin = 0.;
            index[i].end   = static_cast<double>(tdim_i) - 1.;
        } else {
            index[i].begin = static_cast<double>(fdims[i]) / 2.0;
            index[i].end = static_cast<double>(index[i].begin + sdims[i]) - 1.;
        }
        index[i].step = 1.;
    }

    for (int i = baseDim; i < FLY_MAX_DIMS; i++) {
        odims[i]  = max(sdims[i], fdims[i]);
        psdims[i] = sdims[i];
        pfdims[i] = fdims[i];
        index[i]  = fly_span;
    }

    // fft(signal)
    Array<cT> T1 = fft<cT, cT>(S, 1.0, baseDim, psdims.get(), baseDim, true);

    // fft(filter)
    Array<cT> T2 = fft<cT, cT>(F, 1.0, baseDim, pfdims.get(), baseDim, true);

    // fft(signal) * fft(filter)
    T1 = arithOp<cT, fly_mul_t>(T1, T2, odims);

    // ifft(ffit(signal) * fft(filter))
    T1 = fft<cT, cT>(T1, 1.0 / static_cast<double>(count), baseDim, odims.get(),
                     baseDim, false);

    // Index to proper offsets
    T1 = createSubArray<cT>(T1, index);

    if (getInfo(signal).isComplex() || getInfo(filter).isComplex()) {
        return getHandle(cast<T>(T1));
    } else {
        return getHandle(cast<T>(real<convT>(T1)));
    }
}

template<typename T>
inline fly_array fftconvolve(const fly_array &s, const fly_array &f,
                            const bool expand, FLY_BATCH_KIND kind,
                            const int baseDim) {
    if (kind == FLY_BATCH_DIFF) {
        return fftconvolve_fallback<T>(s, f, expand, baseDim);
    } else {
        return getHandle(fftconvolve<T>(getArray<T>(s), castArray<T>(f), expand,
                                        kind, baseDim));
    }
}

FLY_BATCH_KIND identifyBatchKind(const dim4 &sDims, const dim4 &fDims,
                                const int baseDim) {
    dim_t sn = sDims.ndims();
    dim_t fn = fDims.ndims();

    if (sn == baseDim && fn == baseDim) { return FLY_BATCH_NONE; }
    if (sn == baseDim && (fn > baseDim && fn <= FLY_MAX_DIMS)) {
        return FLY_BATCH_RHS;
    }
    if ((sn > baseDim && sn <= FLY_MAX_DIMS) && fn == baseDim) {
        return FLY_BATCH_LHS;
    } else if ((sn > baseDim && sn <= FLY_MAX_DIMS) &&
               (fn > baseDim && fn <= FLY_MAX_DIMS)) {
        bool doesDimensionsMatch = true;
        bool isInterleaved       = true;
        for (dim_t i = baseDim; i < FLY_MAX_DIMS; i++) {
            doesDimensionsMatch &= (sDims[i] == fDims[i]);
            isInterleaved &=
                (sDims[i] == 1 || fDims[i] == 1 || sDims[i] == fDims[i]);
        }
        if (doesDimensionsMatch) { return FLY_BATCH_SAME; }
        return (isInterleaved ? FLY_BATCH_DIFF : FLY_BATCH_UNSUPPORTED);
    } else {
        return FLY_BATCH_UNSUPPORTED;
    }
}

fly_err fft_convolve(fly_array *out, const fly_array signal, const fly_array filter,
                    const bool expand, const int baseDim) {
    try {
        const ArrayInfo &sInfo = getInfo(signal);
        const ArrayInfo &fInfo = getInfo(filter);

        fly_dtype signalType = sInfo.getType();

        const dim4 &sdims = sInfo.dims();
        const dim4 &fdims = fInfo.dims();

        FLY_BATCH_KIND convBT = identifyBatchKind(sdims, fdims, baseDim);

        ARG_ASSERT(1, (convBT != FLY_BATCH_UNSUPPORTED));

        fly_array output;
        switch (signalType) {
            case f64:
                output = fftconvolve<double>(signal, filter, expand, convBT,
                                             baseDim);
                break;
            case f32:
                output =
                    fftconvolve<float>(signal, filter, expand, convBT, baseDim);
                break;
            case u32:
                output =
                    fftconvolve<uint>(signal, filter, expand, convBT, baseDim);
                break;
            case s32:
                output =
                    fftconvolve<int>(signal, filter, expand, convBT, baseDim);
                break;
            case u64:
                output =
                    fftconvolve<uintl>(signal, filter, expand, convBT, baseDim);
                break;
            case s64:
                output =
                    fftconvolve<intl>(signal, filter, expand, convBT, baseDim);
                break;
            case u16:
                output = fftconvolve<ushort>(signal, filter, expand, convBT,
                                             baseDim);
                break;
            case s16:
                output =
                    fftconvolve<short>(signal, filter, expand, convBT, baseDim);
                break;
            case u8:
                output =
                    fftconvolve<uchar>(signal, filter, expand, convBT, baseDim);
                break;
            case b8:
                output =
                    fftconvolve<char>(signal, filter, expand, convBT, baseDim);
                break;
            case c32:
                output = fftconvolve_fallback<cfloat>(signal, filter, expand,
                                                      baseDim);
                break;
            case c64:
                output = fftconvolve_fallback<cdouble>(signal, filter, expand,
                                                       baseDim);
                break;
            default: TYPE_ERROR(1, signalType);
        }
        swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_fft_convolve1(fly_array *out, const fly_array signal,
                        const fly_array filter, const fly_conv_mode mode) {
    return fft_convolve(out, signal, filter, mode == FLY_CONV_EXPAND, 1);
}

fly_err fly_fft_convolve2(fly_array *out, const fly_array signal,
                        const fly_array filter, const fly_conv_mode mode) {
    try {
        if (getInfo(signal).dims().ndims() < 2 &&
            getInfo(filter).dims().ndims() < 2) {
            return fft_convolve(out, signal, filter, mode == FLY_CONV_EXPAND, 1);
        }
        return fft_convolve(out, signal, filter, mode == FLY_CONV_EXPAND, 2);
    }
    CATCHALL;
}

fly_err fly_fft_convolve3(fly_array *out, const fly_array signal,
                        const fly_array filter, const fly_conv_mode mode) {
    try {
        if (getInfo(signal).dims().ndims() < 3 &&
            getInfo(filter).dims().ndims() < 3) {
            return fft_convolve(out, signal, filter, mode == FLY_CONV_EXPAND, 2);
        }
        return fft_convolve(out, signal, filter, mode == FLY_CONV_EXPAND, 3);
    }
    CATCHALL;
}
