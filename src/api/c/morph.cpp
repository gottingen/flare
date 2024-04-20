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

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/indexing_helpers.hpp>
#include <copy.hpp>
#include <fftconvolve.hpp>
#include <handle.hpp>
#include <logic.hpp>
#include <math.hpp>
#include <morph.hpp>
#include <unary.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>

using fly::dim4;
using flare::common::cast;
using flare::common::flip;
using detail::arithOp;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::logicOp;
using detail::scalar;
using detail::uchar;
using detail::uint;
using detail::unaryOp;
using detail::ushort;

template<typename T>
fly_array morph(const fly_array &in, const fly_array &mask, bool isDilation) {
    const Array<T> &input  = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out           = morph<T>(input, filter, isDilation);
    return getHandle(out);
}

template<>
fly_array morph<char>(const fly_array &input, const fly_array &mask,
                     const bool isDilation) {
    using detail::fftconvolve;

#if defined(FLY_CPU)
#if defined(USE_MKL)
    constexpr unsigned fftMethodThreshold = 11;
#else
    constexpr unsigned fftMethodThreshold = 27;
#endif  // defined(USE_MKL)
#elif defined(FLY_CUDA)
    constexpr unsigned fftMethodThreshold = 17;
#endif  // defined(FLY_CPU)

    const Array<float> se = castArray<float>(mask);
    const dim4 &seDims    = se.dims();

    if (seDims[0] <= fftMethodThreshold) {
        auto out =
            morph(getArray<char>(input), castArray<char>(mask), isDilation);
        return getHandle(out);
    }

    DIM_ASSERT(2, (seDims[0] == seDims[1]));

    const Array<char> in = getArray<char>(input);
    const dim4 &inDims   = in.dims();
    const auto paddedSe =
        padArrayBorders(se,
                        {static_cast<dim_t>(seDims[0] % 2 == 0),
                         static_cast<dim_t>(seDims[1] % 2 == 0), 0, 0},
                        {0, 0, 0, 0}, FLY_PAD_ZERO);
    if (isDilation) {
        Array<float> dft =
            fftconvolve(cast<float>(in), paddedSe, false, FLY_BATCH_LHS, 2);

        return getHandle(cast<char>(unaryOp<float, fly_round_t>(dft)));
    } else {
        const Array<char> ONES   = createValueArray(inDims, scalar<char>(1));
        const Array<float> ZEROS = createValueArray(inDims, scalar<float>(0));
        const Array<char> inv    = arithOp<char, fly_sub_t>(ONES, in, inDims);

        Array<float> dft =
            fftconvolve(cast<float>(inv), paddedSe, false, FLY_BATCH_LHS, 2);

        Array<float> rounded = unaryOp<float, fly_round_t>(dft);
        Array<char> thrshd   = logicOp<float, fly_gt_t>(rounded, ZEROS, inDims);
        Array<char> inverted = arithOp<char, fly_sub_t>(ONES, thrshd, inDims);

        return getHandle(inverted);
    }
}

template<typename T>
static inline fly_array morph3d(const fly_array &in, const fly_array &mask,
                               bool isDilation) {
    const Array<T> &input  = getArray<T>(in);
    const Array<T> &filter = castArray<T>(mask);
    Array<T> out           = morph3d<T>(input, filter, isDilation);
    return getHandle(out);
}

fly_err morph(fly_array *out, const fly_array &in, const fly_array &mask,
             bool isDilation) {
    try {
        const ArrayInfo &info  = getInfo(in);
        const ArrayInfo &mInfo = getInfo(mask);
        fly::dim4 dims          = info.dims();
        fly::dim4 mdims         = mInfo.dims();
        dim_t in_ndims         = dims.ndims();
        dim_t mask_ndims       = mdims.ndims();

        DIM_ASSERT(1, (in_ndims >= 2));
        DIM_ASSERT(2, (mask_ndims == 2));

        fly_array output;
        fly_dtype type = info.getType();
        switch (type) {
            case f32: output = morph<float>(in, mask, isDilation); break;
            case f64: output = morph<double>(in, mask, isDilation); break;
            case b8: output = morph<char>(in, mask, isDilation); break;
            case s32: output = morph<int>(in, mask, isDilation); break;
            case u32: output = morph<uint>(in, mask, isDilation); break;
            case s16: output = morph<short>(in, mask, isDilation); break;
            case u16: output = morph<ushort>(in, mask, isDilation); break;
            case u8: output = morph<uchar>(in, mask, isDilation); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err morph3d(fly_array *out, const fly_array &in, const fly_array &mask,
               bool isDilation) {
    try {
        const ArrayInfo &info  = getInfo(in);
        const ArrayInfo &mInfo = getInfo(mask);
        fly::dim4 dims          = info.dims();
        fly::dim4 mdims         = mInfo.dims();
        dim_t in_ndims         = dims.ndims();
        dim_t mask_ndims       = mdims.ndims();

        DIM_ASSERT(1, (in_ndims >= 3));
        DIM_ASSERT(2, (mask_ndims == 3));

        fly_array output;
        fly_dtype type = info.getType();
        switch (type) {
            case f32: output = morph3d<float>(in, mask, isDilation); break;
            case f64: output = morph3d<double>(in, mask, isDilation); break;
            case b8: output = morph3d<char>(in, mask, isDilation); break;
            case s32: output = morph3d<int>(in, mask, isDilation); break;
            case u32: output = morph3d<uint>(in, mask, isDilation); break;
            case s16: output = morph3d<short>(in, mask, isDilation); break;
            case u16: output = morph3d<ushort>(in, mask, isDilation); break;
            case u8: output = morph3d<uchar>(in, mask, isDilation); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_dilate(fly_array *out, const fly_array in, const fly_array mask) {
    return morph(out, in, mask, true);
}

fly_err fly_erode(fly_array *out, const fly_array in, const fly_array mask) {
    return morph(out, in, mask, false);
}

fly_err fly_dilate3(fly_array *out, const fly_array in, const fly_array mask) {
    return morph3d(out, in, mask, true);
}

fly_err fly_erode3(fly_array *out, const fly_array in, const fly_array mask) {
    return morph3d(out, in, mask, false);
}
