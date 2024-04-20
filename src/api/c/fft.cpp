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

#include <backend.hpp>
#include <common/err_common.hpp>
#include <fft_common.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/signal.h>

#include <type_traits>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::multiply_inplace;
using std::conditional;
using std::is_same;

void computePaddedDims(dim4 &pdims, const dim4 &idims, const dim_t npad,
                       dim_t const *const pad) {
    for (int i = 0; i < 4; i++) {
        pdims[i] = (i < static_cast<int>(npad)) ? pad[i] : idims[i];
    }
}

template<typename InType>
fly_array fft(const fly_array in, const double norm_factor, const dim_t npad,
             const dim_t *const pad, int rank, bool direction) {
    using OutType = typename conditional<is_same<InType, double>::value ||
                                             is_same<InType, cdouble>::value,
                                         cdouble, cfloat>::type;
    return getHandle(fft<InType, OutType>(getArray<InType>(in), norm_factor,
                                          npad, pad, rank, direction));
}

fly_err fft(fly_array *out, const fly_array in, const double norm_factor,
           const dim_t npad, const dim_t *const pad, const int rank,
           const bool direction) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();
        const dim4 &dims      = info.dims();

        if (dims.ndims() == 0) { return fly_retain_array(out, in); }

        DIM_ASSERT(1, (dims.ndims() >= rank));

        fly_array output;
        switch (type) {
            case c32:
                output =
                    fft<cfloat>(in, norm_factor, npad, pad, rank, direction);
                break;
            case c64:
                output =
                    fft<cdouble>(in, norm_factor, npad, pad, rank, direction);
                break;
            case f32:
                output =
                    fft<float>(in, norm_factor, npad, pad, rank, direction);
                break;
            case f64:
                output =
                    fft<double>(in, norm_factor, npad, pad, rank, direction);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_fft(fly_array *out, const fly_array in, const double norm_factor,
              const dim_t pad0) {
    const dim_t pad[1] = {pad0};
    return fft(out, in, norm_factor, (pad0 > 0 ? 1 : 0), pad, 1, true);
}

fly_err fly_fft2(fly_array *out, const fly_array in, const double norm_factor,
               const dim_t pad0, const dim_t pad1) {
    const dim_t pad[2] = {pad0, pad1};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 ? 2 : 0), pad, 2,
               true);
}

fly_err fly_fft3(fly_array *out, const fly_array in, const double norm_factor,
               const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 && pad2 > 0 ? 3 : 0),
               pad, 3, true);
}

fly_err fly_ifft(fly_array *out, const fly_array in, const double norm_factor,
               const dim_t pad0) {
    const dim_t pad[1] = {pad0};
    return fft(out, in, norm_factor, (pad0 > 0 ? 1 : 0), pad, 1, false);
}

fly_err fly_ifft2(fly_array *out, const fly_array in, const double norm_factor,
                const dim_t pad0, const dim_t pad1) {
    const dim_t pad[2] = {pad0, pad1};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 ? 2 : 0), pad, 2,
               false);
}

fly_err fly_ifft3(fly_array *out, const fly_array in, const double norm_factor,
                const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 && pad2 > 0 ? 3 : 0),
               pad, 3, false);
}

template<typename T>
void fft_inplace(fly_array in, const double norm_factor, int rank,
                 bool direction) {
    Array<T> &input = getArray<T>(in);
    fft_inplace<T>(input, rank, direction);
    if (norm_factor != 1) { multiply_inplace<T>(input, norm_factor); }
}

fly_err fft_inplace(fly_array in, const double norm_factor, int rank,
                   bool direction) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 dims         = info.dims();

        if (dims.ndims() == 0) { return FLY_SUCCESS; }
        DIM_ASSERT(1, (dims.ndims() >= rank));

        switch (type) {
            case c32:
                fft_inplace<cfloat>(in, norm_factor, rank, direction);
                break;
            case c64:
                fft_inplace<cdouble>(in, norm_factor, rank, direction);
                break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_fft_inplace(fly_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 1, true);
}

fly_err fly_fft2_inplace(fly_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 2, true);
}

fly_err fly_fft3_inplace(fly_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 3, true);
}

fly_err fly_ifft_inplace(fly_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 1, false);
}

fly_err fly_ifft2_inplace(fly_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 2, false);
}

fly_err fly_ifft3_inplace(fly_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 3, false);
}

template<typename InType>
fly_array fft_r2c(const fly_array in, const double norm_factor, const dim_t npad,
                 const dim_t *const pad, const int rank) {
    using OutType = typename conditional<is_same<InType, double>::value,
                                         cdouble, cfloat>::type;
    return getHandle(fft_r2c<InType, OutType>(getArray<InType>(in), norm_factor,
                                              npad, pad, rank));
}

fly_err fft_r2c(fly_array *out, const fly_array in, const double norm_factor,
               const dim_t npad, const dim_t *const pad, const int rank) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 dims         = info.dims();

        if (dims.ndims() == 0) { return fly_retain_array(out, in); }
        DIM_ASSERT(1, (dims.ndims() >= rank));

        fly_array output;
        switch (type) {
            case f32:
                output = fft_r2c<float>(in, norm_factor, npad, pad, rank);
                break;
            case f64:
                output = fft_r2c<double>(in, norm_factor, npad, pad, rank);
                break;
            default: {
                TYPE_ERROR(1, type);
            }
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_fft_r2c(fly_array *out, const fly_array in, const double norm_factor,
                  const dim_t pad0) {
    const dim_t pad[1] = {pad0};
    return fft_r2c(out, in, norm_factor, (pad0 > 0 ? 1 : 0), pad, 1);
}

fly_err fly_fft2_r2c(fly_array *out, const fly_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1) {
    const dim_t pad[2] = {pad0, pad1};
    return fft_r2c(out, in, norm_factor, (pad0 > 0 && pad1 > 0 ? 2 : 0), pad,
                   2);
}

fly_err fly_fft3_r2c(fly_array *out, const fly_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft_r2c(out, in, norm_factor,
                   (pad0 > 0 && pad1 > 0 && pad2 > 0 ? 3 : 0), pad, 3);
}

template<typename InType>
static fly_array fft_c2r(const fly_array in, const double norm_factor,
                        const dim4 &odims, const int rank) {
    using OutType = typename conditional<is_same<InType, cdouble>::value,
                                         double, float>::type;
    return getHandle(fft_c2r<InType, OutType>(getArray<InType>(in), norm_factor,
                                              odims, rank));
}

fly_err fft_c2r(fly_array *out, const fly_array in, const double norm_factor,
               const bool is_odd, const int rank) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 idims        = info.dims();

        if (idims.ndims() == 0) { return fly_retain_array(out, in); }
        DIM_ASSERT(1, (idims.ndims() >= rank));

        dim4 odims = idims;
        odims[0]   = 2 * (odims[0] - 1) + (is_odd ? 1 : 0);

        fly_array output;
        switch (type) {
            case c32:
                output = fft_c2r<cfloat>(in, norm_factor, odims, rank);
                break;
            case c64:
                output = fft_c2r<cdouble>(in, norm_factor, odims, rank);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_fft_c2r(fly_array *out, const fly_array in, const double norm_factor,
                  const bool is_odd) {
    return fft_c2r(out, in, norm_factor, is_odd, 1);
}

fly_err fly_fft2_c2r(fly_array *out, const fly_array in, const double norm_factor,
                   const bool is_odd) {
    return fft_c2r(out, in, norm_factor, is_odd, 2);
}

fly_err fly_fft3_c2r(fly_array *out, const fly_array in, const double norm_factor,
                   const bool is_odd) {
    return fft_c2r(out, in, norm_factor, is_odd, 3);
}

fly_err fly_set_fft_plan_cache_size(size_t cache_size) {
    try {
        detail::setFFTPlanCacheSize(cache_size);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
