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
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <optypes.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>

#include <complex.hpp>

using fly::dim4;
using flare::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::conj;
using detail::imag;
using detail::real;

template<typename To, typename Ti>
static inline fly_array cplx(const fly_array lhs, const fly_array rhs,
                            const dim4 &odims) {
    fly_array res =
        getHandle(cplx<To, Ti>(castArray<Ti>(lhs), castArray<Ti>(rhs), odims));
    return res;
}

fly_err fly_cplx2(fly_array *out, const fly_array lhs, const fly_array rhs,
                bool batchMode) {
    try {
        fly_dtype type = implicit(lhs, rhs);

        if (type == c32 || type == c64) {
            FLY_ERROR("Inputs to cplx2 can not be of complex type", FLY_ERR_ARG);
        }

        if (type != f64) { type = f32; }
        dim4 odims =
            getOutDims(getInfo(lhs).dims(), getInfo(rhs).dims(), batchMode);
        if (odims.ndims() == 0) {
            return fly_create_handle(out, 0, nullptr, type);
        }

        fly_array res;
        switch (type) {
            case f32: res = cplx<cfloat, float>(lhs, rhs, odims); break;
            case f64: res = cplx<cdouble, double>(lhs, rhs, odims); break;
            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_cplx(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (type == c32 || type == c64) {
            FLY_ERROR("Inputs to cplx2 can not be of complex type", FLY_ERR_ARG);
        }
        if (info.ndims() == 0) { return fly_retain_array(out, in); }

        fly_array tmp;
        FLY_CHECK(fly_constant(&tmp, 0, info.ndims(), info.dims().get(), type));

        fly_array res;
        switch (type) {
            case f32: res = cplx<cfloat, float>(in, tmp, info.dims()); break;
            case f64: res = cplx<cdouble, double>(in, tmp, info.dims()); break;

            default: TYPE_ERROR(0, type);
        }

        FLY_CHECK(fly_release_array(tmp));

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_real(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (type != c32 && type != c64) { return fly_retain_array(out, in); }
        if (info.ndims() == 0) { return fly_retain_array(out, in); }

        fly_array res;
        switch (type) {
            case c32:
                res = getHandle(real<float, cfloat>(getArray<cfloat>(in)));
                break;
            case c64:
                res = getHandle(real<double, cdouble>(getArray<cdouble>(in)));
                break;

            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_imag(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (type != c32 && type != c64) {
            return fly_constant(out, 0, info.ndims(), info.dims().get(), type);
        }
        if (info.ndims() == 0) { return fly_retain_array(out, in); }

        fly_array res;
        switch (type) {
            case c32:
                res = getHandle(imag<float, cfloat>(getArray<cfloat>(in)));
                break;
            case c64:
                res = getHandle(imag<double, cdouble>(getArray<cdouble>(in)));
                break;

            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_conjg(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (type != c32 && type != c64) { return fly_retain_array(out, in); }
        if (info.ndims() == 0) { return fly_retain_array(out, in); }

        fly_array res;
        switch (type) {
            case c32:
                res = getHandle(conj<cfloat>(getArray<cfloat>(in)));
                break;
            case c64:
                res = getHandle(conj<cdouble>(getArray<cdouble>(in)));
                break;

            default: TYPE_ERROR(0, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_abs(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &in_info = getInfo(in);
        fly_dtype in_type         = in_info.getType();
        fly_array res;

        // Convert all inputs to floats / doubles
        fly_dtype type = implicit(in_type, f32);
        if (in_type == f16) { type = f16; }
        if (in_info.ndims() == 0) { return fly_retain_array(out, in); }

        switch (type) {
            // clang-format off
            case f32: res = getHandle(detail::abs<float, float>(castArray<float>(in))); break;
            case f64: res = getHandle(detail::abs<double, double>(castArray<double>(in))); break;
            case c32: res = getHandle(detail::abs<float, cfloat>(castArray<cfloat>(in))); break;
            case c64: res = getHandle(detail::abs<double, cdouble>(castArray<cdouble>(in))); break;
            case f16: res = getHandle(detail::abs<half, half>(getArray<half>(in))); break;
            // clang-format on
            default: TYPE_ERROR(1, in_type); break;
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
