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
#include <select.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>

using fly::dim4;
using flare::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createSelectNode;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
fly_array select(const fly_array cond, const fly_array a, const fly_array b,
                const dim4& odims) {
    Array<T> out = createSelectNode(getArray<char>(cond), getArray<T>(a),
                                    getArray<T>(b), odims);
    return getHandle<T>(out);
}

fly_err fly_select(fly_array* out, const fly_array cond, const fly_array a,
                 const fly_array b) {
    try {
        const ArrayInfo& ainfo     = getInfo(a);
        const ArrayInfo& binfo     = getInfo(b);
        const ArrayInfo& cond_info = getInfo(cond);

        if (cond_info.ndims() == 0) { return fly_retain_array(out, cond); }

        ARG_ASSERT(2, ainfo.getType() == binfo.getType());
        ARG_ASSERT(1, cond_info.getType() == b8);

        dim4 adims     = ainfo.dims();
        dim4 bdims     = binfo.dims();
        dim4 cond_dims = cond_info.dims();
        dim4 odims(1, 1, 1, 1);

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(2, (adims[i] == bdims[i] && adims[i] == cond_dims[i]) ||
                              adims[i] == 1 || bdims[i] == 1 ||
                              cond_dims[i] == 1);
            odims[i] = std::max(std::max(adims[i], bdims[i]), cond_dims[i]);
        }

        fly_array res;

        switch (ainfo.getType()) {
            case f32: res = select<float>(cond, a, b, odims); break;
            case f64: res = select<double>(cond, a, b, odims); break;
            case c32: res = select<cfloat>(cond, a, b, odims); break;
            case c64: res = select<cdouble>(cond, a, b, odims); break;
            case s32: res = select<int>(cond, a, b, odims); break;
            case u32: res = select<uint>(cond, a, b, odims); break;
            case s64: res = select<intl>(cond, a, b, odims); break;
            case u64: res = select<uintl>(cond, a, b, odims); break;
            case s16: res = select<short>(cond, a, b, odims); break;
            case u16: res = select<ushort>(cond, a, b, odims); break;
            case u8: res = select<uchar>(cond, a, b, odims); break;
            case b8: res = select<char>(cond, a, b, odims); break;
            case f16: res = select<half>(cond, a, b, odims); break;
            default: TYPE_ERROR(2, ainfo.getType());
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<typename ArrayType, typename ScalarType, bool flip>
fly_array select_scalar(const fly_array cond, const fly_array a,
                       const ScalarType b, const dim4& odims) {
    auto scalar = detail::scalar<ArrayType>(b);
    auto out    = createSelectNode<ArrayType, flip>(
        getArray<char>(cond), getArray<ArrayType>(a), scalar, odims);
    return getHandle(out);
}

template<typename ScalarType, bool IsScalarTrueOutput>
fly_err selectScalar(fly_array* out, const fly_array cond, const fly_array e,
                    const ScalarType c) {
    try {
        const ArrayInfo& einfo = getInfo(e);
        const ArrayInfo& cinfo = getInfo(cond);

        ARG_ASSERT(1, cinfo.getType() == b8);

        dim4 edims     = einfo.dims();
        dim4 cond_dims = cinfo.dims();
        dim4 odims(1);

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cond_dims[i] == edims[i] || cond_dims[i] == 1 ||
                              edims[i] == 1);
            odims[i] = std::max(cond_dims[i], edims[i]);
        }

        fly_array res;

        switch (einfo.getType()) {
            case f16:
                res = select_scalar<half, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case f32:
                res = select_scalar<float, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case f64:
                res = select_scalar<double, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case c32:
                res = select_scalar<cfloat, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case c64:
                res = select_scalar<cdouble, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case s32:
                res = select_scalar<int, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u32:
                res = select_scalar<uint, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case s16:
                res = select_scalar<short, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u16:
                res = select_scalar<ushort, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case s64:
                res = select_scalar<intl, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u64:
                res = select_scalar<uintl, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case u8:
                res = select_scalar<uchar, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            case b8:
                res = select_scalar<char, ScalarType, IsScalarTrueOutput>(
                    cond, e, c, odims);
                break;
            default: TYPE_ERROR((IsScalarTrueOutput ? 3 : 2), einfo.getType());
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_select_scalar_r(fly_array* out, const fly_array cond, const fly_array a,
                          const double b) {
    return selectScalar<double, false>(out, cond, a, b);
}

fly_err fly_select_scalar_r_long(fly_array* out, const fly_array cond,
                               const fly_array a, const long long b) {
    return selectScalar<long long, false>(out, cond, a, b);
}

fly_err fly_select_scalar_r_ulong(fly_array* out, const fly_array cond,
                                const fly_array a, const unsigned long long b) {
    return selectScalar<unsigned long long, false>(out, cond, a, b);
}

fly_err fly_select_scalar_l(fly_array* out, const fly_array cond, const double a,
                          const fly_array b) {
    return selectScalar<double, true>(out, cond, b, a);
}

fly_err fly_select_scalar_l_long(fly_array* out, const fly_array cond,
                               const long long a, const fly_array b) {
    return selectScalar<long long, true>(out, cond, b, a);
}

fly_err fly_select_scalar_l_ulong(fly_array* out, const fly_array cond,
                                const unsigned long long a, const fly_array b) {
    return selectScalar<unsigned long long, true>(out, cond, b, a);
}
