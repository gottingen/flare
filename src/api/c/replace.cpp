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
#include <common/traits.hpp>
#include <handle.hpp>
#include <implicit.hpp>
#include <optypes.hpp>
#include <fly/arith.h>
#include <fly/array.h>
#include <fly/data.h>
#include <fly/defines.h>

#include <select.hpp>

using fly::dim4;
using flare::getCopyOnWriteArray;
using flare::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::select_scalar;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
void replace(fly_array a, const fly_array cond, const fly_array b) {
    select(getCopyOnWriteArray<T>(a), getArray<char>(cond), getArray<T>(a),
           getArray<T>(b));
}

fly_err fly_replace(fly_array a, const fly_array cond, const fly_array b) {
    try {
        const ArrayInfo& ainfo = getInfo(a);
        const ArrayInfo& binfo = getInfo(b);
        const ArrayInfo& cinfo = getInfo(cond);

        if (cinfo.ndims() == 0) { return FLY_SUCCESS; }

        ARG_ASSERT(2, ainfo.getType() == binfo.getType());
        ARG_ASSERT(1, cinfo.getType() == b8);

        DIM_ASSERT(1, ainfo.ndims() >= binfo.ndims());
        DIM_ASSERT(1, cinfo.ndims() == std::min(ainfo.ndims(), binfo.ndims()));

        dim4 adims = ainfo.dims();
        dim4 bdims = binfo.dims();
        dim4 cdims = cinfo.dims();

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cdims[i] == std::min(adims[i], bdims[i]));
            DIM_ASSERT(2, adims[i] == bdims[i] || bdims[i] == 1);
        }

        switch (ainfo.getType()) {
            case f16: replace<half>(a, cond, b); break;
            case f32: replace<float>(a, cond, b); break;
            case f64: replace<double>(a, cond, b); break;
            case c32: replace<cfloat>(a, cond, b); break;
            case c64: replace<cdouble>(a, cond, b); break;
            case s32: replace<int>(a, cond, b); break;
            case u32: replace<uint>(a, cond, b); break;
            case s64: replace<intl>(a, cond, b); break;
            case u64: replace<uintl>(a, cond, b); break;
            case s16: replace<short>(a, cond, b); break;
            case u16: replace<ushort>(a, cond, b); break;
            case u8: replace<uchar>(a, cond, b); break;
            case b8: replace<char>(a, cond, b); break;
            default: TYPE_ERROR(2, ainfo.getType());
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<typename ArrayType, typename ScalarType>
void replace_scalar(fly_array a, const fly_array cond, const ScalarType& b) {
    select_scalar<ArrayType, false>(
        getCopyOnWriteArray<ArrayType>(a), getArray<char>(cond),
        getArray<ArrayType>(a), detail::scalar<ArrayType>(b));
}

template<typename ScalarType>
fly_err replaceScalar(fly_array a, const fly_array cond, const ScalarType b) {
    try {
        const ArrayInfo& ainfo = getInfo(a);
        const ArrayInfo& cinfo = getInfo(cond);

        ARG_ASSERT(1, cinfo.getType() == b8);
        DIM_ASSERT(1, cinfo.ndims() == ainfo.ndims());

        dim4 adims = ainfo.dims();
        dim4 cdims = cinfo.dims();

        for (int i = 0; i < 4; i++) { DIM_ASSERT(1, cdims[i] == adims[i]); }

        switch (ainfo.getType()) {
            case f16: replace_scalar<half>(a, cond, b); break;
            case f32: replace_scalar<float>(a, cond, b); break;
            case f64: replace_scalar<double>(a, cond, b); break;
            case c32: replace_scalar<cfloat>(a, cond, b); break;
            case c64: replace_scalar<cdouble>(a, cond, b); break;
            case s32: replace_scalar<int>(a, cond, b); break;
            case u32: replace_scalar<uint>(a, cond, b); break;
            case s64: replace_scalar<intl>(a, cond, b); break;
            case u64: replace_scalar<uintl>(a, cond, b); break;
            case s16: replace_scalar<short>(a, cond, b); break;
            case u16: replace_scalar<ushort>(a, cond, b); break;
            case u8: replace_scalar<uchar>(a, cond, b); break;
            case b8: replace_scalar<char>(a, cond, b); break;
            default: TYPE_ERROR(2, ainfo.getType());
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_replace_scalar(fly_array a, const fly_array cond, const double b) {
    return replaceScalar(a, cond, b);
}

fly_err fly_replace_scalar_long(fly_array a, const fly_array cond,
                              const long long b) {
    return replaceScalar(a, cond, b);
}

fly_err fly_replace_scalar_ulong(fly_array a, const fly_array cond,
                               const unsigned long long b) {
    return replaceScalar(a, cond, b);
}
