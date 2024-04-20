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
#include <handle.hpp>
#include <set.hpp>
#include <fly/algorithm.h>
#include <fly/defines.h>
#include <complex>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array setUnique(const fly_array in, const bool is_sorted) {
    return getHandle(setUnique(getArray<T>(in), is_sorted));
}

fly_err fly_set_unique(fly_array* out, const fly_array in, const bool is_sorted) {
    try {
        const ArrayInfo& in_info = getInfo(in);

        if (in_info.isEmpty() || in_info.isScalar()) {
            return fly_retain_array(out, in);
        }

        ARG_ASSERT(1, in_info.isVector());

        fly_dtype type = in_info.getType();

        fly_array res;
        switch (type) {
            case f32: res = setUnique<float>(in, is_sorted); break;
            case f64: res = setUnique<double>(in, is_sorted); break;
            case s32: res = setUnique<int>(in, is_sorted); break;
            case u32: res = setUnique<uint>(in, is_sorted); break;
            case s16: res = setUnique<short>(in, is_sorted); break;
            case u16: res = setUnique<ushort>(in, is_sorted); break;
            case s64: res = setUnique<intl>(in, is_sorted); break;
            case u64: res = setUnique<uintl>(in, is_sorted); break;
            case b8: res = setUnique<char>(in, is_sorted); break;
            case u8: res = setUnique<uchar>(in, is_sorted); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline fly_array setUnion(const fly_array first, const fly_array second,
                                const bool is_unique) {
    return getHandle(
        setUnion(getArray<T>(first), getArray<T>(second), is_unique));
}

fly_err fly_set_union(fly_array* out, const fly_array first, const fly_array second,
                    const bool is_unique) {
    try {
        const ArrayInfo& first_info  = getInfo(first);
        const ArrayInfo& second_info = getInfo(second);

        fly_array res;
        if (first_info.isEmpty()) { return fly_retain_array(out, second); }

        if (second_info.isEmpty()) { return fly_retain_array(out, first); }

        ARG_ASSERT(1, (first_info.isVector() || first_info.isScalar()));
        ARG_ASSERT(1, (second_info.isVector() || second_info.isScalar()));

        fly_dtype first_type  = first_info.getType();
        fly_dtype second_type = second_info.getType();

        ARG_ASSERT(1, first_type == second_type);

        switch (first_type) {
            case f32: res = setUnion<float>(first, second, is_unique); break;
            case f64: res = setUnion<double>(first, second, is_unique); break;
            case s32: res = setUnion<int>(first, second, is_unique); break;
            case u32: res = setUnion<uint>(first, second, is_unique); break;
            case s16: res = setUnion<short>(first, second, is_unique); break;
            case u16: res = setUnion<ushort>(first, second, is_unique); break;
            case s64: res = setUnion<intl>(first, second, is_unique); break;
            case u64: res = setUnion<uintl>(first, second, is_unique); break;
            case b8: res = setUnion<char>(first, second, is_unique); break;
            case u8: res = setUnion<uchar>(first, second, is_unique); break;
            default: TYPE_ERROR(1, first_type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline fly_array setIntersect(const fly_array first, const fly_array second,
                                    const bool is_unique) {
    return getHandle(
        setIntersect(getArray<T>(first), getArray<T>(second), is_unique));
}

fly_err fly_set_intersect(fly_array* out, const fly_array first,
                        const fly_array second, const bool is_unique) {
    try {
        const ArrayInfo& first_info  = getInfo(first);
        const ArrayInfo& second_info = getInfo(second);

        // TODO(umar): fix for set intersect from union
        if (first_info.isEmpty()) { return fly_retain_array(out, first); }

        if (second_info.isEmpty()) { return fly_retain_array(out, second); }

        ARG_ASSERT(1, (first_info.isVector() || first_info.isScalar()));
        ARG_ASSERT(1, (second_info.isVector() || second_info.isScalar()));

        fly_dtype first_type  = first_info.getType();
        fly_dtype second_type = second_info.getType();

        ARG_ASSERT(1, first_type == second_type);

        fly_array res;
        switch (first_type) {
            case f32:
                res = setIntersect<float>(first, second, is_unique);
                break;
            case f64:
                res = setIntersect<double>(first, second, is_unique);
                break;
            case s32: res = setIntersect<int>(first, second, is_unique); break;
            case u32: res = setIntersect<uint>(first, second, is_unique); break;
            case s16:
                res = setIntersect<short>(first, second, is_unique);
                break;
            case u16:
                res = setIntersect<ushort>(first, second, is_unique);
                break;
            case s64: res = setIntersect<intl>(first, second, is_unique); break;
            case u64:
                res = setIntersect<uintl>(first, second, is_unique);
                break;
            case b8: res = setIntersect<char>(first, second, is_unique); break;
            case u8: res = setIntersect<uchar>(first, second, is_unique); break;
            default: TYPE_ERROR(1, first_type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
